"""Plan-10 V1 (Option B) two-tower contrastive training.

Single-GPU:    python src/training/train_plan10.py [args]
Multi-GPU:     accelerate launch src/training/train_plan10.py --gather [args]

Architecture: two independent ContrastiveQwen2VL towers (query/target) trained
contrastively on FACap triplets. Both towers are speechQwen2VL + LoRA; the
target tower is fed the locked prompt "Describe this image in detail." (see
src/training/two_tower_model.py).

Differences vs train_plan5.py:
  - Uses TwoTowerSeparateBackbones (parent forward returns (q_emb, t_emb))
  - Dataset loads target images (load_target=True); target embeddings encoded
    on the fly, not looked up from a precomputed FashionCLIP cache
  - Optimizer adds the target tower's LoRA + projection head param groups
    (5 groups total: q_lora, q_proj, t_lora, t_proj, logit_scale)
  - End-of-epoch gallery refresh via encode_gallery_with_tower (distributed)
  - --first-eval-step CLI flag (default 5): triggers the existing eval
    pipeline once after step N; catches OOM / sharding bugs ~2 min into the
    run instead of ~25 min
  - Step-1 grad-receipt assertion: defensive check on both towers' lora_B
    nonzero-grad receipt (catches subtle DDP / requires_grad regressions
    early, even though Option B does not have the PEFT set_adapter footgun
    that Option A would).

See Documentation/Plan_10_20260510.md for the full design and rationale.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

try:
    import setproctitle
    setproctitle.setproctitle("CIR")
except ImportError:
    pass

import numpy as np
import torch
import torch.utils.data as tud
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed

from src.data.contrastive_dataset import FacapContrastiveDataset, contrastive_collate
from src.data.facap_dataset import FacapDataset
from src.training.two_tower_model import (
    TwoTowerSeparateBackbones,
    TwoTowerSharedBackbone,
    TARGET_PROMPT,
)
from src.training.loss import SymmetricInfoNCE
from src.training.online_eval import (
    encode_gallery_with_tower,
    make_gallery_db,
    run_dev_loss_two_tower,
    run_dev_probe,
    run_retrieval_eval,
)
from src.training.target_cache import _gallery_ids_and_paths
from src.baseline.eval import format_metrics_table


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plan-10 V1 two-tower training")
    p.add_argument("--arch", choices=["separate", "shared"], default="separate",
                   help="Two-tower architecture variant. 'separate' = Option B "
                        "(two independent ContrastiveQwen2VL towers, default for "
                        "back-compat). 'shared' = Option A (one shared backbone + "
                        "two PEFT LoRA adapters; gradient checkpointing OFF — see "
                        "Progress_11 §Appendix C).")
    p.add_argument("--run-dir", type=Path,
                   default=Path("runs_local_plan10/run_default"),
                   help="Output directory. On server10 use runs_local_plan10/, "
                        "on server6 use runs/plan10/. See Plan_10 §10.4.")
    p.add_argument("--d-target", type=int, default=512,
                   help="Projection head output dim (V1 default 512)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Per-GPU training batch size (Option B default 8 on A6000)")
    p.add_argument("--gallery-batch-size", type=int, default=8,
                   help="Per-GPU batch size for end-of-epoch gallery refresh "
                        "(can be larger than train bs since no grads are tracked)")
    p.add_argument("--lr", type=float, default=2e-5, help="LoRA LR (both towers)")
    p.add_argument("--lr-proj", type=float, default=1e-4, help="Projection head LR")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-epochs", type=int, default=18)
    p.add_argument("--eval-epochs", type=float, default=0.5,
                   help="Run dev eval every this many epochs (default 0.5)")
    p.add_argument("--first-eval-step", type=int, default=5,
                   help="Trigger the existing eval pipeline once after step N "
                        "(default 5). Catches OOM / sharding bugs early without a "
                        "parallel smoke harness. Set 0 to disable.")
    p.add_argument("--dev-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-mod-len", type=int, default=512)
    p.add_argument("--gather", action="store_true",
                   help="all_gather cross-GPU negatives (multi-GPU mode)")
    p.add_argument("--lr-schedule", choices=["none", "cosine"], default="none",
                   help="LR schedule. Default 'none' per Plan-7 finding.")
    p.add_argument("--resume-from", type=Path, default=None,
                   help="ckpt_epochN/ dir to resume from")
    p.add_argument("--start-epoch", type=int, default=0,
                   help="0-based epoch index when resuming")
    p.add_argument("--wandb-project", default="fashion-retrieval-agent")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step-1 grad-receipt assertion (Plan-10 §10.1.b)
# ---------------------------------------------------------------------------

def _check_grad_receipt(unwrapped, accelerator) -> None:
    """Assert both towers' lora_B params received gradient after step 1.

    PEFT init: lora_A is Kaiming (nonzero), lora_B is zero. After the
    first backward, dloss/dA = (...) @ B.T = 0 exactly. So at step 1
    we can only check lora_B for nonzero grad (lora_A is correctly zero
    by design).

    This is overkill for Option B (no adapter switching, no requires_grad
    footgun), but cheap and serves as a sanity check on DDP grad flow,
    optimizer wiring, and the parent forward path's correctness.
    """
    if not accelerator.is_main_process:
        return

    def _adapter_for(name: str) -> str | None:
        # Option A naming (shared backbone, PEFT adapters "query"/"target")
        if ".query." in name:
            return "query"
        if ".target." in name:
            return "target"
        # Option B naming (separate ContrastiveQwen2VL instances under
        # query_tower / target_tower module prefixes, PEFT adapter "default")
        if name.startswith("query_tower.") or ".query_tower." in name:
            return "query"
        if name.startswith("target_tower.") or ".target_tower." in name:
            return "target"
        return None

    expected = {"query", "target"}
    by_adapter: dict[str, dict[str, list]] = {
        a: {"lora_A": [], "lora_B": []} for a in expected
    }
    for n, p in unwrapped.named_parameters():
        if "lora_" not in n:
            continue
        adapter = _adapter_for(n)
        if adapter is None:
            continue
        kind = "lora_A" if "lora_A" in n else "lora_B" if "lora_B" in n else None
        if kind is None:
            continue
        by_adapter[adapter][kind].append((n, p))

    for adapter in expected:
        a_count = len(by_adapter[adapter]["lora_A"])
        b_count = len(by_adapter[adapter]["lora_B"])
        assert a_count > 0 and b_count > 0, (
            f"[Plan-10 grad-receipt] adapter '{adapter}' has no lora_A/lora_B "
            f"params (A={a_count}, B={b_count}). Was add_adapter / "
            f"get_peft_model called?"
        )
        for n, p in by_adapter[adapter]["lora_A"] + by_adapter[adapter]["lora_B"]:
            assert p.requires_grad, (
                f"[Plan-10 grad-receipt] {n} has requires_grad=False at step 1. "
                f"PEFT set_adapter footgun (see Plan_10 §4.3)?"
            )
        b_with_grad = sum(
            1 for _, p in by_adapter[adapter]["lora_B"]
            if p.grad is not None and p.grad.abs().sum().item() > 0
        )
        # Relaxed: PEFT injects LoRA on every module matching target_modules.
        # speechQwen2VL's vision/audio sub-modules also match q/k/v/o_proj but
        # may not participate in the image+text forward (frozen vision encoder,
        # unused audio path). What matters is the LM layers receive gradient
        # — proven by ANY nonzero lora_B grad in this adapter. The active
        # fraction is reported for observability.
        assert b_with_grad > 0, (
            f"[Plan-10 grad-receipt] adapter '{adapter}': 0/{b_count} lora_B "
            f"params received gradient. Did this adapter participate in the "
            f"forward graph at all?"
        )
        by_adapter[adapter]["_active"] = b_with_grad

    accelerator.print(
        f"[Plan-10 grad-receipt OK] "
        + ", ".join(
            f"{a}: {by_adapter[a]['_active']}/{len(by_adapter[a]['lora_B'])} "
            f"lora_B got grad"
            for a in expected
        )
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    log_with = [] if args.no_wandb else ["wandb"]
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with=log_with, kwargs_handlers=[ddp_kwargs])
    set_seed(args.seed)

    accelerator.print(f"=== Plan-10 V1 (Option B) two-tower training ===")
    accelerator.print(f"run_dir: {args.run_dir}")
    accelerator.print(
        f"gather: {args.gather} | world_size: {accelerator.num_processes} | "
        f"bs/GPU: {args.batch_size} | eff. neg: "
        f"{args.batch_size * accelerator.num_processes}"
    )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    base_ds = FacapDataset(category="dress", split="train")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    dev_slice_json = args.run_dir / "dev_slice.json"

    train_ds = FacapContrastiveDataset(
        base=base_ds,
        dev_seed=args.dev_seed,
        dev_slice_json=dev_slice_json if accelerator.is_main_process else None,
        load_target=True,        # Plan-10: yield tgt_image PIL per item
    )
    accelerator.print(train_ds.summary())

    train_loader = tud.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=contrastive_collate,
        drop_last=True,
        pin_memory=False,
    )

    # ------------------------------------------------------------------
    # Canonical gallery ids/paths (same source as target_cache.build_target_cache)
    # ------------------------------------------------------------------
    pairs = _gallery_ids_and_paths(base_ds)
    gallery_ids: list[str] = [gid for gid, _ in pairs]
    gallery_paths: list[Path] = [p for _, p in pairs]
    # Integer index for each gallery ID — multi-positive InfoNCE mask
    tid_to_idx: dict[str, int] = {gid: i for i, gid in enumerate(gallery_ids)}
    accelerator.print(
        f"Gallery: {len(gallery_ids)} images (canonical order from "
        f"target_cache._gallery_ids_and_paths)"
    )

    # ------------------------------------------------------------------
    # Model + loss
    # ------------------------------------------------------------------
    if args.arch == "shared":
        model = TwoTowerSharedBackbone(
            d_target=args.d_target,
            device_map=f"cuda:{accelerator.local_process_index}",
        )
    else:  # "separate" (default — Option B)
        model = TwoTowerSeparateBackbones(
            d_target=args.d_target,
            device_map=f"cuda:{accelerator.local_process_index}",
        )
    loss_fn = SymmetricInfoNCE()
    device = accelerator.device

    # ------------------------------------------------------------------
    # Optimizer: 5 param groups (q_lora, q_proj, t_lora, t_proj, logit_scale)
    # ------------------------------------------------------------------
    q_lora, q_proj, t_lora, t_proj = model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        [
            {"params": q_lora,                "lr": args.lr},
            {"params": q_proj,                "lr": args.lr_proj},
            {"params": t_lora,                "lr": args.lr},
            {"params": t_proj,                "lr": args.lr_proj},
            {"params": [loss_fn.logit_scale], "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    # ------------------------------------------------------------------
    # Accelerate prepare
    # ------------------------------------------------------------------
    model, loss_fn, optimizer, train_loader = accelerator.prepare(
        model, loss_fn, optimizer, train_loader
    )

    # ------------------------------------------------------------------
    # LR scheduler (after prepare so len(train_loader) is final)
    # ------------------------------------------------------------------
    _total_steps = len(train_loader) * args.max_epochs
    if args.lr_schedule == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=_total_steps, eta_min=0)
    else:
        scheduler = None

    # ------------------------------------------------------------------
    # Resume from checkpoint (NAS-shared path; all ranks load)
    # ------------------------------------------------------------------
    if args.resume_from is not None:
        from peft import load_peft_weights, set_peft_model_state_dict
        ckpt = args.resume_from
        unwrapped_for_load = accelerator.unwrap_model(model)
        for tower_name in ("query_tower", "target_tower"):
            tower = getattr(unwrapped_for_load, tower_name)
            peft_weights = load_peft_weights(str(ckpt / tower_name))
            set_peft_model_state_dict(tower.vlm, peft_weights)
            tower.proj.load_state_dict(
                torch.load(str(ckpt / f"head_{tower_name.split('_')[0]}.pt"),
                           map_location=device)
            )
        ls_pt = ckpt / "logit_scale.pt"
        if ls_pt.exists():
            accelerator.unwrap_model(loss_fn).load_state_dict(
                torch.load(str(ls_pt), map_location=device)
            )
        accelerator.print(f"Resumed from checkpoint: {ckpt}")
        if scheduler is not None:
            resume_steps = args.start_epoch * len(train_loader)
            for _ in range(resume_steps):
                scheduler.step()

    # ------------------------------------------------------------------
    # W&B init
    # ------------------------------------------------------------------
    if not args.no_wandb and accelerator.is_main_process:
        def _gpu_class() -> str:
            try:
                name = torch.cuda.get_device_name(0).lower()
            except Exception:
                return "gpu"
            # both "RTX 6000 Ada" and "RTX A6000" are treated as "A6000-class"
            for substring, label in [
                ("rtx 6000 ada", "A6000"),
                ("a6000",        "A6000"),
                ("a100",         "A100"),
                ("h100",         "H100"),
                ("rtx 3090",     "3090"),
                ("rtx 4090",     "4090"),
                ("l40",          "L40"),
            ]:
                if substring in name:
                    return label
            return "gpu"
        date_str = time.strftime("%Y%m%d")
        arch_label = "shared" if args.arch == "shared" else "separate"
        option_tag = "option-a" if args.arch == "shared" else "option-b"
        run_name = args.wandb_run_name or (
            f"plan10/v1_{arch_label}_bs{args.batch_size}"
            f"_{accelerator.num_processes}x{_gpu_class()}_{date_str}"
        )
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {
                "name": run_name,
                "tags": ["phase-b", "plan10", "two-tower", option_tag],
            }},
        )

    # ------------------------------------------------------------------
    # Initial gallery encoding (BEFORE training so first eval can run)
    # ------------------------------------------------------------------
    accelerator.print("=== Initial gallery encoding (step 0) ===")
    unwrapped_model = accelerator.unwrap_model(model)
    initial_embs = encode_gallery_with_tower(
        unwrapped_model,
        gallery_ids=gallery_ids,
        gallery_paths=gallery_paths,
        batch_size=args.gallery_batch_size,
        accelerator=accelerator,
        out_dir=args.run_dir if accelerator.is_main_process else None,
        epoch_tag=0,
    )
    gallery_lookup: dict[str, np.ndarray] = {
        gid: initial_embs[i] for i, gid in enumerate(gallery_ids)
    }
    gallery_db = make_gallery_db(initial_embs, gallery_ids)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.max_epochs

    def _should_eval(step: int) -> bool:
        # First-eval trigger (catches OOM / sharding bugs ~2 min into the run)
        if args.first_eval_step > 0 and step == args.first_eval_step:
            return True
        if step == 0:
            return True
        prev = (step - 1) / steps_per_epoch
        curr = step / steps_per_epoch
        return math.floor(curr / args.eval_epochs) > math.floor(prev / args.eval_epochs)

    def _run_eval(step: int, epoch_frac: float) -> None:
        """Full dev + headline eval; main-process only. Reads the current
        `gallery_db` binding from the enclosing scope so end-of-epoch
        refreshes are picked up.
        """
        if not accelerator.is_main_process:
            return
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_loss_fn = accelerator.unwrap_model(loss_fn)

        dev_loss = run_dev_loss_two_tower(
            unwrapped_model, unwrapped_loss_fn,
            train_ds.dev_items, tid_to_idx, base_ds, device,
        )
        probe = run_dev_probe(
            unwrapped_model, train_ds.dev_items, gallery_db, base_ds,
            train_ds.train_mod_texts, seed=args.dev_seed,
        )
        normal = probe["normal"]
        stripped = probe["mod_stripped"]
        shuffled = probe["mod_shuffled"]
        sensitivity_gap = normal.recall[10] - stripped.recall[10]
        headline_result = run_retrieval_eval(
            unwrapped_model, train_ds.headline_items, gallery_db, base_ds,
        )

        metric_row = {
            "step": step,
            "epoch_frac": epoch_frac,
            "dev/loss":            dev_loss,
            "dev/r1_normal":       normal.recall[1],
            "dev/r5_normal":       normal.recall[5],
            "dev/r10_normal":      normal.recall[10],
            "dev/r50_normal":      normal.recall[50],
            "dev/median_rank":     normal.median_rank,
            "dev/r10_stripped":    stripped.recall[10],
            "dev/r10_shuffled":    shuffled.recall[10],
            "dev/sensitivity_gap": sensitivity_gap,
            "headline/r1":          headline_result.recall[1],
            "headline/r5":          headline_result.recall[5],
            "headline/r10":         headline_result.recall[10],
            "headline/r50":         headline_result.recall[50],
            "headline/median_rank": headline_result.median_rank,
        }
        all_metrics.append(metric_row)
        accelerator.log(
            {k: v for k, v in metric_row.items() if k != "step"},
            step=step,
        )
        accelerator.print(
            f"\n[DEV @step {step}]\n"
            f"  dev/loss={dev_loss:.4f}\n"
            + format_metrics_table(probe["normal"])
            + f"\n  sensitivity gap (normal−stripped): {sensitivity_gap:+.4f}"
            + f"\n=== Headline ===\n"
            + format_metrics_table(headline_result)
        )
        model.train()

    accelerator.print(
        f"steps/epoch={steps_per_epoch}  eval_every={args.eval_epochs:.2f} epoch  "
        f"first_eval_step={args.first_eval_step}  total_steps={total_steps}"
    )

    global_step = args.start_epoch * steps_per_epoch
    all_metrics: list[dict] = []

    for epoch in range(args.start_epoch, args.max_epochs):
        model.train()
        for batch in train_loader:
            cand_images = batch["cand_images"]
            mod_texts = batch["mod_texts"]
            tgt_images = batch["tgt_images"]
            target_ids: list[str] = batch["target_ids"]

            # DDP-safe forward: parent module's forward returns (q_emb, t_emb)
            q_embs, t_embs = model(cand_images, mod_texts, tgt_images)

            tid_tensor = torch.tensor(
                [tid_to_idx[tid] for tid in target_ids],
                dtype=torch.int64, device=device,
            )
            loss = loss_fn(q_embs, t_embs, gather=args.gather, target_ids=tid_tensor)

            if not torch.isfinite(loss):
                accelerator.print(f"NaN/Inf loss at step {global_step}! Stopping.")
                accelerator.end_training()
                return

            optimizer.zero_grad()
            accelerator.backward(loss)

            grad_norm = None
            if global_step % 20 == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float("inf")
                ).item()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            accelerator.unwrap_model(loss_fn).clamp_logit_scale()

            # ----------------------------------------------------------
            # Step-1 grad-receipt assertion (Plan-10 §10.1.b)
            # ----------------------------------------------------------
            if global_step == 0:
                _check_grad_receipt(accelerator.unwrap_model(model), accelerator)

            # ----------------------------------------------------------
            # Logging
            # ----------------------------------------------------------
            if global_step % 20 == 0:
                unwrapped_loss = accelerator.unwrap_model(loss_fn)
                log_dict = {
                    "train/loss":        loss.item(),
                    "train/logit_scale": unwrapped_loss.logit_scale.exp().item(),
                    "train/epoch":       global_step / steps_per_epoch,
                    "train/lr_lora_q":   optimizer.param_groups[0]["lr"],
                    "train/lr_proj_q":   optimizer.param_groups[1]["lr"],
                    "train/lr_lora_t":   optimizer.param_groups[2]["lr"],
                    "train/lr_proj_t":   optimizer.param_groups[3]["lr"],
                    "train/grad_norm":   grad_norm,
                }
                accelerator.log(log_dict, step=global_step)
                accelerator.print(
                    f"[{epoch}:{global_step}]  loss={loss.item():.4f}  "
                    f"τ_inv={unwrapped_loss.logit_scale.exp().item():.2f}"
                )

            # ----------------------------------------------------------
            # Eval (uses MOST RECENT gallery cache from initial or last epoch refresh)
            # ----------------------------------------------------------
            if _should_eval(global_step) and accelerator.is_main_process:
                _run_eval(global_step, global_step / steps_per_epoch)

            global_step += 1

        # ------------------------------------------------------------------
        # End of epoch: (a) refresh gallery (all-rank), (b) checkpoint (rank 0)
        # ------------------------------------------------------------------
        accelerator.print(f"=== End of epoch {epoch + 1}: refreshing gallery ===")
        unwrapped_model = accelerator.unwrap_model(model)
        fresh_embs = encode_gallery_with_tower(
            unwrapped_model,
            gallery_ids=gallery_ids,
            gallery_paths=gallery_paths,
            batch_size=args.gallery_batch_size,
            accelerator=accelerator,
            out_dir=args.run_dir if accelerator.is_main_process else None,
            epoch_tag=epoch + 1,
        )
        # Overwrite in-memory lookup + gallery_db (all ranks rebuild)
        gallery_lookup = {gid: fresh_embs[i] for i, gid in enumerate(gallery_ids)}
        gallery_db = make_gallery_db(fresh_embs, gallery_ids)

        if accelerator.is_main_process:
            ckpt_dir = args.run_dir / f"ckpt_epoch{epoch + 1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            if args.arch == "shared":
                # Option A: one PeftModel with two adapters under
                # unwrapped_model.vlm; save_pretrained dumps BOTH adapters
                # into subdirs adapter_name/ inside the destination.
                unwrapped_model.vlm.save_pretrained(
                    str(ckpt_dir / "shared_backbone")
                )
                torch.save(
                    unwrapped_model.head_query.state_dict(),
                    ckpt_dir / "head_query.pt",
                )
                torch.save(
                    unwrapped_model.head_target.state_dict(),
                    ckpt_dir / "head_target.pt",
                )
            else:
                # Option B: two ContrastiveQwen2VL instances as attributes.
                for tower_name in ("query_tower", "target_tower"):
                    tower = getattr(unwrapped_model, tower_name)
                    tower.vlm.save_pretrained(str(ckpt_dir / tower_name))
                    torch.save(
                        tower.proj.state_dict(),
                        ckpt_dir / f"head_{tower_name.split('_')[0]}.pt",
                    )
            torch.save(
                accelerator.unwrap_model(loss_fn).state_dict(),
                ckpt_dir / "logit_scale.pt",
            )

    # ------------------------------------------------------------------
    # Final eval after the last epoch's gallery refresh.
    # The in-loop schedule misses this because the epoch-boundary trigger
    # fires at the START of the next epoch — which never happens after
    # the final epoch. Captures the post-refresh R@K against the freshest
    # gallery embeddings.
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        accelerator.print("=== Final eval (post-last-epoch gallery refresh) ===")
        _run_eval(global_step, float(args.max_epochs))

    # ------------------------------------------------------------------
    # Final metrics JSON
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        metrics_path = args.run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        accelerator.print(f"\nMetrics saved to {metrics_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
