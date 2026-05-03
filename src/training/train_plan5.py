"""Plan-5 contrastive training loop.

Single-GPU:    python src/training/train_plan5.py [args]
Multi-GPU:     accelerate launch src/training/train_plan5.py --gather [args]

Key flags:
  --profile        VRAM sweep at batch sizes 16/24/32/48/64, then exit
  --smoke          50-step plumbing test on 64 fixed triplets
  --gather         enable cross-GPU all_gather in InfoNCE (multi-GPU mode)
  --cache-dir      directory containing target_emb_cache_*.npy (default: runs/plan5)
  --encoder-id     target encoder (default: marqo-fashionclip)
  --batch-size     true contrastive batch (no grad-accum for the loss in v1)
  --run-dir        output directory for checkpoints, metrics, W&B artifacts
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as tud
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed

from src.data.contrastive_dataset import FacapContrastiveDataset, contrastive_collate
from src.data.facap_dataset import FacapDataset
from src.training.contrastive_model import ContrastiveQwen2VL
from src.training.loss import SymmetricInfoNCE
from src.training.online_eval import (
    harness_sanity,
    make_gallery_db,
    run_dev_probe,
    run_retrieval_eval,
)
from src.training.target_cache import load_target_cache, make_gallery_lookup
from src.baseline.eval import format_metrics_table


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plan-5 contrastive training")
    p.add_argument("--run-dir", type=Path, default=Path("runs/plan5/run_default"))
    p.add_argument("--cache-dir", type=Path, default=Path("runs/plan5"))
    p.add_argument("--encoder-id", default="marqo-fashionclip")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5, help="LoRA LR")
    p.add_argument("--lr-proj", type=float, default=1e-4, help="Projection head LR")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-epochs", type=int, default=3)
    p.add_argument("--eval-epochs", type=float, default=0.5,
                   help="Run dev eval every this many epochs (default 0.5)")
    p.add_argument("--dev-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-mod-len", type=int, default=512)
    p.add_argument("--gather", action="store_true",
                   help="all_gather cross-GPU negatives (multi-GPU mode)")
    p.add_argument("--smoke", action="store_true",
                   help="50-step plumbing test on 64 fixed triplets, then exit")
    p.add_argument("--profile", action="store_true",
                   help="VRAM sweep at batch sizes 2/4/6/8/12/16, then exit")
    p.add_argument("--d-target", type=int, default=None,
                   help="Override embedding dim (avoids loading cache for --profile)")
    p.add_argument("--wandb-project", default="fashion-retrieval-agent")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--resume-from", type=Path, default=None,
                   help="ckpt_epochN/ dir to resume from (loads vlm_lora + proj_head.pt)")
    p.add_argument("--start-epoch", type=int, default=0,
                   help="0-based epoch index to start from (set to N when resuming from ckpt_epochN)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# VRAM profile
# ---------------------------------------------------------------------------

def run_profile(model: ContrastiveQwen2VL, base_ds: FacapDataset) -> None:
    """Forward+backward at increasing batch sizes; print peak VRAM table."""
    import gc
    from src.training.loss import SymmetricInfoNCE

    loss_fn = SymmetricInfoNCE().to(next(model.parameters()).device)
    # Use first available images as dummy input
    dummy_item = base_ds[0]
    dummy_img = base_ds.load_image(dummy_item, "candidate")
    dummy_txt = dummy_item["modification_text"]

    print("\nVRAM profile (3090 target: ≤ 20 GB / A6000 target: ≤ 40 GB)\n" + "-" * 45)
    print(f"{'batch_size':>12}  {'peak VRAM (GB)':>16}  {'time (s)':>10}")
    print("-" * 45)

    for bs in (2, 4, 6, 8, 12, 16, 24, 32):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            imgs = [dummy_img] * bs
            txts = [dummy_txt] * bs
            dummy_t = torch.zeros(bs, model.d_target,
                                  device=next(model.parameters()).device)
            t0 = time.time()
            model.train()
            q = model(imgs, txts)
            loss = loss_fn(q, dummy_t)
            loss.backward()
            dt = time.time() - t0
            model.zero_grad()
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"{bs:>12}  {peak:>16.1f}  {dt:>10.1f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>12}  {'OOM':>16}  {'—':>10}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    print("-" * 45)
    print("Pick the largest batch_size with peak ≤ ~40 GB (leaves headroom).")
    print("Then re-run with: --batch-size <N>")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke(
    model: ContrastiveQwen2VL,
    loss_fn: SymmetricInfoNCE,
    gallery_lookup: dict[str, np.ndarray],
    base_ds: FacapDataset,
    device: torch.device,
    gather: bool,
    max_mod_len: int,
) -> None:
    """50 steps on 4 fixed distinct triplets. Loss must decrease (windowed avg, no NaN/Inf)."""
    # Pick 4 distinct items with different target_ids — identical batches produce
    # degenerate uniform logits where gradients cancel and loss stays at log(B).
    smoke_items = []
    seen_tids: set[str] = set()
    for i in range(len(base_ds)):
        it = base_ds[i]
        if it["target_id"] not in seen_tids and it["target_id"] in gallery_lookup:
            smoke_items.append(it)
            seen_tids.add(it["target_id"])
        if len(smoke_items) == 4:
            break
    if len(smoke_items) < 4:
        raise RuntimeError("Not enough distinct gallery items for smoke test.")

    smoke_imgs = [base_ds.load_image(it, "candidate") for it in smoke_items]
    smoke_txts = [it["modification_text"] for it in smoke_items]
    smoke_tids = [it["target_id"] for it in smoke_items]

    optimizer = torch.optim.AdamW(
        list(model.vlm.parameters()) + list(model.proj.parameters())
        + [loss_fn.logit_scale],
        lr=2e-5,
    )

    model.train()
    losses: list[float] = []
    print("Smoke test: 50 steps on 4 fixed distinct triplets...")
    for step in range(50):
        imgs = smoke_imgs
        txts = smoke_txts
        tids = smoke_tids

        t_embs = torch.stack(
            [torch.from_numpy(gallery_lookup[tid]) for tid in tids]
        ).to(device)

        q_embs = model(imgs, txts, max_mod_len=max_mod_len)
        loss = loss_fn(q_embs, t_embs, gather=gather)

        if not torch.isfinite(loss):
            print(f"  step {step}: NaN/Inf loss! Smoke FAILED.")
            return

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_fn.clamp_logit_scale()
        losses.append(loss.item())

        if step % 10 == 9:
            window_avg = sum(losses[-10:]) / 10
            print(
                f"  step {step + 1:3d}  loss={loss.item():.4f}  "
                f"10-step avg={window_avg:.4f}  "
                f"logit_scale={loss_fn.logit_scale.exp().item():.3f}"
            )

    # Check windowed average is decreasing
    first_half = sum(losses[:25]) / 25
    second_half = sum(losses[25:]) / 25
    passed = second_half < first_half
    print(
        f"\nSmoke {'PASSED' if passed else 'FAILED'}: "
        f"first-half avg={first_half:.4f}, second-half avg={second_half:.4f}"
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Accelerator — handles single-GPU / multi-GPU transparently
    log_with = [] if args.no_wandb else ["wandb"]
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with=log_with, kwargs_handlers=[ddp_kwargs])
    set_seed(args.seed)

    accelerator.print(f"=== Plan-5 contrastive training ===")
    accelerator.print(f"run_dir: {args.run_dir}")
    accelerator.print(f"gather: {args.gather}  | world_size: {accelerator.num_processes}")

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
    # Target embedding cache (skipped for --profile with --d-target override)
    # ------------------------------------------------------------------
    if args.profile and args.d_target is not None:
        d_target = args.d_target
        embeddings, gallery_ids, gallery_lookup, gallery_db = None, [], {}, None
        accelerator.print(f"Profile mode: d_target={d_target} (from --d-target, cache skipped)")
    else:
        embeddings, gallery_ids, d_target = load_target_cache(args.cache_dir, args.encoder_id)
        if args.d_target is not None and args.d_target != d_target:
            raise ValueError(f"--d-target {args.d_target} conflicts with cache dim {d_target}")
        gallery_lookup = make_gallery_lookup(embeddings, gallery_ids)
        gallery_db = make_gallery_db(embeddings, gallery_ids)
        # Integer index for each gallery ID — used by multi-positive InfoNCE mask
        tid_to_idx: dict[str, int] = {gid: i for i, gid in enumerate(gallery_ids)}
        accelerator.print(
            f"Gallery cache: {len(gallery_ids)} images, dim={d_target}, "
            f"{embeddings.nbytes / 1e6:.0f} MB"
        )

    # ------------------------------------------------------------------
    # Model + loss
    # ------------------------------------------------------------------
    model = ContrastiveQwen2VL(
        d_target=d_target,
        device_map=f"cuda:{accelerator.local_process_index}",
    )
    loss_fn = SymmetricInfoNCE()

    device = accelerator.device

    # ------------------------------------------------------------------
    # Profile (exits early)
    # ------------------------------------------------------------------
    if args.profile:
        run_profile(model.to(device), base_ds)
        return

    # ------------------------------------------------------------------
    # Smoke test (exits early)
    # ------------------------------------------------------------------
    if args.smoke:
        model = model.to(device)
        loss_fn = loss_fn.to(device)
        harness_sanity(gallery_db)
        run_smoke(model, loss_fn, gallery_lookup, base_ds, device,
                  args.gather, args.max_mod_len)
        return

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------
    lora_params, proj_params = model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr},
            {"params": proj_params, "lr": args.lr_proj},
            {"params": [loss_fn.logit_scale], "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    # ------------------------------------------------------------------
    # Accelerate prepare (wraps model, loss, optimizer, loader)
    # ------------------------------------------------------------------
    model, loss_fn, optimizer, train_loader = accelerator.prepare(
        model, loss_fn, optimizer, train_loader
    )

    # ------------------------------------------------------------------
    # Resume from checkpoint (all ranks load; NAS path is shared)
    # ------------------------------------------------------------------
    if args.resume_from is not None:
        from peft import load_peft_weights, set_peft_model_state_dict
        ckpt = args.resume_from
        unwrapped_for_load = accelerator.unwrap_model(model)
        peft_weights = load_peft_weights(str(ckpt / "vlm_lora"))
        set_peft_model_state_dict(unwrapped_for_load.vlm, peft_weights)
        unwrapped_for_load.proj.load_state_dict(
            torch.load(str(ckpt / "proj_head.pt"), map_location=accelerator.device)
        )
        loss_fn_pt = ckpt / "loss_fn.pt"
        if loss_fn_pt.exists():
            accelerator.unwrap_model(loss_fn).load_state_dict(
                torch.load(str(loss_fn_pt), map_location=accelerator.device)
            )
        accelerator.print(f"Resumed from checkpoint: {ckpt}")

    # ------------------------------------------------------------------
    # W&B init
    # ------------------------------------------------------------------
    if not args.no_wandb and accelerator.is_main_process:
        run_name = args.wandb_run_name or (
            f"plan5_{args.encoder_id}_lora{ContrastiveQwen2VL.__module__}"
            f"_eb{args.batch_size * accelerator.num_processes}"
        )
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"name": run_name, "tags": ["phase-b", "single-gpu", "infonce"]}},
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    steps_per_epoch = len(train_loader)
    alive_step = max(1, steps_per_epoch // 4)   # 0.25 epoch
    total_steps = steps_per_epoch * args.max_epochs

    def _should_eval(step: int) -> bool:
        """True when step crosses a multiple of eval_epochs epoch boundaries."""
        if step == 0:
            return True
        prev = (step - 1) / steps_per_epoch
        curr = step / steps_per_epoch
        return math.floor(curr / args.eval_epochs) > math.floor(prev / args.eval_epochs)

    accelerator.print(
        f"steps/epoch={steps_per_epoch}  eval_every={args.eval_epochs:.2f} epoch  "
        f"alive_check={alive_step}  total_steps={total_steps}"
    )

    global_step = args.start_epoch * steps_per_epoch
    all_metrics: list[dict] = []

    for epoch in range(args.start_epoch, args.max_epochs):
        model.train()
        for batch in train_loader:
            cand_images: list = batch["cand_images"]
            mod_texts: list[str] = batch["mod_texts"]
            target_ids: list[str] = batch["target_ids"]

            # Look up frozen target embeddings
            t_embs = torch.stack(
                [torch.from_numpy(gallery_lookup[tid]) for tid in target_ids]
            ).to(device)

            # Forward + loss
            q_embs = model(cand_images, mod_texts, max_mod_len=args.max_mod_len)
            tid_tensor = torch.tensor(
                [tid_to_idx[tid] for tid in target_ids], dtype=torch.int64, device=device
            )
            loss = loss_fn(q_embs, t_embs, gather=args.gather, target_ids=tid_tensor)

            # Hard stop on numerical failure
            if not torch.isfinite(loss):
                accelerator.print(f"NaN/Inf loss at step {global_step}! Stopping.")
                accelerator.end_training()
                return

            optimizer.zero_grad()
            accelerator.backward(loss)

            # Grad norm before step (only compute when we'll log it)
            grad_norm = None
            if global_step % 20 == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float("inf")
                ).item()

            optimizer.step()

            # Clamp logit_scale after every step (all ranks — clamp modifies .data
            # directly, bypassing DDP sync, so every rank must clamp independently)
            accelerator.unwrap_model(loss_fn).clamp_logit_scale()

            # ----------------------------------------------------------
            # Logging
            # ----------------------------------------------------------
            if global_step % 20 == 0:
                unwrapped_loss = accelerator.unwrap_model(loss_fn)
                log_dict = {
                    "train/loss":       loss.item(),
                    "train/logit_scale": unwrapped_loss.logit_scale.exp().item(),
                    "train/epoch":      global_step / steps_per_epoch,
                    "train/lr_lora":    optimizer.param_groups[0]["lr"],
                    "train/lr_proj":    optimizer.param_groups[1]["lr"],
                    "train/grad_norm":  grad_norm,
                }
                accelerator.log(log_dict, step=global_step)
                accelerator.print(
                    f"[{epoch}:{global_step}]  "
                    f"loss={loss.item():.4f}  "
                    f"τ_inv={unwrapped_loss.logit_scale.exp().item():.2f}"
                )

            # ----------------------------------------------------------
            # Dev + headline eval every eval_epochs
            # Both run together so every checkpoint has aligned metrics.
            # ----------------------------------------------------------
            if _should_eval(global_step) and accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)

                # Dev eval (500 queries, 3 conditions)
                probe = run_dev_probe(
                    unwrapped_model,
                    train_ds.dev_items,
                    gallery_db,
                    base_ds,
                    train_ds.train_mod_texts,
                    seed=args.dev_seed,
                )
                normal   = probe["normal"]
                stripped = probe["mod_stripped"]
                shuffled = probe["mod_shuffled"]
                sensitivity_gap = normal.recall[10] - stripped.recall[10]

                # Headline eval (1000 queries, authoritative)
                headline_result = run_retrieval_eval(
                    unwrapped_model,
                    train_ds.headline_items,
                    gallery_db,
                    base_ds,
                )

                metric_row = {
                    "step": global_step,
                    "epoch_frac": global_step / steps_per_epoch,
                    "dev/r1_normal":      normal.recall[1],
                    "dev/r5_normal":      normal.recall[5],
                    "dev/r10_normal":     normal.recall[10],
                    "dev/r50_normal":     normal.recall[50],
                    "dev/median_rank":    normal.median_rank,
                    "dev/r10_stripped":   stripped.recall[10],
                    "dev/r10_shuffled":   shuffled.recall[10],
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
                    step=global_step,
                )
                accelerator.print(
                    f"\n[DEV @step {global_step}]\n"
                    + format_metrics_table(probe["normal"])
                    + f"\n  sensitivity gap (normal−stripped): {sensitivity_gap:+.4f}"
                    + f"\n=== Headline ===\n"
                    + format_metrics_table(headline_result)
                )

                # Warning checks
                if global_step >= alive_step:
                    if normal.recall[10] < 0.20:
                        accelerator.print(
                            f"[WARNING] Alive bar: dev R@10={normal.recall[10]:.4f} < 0.20 "
                            f"at {global_step} steps (0.25 epoch). "
                            f"Check loss curve before stopping manually."
                        )
                    if sensitivity_gap <= 0:
                        accelerator.print(
                            f"[WARNING] Sensitivity gap ≤ 0 at {global_step} steps. "
                            f"Model may be doing visual-NN retrieval. "
                            f"Inspect mod-stripped vs normal trajectory."
                        )

                model.train()

            global_step += 1

        # ------------------------------------------------------------------
        # End of epoch: checkpoint only (eval already ran at last _should_eval)
        # ------------------------------------------------------------------
        if accelerator.is_main_process:

            # Save checkpoint
            ckpt_dir = args.run_dir / f"ckpt_epoch{epoch + 1}"
            unwrapped_model.vlm.save_pretrained(str(ckpt_dir / "vlm_lora"))
            torch.save(
                unwrapped_model.proj.state_dict(),
                ckpt_dir / "proj_head.pt",
            )
            torch.save(
                accelerator.unwrap_model(loss_fn).state_dict(),
                ckpt_dir / "loss_fn.pt",
            )

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
