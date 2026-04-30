"""VLM caption generators for the text-modification retrieval baseline.

Four backends behind one interface so the orchestrator never branches:

    MockVLMCaptioner       — local, deterministic, no model. For plumbing tests.
    OracleCaptioner        — returns the ground-truth target caption.
                             If retrieval doesn't get Recall@1 ~= 1.0 with this,
                             the bug is in encoder/index/retrieve/eval, not VLM.
    Qwen2VLCaptioner       — vanilla `Qwen/Qwen2-VL-7B-Instruct` (text-only).
                             Reference baseline number, server-only.
    SpeechQwen2VLCaptioner — Stage-1+Stage-2 speechQwen2VL in text-only mode.
                             User's headline baseline (preserves Stage 1->2 narrative),
                             server-only.

Both real backends share the prompt template + generation kwargs declared as
class constants. They check VRAM up front and raise a clear "server-only"
error on machines that can't host Qwen2-VL-7B at bf16 (~14-15 GB).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image

# Both real backends need >= this many GB of total VRAM to run Qwen2-VL-7B at bf16.
MIN_VRAM_GB_FOR_QWEN2VL_7B = 14.0


def _check_can_host_qwen2vl_7b(model_label: str) -> None:
    """Raise RuntimeError if this machine can't hold Qwen2-VL-7B at bf16."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"server-only: {model_label} needs CUDA; this machine has no GPU"
        )
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gb < MIN_VRAM_GB_FOR_QWEN2VL_7B:
        raise RuntimeError(
            f"server-only: {model_label} needs >= {MIN_VRAM_GB_FOR_QWEN2VL_7B} GB VRAM "
            f"at bf16; this machine has {total_gb:.1f} GB. "
            f"Run on the GPU server."
        )


class VLMCaptioner(ABC):
    """All backends consume the full FacapDataset item dict.

    Item keys: candidate_image_path, modification_text, target_image_path,
               target_caption, target_id, candidate_id.
    """

    @abstractmethod
    def caption(self, item: dict[str, Any]) -> str: ...


class MockVLMCaptioner(VLMCaptioner):
    """Templates the modification text into a caption-shaped string.

    Deliberately weak: the point is to verify that the *pipeline* runs
    (encoder, retrieval, eval, IO) without touching a GPU. Recall@K will
    be poor; that's expected.
    """

    TEMPLATE = "A fashion item that is {modification}"

    def caption(self, item: dict[str, Any]) -> str:
        return self.TEMPLATE.format(modification=item["modification_text"])


class OracleCaptioner(VLMCaptioner):
    """Returns the ground-truth target caption.

    With a properly built caption DB (target caption included as a row),
    this backend should produce Recall@1 ~= 1.0. Failure here means
    the bug is upstream of the VLM.
    """

    def caption(self, item: dict[str, Any]) -> str:
        return item["target_caption"]


class _Qwen2VLLikeCaptioner(VLMCaptioner):
    """Shared logic for vanilla Qwen2-VL and speechQwen2VL (text-only mode)."""

    PROMPT_TEMPLATE = (
        "Given the reference fashion image and the modification instruction, "
        "write a concise caption describing the target fashion item after "
        "applying the modification."
    )
    GENERATION_KWARGS = {
        "max_new_tokens": 256,
        "num_beams": 1,
        "do_sample": False,
    }

    # Override per backend.
    MODEL_LABEL: str = "Qwen2-VL-7B"
    BASE_REPO: str = ""
    LORA_REPO: str | None = None

    def __init__(self, image_cache_root: Path | str | None = None) -> None:
        _check_can_host_qwen2vl_7b(self.MODEL_LABEL)
        self.image_cache_root = Path(image_cache_root) if image_cache_root else None
        self._load_model()

    def _load_model(self) -> None:
        import torch
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

        self._torch = torch
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.BASE_REPO,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.config.use_cache = True
        if self.LORA_REPO:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.LORA_REPO)
        self.model.eval()
        self.processor = Qwen2VLProcessor.from_pretrained(self.BASE_REPO)

    def _resolve_image_path(self, item: dict[str, Any]) -> Path:
        image_id = item["candidate_id"]
        if self.image_cache_root is None:
            raise RuntimeError(
                "image_cache_root not set; cannot resolve candidate image"
            )
        path = self.image_cache_root / f"{image_id}.jpeg"
        if not path.exists():
            raise FileNotFoundError(
                f"candidate image {image_id} not in cache ({path}); "
                f"run src/baseline/prepare_images.py before inference"
            )
        return path

    def caption(self, item: dict[str, Any]) -> str:
        path = self._resolve_image_path(item)
        image = Image.open(path).convert("RGB")
        modification = item["modification_text"]

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": str(path)},
                {"type": "text", "text": f"{self.PROMPT_TEMPLATE}\n\nModification: {modification}"},
            ]},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        batch = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        batch = {
            k: v.to(self.model.device) if hasattr(v, "to") else v
            for k, v in batch.items()
        }
        with self._torch.inference_mode():
            output_ids = self.model.generate(**batch, **self.GENERATION_KWARGS)
        prompt_len = batch["input_ids"].shape[1]
        return self.processor.batch_decode(
            output_ids[:, prompt_len:], skip_special_tokens=True
        )[0].strip()


class Qwen2VLCaptioner(_Qwen2VLLikeCaptioner):
    """Vanilla Qwen2-VL-7B-Instruct in text+image mode (no audio)."""

    MODEL_LABEL = "Qwen2-VL-7B-Instruct (vanilla)"
    BASE_REPO = "Qwen/Qwen2-VL-7B-Instruct"
    LORA_REPO = None


class SpeechQwen2VLCaptioner(_Qwen2VLLikeCaptioner):
    """User's Stage-1+Stage-2 speechQwen2VL, used in text-only mode here.

    Audio path is intentionally ignored — Plan_2 calls this the headline
    baseline because it preserves the Stage 1 -> Stage 2 narrative
    (the model the user trained, just without speech input).
    """

    MODEL_LABEL = "speechQwen2VL (Stage-1 base + Stage-2 LoRA)"
    BASE_REPO = "DanJZY/Qwen2-VL-7B-Speech"
    LORA_REPO = "DanJZY/Qwen2-VL-7B-Speech-LoRA"


_BACKENDS: dict[str, type[VLMCaptioner]] = {
    "mock": MockVLMCaptioner,
    "oracle": OracleCaptioner,
    "qwen2vl": Qwen2VLCaptioner,
    "speechqwen2vl": SpeechQwen2VLCaptioner,
}


def make_captioner(name: str, **kwargs: Any) -> VLMCaptioner:
    if name not in _BACKENDS:
        raise ValueError(f"unknown VLM backend {name!r}; known: {sorted(_BACKENDS)}")
    cls = _BACKENDS[name]
    if cls in (MockVLMCaptioner, OracleCaptioner):
        return cls()
    return cls(**kwargs)
