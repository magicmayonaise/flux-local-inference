"""Inference configuration for FLUX.1 [schnell] on 8 GB Turing.

Defaults are tuned for an RTX 2070 (compute 7.5, 8 GB VRAM). The values are
not arbitrary; each is documented with the reason it has the value it does.
Changing them safely requires understanding *why*, hence the comments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

# Repo root resolved from this file, not cwd, so the same paths work whether
# the user invokes via `python -m src.generate` or from a notebook.
ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class InferenceConfig:
    # FLUX.1 [schnell] - Apache 2.0, 4-step distilled, gated on HF.
    # The dev variant requires a research license and CFG, neither of which
    # we want for a portfolio demo.
    model_repo: str = "black-forest-labs/FLUX.1-schnell"

    # Local caches live in repo for reproducibility; .gitignore excludes them.
    model_cache_dir: Path = field(default_factory=lambda: ROOT / "models")
    output_dir: Path = field(default_factory=lambda: ROOT / "outputs")

    device: str = "cuda"

    # Diffusers loads FLUX weights in bf16 on disk; loading in bf16 keeps the
    # checkpoint and the in-memory tensors bit-identical (no dtype conversion
    # at load time). The dtype the pipeline *computes* in is set separately
    # below, after sequential offload is configured. See pipeline.py.
    load_dtype: str = "bfloat16"

    # Turing (compute 7.5) has no native bf16 tensor cores - bf16 ops are
    # emulated and are markedly slower than fp16. We cast to fp16 for the
    # actual forward pass. This is the single most important decision in
    # this config for performance on 8 GB.
    compute_dtype: str = "float16"

    # 768 x 1360 ~= 1.044M pixels is the canonical low-VRAM example in the
    # HF FLUX docs. 1024 x 1024 (1.05M) occasionally OOMs after vae decode
    # on 8 GB. Keep this conservative for the default; benchmark.py uses it.
    height: int = 768
    width: int = 1360

    # schnell is timestep-distilled to 4 inference steps. Going lower harms
    # quality; going higher just wastes time without improving output.
    num_inference_steps: int = 4

    # schnell is *not* a CFG model. The distilled timesteps already encode
    # the implicit guidance; setting guidance_scale > 0 produces artifacts.
    guidance_scale: float = 0.0

    # schnell's text encoder is trained with a 256-token context; the dev
    # variant relaxes this to 512. Going above 256 here is a silent quality
    # regression on schnell.
    max_sequence_length: int = 256

    seed: int = 42

    # VAE slicing/tiling trade a small compute cost for a meaningful VRAM
    # reduction during decode. Both are required on 8 GB at 1360-wide.
    # Optional LoRA. None = base model only. Compatibility caveat: most public
    # FLUX LoRAs are trained for FLUX-dev; some work on schnell with degraded
    # quality at 4 steps, some don't load at all. The neuro-imagery LoRA in the
    # README's Future work section is the right long-term answer.
    lora_repo: Optional[str] = None
    # 0 == lora off; 1.0 == full strength; >1.0 == over-application.
    lora_scale: float = 1.0
    # Cache dir for downloaded LoRA weights. Gitignored.
    lora_cache_dir: Path = field(default_factory=lambda: ROOT / "models" / "loras")

    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = True

    # "sequential" offloads each module to CPU between forwards; the lowest
    # VRAM strategy and the one diffusers documents for <16 GB cards. "model"
    # is faster but uses more VRAM; "none" is for >24 GB cards only.
    offload_strategy: Literal["sequential", "model", "none"] = "sequential"

    def __post_init__(self) -> None:
        # output_dir must exist before generate() writes to it; main cache_dir
        # and lora_cache_dir are created lazily by huggingface_hub on download.
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # FLUX uses a patchify factor of 16 for the transformer; non-multiple
        # dimensions are silently rounded inside diffusers, which makes
        # reproducibility brittle. Reject them at the boundary instead.
        for side, value in (("height", self.height), ("width", self.width)):
            if value % 16 != 0:
                raise ValueError(
                    f"{side}={value} must be a multiple of 16 "
                    f"(FLUX patch size); got remainder {value % 16}"
                )
