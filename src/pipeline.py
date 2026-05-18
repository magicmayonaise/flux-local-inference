"""FluxGenerator: thin wrapper around diffusers.FluxPipeline.

The load order in `__init__` matters and is not interchangeable. See the
inline comments for why each step happens when it does. This is the part
of the code most worth reading carefully if anything misbehaves on 8 GB.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch
from diffusers import FluxPipeline
from PIL import Image

from src.config import InferenceConfig

logger = logging.getLogger(__name__)


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class FluxGenerator:
    """Pipeline wrapper with a Turing-safe initialisation sequence.

    The constructor performs the only ordering that the HF docs say works on
    sub-16-GB cards: load -> offload -> vae slicing/tiling -> cast. Diverging
    from this order causes either OOM at load time or a silent re-upload of
    every module to GPU memory.
    """

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        load_dtype = _DTYPE_MAP[config.load_dtype]
        compute_dtype = _DTYPE_MAP[config.compute_dtype]

        # 1) Load in bf16 to match the on-disk checkpoint exactly. Loading
        #    directly in fp16 would force a dtype cast for every weight and
        #    briefly double peak memory.
        pipe = FluxPipeline.from_pretrained(
            config.model_repo,
            torch_dtype=load_dtype,
            cache_dir=str(config.model_cache_dir),
        )

        # 2) Sequential offload MUST be enabled before the fp16 cast.
        #    enable_sequential_cpu_offload() builds an accelerate hook graph
        #    based on the pipeline's current device map; if you cast first,
        #    the hooks treat the GPU as the resident device and re-upload
        #    everything on the first forward, defeating the point.
        if config.offload_strategy == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif config.offload_strategy == "model":
            pipe.enable_model_cpu_offload()
        # "none": leave on CPU, the user is responsible for .to("cuda")

        # 3) VAE slicing + tiling reduces decode-time VRAM from ~6 GB to
        #    ~1.5 GB on a 1360-wide image. Order vs. offload doesn't matter
        #    for correctness; doing it here keeps the post-condition obvious.
        if config.enable_vae_slicing:
            pipe.vae.enable_slicing()
        if config.enable_vae_tiling:
            pipe.vae.enable_tiling()

        # 4) Now cast to fp16 for compute. The offload hooks see the new
        #    dtype on the next forward and move fp16 tensors instead of bf16.
        pipe.to(compute_dtype)

        # 5) Intentionally NOT calling pipe.to("cuda"). Sequential offload
        #    manages device placement itself; a manual .to("cuda") here
        #    would defeat the whole strategy and OOM at load.

        self.pipe = pipe

        # 6) Optional LoRA. Diffusers documents loading LoRAs AFTER offload
        #    enable; the adapter weights then get the same offload hooks as
        #    the base modules they patch.
        # active_lora is (repo, scale) so swap_lora() can compare and no-op
        # when the same LoRA is requested at the same scale.
        self.active_lora: Optional[tuple[str, float]] = None
        if config.lora_repo:
            self._apply_lora(config.lora_repo, config.lora_scale)

    def _apply_lora(self, repo: str, scale: float) -> None:
        """Load and activate a single LoRA. Raises ValueError with a clear
        message if the LoRA isn't compatible with FLUX (the most common
        failure mode is loading an SDXL or FLUX-dev LoRA that schnell's
        transformer rejects).
        """
        try:
            self.pipe.load_lora_weights(
                repo,
                adapter_name="active",
                cache_dir=str(self.config.lora_cache_dir),
            )
            self.pipe.set_adapters(["active"], adapter_weights=[scale])
            self.active_lora = (repo, scale)
            logger.info("Loaded LoRA %s @ scale=%.2f", repo, scale)
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                f"Failed to load LoRA {repo!r}: {e}. "
                "Most public FLUX LoRAs are trained for FLUX-dev and may not "
                "load on schnell. Check the LoRA's model card on Hugging Face."
            ) from e

    def swap_lora(self, repo: Optional[str], scale: float = 1.0) -> None:
        """Public API used by the UI: change the active LoRA, or remove it.

        repo == None       -> unload current LoRA (back to base model)
        repo == active     -> no-op if scale unchanged; update scale otherwise
        repo == something  -> unload current (if any) and load the new one
        """
        if repo is None:
            if self.active_lora is not None:
                self.pipe.unload_lora_weights()
                self.active_lora = None
            return

        current = self.active_lora
        if current and current[0] == repo:
            if current[1] != scale:
                self.pipe.set_adapters(["active"], adapter_weights=[scale])
                self.active_lora = (repo, scale)
            return

        if self.active_lora is not None:
            self.pipe.unload_lora_weights()
            self.active_lora = None
        self._apply_lora(repo, scale)

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generate one image. Returns the PIL image; does not save to disk."""
        c = self.config
        # CPU generator (not cuda) so seeds reproduce across machines and
        # across torch CUDA versions. The small RNG-state transfer cost is
        # negligible next to the 4 transformer forwards.
        gen = torch.Generator(device="cpu").manual_seed(
            seed if seed is not None else c.seed
        )

        try:
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                height=c.height,
                width=c.width,
                num_inference_steps=c.num_inference_steps,
                guidance_scale=c.guidance_scale,
                max_sequence_length=c.max_sequence_length,
                generator=gen,
            )
        except torch.cuda.OutOfMemoryError as e:
            logger.error(
                "CUDA OOM at %dx%d with %s offload. Try smaller dimensions "
                "(both multiples of 16), or run `torch.cuda.empty_cache()` "
                "and retry. Original error: %s",
                c.width,
                c.height,
                c.offload_strategy,
                e,
            )
            raise

        return result.images[0]

    def generate_with_metrics(
        self, prompt: str, seed: Optional[int] = None
    ) -> tuple[Image.Image, dict]:
        """Time + peak VRAM around a single generate(). Used by benchmark.py
        and src.ui. `seed` is forwarded to generate(); None falls back to the
        config default.

        We reset_peak_memory_stats before and read max_memory_allocated after.
        This is the right call on Turing because cached but unallocated blocks
        are not counted - we want true working-set, not reserved.
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        image = self.generate(prompt, seed=seed)
        elapsed = time.perf_counter() - t0
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        # `seed` may be None at the API surface; record the value actually used
        # so history records and CLI output always know the effective seed.
        effective_seed = seed if seed is not None else self.config.seed
        return image, {
            "elapsed_s": elapsed,
            "peak_vram_gb": peak_vram_gb,
            "seed": effective_seed,
            # active_lora is (repo, scale) or None; surface both for history/UI.
            "lora_repo": self.active_lora[0] if self.active_lora else None,
            "lora_scale": self.active_lora[1] if self.active_lora else None,
        }
