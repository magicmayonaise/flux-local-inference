"""CLI entry point: `python -m src.generate --prompt "..."`.

Loads .env first so HF_TOKEN propagates into huggingface_hub on first call.
Prints elapsed + peak VRAM after each generation so users can sanity-check
their numbers against the benchmark.md committed in the README.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

# Must run before any HF import so HUGGINGFACE_HUB_TOKEN / HF_TOKEN are visible.
load_dotenv()

from src.config import InferenceConfig  # noqa: E402 - intentional, see above
from src.history import save_image_with_metadata  # noqa: E402
from src.pipeline import FluxGenerator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m src.generate",
        description="Generate one image from a prompt using FLUX.1 [schnell].",
    )
    p.add_argument("--prompt", required=True, help="Text prompt.")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (defaults to outputs/<seed>.png).",
    )
    p.add_argument("--seed", type=int, default=None, help="Override config seed.")
    p.add_argument("--height", type=int, default=None, help="Multiple of 16.")
    p.add_argument("--width", type=int, default=None, help="Multiple of 16.")
    p.add_argument(
        "--lora",
        type=str,
        default=None,
        dest="lora_repo",
        help="HuggingFace repo ID or local path to a FLUX LoRA. "
        "Most public LoRAs target FLUX-dev; compatibility with schnell varies.",
    )
    p.add_argument(
        "--lora-scale",
        type=float,
        default=None,
        help="LoRA strength multiplier (default: 1.0). 0 = off, >1 = over-applied.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    config = InferenceConfig()
    # Apply CLI overrides via replace() to preserve frozen-dataclass invariants
    # (this re-runs __post_init__ and re-validates multiples-of-16).
    overrides = {
        k: v
        for k, v in {
            "seed": args.seed,
            "height": args.height,
            "width": args.width,
            "lora_repo": args.lora_repo,
            "lora_scale": args.lora_scale,
        }.items()
        if v is not None
    }
    if overrides:
        config = replace(config, **overrides)

    out_path = args.out or (config.output_dir / f"flux_{config.seed}.png")

    gen = FluxGenerator(config)
    image, metrics = gen.generate_with_metrics(args.prompt, seed=args.seed)
    # save_image_with_metadata embeds prompt+seed+config in PNG tEXt chunks
    # and appends to outputs/history.jsonl. Same provenance story as the UI.
    save_image_with_metadata(
        image,
        out_path,
        prompt=args.prompt,
        seed=metrics["seed"],
        metrics=metrics,
        config=config,
    )

    print(
        f"saved {out_path}  "
        f"elapsed={metrics['elapsed_s']:.1f}s  "
        f"peak_vram={metrics['peak_vram_gb']:.2f} GB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
