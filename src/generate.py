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
        }.items()
        if v is not None
    }
    if overrides:
        config = replace(config, **overrides)

    out_path = args.out or (config.output_dir / f"flux_{config.seed}.png")

    gen = FluxGenerator(config)
    image, metrics = gen.generate_with_metrics(args.prompt)
    image.save(out_path)

    print(
        f"saved {out_path}  "
        f"elapsed={metrics['elapsed_s']:.1f}s  "
        f"peak_vram={metrics['peak_vram_gb']:.2f} GB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
