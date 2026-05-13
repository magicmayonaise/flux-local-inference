"""Minimal end-to-end smoke test: generate the smallest valid image.

Goal: prove the *pipeline plumbing* works before anyone waits 4+ minutes
on a real generation. We use 512x512 at 1 step so the test runs in well
under a minute even on Turing.

This is the gate before benchmark.py. If smoke_test fails, fixing the
pipeline takes priority over anything else.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from PIL import Image

from src.config import InferenceConfig
from src.pipeline import FluxGenerator

# Tunable thresholds for the asserts at the bottom. Chosen empirically:
#   MIN_FILE_BYTES: a 512x512 PNG of any content is well above 10 KB
#   MAX_VRAM_GB:    we want headroom for the real 768x1360 default; 7.5 GB
#                   peak at 512x512 means 768x1360 will surely OOM.
MIN_FILE_BYTES = 10 * 1024
MAX_VRAM_GB = 7.5


def main() -> int:
    # 1) Reuse check_environment as a pre-condition. Failing here makes the
    #    later asserts useless. Use a subprocess so we get the same exit-code
    #    semantics a user would.
    repo = Path(__file__).resolve().parent.parent
    env_check = subprocess.run(
        [sys.executable, str(repo / "scripts" / "check_environment.py")],
    )
    if env_check.returncode != 0:
        print("smoke_test: environment check FAILED, aborting.", file=sys.stderr)
        return env_check.returncode

    # 2) Confirm a model_index.json exists - cheapest possible weights check.
    cfg = InferenceConfig()
    candidates = list(cfg.model_cache_dir.rglob("model_index.json"))
    if not candidates:
        print(
            f"smoke_test: no model_index.json under {cfg.model_cache_dir}. "
            "Run `uv run python scripts/download_model.py` first.",
            file=sys.stderr,
        )
        return 5

    # 3) Override to the cheapest valid config: 512x512, 1 step.
    cfg = replace(cfg, height=512, width=512, num_inference_steps=1)
    out_path = cfg.output_dir / "smoke_test.png"

    t0 = time.perf_counter()
    gen = FluxGenerator(cfg)
    image, metrics = gen.generate_with_metrics("test")
    image.save(out_path)
    elapsed = time.perf_counter() - t0

    # 4) Hard asserts. We want failure here to be loud and specific.
    assert out_path.exists(), f"output {out_path} not written"
    size = out_path.stat().st_size
    assert size > MIN_FILE_BYTES, f"output suspiciously small: {size} bytes"
    # Re-open with PIL to confirm a valid PNG and correct dimensions.
    with Image.open(out_path) as im:
        assert im.size == (cfg.width, cfg.height), (
            f"size mismatch: got {im.size}, want ({cfg.width}, {cfg.height})"
        )
    assert metrics["peak_vram_gb"] < MAX_VRAM_GB, (
        f"peak VRAM {metrics['peak_vram_gb']:.2f} GB exceeds {MAX_VRAM_GB} GB "
        "at 512x512; default 768x1360 will OOM"
    )

    print(
        f"SMOKE TEST PASSED  elapsed_total={elapsed:.1f}s  "
        f"gen_only={metrics['elapsed_s']:.1f}s  "
        f"peak_vram={metrics['peak_vram_gb']:.2f} GB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
