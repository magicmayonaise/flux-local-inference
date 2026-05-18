"""Unit tests for src.history - no real model, no real Gradio.

Verifies that:
  - the PNG is written with the expected tEXt chunks (provenance survives
    moving the file out of the project)
  - history.jsonl gets one well-formed JSON row per save
  - InferenceConfig serialises cleanly into the JSONL record
"""

from __future__ import annotations

import json

from PIL import Image

from src.config import InferenceConfig
from src.history import save_image_with_metadata, ui_filename


def _make_image() -> Image.Image:
    # 1x1 RGB - cheapest possible image. We're testing the metadata path,
    # not the bytes.
    return Image.new("RGB", (1, 1), color=(123, 45, 67))


def test_png_contains_prompt_and_seed_in_text_chunks(tmp_path):
    out = tmp_path / "x.png"
    save_image_with_metadata(
        _make_image(),
        out,
        prompt="a fruit fly brain",
        seed=42,
        metrics={"elapsed_s": 1.23, "peak_vram_gb": 1.80, "seed": 42},
        config=InferenceConfig(output_dir=tmp_path),  # type: ignore[call-arg]
    )
    with Image.open(out) as im:
        # `text` is the merged dict of tEXt + iTXt + zTXt chunks.
        text = im.text  # type: ignore[attr-defined]
    assert text["prompt"] == "a fruit fly brain"
    assert text["seed"] == "42"
    assert text["generator"] == "flux-local-inference"
    # InferenceConfig fields should be serialised into the chunks too.
    assert text["width"] == "1360"
    assert text["height"] == "768"


def test_jsonl_manifest_appends_one_row_per_save(tmp_path):
    cfg = InferenceConfig(output_dir=tmp_path)  # type: ignore[call-arg]
    for i in range(3):
        save_image_with_metadata(
            _make_image(),
            tmp_path / f"x_{i}.png",
            prompt=f"prompt {i}",
            seed=100 + i,
            metrics={"elapsed_s": 1.0, "peak_vram_gb": 1.0, "seed": 100 + i},
            config=cfg,
        )
    rows = (tmp_path / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 3
    parsed = [json.loads(r) for r in rows]
    assert [p["seed"] for p in parsed] == [100, 101, 102]
    assert [p["prompt"] for p in parsed] == ["prompt 0", "prompt 1", "prompt 2"]
    # Path-typed config fields must serialise (no TypeError when json.dumps).
    assert isinstance(parsed[0]["config"]["model_cache_dir"], str)
    assert parsed[0]["schema_version"] == 1


def test_ui_filename_format():
    name = ui_filename(seed=42)
    assert name.startswith("ui_")
    assert name.endswith("_seed42.png")
    # timestamp segment is YYYYMMDD-HHMMSS = 15 chars
    timestamp = name[len("ui_") : -len("_seed42.png")]
    assert len(timestamp) == 15
    assert timestamp[8] == "-"
