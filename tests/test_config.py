"""Config invariants and dimension validation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.config import InferenceConfig


def test_defaults_within_expected_ranges():
    c = InferenceConfig()
    # schnell-specific invariants - if any of these drift, something is wrong.
    assert c.num_inference_steps == 4, "schnell is 4-step distilled"
    assert c.guidance_scale == 0.0, "schnell does not use CFG"
    assert c.max_sequence_length <= 256, "schnell text encoder is 256-token"
    assert c.compute_dtype == "float16", "Turing-safe compute dtype"
    # 8 GB sanity: 768 x 1360 ~= 1.04M pixels, well-tested for our hardware.
    assert c.height * c.width <= 1_050_000


def test_dimensions_must_be_multiples_of_16():
    with pytest.raises(ValueError, match="multiple of 16"):
        InferenceConfig(height=767, width=1360)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="multiple of 16"):
        InferenceConfig(height=768, width=1361)  # type: ignore[call-arg]


def test_replace_revalidates_dimensions():
    # replace() bypasses __init__ but should still trigger __post_init__,
    # so override paths get the same validation as direct construction.
    base = InferenceConfig()
    with pytest.raises(ValueError, match="multiple of 16"):
        replace(base, height=513)


def test_output_dir_is_created(tmp_path, monkeypatch):
    # The output_dir defaults via field(default_factory) to ROOT/outputs.
    # We can't easily redirect that without touching the module, but we
    # CAN verify __post_init__ creates the directory if missing by using
    # a target that doesn't yet exist.
    target = tmp_path / "fresh_outputs"
    assert not target.exists()
    c = InferenceConfig(output_dir=target)  # type: ignore[call-arg]
    assert c.output_dir == target
    assert target.is_dir()
