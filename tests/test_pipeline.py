"""Mock-based tests for FluxGenerator init order.

We never load real FLUX weights in tests - that's a 24 GB download and a
~30 s cold init even when cached. We monkey-patch FluxPipeline.from_pretrained
to return a recording stub and assert the methods are called in the order
the HF docs document as safe on <16 GB cards.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.config import InferenceConfig


class _StubPipe:
    """Records ordered method calls so we can assert the load sequence."""

    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []
        # vae is itself a stub so .enable_slicing/.enable_tiling appear
        # in the *same* call list as the pipe-level methods.
        self.vae = MagicMock()
        self.vae.enable_slicing = self._make_recorder("vae.enable_slicing")
        self.vae.enable_tiling = self._make_recorder("vae.enable_tiling")
        self.enable_sequential_cpu_offload = self._make_recorder(
            "enable_sequential_cpu_offload"
        )
        self.enable_model_cpu_offload = self._make_recorder("enable_model_cpu_offload")
        self.to = self._make_recorder("to")

    def _make_recorder(self, name):
        def recorder(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self

        return recorder


@pytest.fixture
def stub_pipe(monkeypatch):
    pipe = _StubPipe()
    from src import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module.FluxPipeline,
        "from_pretrained",
        classmethod(lambda cls, *a, **kw: pipe),
    )
    return pipe


def test_init_calls_offload_before_dtype_cast(stub_pipe):
    """Sequential offload must precede the fp16 cast - this is the bit
    that breaks silently on 8 GB if reordered. The HF docs are explicit.
    """
    from src.pipeline import FluxGenerator

    FluxGenerator(InferenceConfig())
    names = [c[0] for c in stub_pipe.calls]

    assert "enable_sequential_cpu_offload" in names
    assert "to" in names
    assert names.index("enable_sequential_cpu_offload") < names.index("to"), (
        f"expected sequential_cpu_offload before to(dtype); got {names}"
    )


def test_init_enables_vae_slicing_and_tiling(stub_pipe):
    from src.pipeline import FluxGenerator

    FluxGenerator(InferenceConfig())
    names = [c[0] for c in stub_pipe.calls]
    assert "vae.enable_slicing" in names
    assert "vae.enable_tiling" in names


def test_init_does_not_call_pipe_to_cuda(stub_pipe):
    """Sequential offload manages device placement. A literal .to("cuda")
    would defeat that and OOM. Verify the only `to` call is the dtype cast.
    """
    from src.pipeline import FluxGenerator
    import torch

    FluxGenerator(InferenceConfig())
    to_calls = [c for c in stub_pipe.calls if c[0] == "to"]
    assert len(to_calls) == 1
    (_, args, kwargs) = to_calls[0]
    # The single positional arg should be a torch.dtype, not "cuda".
    assert args and isinstance(args[0], torch.dtype)
    assert args[0] == torch.float16, "Turing wants fp16 compute, got something else"
