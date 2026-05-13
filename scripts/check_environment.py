"""Environment validation for FLUX.1 [schnell] local inference.

Runs a battery of checks on the host machine: Python version, CUDA availability,
GPU name, VRAM, compute capability, dtype support, key library imports, HF auth
state, and disk space. Designed to be the first thing a user (or future me)
runs when something looks wrong.

Exit codes:
    0 -> all checks PASS or only WARN
    1 -> at least one FAIL

WARNs surface to stderr so CI can grep for them without flagging a build red.

Why this script exists: the FLUX pipeline crashes opaquely on Turing if loaded
with the wrong dtype, on a wrong cu* torch wheel, or with insufficient VRAM. A
fast pre-check is cheaper than a 30 s pipeline init that ends in OOM.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

# Resolved at module load: lets the script run from any cwd and still
# report disk space on the project's models/ directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Substring expected in torch.cuda.get_device_name(0). Overridable so the same
# script can be used unmodified on a 4090, A100, etc.
EXPECTED_GPU_SUBSTRING = os.environ.get("FLUX_EXPECTED_GPU", "RTX 2070")

# 8192 MiB - some margin for the WSL/driver overhead reported by nvidia-smi.
MIN_VRAM_MIB = 7500

# 24 GB FLUX weights + 4-5 GB venv + 1 GB git history headroom.
MIN_FREE_DISK_GB = 30


@dataclass
class Result:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str


def _row(name: str, status: str, detail: str) -> Result:
    return Result(name=name, status=status, detail=detail)


def check_python() -> Result:
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 10)
    return _row(
        "python>=3.10",
        "PASS" if ok else "FAIL",
        f"running {major}.{minor}.{sys.version_info.micro}",
    )


def check_torch_cuda() -> tuple[Result, "object|None"]:
    """Return (result, torch_module_or_None).

    Bundling the torch import here means a missing/broken torch surfaces as a
    single FAIL rather than an unhandled ImportError further down.
    """
    try:
        import torch
    except Exception as e:  # noqa: BLE001 - we want any import-time failure
        return _row("torch import", "FAIL", f"{type(e).__name__}: {e}"), None

    if not torch.cuda.is_available():
        return (
            _row(
                "torch CUDA",
                "FAIL",
                f"torch {torch.__version__} built against CUDA "
                f"{torch.version.cuda}, but cuda.is_available()=False",
            ),
            torch,
        )

    return (
        _row(
            "torch CUDA",
            "PASS",
            f"torch {torch.__version__} / CUDA {torch.version.cuda}",
        ),
        torch,
    )


def check_gpu_name(torch) -> Result:
    if torch is None or not torch.cuda.is_available():
        return _row("GPU name", "FAIL", "no CUDA device")
    name = torch.cuda.get_device_name(0)
    ok = EXPECTED_GPU_SUBSTRING.lower() in name.lower()
    return _row(
        "GPU name",
        "PASS" if ok else "WARN",
        f"{name!r} (expected substring {EXPECTED_GPU_SUBSTRING!r})",
    )


def check_vram(torch) -> Result:
    if torch is None or not torch.cuda.is_available():
        return _row("VRAM", "FAIL", "no CUDA device")
    props = torch.cuda.get_device_properties(0)
    total_mib = props.total_memory // (1024 * 1024)
    ok = total_mib >= MIN_VRAM_MIB
    return _row(
        "VRAM",
        "PASS" if ok else "FAIL",
        f"{total_mib} MiB total (need >= {MIN_VRAM_MIB})",
    )


def check_compute_capability(torch) -> Result:
    """Compute-capability gate.

    Compute < 7.0 means no fp16 tensor cores at all -> FLUX inference is
    impractically slow. Compute < 8.0 means no native bf16 (Turing emulates
    it via a software path that is markedly slower than native fp16). We
    accept >= 7.0 and downgrade to WARN below 8.0 so the user knows why
    `compute_dtype` should stay float16.
    """
    if torch is None or not torch.cuda.is_available():
        return _row("compute capability", "FAIL", "no CUDA device")
    major, minor = torch.cuda.get_device_capability(0)
    cap = float(f"{major}.{minor}")
    if cap < 7.0:
        return _row("compute capability", "FAIL", f"{cap} - too old, no fp16 cores")
    if cap < 8.0:
        return _row(
            "compute capability",
            "WARN",
            f"{cap} - pre-Ampere, no native bf16 (emulated); use float16",
        )
    return _row("compute capability", "PASS", f"{cap} - native bf16 available")


def check_bf16_support(torch) -> Result:
    """torch.cuda.is_bf16_supported() reports software-emulated bf16 as True
    on Turing in torch >= 2.4. Informational only - the compute-capability
    check is the load-bearing signal.
    """
    if torch is None or not torch.cuda.is_available():
        return _row("bf16 support", "FAIL", "no CUDA device")
    is_supported = bool(torch.cuda.is_bf16_supported())
    return _row(
        "bf16 support",
        "WARN" if is_supported else "WARN",  # always informational
        f"torch.cuda.is_bf16_supported()={is_supported} (informational)",
    )


def check_imports() -> Result:
    """Verify diffusers/transformers/accelerate import cleanly.

    These pull in compiled extensions on first import; a botched install
    surfaces here, not 30 s into pipeline init.
    """
    missing = []
    for mod in ("diffusers", "transformers", "accelerate"):
        try:
            importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{mod} ({type(e).__name__}: {e})")
    if missing:
        return _row("library imports", "FAIL", "; ".join(missing))
    return _row("library imports", "PASS", "diffusers, transformers, accelerate")


def check_hf_auth() -> Result:
    """FLUX.1-schnell is gated; auth is required for download but not for
    code import. WARN keeps this script useful in air-gapped CI runs.
    """
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return _row("HF auth", "PASS", "HF_TOKEN env var set")
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return _row("HF auth", "PASS", f"token file at {token_file}")
    return _row(
        "HF auth",
        "WARN",
        "no HF_TOKEN and no ~/.cache/huggingface/token; "
        "run `huggingface-cli login` before download",
    )


def check_disk_space() -> Result:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(MODELS_DIR).free
    free_gb = free_bytes / (1024**3)
    ok = free_gb >= MIN_FREE_DISK_GB
    return _row(
        "disk space",
        "PASS" if ok else "FAIL",
        f"{free_gb:.1f} GB free at {MODELS_DIR} (need >= {MIN_FREE_DISK_GB})",
    )


def main() -> int:
    results: list[Result] = []
    results.append(check_python())
    torch_result, torch_mod = check_torch_cuda()
    results.append(torch_result)
    results.append(check_gpu_name(torch_mod))
    results.append(check_vram(torch_mod))
    results.append(check_compute_capability(torch_mod))
    results.append(check_bf16_support(torch_mod))
    results.append(check_imports())
    results.append(check_hf_auth())
    results.append(check_disk_space())

    # Fixed-width table; intentionally no rich/tabulate dep.
    name_w = max(len(r.name) for r in results)
    detail_w = max(len(r.detail) for r in results)
    sep = "+-" + "-" * name_w + "-+--------+-" + "-" * detail_w + "-+"
    print(sep)
    print(f"| {'check'.ljust(name_w)} | {'status':<6} | {'detail'.ljust(detail_w)} |")
    print(sep)
    for r in results:
        print(
            f"| {r.name.ljust(name_w)} | {r.status:<6} | {r.detail.ljust(detail_w)} |"
        )
    print(sep)

    fails = [r for r in results if r.status == "FAIL"]
    warns = [r for r in results if r.status == "WARN"]

    for w in warns:
        print(f"WARN: {w.name}: {w.detail}", file=sys.stderr)

    if fails:
        for f in fails:
            print(f"FAIL: {f.name}: {f.detail}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
