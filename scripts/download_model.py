"""Download FLUX.1 [schnell] weights into models/ via huggingface_hub.

Avoiding pipe.from_pretrained() here means the user gets a clear progress
bar, can interrupt and resume, and we can scope the download to just the
files diffusers actually loads (skipping ONNX/msgpack/bin duplicates).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from huggingface_hub.utils import (
    GatedRepoError,
    RepositoryNotFoundError,
)

load_dotenv()

from src.config import InferenceConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# diffusers reads safetensors + the JSON config tree + tokenizer .model files.
# We exclude the redundant on-disk formats (~8 GB saved):
#   *.bin       - pytorch_model.bin pickled tensors, redundant with safetensors
#   *.msgpack   - flax weights
#   *.onnx      - ONNX export
ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.model", "*.py"]
IGNORE_PATTERNS = ["*.bin", "*.msgpack", "*.onnx"]


def _dir_size_gb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / (1024**3)


def main() -> int:
    config = InferenceConfig()
    target = config.model_cache_dir / config.model_repo.replace("/", "__")
    target.mkdir(parents=True, exist_ok=True)

    logger.info("downloading %s into %s", config.model_repo, target)
    try:
        local_dir = snapshot_download(
            repo_id=config.model_repo,
            local_dir=str(target),
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
        )
    except GatedRepoError:
        print(
            "FAIL: FLUX.1-schnell is gated. Steps to unblock:\n"
            "  1. Open https://huggingface.co/black-forest-labs/FLUX.1-schnell\n"
            "  2. Log in and click 'Agree and access repository'\n"
            "  3. Create a Read token at https://huggingface.co/settings/tokens\n"
            "  4. Run: huggingface-cli login (and paste the token)\n"
            "  5. Re-run this script.",
            file=sys.stderr,
        )
        return 2
    except RepositoryNotFoundError as e:
        # 401/403 land here on hub-hub auth failures; same user-facing fix.
        print(
            f"FAIL: repository not found or no access ({e}). Same fix as the "
            "gated-repo case: log in to HF, accept the license, and run "
            "`huggingface-cli login`.",
            file=sys.stderr,
        )
        return 3
    except ConnectionError as e:
        print(
            f"FAIL: network error ({e}). Check VPN / corporate proxy / firewall "
            "rules between this host and huggingface.co.",
            file=sys.stderr,
        )
        return 4

    size_gb = _dir_size_gb(Path(local_dir))
    print(f"downloaded {size_gb:.1f} GB into {local_dir}")
    # Sanity: pipeline init needs model_index.json. Surfacing its presence
    # here saves a confusing failure later in smoke_test.
    if not (Path(local_dir) / "model_index.json").exists():
        print(
            "WARN: model_index.json not found after download; diffusers may "
            "fail to load the pipeline.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
