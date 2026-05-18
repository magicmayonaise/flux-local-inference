"""Local provenance: save each generation with embedded PNG metadata and an
append-only JSONL manifest.

Two pieces of state land next to every saved image:

1. **PNG tEXt chunks** embedded in the file itself (prompt, seed, model,
   dimensions, steps, elapsed_s, peak_vram_gb). This survives moving the
   file out of the project - `identify -verbose foo.png` or `exiftool` will
   show the provenance, and so will `PIL.Image.open(path).text`.

2. **outputs/history.jsonl** - append-only, one JSON object per line. Easy to
   `jq` for "what did I generate yesterday with prompt X" without parsing
   100 PNG headers. Gitignored.

Both writes are best-effort: if the JSONL append fails, the PNG is still
saved with embedded metadata, so no provenance is lost.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.PngImagePlugin import PngInfo

logger = logging.getLogger(__name__)

# Schema version embedded in every JSONL row so future readers can branch
# on format changes without guessing.
HISTORY_SCHEMA_VERSION = 1


def _serialise_config(config: Any) -> dict:
    """Best-effort serialisation of InferenceConfig (or any dataclass / dict).

    Path values are coerced to str so the JSONL stays json-encodable.
    """
    if is_dataclass(config):
        d = asdict(config)
    elif isinstance(config, dict):
        d = dict(config)
    else:
        return {}
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}


def _build_pnginfo(prompt: str, seed: int, metrics: dict, config: Any) -> PngInfo:
    """Embed prompt + seed + key config + metrics in PNG tEXt chunks.

    Chunks are intentionally flat (no nested JSON) so common viewers
    (`identify -verbose`, exiftool, Affinity, Photoshop's metadata panel)
    display them cleanly. The full record lives in history.jsonl.
    """
    info = PngInfo()
    info.add_text("prompt", prompt)
    info.add_text("seed", str(seed))
    info.add_text("generator", "flux-local-inference")
    info.add_text("schema_version", str(HISTORY_SCHEMA_VERSION))
    cfg = _serialise_config(config)
    for key in (
        "model_repo",
        "width",
        "height",
        "num_inference_steps",
        "guidance_scale",
    ):
        if key in cfg:
            info.add_text(key, str(cfg[key]))
    info.add_text("elapsed_s", f"{metrics.get('elapsed_s', 0):.2f}")
    info.add_text("peak_vram_gb", f"{metrics.get('peak_vram_gb', 0):.2f}")
    return info


def save_image_with_metadata(
    image: Image.Image,
    path: Path,
    *,
    prompt: str,
    seed: int,
    metrics: dict,
    config: Any,
) -> Path:
    """Save `image` to `path` with provenance embedded; append manifest row.

    Path is created if needed. The manifest lives at `path.parent / history.jsonl`
    so calling this with a different output_dir creates an independent log.
    Returns the path actually written (== path).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pnginfo = _build_pnginfo(prompt, seed, metrics, config)
    image.save(path, format="PNG", pnginfo=pnginfo)

    record = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "filename": path.name,
        "prompt": prompt,
        "seed": int(seed),
        "elapsed_s": float(metrics.get("elapsed_s", 0.0)),
        "peak_vram_gb": float(metrics.get("peak_vram_gb", 0.0)),
        "config": _serialise_config(config),
    }
    manifest = path.parent / "history.jsonl"
    try:
        with manifest.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        # Best-effort: the PNG is already on disk with embedded metadata, so
        # provenance is preserved even if the JSONL append fails (rare - usually
        # only happens on a read-only filesystem).
        logger.warning("Failed to append to %s: %s", manifest, e)
    return path


def ui_filename(seed: int, when: datetime | None = None) -> str:
    """Canonical UI filename: ui_YYYYMMDD-HHMMSS_seed{N}.png."""
    when = when or datetime.now()
    return f"ui_{when.strftime('%Y%m%d-%H%M%S')}_seed{int(seed)}.png"
