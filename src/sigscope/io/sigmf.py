"""Minimal, safe SigMF (.sigmf-meta) sidecar reader.

Covers: FR-02 (SigMF metadata when present), SR-02 (json.load only, never
pickle/eval/exec on external data).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sigscope.core.exceptions import MalformedFileError

# SigMF core:datatype strings we understand, mapped to our internal dtype aliases.
_SIGMF_DATATYPE_MAP = {
    "ci8": "int8",
    "ci16_le": "int16",
    "ci16": "int16",
    "cf32_le": "complex64",
    "cf32": "complex64",
}


def find_sigmf_sidecar(data_path: str | Path) -> Path | None:
    """Return the .sigmf-meta path for a data file if one exists on disk."""
    data_path = Path(data_path)
    candidates = [data_path.with_suffix(".sigmf-meta"), Path(str(data_path) + ".sigmf-meta")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_sigmf_meta(meta_path: str | Path) -> dict[str, Any]:
    """Parse a .sigmf-meta file into {sample_rate, center_freq, dtype} (any may be None)."""
    meta_path = Path(meta_path)
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise MalformedFileError(f"{meta_path}: not a valid JSON SigMF file ({exc})") from exc
    if not isinstance(raw, dict) or "global" not in raw:
        raise MalformedFileError(f"{meta_path}: missing top-level 'global' object")

    glob = raw["global"]
    sigmf_dtype = glob.get("core:datatype")
    result: dict[str, Any] = {
        "sample_rate": glob.get("core:sample_rate"),
        "dtype": _SIGMF_DATATYPE_MAP.get(sigmf_dtype),
        "center_freq": None,
    }
    captures = raw.get("captures")
    if isinstance(captures, list) and captures and isinstance(captures[0], dict):
        result["center_freq"] = captures[0].get("core:frequency")
    return result
