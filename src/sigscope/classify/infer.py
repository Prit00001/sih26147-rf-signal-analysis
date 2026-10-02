"""ONNX Runtime inference for the modulation classifier.

Covers: SR-02 (SHA-256 verified against a manifest before load; never
pickle/eval/exec on external data -- json + onnxruntime only).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from sigscope.core.exceptions import SigscopeError


class ModelIntegrityError(SigscopeError):
    """Raised when a model file's SHA-256 does not match its manifest."""


@dataclass
class ClassifierBundle:
    session: ort.InferenceSession
    class_names: list[str]
    window_length: int
    temperature: float


def load_classifier(manifest_path: str | Path) -> ClassifierBundle:
    """Load the ONNX classifier, refusing to proceed if its checksum doesn't match."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    onnx_path = manifest_path.parent / manifest["onnx_file"]
    actual_hash = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    if actual_hash != manifest["sha256"]:
        raise ModelIntegrityError(
            f"{onnx_path}: SHA-256 mismatch (manifest says {manifest['sha256']}, file hashes to "
            f"{actual_hash}); refusing to load a model that does not match its manifest"
        )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return ClassifierBundle(
        session=session,
        class_names=manifest["class_names"],
        window_length=manifest["window_length"],
        temperature=manifest["temperature"],
    )


def _softmax(logits: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return cast(npt.NDArray[np.float32], (exp / np.sum(exp)).astype(np.float32))


def classify_cnn(bundle: ClassifierBundle, iq_tensor: npt.NDArray[np.float32]) -> tuple[str, float, dict[str, float]]:
    """``iq_tensor``: [2, L] float32. Returns (label, confidence, per-class probabilities)."""
    input_name = bundle.session.get_inputs()[0].name
    logits = bundle.session.run(None, {input_name: iq_tensor[None, :, :].astype(np.float32)})[0][0]
    calibrated = logits / bundle.temperature
    probs = _softmax(calibrated)
    best_idx = int(np.argmax(probs))
    label_probs = dict(zip(bundle.class_names, probs.tolist(), strict=True))
    return bundle.class_names[best_idx], float(probs[best_idx]), label_probs
