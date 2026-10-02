"""Train the modulation CNN and export it to ONNX with a SHA-256 manifest.

Covers: FR-06 (CNN classifier, calibrated confidence), SR-02 (ONNX + checksum,
never pickle). Train-time only: requires PyTorch (`pip install -e ".[train]"`).

Run: python -m sigscope.classify.train --out models/
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from sigscope.classify.dataset import generate_dataset
from sigscope.classify.model import ModulationCNN


def _split(
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.int64],
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 0,
) -> tuple[tuple[Any, Any], tuple[Any, Any], tuple[Any, Any]]:
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx, val_idx, test_idx = idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]
    return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx]), (x[test_idx], y[test_idx])


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """Single-parameter temperature scaling calibration (Guo et al. 2017)."""
    temperature = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = cast(torch.Tensor, nll(logits / temperature.clamp(min=0.05), labels))
        loss.backward()  # type: ignore[no-untyped-call]  # torch autograd is partially untyped upstream
        return loss

    optimizer.step(closure)  # type: ignore[no-untyped-call]  # torch LBFGS.step is partially untyped upstream
    return float(temperature.clamp(min=0.05).item())


def train(
    out_dir: str | Path,
    *,
    examples_per_class_snr: int = 40,
    window_length: int = 512,
    epochs: int = 8,
    batch_size: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y, class_names = generate_dataset(
        examples_per_class_snr=examples_per_class_snr, window_length=window_length, seed=seed
    )
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = _split(x, y, seed=seed)

    torch.manual_seed(seed)
    model = ModulationCNN(num_classes=len(class_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    n = len(y_train_t)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = perm[start : start + batch_size]
            xb, yb = x_train_t[batch_idx], y_train_t[batch_idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_idx)
        print(f"epoch {epoch + 1}/{epochs} loss={total_loss / n:.4f}")

    model.eval()
    with torch.no_grad():
        val_logits = model(torch.from_numpy(x_val))
        test_logits = model(torch.from_numpy(x_test))
    val_labels = torch.from_numpy(y_val)
    test_labels = torch.from_numpy(y_test)

    temperature = fit_temperature(val_logits, val_labels)

    with torch.no_grad():
        test_pred = torch.argmax(test_logits, dim=1)
        test_acc = float((test_pred == test_labels).float().mean().item())

    onnx_path = out_dir / "modulation_cnn.onnx"
    dummy_input = (torch.zeros(1, 2, window_length),)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["iq"],
        output_names=["logits"],
        opset_version=18,
    )

    checksum = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    manifest = {
        "onnx_file": onnx_path.name,
        "sha256": checksum,
        "class_names": class_names,
        "window_length": window_length,
        "temperature": temperature,
        "test_accuracy": test_acc,
        "num_train_examples": int(n),
    }
    (out_dir / "modulation_cnn.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models")
    parser.add_argument("--examples-per-class-snr", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = train(args.out, examples_per_class_snr=args.examples_per_class_snr, epochs=args.epochs, seed=args.seed)
    print(json.dumps(result, indent=2))
