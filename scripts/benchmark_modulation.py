"""Accuracy-vs-SNR benchmark for the modulation classifier ensemble.

Covers: NFR-02 (report classification accuracy vs SNR). Uses freshly generated
test signals (different seeds from training) against the trained model in
models/modulation_cnn.manifest.json.

Run: python scripts/benchmark_modulation.py --model models/modulation_cnn.manifest.json --out reports/
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sigscope.classify.ensemble import ensemble_classify
from sigscope.classify.infer import load_classifier
from sigscope.synth.generator import ALL_MODULATIONS, generate_signal

SNR_GRID_DB = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0]


def run_benchmark(manifest_path: Path, *, trials_per_cell: int = 30, seed: int = 999) -> list[dict[str, object]]:
    bundle = load_classifier(manifest_path)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for mod in ALL_MODULATIONS:
        for snr_db in SNR_GRID_DB:
            correct = 0
            for _ in range(trials_per_cell):
                sig, _ = generate_signal(
                    mod,  # type: ignore[arg-type]
                    num_symbols=250,
                    sample_rate=1_000_000.0,
                    symbol_rate=100_000.0,
                    snr_db=snr_db,
                    cfo_hz=float(rng.uniform(-500, 500)),
                    timing_offset_frac=float(rng.uniform(-0.3, 0.3)),
                    iq_imbalance_gain_db=float(rng.uniform(-0.5, 0.5)),
                    iq_imbalance_phase_deg=float(rng.uniform(-3, 3)),
                    seed=int(rng.integers(0, 2**31 - 1)),
                )
                label, _confidence, _scores = ensemble_classify(
                    sig.samples, bundle, sample_rate=sig.sample_rate, symbol_rate_hz=100_000.0
                )
                if label == mod:
                    correct += 1
            rows.append({"modulation": mod, "snr_db": snr_db, "accuracy": correct / trials_per_cell})
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["modulation", "snr_db", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict[str, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for mod in ALL_MODULATIONS:
        mod_rows = sorted((r for r in rows if r["modulation"] == mod), key=lambda r: r["snr_db"])  # type: ignore[arg-type]
        snrs = [r["snr_db"] for r in mod_rows]
        accs = [r["accuracy"] for r in mod_rows]
        ax.plot(snrs, accs, marker="o", label=mod)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Modulation classifier: accuracy vs SNR (measured, not asserted)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/modulation_cnn.manifest.json")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--trials-per-cell", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = run_benchmark(Path(args.model), trials_per_cell=args.trials_per_cell)
    write_csv(rows, out_dir / "modulation_accuracy_vs_snr.csv")
    write_plot(rows, out_dir / "modulation_accuracy_vs_snr.png")

    overall = sum(r["accuracy"] for r in rows) / len(rows)  # type: ignore[misc]
    print(f"overall mean accuracy across all classes/SNRs: {overall:.3f}")
    print(f"wrote {out_dir / 'modulation_accuracy_vs_snr.csv'}")
    print(f"wrote {out_dir / 'modulation_accuracy_vs_snr.png'}")
