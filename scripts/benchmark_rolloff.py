"""Roll-off estimator (PSD band-edge shape fit) accuracy-vs-SNR benchmark.

Covers: item 4 of the dashboard-review task -- measures estimate_rolloff's
real error against generator ground truth across SNR and true rolloff, and
writes a real report (never typed-in numbers) the dashboard can show.

Run: .venv/bin/python scripts/benchmark_rolloff.py --out reports
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sigscope.estimate.blind import estimate_rolloff
from sigscope.synth.generator import generate_signal

SNR_GRID_DB = [-5.0, 0.0, 5.0, 10.0, 15.0, 18.0, 20.0, 25.0]
TRUE_ROLLOFFS = [0.2, 0.35, 0.5]


def run_benchmark(trials_per_cell: int = 20, seed: int = 2024) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for snr_db in SNR_GRID_DB:
        errs = []
        for _ in range(trials_per_cell):
            true_rolloff = float(rng.choice(TRUE_ROLLOFFS))
            sig, _gt = generate_signal(
                "qpsk",
                num_symbols=2000,
                sample_rate=1_000_000.0,
                symbol_rate=100_000.0,
                snr_db=snr_db,
                rolloff=true_rolloff,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            est, _conf = estimate_rolloff(sig.samples, sig.sample_rate, 100_000.0)
            errs.append(abs(est - true_rolloff))
        errs_arr = np.array(errs)
        rows.append({"snr_db": snr_db, "mae": float(errs_arr.mean()), "max_error": float(errs_arr.max())})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    rows = run_benchmark()
    csv_path = out_dir / "rolloff_error_vs_snr.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["snr_db", "mae", "max_error"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")
    for r in rows:
        print(f"  SNR {r['snr_db']:+5.1f} dB: MAE={r['mae']:.3f}  max={r['max_error']:.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    snrs = [r["snr_db"] for r in rows]
    maes = [r["mae"] for r in rows]
    ax.plot(snrs, maes, marker="o", label="MAE")
    ax.axhline(0.1, color="red", linestyle="--", label="+/-0.1 target (15 dB)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Roll-off absolute error")
    ax.set_title("Roll-off estimator (PSD band-edge shape fit): error vs SNR (measured, not asserted)")
    ax.legend()
    fig.tight_layout()
    png_path = out_dir / "rolloff_error_vs_snr.png"
    fig.savefig(png_path, dpi=110)
    print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
