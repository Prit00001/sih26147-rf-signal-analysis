"""BER vs SNR benchmark for the demodulation chain, vs theoretical curves.

Covers: NFR-02 (report BER after decoding vs SNR). See tests/demod_test_utils.py
and demod/theory.py docstrings for the measured-SNR-vs-theory methodology.

Run: python scripts/benchmark_ber.py --out reports
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from demod_test_utils import best_aligned_ser, measured_symbol_snr_db, tx_symbol_indices  # noqa: E402

from sigscope.demod.fsk import demodulate_fsk  # noqa: E402
from sigscope.demod.psk_qam import demodulate_psk_qam  # noqa: E402
from sigscope.demod.theory import theoretical_ber  # noqa: E402
from sigscope.synth.generator import FSK_BITS_PER_SYMBOL, generate_signal  # noqa: E402

LINEAR_MODS = ["bpsk", "qpsk", "8psk", "16qam", "64qam"]
FSK_MODS = ["2fsk", "4fsk", "8fsk"]
SNR_GRID_DB = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]

# P6: one fixed, documented impaired-channel preset (not a CLI-tunable knob
# per impairment -- this benchmark's job is "does the chain still work under
# a representative set of real-world channel impairments", not a full
# multi-dimensional sweep). A 2-tap multipath (one -6dB echo 3 samples
# behind the direct path), slow flat Rayleigh-like fading, and a receiver
# clock drift this project's demod chain previously had nothing to track
# (see sync.py's Gardner recovery docstring).
IMPAIRED_CHANNEL_KWARGS: dict[str, object] = {
    "multipath_taps": ((3, -6.0, 40.0),),
    "fading_rate_hz": 5.0,
    "timing_drift_ppm": 20.0,
}


def run_linear(mod: str, snr_db: float, seed: int, *, impaired: bool = False) -> dict[str, float]:
    sig, gt = generate_signal(
        mod,
        num_symbols=4000,
        sample_rate=1_000_000.0,
        symbol_rate=100_000.0,
        snr_db=snr_db,
        seed=seed,
        **(IMPAIRED_CHANNEL_KWARGS if impaired else {}),
    )
    result = demodulate_psk_qam(sig.samples, mod, sig.sample_rate, gt.symbol_rate)
    tx_symbols = tx_symbol_indices(gt.bits, mod)
    ser, rot, shift, _length = best_aligned_ser(result.symbols, tx_symbols, mod)
    measured_snr = measured_symbol_snr_db(result.symbols, tx_symbols, mod, rot, shift)
    return {"ser": ser, "measured_snr_db": measured_snr, "theory_ber": theoretical_ber(mod, measured_snr)}


def run_fsk(mod: str, snr_db: float, seed: int, *, impaired: bool = False) -> dict[str, float]:
    sig, gt = generate_signal(
        mod,
        num_symbols=4000,
        sample_rate=1_000_000.0,
        symbol_rate=100_000.0,
        snr_db=snr_db,
        seed=seed,
        **(IMPAIRED_CHANNEL_KWARGS if impaired else {}),
    )
    result = demodulate_fsk(sig.samples, mod, sig.sample_rate, gt.symbol_rate)
    bps = FSK_BITS_PER_SYMBOL[mod]
    tx_bits = np.array(gt.bits)
    weights = 1 << np.arange(bps - 1, -1, -1)
    n_full = len(tx_bits) // bps
    tx_symbols = (tx_bits[: n_full * bps].reshape(-1, bps) @ weights).astype(np.int64)
    a, b = result.decided_indices, tx_symbols[: len(result.decided_indices)]
    length = min(len(a), len(b))
    ser = float(np.mean(a[:length] != b[:length]))
    # No closed-form theoretical BER implemented for FSK in demod/theory.py.
    return {"ser": ser, "measured_snr_db": snr_db, "theory_ber": float("nan")}


def run_benchmark(seed: int = 42, *, impaired: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mod in LINEAR_MODS:
        for snr_db in SNR_GRID_DB:
            r = run_linear(mod, snr_db, seed, impaired=impaired)
            rows.append({"modulation": mod, "nominal_snr_db": snr_db, **r})
    for mod in FSK_MODS:
        for snr_db in SNR_GRID_DB:
            r = run_fsk(mod, snr_db, seed, impaired=impaired)
            rows.append({"modulation": mod, "nominal_snr_db": snr_db, **r})
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["modulation", "nominal_snr_db", "ser", "measured_snr_db", "theory_ber"])
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict[str, object]], path: Path) -> None:
    # Dark theme matching the web dashboard's own chart styling
    # (webui/server.py's _DARK_RC) -- this PNG is embedded directly in that
    # dark-themed page's Benchmarks tab.
    with plt.rc_context(
        {
            "figure.facecolor": "#111113", "axes.facecolor": "#111113", "savefig.facecolor": "#111113",
            "axes.edgecolor": "#2a2a2e", "axes.labelcolor": "#f3f3f5", "axes.titlecolor": "#f3f3f5",
            "text.color": "#f3f3f5", "xtick.color": "#93939d", "ytick.color": "#93939d",
            "legend.labelcolor": "#f3f3f5", "grid.color": "#2a2a2e",
        }
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for mod in LINEAR_MODS + FSK_MODS:
            mod_rows = sorted((r for r in rows if r["modulation"] == mod), key=lambda r: r["nominal_snr_db"])  # type: ignore[arg-type]
            snrs = [r["nominal_snr_db"] for r in mod_rows]
            sers = [max(r["ser"], 1e-4) for r in mod_rows]  # type: ignore[type-var]
            ax.semilogy(snrs, sers, marker="o", label=mod, linewidth=1.5, markersize=4)
        ax.set_xlabel("Generator nominal SNR (dB)")
        ax.set_ylabel("Symbol error rate (log scale, floor 1e-4)")
        ax.set_title("Demodulator SER vs SNR (measured)")
        ax.legend(fontsize=8, ncol=2, facecolor="#19191c", edgecolor="#2a2a2e")
        ax.grid(alpha=0.5, which="both")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports")
    parser.add_argument(
        "--impaired", action="store_true", help="apply the P6 multipath+fading+timing-drift channel preset"
    )
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_impaired" if args.impaired else ""
    rows = run_benchmark(impaired=args.impaired)
    write_csv(rows, out_dir / f"demod_ber_vs_snr{suffix}.csv")
    write_plot(rows, out_dir / f"demod_ber_vs_snr{suffix}.png")
    print(f"wrote {out_dir / f'demod_ber_vs_snr{suffix}.csv'}")
    print(f"wrote {out_dir / f'demod_ber_vs_snr{suffix}.png'}")
