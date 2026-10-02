"""sigscope command-line interface.

Covers: FR-01/FR-02 (ingest), FR-05/FR-09/FR-14 (spectrum data for the GUI),
FR-16 (generate), NFR-03 (usable without writing code), SR-03 (fully offline).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from sigscope.classify.ensemble import ensemble_classify
from sigscope.classify.infer import load_classifier
from sigscope.core.exceptions import SigscopeError
from sigscope.core.signal import Signal
from sigscope.estimate.blind import estimate_rolloff, estimate_snr_m2m4, estimate_symbol_rate_hz
from sigscope.io.iq_reader import read_iq
from sigscope.io.wav_reader import read_wav
from sigscope.spectral.analysis import estimate_cfo, occupied_bandwidth, sample_rate_sanity_check, waterfall, welch_psd
from sigscope.synth.generator import generate_signal, write_pair

_DEFAULT_MODEL_MANIFEST = Path(__file__).resolve().parents[2] / "models" / "modulation_cnn.manifest.json"


def _load_any(
    path: Path, *, dtype: str | None, sample_rate: float | None, center_freq: float | None
) -> Signal:
    if path.suffix.lower() == ".wav":
        return read_wav(path)
    return read_iq(path, dtype=dtype, sample_rate=sample_rate, center_freq=center_freq)


def _cmd_ingest(args: argparse.Namespace) -> int:
    path = Path(args.file)
    signal = _load_any(path, dtype=args.dtype, sample_rate=args.sample_rate, center_freq=args.center_freq)
    report: dict[str, Any] = {
        "path": str(path),
        "source_format": signal.source_format.value,
        "sample_rate": signal.sample_rate,
        "center_freq": signal.center_freq,
        "num_samples": signal.num_samples,
        "duration_seconds": signal.duration_seconds,
        "provenance": signal.provenance,
        "confidence": signal.confidence,
    }
    print(json.dumps(report, indent=2))
    return 0


def _cmd_spectrum(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(args.file)
    signal = _load_any(path, dtype=args.dtype, sample_rate=args.sample_rate, center_freq=args.center_freq)

    freqs, psd = welch_psd(signal.samples, signal.sample_rate)
    times, wf_freqs, magnitude_db = waterfall(signal.samples, signal.sample_rate)
    low, high, bw = occupied_bandwidth(freqs, psd)
    cfo = estimate_cfo(freqs, psd)
    sanity = sample_rate_sanity_check(signal.sample_rate, bw)

    fig, (ax_psd, ax_wf) = plt.subplots(2, 1, figsize=(9, 7))
    ax_psd.plot(freqs, 10 * np.log10(psd + 1e-20))
    ax_psd.axvspan(low, high, color="orange", alpha=0.2, label=f"99% BW = {bw:.1f} Hz")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD (dB)")
    ax_psd.set_title(f"{path.name} -- Welch PSD (est. CFO {cfo:.1f} Hz)")
    ax_psd.legend(loc="upper right")

    mesh = ax_wf.pcolormesh(times, wf_freqs, magnitude_db, shading="auto")
    ax_wf.set_xlabel("Time (s)")
    ax_wf.set_ylabel("Frequency (Hz)")
    ax_wf.set_title("Waterfall (STFT magnitude, dB)")
    fig.colorbar(mesh, ax=ax_wf)

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    plt.close(fig)

    summary = {
        "occupied_bandwidth_hz": bw,
        "band_low_hz": low,
        "band_high_hz": high,
        "estimated_cfo_hz": cfo,
        "sample_rate_sanity": dataclasses.asdict(sanity),
        "image": str(args.out),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    path = Path(args.file)
    signal = _load_any(path, dtype=args.dtype, sample_rate=args.sample_rate, center_freq=args.center_freq)

    symbol_rate_hz, symbol_rate_conf = estimate_symbol_rate_hz(signal.samples, signal.sample_rate)
    snr_db, snr_conf = estimate_snr_m2m4(signal.samples)
    rolloff, rolloff_conf = estimate_rolloff(signal.samples, signal.sample_rate, symbol_rate_hz)

    report: dict[str, Any] = {
        "path": str(path),
        "sample_rate": {"value": signal.sample_rate, "confidence": signal.confidence.get("sample_rate", 1.0)},
        "symbol_rate_hz": {"value": symbol_rate_hz, "confidence": symbol_rate_conf},
        "snr_db": {"value": snr_db, "confidence": snr_conf},
        "rolloff": {"value": rolloff, "confidence": rolloff_conf},
    }

    manifest_path = Path(args.model) if args.model else _DEFAULT_MODEL_MANIFEST
    if manifest_path.is_file():
        bundle = load_classifier(manifest_path)
        label, confidence, all_scores = ensemble_classify(
            signal.samples, bundle, sample_rate=signal.sample_rate, symbol_rate_hz=symbol_rate_hz
        )
        report["modulation"] = {"value": label, "confidence": confidence, "all_scores": all_scores}
    else:
        report["modulation"] = {
            "value": None,
            "confidence": 0.0,
            "note": f"no trained model manifest found at {manifest_path}; run `python -m sigscope.classify.train`",
        }

    print(json.dumps(report, indent=2))
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    signal, ground_truth = generate_signal(
        args.mod,
        num_symbols=args.num_symbols,
        sample_rate=args.sample_rate,
        symbol_rate=args.symbol_rate,
        snr_db=args.snr,
        cfo_hz=args.cfo,
        timing_offset_frac=args.timing_offset,
        iq_imbalance_gain_db=args.iq_gain_db,
        iq_imbalance_phase_deg=args.iq_phase_deg,
        seed=args.seed,
    )
    name = args.name or f"{args.mod}_snr{args.snr:g}_seed{args.seed if args.seed is not None else 0}"
    paths = write_pair(signal, ground_truth, args.out, name)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sigscope", description="Offline RF signal analysis toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_raw_iq_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dtype", choices=["int8", "int16", "float32", "complex64"], default=None)
        p.add_argument("--sample-rate", type=float, default=None)
        p.add_argument("--center-freq", type=float, default=None)

    p_ingest = sub.add_parser("ingest", help="Load a .wav or .IQ file and print its metadata as JSON")
    p_ingest.add_argument("file")
    add_raw_iq_args(p_ingest)
    p_ingest.set_defaults(func=_cmd_ingest)

    p_spectrum = sub.add_parser("spectrum", help="Render PSD + waterfall for a .wav or .IQ file")
    p_spectrum.add_argument("file")
    p_spectrum.add_argument("--out", required=True, help="Output PNG path")
    add_raw_iq_args(p_spectrum)
    p_spectrum.set_defaults(func=_cmd_spectrum)

    p_analyze = sub.add_parser("analyze", help="Estimate parameters and classify modulation for a .wav or .IQ file")
    p_analyze.add_argument("file")
    p_analyze.add_argument("--model", default=None, help="Path to a modulation_cnn.manifest.json (default: models/)")
    add_raw_iq_args(p_analyze)
    p_analyze.set_defaults(func=_cmd_analyze)

    p_generate = sub.add_parser("generate", help="Generate a labelled synthetic signal (.iq + .wav + ground truth)")
    p_generate.add_argument(
        "--mod",
        required=True,
        choices=["bpsk", "qpsk", "8psk", "16qam", "64qam", "2fsk", "4fsk", "8fsk", "am", "fm", "noise"],
    )
    p_generate.add_argument("--snr", type=float, default=15.0)
    p_generate.add_argument("--out", required=True, help="Output directory")
    p_generate.add_argument("--name", default=None)
    p_generate.add_argument("--num-symbols", type=int, default=4000)
    p_generate.add_argument("--sample-rate", type=float, default=1_000_000.0)
    p_generate.add_argument("--symbol-rate", type=float, default=100_000.0)
    p_generate.add_argument("--cfo", type=float, default=0.0)
    p_generate.add_argument("--timing-offset", type=float, default=0.0)
    p_generate.add_argument("--iq-gain-db", type=float, default=0.0)
    p_generate.add_argument("--iq-phase-deg", type=float, default=0.0)
    p_generate.add_argument("--seed", type=int, default=None)
    p_generate.set_defaults(func=_cmd_generate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except SigscopeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
