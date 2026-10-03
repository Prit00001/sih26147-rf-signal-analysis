"""Reed-Solomon decode time: before (combinatorial search) vs after
(Berlekamp-Massey + Chien search + Forney) -- see reed_solomon.py's
decode()/`_decode_combinatorial` docstrings for the algorithms themselves.

Run: .venv/bin/python scripts/benchmark_rs_decode.py --out reports
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from sigscope.fec.reed_solomon import RSCode, UncorrectableError

# (m, n, k) -- RS(255,223) is the CCSDS-standard code this project's own
# README cites as the target this upgrade exists for; the smaller codes are
# included so the OLD decoder's numbers can still be measured at all (it is
# not feasible to even run it at RS(255,223) scale -- C(255,16) candidate
# subsets -- which is the entire point of the "before" column being blank
# there, not a benchmark oversight).
CONFIGS: list[tuple[int, int, int]] = [(4, 15, 9), (5, 31, 21), (8, 255, 223)]
TRIALS_PER_CONFIG = 30
# The old decoder's cost is O(C(n,t)): MEASURED, a single RS(63,51) (t=6)
# decode at full t already exceeds 120s, so it is only timed at all for
# n<=31, and with fewer trials there (C(31,5)~1.7e5 candidates per call is
# still seconds, not milliseconds).
OLD_DECODER_MAX_N = 31
OLD_DECODER_TRIALS = 5


def _time_decode(decode_fn, codeword: np.ndarray, num_err: int, rng: np.random.Generator, field_size: int) -> float:
    received = codeword.copy()
    positions = rng.choice(len(codeword), size=num_err, replace=False)
    for p in positions:
        received[p] = int(received[p]) ^ int(rng.integers(1, field_size))
    t0 = time.perf_counter()
    try:
        decode_fn(received)
    except UncorrectableError:
        pass
    return (time.perf_counter() - t0) * 1000.0


def run_benchmark() -> list[dict[str, object]]:
    rng = np.random.default_rng(2024)
    rows: list[dict[str, object]] = []
    for m, n, k in CONFIGS:
        rs = RSCode(m=m, n=n, k=k)
        field_size = 1 << m
        new_times = []
        old_times = []
        for _ in range(TRIALS_PER_CONFIG):
            message = rng.integers(0, field_size, rs.k)
            codeword = rs.encode(message)
            num_err = rs.t  # worst-case (full correction radius) decode time
            new_times.append(_time_decode(rs.decode, codeword, num_err, rng, field_size))
            if n <= OLD_DECODER_MAX_N and len(old_times) < OLD_DECODER_TRIALS:
                old_times.append(_time_decode(rs._decode_combinatorial, codeword, num_err, rng, field_size))
        row = {
            "m": m, "n": n, "k": k, "t": rs.t,
            "new_mean_ms": round(float(np.mean(new_times)), 4),
            "new_max_ms": round(float(np.max(new_times)), 4),
            "old_mean_ms": round(float(np.mean(old_times)), 4) if old_times else None,
            "old_max_ms": round(float(np.max(old_times)), 4) if old_times else None,
        }
        rows.append(row)
        old_str = f"{row['old_mean_ms']} ms" if row["old_mean_ms"] is not None else "not feasible to run"
        print(
            f"RS(n={n},k={k},m={m}, t={rs.t}): new={row['new_mean_ms']} ms (max {row['new_max_ms']}), "
            f"old={old_str}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = run_benchmark()
    csv_path = args.out / "rs_decode_time_before_after.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
