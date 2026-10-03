"""TEMPORARY diagnostic script -- not part of the project, to be deleted
once the Linux-CI-only empty-hard_bits failure is root-caused. Prints
intermediate array lengths/stats through the demod chain for a plain QPSK
signal, to find where it goes empty on Linux but not macOS.
"""

from __future__ import annotations

import numpy as np

from sigscope.demod.sync import estimate_cfo_mth_power, matched_filter, gardner_timing_recovery, decision_directed_carrier_recovery
from sigscope.demod.psk_qam import demodulate_psk_qam
from sigscope.synth.generator import generate_signal, constellation_for

sig, gt = generate_signal(
    "qpsk", num_symbols=4000, sample_rate=1_000_000.0, symbol_rate=100_000.0, snr_db=15.0, seed=101
)
print("platform numpy version:", np.__version__)
print("len(sig.samples):", len(sig.samples), "dtype:", sig.samples.dtype)
print("sample_rate:", sig.sample_rate, "symbol_rate:", gt.symbol_rate)
sps = sig.sample_rate / gt.symbol_rate
print("sps (float):", sps, "round(sps):", round(sps))

cfo_est = estimate_cfo_mth_power(sig.samples, sig.sample_rate, 4)
print("cfo_est:", cfo_est)
n = np.arange(len(sig.samples))
coarse = (sig.samples.astype(np.complex128) * np.exp(-1j * 2 * np.pi * cfo_est * n / sig.sample_rate)).astype(np.complex64)
print("coarse: any nan?", np.any(np.isnan(coarse)), "any inf?", np.any(np.isinf(coarse)))

filtered = matched_filter(coarse, 0.35, 8, int(round(sps)))
print("len(filtered):", len(filtered), "any nan?", np.any(np.isnan(filtered)))

symbols = gardner_timing_recovery(filtered, sps, gain=0.002)
print("len(symbols) after gardner:", len(symbols))

result = demodulate_psk_qam(sig.samples, "qpsk", sig.sample_rate, gt.symbol_rate)
print("len(result.hard_bits):", len(result.hard_bits))
print("len(result.symbols):", len(result.symbols))
