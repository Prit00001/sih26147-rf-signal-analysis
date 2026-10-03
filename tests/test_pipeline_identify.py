"""Integration tests: interleaver/FEC identification end-to-end through the
real ingest -> demod -> identify pipeline (pipeline_core.run_full_pipeline),
not just on synthetic bit arrays in isolation.

Covers: FR-07, FR-08 (identification wired into the shared pipeline used by
both GUIs). Real bugs found and fixed while building this (see fec/identify.py
and pipeline_core.py docstrings): (1) FEC identification must run on the
DE-interleaved stream when an interleaver is found -- a block code's
row-wise redundancy is invisible in the raw interleaved stream; (2)
identify_rs's naive argmax picked the weakest nested k (n-k smaller = fewer
constraints checked = trivially also satisfied) instead of the true one.
"""

from __future__ import annotations

import numpy as np
import pytest

from sigscope.fec.convolutional import conv_encode
from sigscope.fec.interleave import block_interleave
from sigscope.fec.reed_solomon import RSCode
from sigscope.pipeline_core import run_full_pipeline
from sigscope.synth.generator import generate_signal, write_pair

pytest.importorskip("sigscope._native", reason="native extension not built")


def test_convolutional_code_identified_through_full_pipeline(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(2)
    msg_bits = rng.integers(0, 2, 4000)
    coded = conv_encode(msg_bits, 7, [0o171, 0o133])
    num_symbols = len(coded) // 2
    sig, gt = generate_signal(
        "qpsk",
        num_symbols=num_symbols,
        sample_rate=1_000_000.0,
        symbol_rate=100_000.0,
        snr_db=18.0,
        bits=coded[: num_symbols * 2].astype(np.int64),
        seed=2,
    )
    paths = write_pair(sig, gt, tmp_path, "conv_demo")
    result = run_full_pipeline(str(paths["wav"]), modulation_override="qpsk")
    assert result.fec_label == "convolutional (rate1/2_K7)"
    assert result.fec_confidence > 0.8
    # Corrected (a real bug caught while building a full RS+interleave+conv
    # demo): a convolutional match on the raw stream does NOT prove there is
    # no interleaving, because conv coding is typically the innermost
    # transform before modulation -- an outer code + interleaver could still
    # be hidden behind it, undetectable by the GF2-rank interleaver check
    # (which cannot see through a conv transform). Only a direct RS match
    # soundly proves "no interleaving" (RS codeword validity is bit-exact
    # and any permutation would destroy it). This demo has no interleaver
    # AND no outer code, but the pipeline correctly can't prove that from a
    # conv-only match alone -- "unidentified" is the honest answer.
    assert result.interleaver_label == "unidentified"


def test_rs_plus_interleaver_identified_through_full_pipeline(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(3)
    rs = RSCode(m=4, n=15, k=9)
    symbols = np.concatenate([rs.encode(rng.integers(0, 16, rs.k)) for _ in range(60)])
    bits = ((symbols[:, None] >> np.arange(3, -1, -1)) & 1).reshape(-1).astype(np.int64)
    cols = 60
    rows = len(bits) // cols
    interleaved = block_interleave(bits[: rows * cols], rows, cols)
    num_symbols = len(interleaved) // 2
    sig, gt = generate_signal(
        "qpsk",
        num_symbols=num_symbols,
        sample_rate=1_000_000.0,
        symbol_rate=100_000.0,
        snr_db=18.0,
        bits=interleaved[: num_symbols * 2].astype(np.int64),
        seed=3,
    )
    paths = write_pair(sig, gt, tmp_path, "rs_interleave_demo")
    result = run_full_pipeline(str(paths["wav"]), modulation_override="qpsk")
    assert result.interleaver_label == "block (period=60)"
    assert result.fec_label == "reed-solomon (n=15, k=9, m=4)"
    # Regression test for a real bug: RS decode must run on the SAME stream
    # the match was found on (de-interleaved, since an interleaver was
    # identified here) -- decoding the still-interleaved raw bits as if they
    # were direct RS codewords previously made 55/60 blocks look
    # "uncorrectable" even at 0% channel BER (confirmed via ground-truth
    # comparison). At 18dB SNR, the large majority of blocks must decode
    # cleanly once the correct stream is used.
    assert result.fec_blocks_total is not None and result.fec_blocks_total > 0
    assert result.fec_blocks_uncorrectable is not None
    assert result.fec_blocks_uncorrectable < result.fec_blocks_total * 0.5


def test_plain_uncoded_signal_honestly_reports_unidentified(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """FEC deliberately stays a two-outcome (identified/unidentified) result:
    a third "none present" outcome was attempted via the bit-statistics
    classifier and REJECTED after testing proved it unreliable on exactly
    this uncoded case (see fec/identify.py's BitStatsCentroidClassifier
    docstring) -- "unidentified" is the honest answer, not a confident wrong
    one. Interleaving on uncoded data is likewise undetectable by this method
    (a permutation of bits with no redundancy changes nothing statistically)."""
    sig, gt = generate_signal(
        "qpsk", num_symbols=4000, sample_rate=1_000_000.0, symbol_rate=100_000.0, snr_db=18.0, seed=4
    )
    paths = write_pair(sig, gt, tmp_path, "plain_demo")
    result = run_full_pipeline(str(paths["wav"]), modulation_override="qpsk")
    assert result.interleaver_label == "unidentified"
    assert result.fec_label == "unidentified"


def test_interleaver_under_a_conv_code_is_not_falsely_claimed_absent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Regression test for a real bug: RS (outer) -> interleave -> conv
    (inner) is the architecturally correct concatenated chain, and a
    confident convolutional match on the raw received stream must NOT be
    read as proof the interleaver is absent -- conv coding sits between the
    channel and whatever outer code+interleaver might exist, so a conv match
    alone says nothing about what's hidden behind it. Before the first fix,
    this produced a false "none present" claim for an interleaver that WAS
    applied.

    Since then, pipeline_core gained a second fix that goes further than
    just not-lying: once a convolutional code is confidently identified, it
    is Viterbi-decoded, and the SAME interleaver+FEC search is tried one
    level deeper on the decoded payload -- recovering the full chain instead
    of stopping at "unidentified but not falsely absent". This signal's
    interleaver and outer RS code are both now actually found.
    """
    rng = np.random.default_rng(42)
    rs = RSCode(m=4, n=15, k=9)
    rs_symbols = np.concatenate([rs.encode(rng.integers(0, 16, rs.k)) for _ in range(60)])
    rs_bits = ((rs_symbols[:, None] >> np.arange(3, -1, -1)) & 1).reshape(-1).astype(np.int64)
    cols = 60
    rows = len(rs_bits) // cols
    interleaved = block_interleave(rs_bits[: rows * cols], rows, cols)
    coded = conv_encode(interleaved, 7, [0o171, 0o133])
    num_symbols = len(coded) // 2
    sig, gt = generate_signal(
        "qpsk",
        num_symbols=num_symbols,
        sample_rate=1_000_000.0,
        symbol_rate=100_000.0,
        snr_db=18.0,
        bits=coded[: num_symbols * 2].astype(np.int64),
        seed=42,
    )
    paths = write_pair(sig, gt, tmp_path, "rs_interleave_conv_demo")
    result = run_full_pipeline(str(paths["wav"]), modulation_override="qpsk")
    assert result.fec_label == "convolutional (rate1/2_K7) + reed-solomon (n=15, k=9, m=4)"
    assert result.interleaver_label == "block (period=60)"
