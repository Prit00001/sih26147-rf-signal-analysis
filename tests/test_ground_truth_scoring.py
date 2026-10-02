"""Tests for the ground-truth comparison SCORING logic itself (not just that
the panel renders) -- in particular the rule that a genuinely ABSENT ground-
truth feature (no framing/code applied) correctly scored as "not detected"
by the dashboard counts as a MATCH, not a miss.
"""

from __future__ import annotations

from sigscope.webui.server import _NOT_DETECTED_THRESHOLD, _absent_or_value_row


def test_absent_expected_and_not_detected_is_a_match() -> None:
    row = _absent_or_value_row("frame_length", None, 18, 0.04)  # 4% < threshold -> "not detected"
    assert row["match"] is True
    assert row["expected"] == "none"
    assert row["detected"] == "not detected"


def test_absent_expected_but_confidently_detected_is_a_mismatch() -> None:
    """If ground truth says absent but the pipeline confidently reports a
    value anyway, that is a real false positive -- must NOT be scored as a
    match just because the expected side says "none"."""
    row = _absent_or_value_row("frame_length", None, 18, 0.9)
    assert row["match"] is False
    assert row["detected"] == "18"


def test_present_expected_and_correctly_detected_is_a_match() -> None:
    row = _absent_or_value_row("frame_length", 72, 72, 0.9)
    assert row["match"] is True


def test_present_expected_but_wrong_value_is_a_mismatch() -> None:
    row = _absent_or_value_row("frame_length", 72, 18, 0.9)
    assert row["match"] is False


def test_present_expected_but_not_detected_is_a_mismatch() -> None:
    """Ground truth says the feature IS there; the dashboard saying "not
    detected" is a real miss, not something to be scored generously."""
    row = _absent_or_value_row("frame_length", 72, 0, 0.04)
    assert row["match"] is False
    assert row["detected"] == "not detected"


def test_threshold_boundary_is_exclusive_of_not_detected() -> None:
    row_at_threshold = _absent_or_value_row("frame_length", None, 5, _NOT_DETECTED_THRESHOLD)
    assert row_at_threshold["match"] is False  # confidence == threshold is NOT "below" it -> confidently detected
    row_just_below = _absent_or_value_row("frame_length", None, 5, _NOT_DETECTED_THRESHOLD - 0.001)
    assert row_just_below["match"] is True
