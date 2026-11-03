"""What "done" means for evals/grading.py."""

import pytest

from wavecheck.evals import grading
from wavecheck.evals.tasks import Task


def test_parse_takes_the_last_answer_block():
    text = "ANSWER: 1\nOn reflection...\nANSWER: -2.31\nCONFIDENCE: 0.8"
    assert grading.parse_answer(text) == ("-2.31", 0.8)


def test_parse_without_confidence():
    assert grading.parse_answer("Reasoning.\nANSWER: YES") == ("YES", None)


def test_parse_without_answer():
    assert grading.parse_answer("I am not sure.") == (None, None)


def test_numeric_grading_uses_relative_tolerance():
    t = Task(
        id="t", family="resonance_shift", prompt="", answer=-2.2, kind="numeric", tolerance_rel=0.02
    )
    assert grading.grade(t, "-2.23") is True
    assert grading.grade(t, "-2.0") is False
    assert grading.grade(t, "about minus two") is None


def test_yesno_grading_is_case_insensitive():
    t = Task(id="t", family="report_check", prompt="", answer="NO", kind="yesno")
    assert grading.grade(t, "no") is True
    assert grading.grade(t, "YES") is False


def test_brier_score():
    assert grading.brier([1.0, 0.0], [True, False]) == pytest.approx(0.0)
    assert grading.brier([0.5, 0.5], [True, False]) == pytest.approx(0.25)


def test_ece_is_zero_when_calibrated():
    conf = [0.25] * 4 + [0.75] * 4
    correct = [True, False, False, False] + [True, True, True, False]
    assert grading.expected_calibration_error(conf, correct, n_bins=2) == pytest.approx(0.0)


def test_ece_measures_overconfidence():
    conf, correct = [0.9] * 10, [True] * 5 + [False] * 5
    assert grading.expected_calibration_error(conf, correct, n_bins=10) == pytest.approx(0.4)
