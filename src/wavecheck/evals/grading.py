"""Parse model answers and score them.

TODO: implement; tests/test_grading.py defines "done".
"""

from __future__ import annotations

from collections.abc import Sequence

from wavecheck.evals.tasks import Task


def parse_answer(text: str) -> tuple[str | None, float | None]:
    """Return (answer, confidence) from the LAST "ANSWER:" and "CONFIDENCE:" lines.

    The answer is the stripped text after "ANSWER:". The confidence is a float in [0, 1], or None
    if missing or unparseable. Returns (None, None) when there is no ANSWER line.
    """
    raise NotImplementedError


def grade(task: Task, answer: str | None) -> bool | None:
    """Score one answer.

    numeric: True if |value / task.answer - 1| <= task.tolerance_rel.
    yesno: case-insensitive match with task.answer.
    Returns None when the answer cannot be parsed; report these separately, never as wrong.
    """
    raise NotImplementedError


def brier(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Mean squared difference between confidence and outcome (1 if correct, else 0)."""
    raise NotImplementedError


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> float:
    """Equal-width bins on [0, 1] (a confidence of 1.0 falls in the last bin).

    ECE = sum over non-empty bins of (bin size / total) x |bin accuracy - bin mean confidence|.
    """
    raise NotImplementedError
