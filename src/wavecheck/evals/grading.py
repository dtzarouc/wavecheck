"""Parse model answers and score them.

Unparseable answers are graded None and reported separately, never as wrong.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from wavecheck.evals.tasks import Task


def parse_answer(text: str) -> tuple[str | None, float | None]:
    """Return (answer, confidence) from the LAST "ANSWER:" and "CONFIDENCE:" lines.

    The answer is the stripped text after "ANSWER:". The confidence is a float in [0, 1], or None
    if missing or unparseable. Returns (None, None) when there is no ANSWER line.
    """
    answers = re.findall(r"^\s*\**ANSWER\**\s*:\s*(.+?)\s*$", text, flags=re.M | re.I)
    if not answers:
        return None, None
    answer = answers[-1].strip().strip("*").strip()
    confidences = re.findall(r"^\s*\**CONFIDENCE\**\s*:\s*(.+?)\s*$", text, flags=re.M | re.I)
    confidence = None
    if confidences:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*(%?)", confidences[-1])
        if m:
            value = float(m.group(0).rstrip("% "))
            if m.group(1) == "%":
                value /= 100.0
            confidence = value if 0.0 <= value <= 1.0 else None
    return answer, confidence


def grade(task: Task, answer: str | None) -> bool | None:
    """Score one answer.

    numeric: True if |value / task.answer - 1| <= task.tolerance_rel.
    yesno: case-insensitive match with task.answer.
    Returns None when the answer cannot be parsed; report these separately, never as wrong.
    """
    if answer is None:
        return None
    if task.kind == "yesno":
        word = re.match(r"\W*(yes|no)\b", answer, flags=re.I)
        return None if word is None else word.group(1).upper() == task.answer
    # numeric: the answer must start with a number (unicode minus and "eps =" prefixes allowed)
    cleaned = answer.replace("\u2212", "-").replace("\u2248", "=").replace(" ", "")
    m = re.match(r"^(?:\S*?=)?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", cleaned)
    if m is None:
        return None
    value = float(m.group(1))
    return bool(abs(value / float(task.answer) - 1.0) <= float(task.tolerance_rel))


def brier(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Mean squared difference between confidence and outcome (1 if correct, else 0)."""
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    return float(np.mean((c - y) ** 2))


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> float:
    """Equal-width bins on [0, 1] (a confidence of 1.0 falls in the last bin).

    ECE = sum over non-empty bins of (bin size / total) x |bin accuracy - bin mean confidence|.
    """
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    # np.minimum puts confidence 1.0 into the last bin instead of an extra one.
    bins = np.minimum((c * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in np.unique(bins):
        sel = bins == b
        ece += sel.mean() * abs(y[sel].mean() - c[sel].mean())
    return float(ece)
