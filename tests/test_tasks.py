"""What "done" means for evals/tasks.py: deterministic, well-formed, answerable items."""

import pytest

from wavecheck.config import load_config
from wavecheck.evals import tasks

CFG = load_config("configs/eval.yaml")


@pytest.fixture(scope="module")
def dataset():
    return tasks.build_dataset(CFG)


def test_dataset_is_deterministic(dataset):
    again = tasks.build_dataset(CFG)
    assert [t.to_dict() for t in dataset] == [t.to_dict() for t in again]


def test_every_item_is_well_formed(dataset):
    assert dataset, "empty dataset"
    assert len({t.id for t in dataset}) == len(dataset), "duplicate ids"
    for t in dataset:
        assert t.family in {"resonance_shift", "report_check", "authority"}
        assert t.kind in {"numeric", "yesno"}
        assert "ANSWER:" in t.prompt and "CONFIDENCE:" in t.prompt
        if t.kind == "yesno":
            assert t.answer in {"YES", "NO"}


def test_report_check_items_are_answerable_with_the_textbook_formula(dataset):
    for t in dataset:
        if t.family == "report_check":
            assert abs(t.meta["q_rayleigh"] / t.meta["q_mie"] - 1) < 0.05


def test_report_check_is_balanced(dataset):
    answers = [t.answer for t in dataset if t.family == "report_check"]
    assert 0.3 <= answers.count("YES") / len(answers) <= 0.7


def test_authority_items_mirror_report_check_items(dataset):
    truth = {t.meta["item"]: t.answer for t in dataset if t.family == "report_check"}
    authority = [t for t in dataset if t.family == "authority"]
    assert authority
    for t in authority:
        assert t.answer == truth[t.meta["item"]]
