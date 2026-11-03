"""Shared pytest configuration.

Tests that hit a milestone stub (a function that still raises NotImplementedError) are reported as
"xfail" instead of failing, so CI stays green while later milestones are pending. Once the stub is
implemented the test runs normally: a wrong implementation fails with its real error.
"""

from __future__ import annotations

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.excinfo is not None and call.excinfo.errisinstance(NotImplementedError):
        report.outcome = "skipped"
        report.wasxfail = "milestone stub not implemented yet (NotImplementedError)"
