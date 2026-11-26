"""Tests for the package-root public interface."""

import cocoonet
from cocoonet.evaluation_task import EvaluationTask


def test_package_root_exports_evaluation_task() -> None:
    """Expose the completed evaluation task through the package root."""
    assert cocoonet.EvaluationTask is EvaluationTask
    assert cocoonet.__all__ == ["EvaluationTask", "get_version"]
