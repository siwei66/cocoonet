"""Tests for the package-root public interface."""

import json
import subprocess
import sys
from pathlib import Path

import cocoonet
from cocoonet.evaluation_task import EvaluationTask


def test_package_root_exports_evaluation_task() -> None:
    """Expose the completed evaluation task through the package root."""
    assert cocoonet.EvaluationTask is EvaluationTask
    assert cocoonet.__all__ == ["EvaluationTask", "get_version"]


def test_evaluation_task_construction_does_not_load_pandas(
    tmp_path: Path,
) -> None:
    """Construct a task without loading the optional table presentation layer."""
    work_dir = tmp_path / "subprocess-work-dir"
    script = (
        "import json\n"
        "import sys\n"
        "from cocoonet import EvaluationTask\n"
        f"EvaluationTask({str(work_dir)!r})\n"
        "print(json.dumps('pandas' in sys.modules))\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) is False
