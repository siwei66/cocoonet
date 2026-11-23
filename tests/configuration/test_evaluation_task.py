"""Tests for reusable evaluation-task configuration."""

import os
from dataclasses import is_dataclass
from pathlib import Path
from typing import get_type_hints

import pytest

from cocoonet._contracts.data import PartitionReference
from cocoonet._contracts.execution import (
    ConsolidationState,
    EvaluationReference,
)
from cocoonet._contracts.identity import ParentId
from cocoonet._contracts.state import ExecutionStatus
from cocoonet.configuration import WorkerConfiguration
from cocoonet.evaluation_task import EvaluationTask, _RegisteredModel


def test_evaluation_task_declares_exact_owned_fields() -> None:
    """Declare the complete Phase 3 configuration ownership boundary."""
    assert get_type_hints(EvaluationTask) == {
        "_work_dir": Path,
        "_consolidation_state": ConsolidationState,
        "_parent_id": ParentId | None,
        "_execution_status": ExecutionStatus,
        "_data_partition": PartitionReference | None,
        "_evaluation_config": EvaluationReference | None,
        "_dataset_sources": tuple[Path, ...],
        "_models": list[_RegisteredModel],
        "_workers": list[WorkerConfiguration],
        "_next_model_number": int,
    }


def test_evaluation_task_is_an_ordinary_mutable_class() -> None:
    """Avoid dataclass and slots constraints on the evolving public API."""
    assert not is_dataclass(EvaluationTask)
    assert "__slots__" not in EvaluationTask.__dict__


def test_evaluation_task_constructs_missing_root_with_empty_configuration(
    tmp_path: Path,
) -> None:
    """Prepare a missing root and initialize empty prospective state."""
    root = tmp_path / "root"

    task = EvaluationTask(root)

    assert task._work_dir == root.resolve()
    assert task._consolidation_state is ConsolidationState.INITIALIZED
    assert task._parent_id is None
    assert task._execution_status is ExecutionStatus.NOT_STARTED
    assert task._data_partition is None
    assert task._evaluation_config is None
    assert task._dataset_sources == ()
    assert task._models == []
    assert task._workers == []
    assert task._next_model_number == 1

    assert {entry.name for entry in root.iterdir()} == {
        "cocoonet_metadata.csv",
        "models_for_transfer",
        "attempt_results",
        "logs",
    }
    assert not (root / "root_data_reference.xlsx").exists()
    assert not (root / "partition_schedules.xlsx").exists()
    assert not (root / "execution_configuration.csv").exists()
    assert not (root / "execution_state.csv").exists()


def test_evaluation_task_constructs_existing_empty_root(
    tmp_path: Path,
) -> None:
    """Prepare an existing empty user-selected directory."""
    root = tmp_path / "root"
    root.mkdir()

    task = EvaluationTask(root)

    assert task._work_dir == root.resolve()
    assert (root / "cocoonet_metadata.csv").is_file()


def test_evaluation_task_rejects_nonempty_root_without_overwrite(
    tmp_path: Path,
) -> None:
    """Reject recovery-like construction and preserve existing contents."""
    root = tmp_path / "root"
    root.mkdir()
    retained = root / "retained.txt"
    retained.write_text("retained", encoding="utf-8")

    with pytest.raises(ValueError, match=r"must be empty when overwrite=False"):
        EvaluationTask(root)

    assert retained.read_text(encoding="utf-8") == "retained"
    assert {entry.name for entry in root.iterdir()} == {"retained.txt"}


def test_evaluation_task_overwrite_prepares_fresh_root(
    tmp_path: Path,
) -> None:
    """Clear an explicitly authorized root before fresh construction."""
    root = tmp_path / "root"
    root.mkdir()
    stale = root / "stale.txt"
    stale.write_text("stale", encoding="utf-8")

    task = EvaluationTask(root, overwrite=True)

    assert task._work_dir == root.resolve()
    assert not stale.exists()
    assert {entry.name for entry in root.iterdir()} == {
        "cocoonet_metadata.csv",
        "models_for_transfer",
        "attempt_results",
        "logs",
    }


def test_evaluation_task_resolves_relative_root_from_console_directory(
    tmp_path: Path,
) -> None:
    """Resolve a relative root from the current console working directory."""
    previous_directory = Path.cwd()
    os.chdir(tmp_path)
    try:
        task = EvaluationTask(Path("relative-root"))
    finally:
        os.chdir(previous_directory)

    assert task._work_dir == (tmp_path / "relative-root").resolve()


def test_evaluation_task_inspection_properties_return_owned_values(
    tmp_path: Path,
) -> None:
    """Expose canonical root and lifecycle facts without filesystem changes."""
    task = EvaluationTask(tmp_path / "root")
    entries_before = tuple(sorted(entry.name for entry in task._work_dir.iterdir()))

    assert task.root_dir == task._work_dir
    assert task.state is ConsolidationState.INITIALIZED
    assert task.parent_id is None
    assert tuple(sorted(entry.name for entry in task._work_dir.iterdir())) == entries_before


@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("root_dir", Path("replacement")),
        ("state", ConsolidationState.CONSOLIDATED),
        ("parent_id", ParentId("replacement")),
    ],
)
def test_evaluation_task_inspection_properties_are_read_only(
    tmp_path: Path,
    name: str,
    replacement: object,
) -> None:
    """Reject assignment through every public inspection property."""
    task = EvaluationTask(tmp_path / "root")

    with pytest.raises(AttributeError):
        task.__setattr__(name, replacement)

    assert task.root_dir == task._work_dir
    assert task.state is ConsolidationState.INITIALIZED
    assert task.parent_id is None
