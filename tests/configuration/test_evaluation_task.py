"""Tests for reusable evaluation-task configuration."""

from dataclasses import is_dataclass
from pathlib import Path
from typing import get_type_hints

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
