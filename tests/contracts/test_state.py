"""Tests for orthogonal execution-state contracts."""

from dataclasses import FrozenInstanceError

import pytest

from cocoonet._contracts.identity import ParentId
from cocoonet._contracts.state import AttemptStatus, ExecutionStatus, ParentProgress


def test_execution_status_has_exact_string_vocabulary() -> None:
    """Expose only the planned parent-execution states."""
    assert tuple(ExecutionStatus) == (
        ExecutionStatus.NOT_STARTED,
        ExecutionStatus.RUNNING,
        ExecutionStatus.INTERRUPTED,
        ExecutionStatus.SETTLED,
    )
    assert ExecutionStatus.NOT_STARTED == "not_started"
    assert ExecutionStatus.RUNNING == "running"
    assert ExecutionStatus.INTERRUPTED == "interrupted"
    assert ExecutionStatus.SETTLED == "settled"


def test_parent_progress_preserves_execution_observation() -> None:
    """Preserve the parent identity and root-observed execution status."""
    progress = ParentProgress(
        parent_id=ParentId("parent-a"),
        execution_status=ExecutionStatus.RUNNING,
    )
    same = ParentProgress(
        parent_id=ParentId("parent-a"),
        execution_status=ExecutionStatus.RUNNING,
    )
    different_status = ParentProgress(
        parent_id=ParentId("parent-a"),
        execution_status=ExecutionStatus.SETTLED,
    )

    assert progress.parent_id == ParentId("parent-a")
    assert progress.execution_status is ExecutionStatus.RUNNING
    assert progress == same
    assert progress != different_status


def test_parent_progress_rejects_field_mutation() -> None:
    """Prevent reassignment of a parent execution observation."""
    progress = ParentProgress(
        parent_id=ParentId("parent-a"),
        execution_status=ExecutionStatus.RUNNING,
    )

    with pytest.raises(FrozenInstanceError):
        progress.__setattr__(
            "execution_status",
            ExecutionStatus.SETTLED,
        )


def test_attempt_status_has_exact_string_vocabulary() -> None:
    """Expose only the planned worker-side attempt-processing states."""
    assert tuple(AttemptStatus) == (
        AttemptStatus.ALLOCATED,
        AttemptStatus.RUNNING,
        AttemptStatus.RESULT_READY,
        AttemptStatus.STOPPED,
    )
    assert AttemptStatus.ALLOCATED == "allocated"
    assert AttemptStatus.RUNNING == "running"
    assert AttemptStatus.RESULT_READY == "result_ready"
    assert AttemptStatus.STOPPED == "stopped"
