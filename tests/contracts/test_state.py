"""Tests for orthogonal execution-state contracts."""

from dataclasses import FrozenInstanceError

import pytest

from cocoonet._contracts.execution import AttemptInputs
from cocoonet._contracts.identity import ArtifactId, AttemptId, ModelId, ParentId, WorkerId
from cocoonet._contracts.state import (
    AttemptProgress,
    AttemptStatus,
    ExecutionStatus,
    ParentProgress,
    StopDisposition,
    StopReason,
    StopStatus,
)


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


def test_stop_status_has_exact_string_vocabulary() -> None:
    """Expose only the planned cooperative-stop handling states."""
    assert tuple(StopStatus) == (
        StopStatus.NONE,
        StopStatus.REQUESTED,
        StopStatus.OBSERVED,
    )
    assert StopStatus.NONE == "none"
    assert StopStatus.REQUESTED == "requested"
    assert StopStatus.OBSERVED == "observed"


def test_stop_reason_has_exact_string_vocabulary() -> None:
    """Expose only the planned reasons for cooperative attempt stopping."""
    assert tuple(StopReason) == (
        StopReason.PARENT_INTERRUPTED,
        StopReason.PARENT_CANCELLED,
        StopReason.PARENT_SUPERSEDED,
        StopReason.DUPLICATE_ACCEPTED,
    )
    assert StopReason.PARENT_INTERRUPTED == "parent_interrupted"
    assert StopReason.PARENT_CANCELLED == "parent_cancelled"
    assert StopReason.PARENT_SUPERSEDED == "parent_superseded"
    assert StopReason.DUPLICATE_ACCEPTED == "duplicate_accepted"


def test_stop_disposition_preserves_stop_observation() -> None:
    """Preserve the cooperative-stop status and its associated reason."""
    disposition = StopDisposition(
        status=StopStatus.REQUESTED,
        reason=StopReason.PARENT_INTERRUPTED,
    )
    same = StopDisposition(
        status=StopStatus.REQUESTED,
        reason=StopReason.PARENT_INTERRUPTED,
    )
    no_stop = StopDisposition(
        status=StopStatus.NONE,
        reason=None,
    )

    assert disposition.status is StopStatus.REQUESTED
    assert disposition.reason is StopReason.PARENT_INTERRUPTED
    assert disposition == same
    assert disposition != no_stop


def test_stop_disposition_rejects_field_mutation() -> None:
    """Prevent reassignment of a cooperative-stop observation."""
    disposition = StopDisposition(
        status=StopStatus.REQUESTED,
        reason=StopReason.PARENT_CANCELLED,
    )

    with pytest.raises(FrozenInstanceError):
        disposition.__setattr__(
            "status",
            StopStatus.OBSERVED,
        )


def test_attempt_progress_preserves_attempt_observation() -> None:
    """Preserve routing, processing, stop, start, and result observations."""
    inputs = AttemptInputs(
        attempt_id=AttemptId("attempt-a"),
        parent_id=ParentId("parent-a"),
        model_id=ModelId("model-a"),
        worker_id=WorkerId("worker-a"),
    )
    stop = StopDisposition(
        status=StopStatus.REQUESTED,
        reason=StopReason.DUPLICATE_ACCEPTED,
    )
    progress = AttemptProgress(
        inputs=inputs,
        status=AttemptStatus.RESULT_READY,
        stop=stop,
        first_started_at=1_700_000_000.5,
        result_artifact_id=ArtifactId("artifact-a"),
    )
    same = AttemptProgress(
        inputs=inputs,
        status=AttemptStatus.RESULT_READY,
        stop=stop,
        first_started_at=1_700_000_000.5,
        result_artifact_id=ArtifactId("artifact-a"),
    )

    assert progress.inputs == inputs
    assert progress.status is AttemptStatus.RESULT_READY
    assert progress.stop == stop
    assert progress.first_started_at == 1_700_000_000.5
    assert progress.result_artifact_id == ArtifactId("artifact-a")
    assert progress == same


def test_attempt_progress_rejects_field_mutation() -> None:
    """Prevent reassignment of an attempt-progress observation."""
    progress = AttemptProgress(
        inputs=AttemptInputs(
            attempt_id=AttemptId("attempt-a"),
            parent_id=ParentId("parent-a"),
            model_id=ModelId("model-a"),
            worker_id=WorkerId("worker-a"),
        ),
        status=AttemptStatus.ALLOCATED,
        stop=StopDisposition(
            status=StopStatus.NONE,
            reason=None,
        ),
        first_started_at=None,
        result_artifact_id=None,
    )

    with pytest.raises(FrozenInstanceError):
        progress.__setattr__(
            "status",
            AttemptStatus.RUNNING,
        )
