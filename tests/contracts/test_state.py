"""Tests for orthogonal execution-state contracts."""

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from cocoonet._contracts.data import PartitionReference
from cocoonet._contracts.execution import (
    AttemptInputs,
    EvaluationReference,
    ModelFamily,
    ModelReference,
    ParentEvaluation,
    TaskKind,
)
from cocoonet._contracts.identity import (
    ArtifactId,
    AttemptId,
    DatasetId,
    EvaluationConfigId,
    ModelId,
    ParentId,
    PartitionId,
    WorkerId,
)
from cocoonet._contracts.state import (
    AttemptProgress,
    AttemptStatus,
    CompatibilityStatus,
    ExecutionStatus,
    OwnershipStatus,
    ParentContextReadiness,
    ParentProgress,
    StopDisposition,
    StopReason,
    StopStatus,
    WorkerAvailability,
    WorkerProcessStatus,
    validate_attempt_progress,
    validate_parent_progress,
    validate_stop_disposition,
)


def _valid_parent_evaluation() -> ParentEvaluation:
    """Build a valid fixed parent context for state-validator tests."""
    partition = PartitionReference(
        partition_id=PartitionId("partition-a"),
        dataset_id=DatasetId("dataset-a"),
    )
    evaluation = EvaluationReference(
        configuration_id=EvaluationConfigId("evaluation-a"),
        task_kind=TaskKind.REGRESSION,
        class_labels=(),
        influence_requested=False,
    )
    model = ModelReference(
        model_id=ModelId("model-a"),
        model_label="model-a",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.REGRESSION,
    )
    return ParentEvaluation(
        parent_id=ParentId("parent-a"),
        partition=partition,
        evaluation=evaluation,
        models=(model,),
    )


def _attempt_progress(
    *,
    parent: ParentEvaluation,
    status: AttemptStatus,
    stop: StopDisposition,
    first_started_at: float | None,
    result_artifact_id: ArtifactId | None,
) -> AttemptProgress:
    """Build an attempt-progress candidate with fixed valid routing."""
    return AttemptProgress(
        inputs=AttemptInputs(
            attempt_id=AttemptId("attempt-a"),
            parent_id=parent.parent_id,
            model_id=parent.models[0].model_id,
            worker_id=WorkerId("worker-a"),
        ),
        status=status,
        stop=stop,
        first_started_at=first_started_at,
        result_artifact_id=result_artifact_id,
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


@pytest.mark.parametrize("status", tuple(ExecutionStatus))
def test_validate_parent_progress_accepts_each_execution_status(
    status: ExecutionStatus,
) -> None:
    """Accept every declared parent-execution status without inferring state."""
    progress = ParentProgress(
        parent_id=ParentId("parent-a"),
        execution_status=status,
    )

    validate_parent_progress(progress)
    assert progress.execution_status is status


def test_validate_parent_progress_rejects_empty_parent_identity() -> None:
    """Reject a parent observation whose parent identity is empty."""
    progress = ParentProgress(
        parent_id=ParentId(""),
        execution_status=ExecutionStatus.RUNNING,
    )

    with pytest.raises(ValueError) as error:
        validate_parent_progress(progress)

    assert str(error.value) == "parent_id must be a nonempty string; got an empty string."


def test_validate_parent_progress_rejects_invalid_execution_status() -> None:
    """Reject a value outside the declared parent-execution vocabulary."""
    progress = ParentProgress(
        parent_id=ParentId("parent-a"),
        execution_status=cast(ExecutionStatus, "settled"),
    )

    with pytest.raises(TypeError) as error:
        validate_parent_progress(progress)

    message = str(error.value)
    assert "execution_status" in message
    assert "ExecutionStatus" in message
    assert "builtins.str" in message
    assert "'settled'" in message


def test_validate_parent_progress_rejects_wrong_outer_record() -> None:
    """Reject a value that is not a ParentProgress record."""
    with pytest.raises(TypeError) as error:
        validate_parent_progress(cast(ParentProgress, None))

    message = str(error.value)
    assert "ParentProgress" in message
    assert "NoneType" in message
    assert "None" in message


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


def test_validate_stop_disposition_accepts_none_without_reason() -> None:
    """Accept no cooperative stop when no reason is recorded."""
    disposition = StopDisposition(
        status=StopStatus.NONE,
        reason=None,
    )

    validate_stop_disposition(disposition)


@pytest.mark.parametrize(
    "status",
    (
        StopStatus.REQUESTED,
        StopStatus.OBSERVED,
    ),
)
@pytest.mark.parametrize("reason", tuple(StopReason))
def test_validate_stop_disposition_accepts_stop_with_each_reason(
    status: StopStatus,
    reason: StopReason,
) -> None:
    """Accept every reason for requested and observed cooperative stops."""
    disposition = StopDisposition(
        status=status,
        reason=reason,
    )

    validate_stop_disposition(disposition)


@pytest.mark.parametrize(
    ("disposition", "expected_message"),
    (
        (
            StopDisposition(
                status=StopStatus.NONE,
                reason=StopReason.PARENT_CANCELLED,
            ),
            (
                "Invalid stop disposition:\n"
                "  status: none\n"
                "  expected: reason to be None\n"
                "  got: reason is parent_cancelled"
            ),
        ),
        (
            StopDisposition(
                status=StopStatus.REQUESTED,
                reason=None,
            ),
            (
                "Invalid stop disposition:\n"
                "  status: requested\n"
                "  expected: reason to be a StopReason\n"
                "  got: reason is None"
            ),
        ),
        (
            StopDisposition(
                status=StopStatus.OBSERVED,
                reason=None,
            ),
            (
                "Invalid stop disposition:\n"
                "  status: observed\n"
                "  expected: reason to be a StopReason\n"
                "  got: reason is None"
            ),
        ),
    ),
)
def test_validate_stop_disposition_rejects_inconsistent_pair(
    disposition: StopDisposition,
    expected_message: str,
) -> None:
    """Reject each invalid stop status/reason relationship."""
    with pytest.raises(ValueError) as error:
        validate_stop_disposition(disposition)

    assert str(error.value) == expected_message


@pytest.mark.parametrize(
    ("disposition", "field", "expected_type"),
    (
        (
            StopDisposition(
                status=cast(StopStatus, "requested"),
                reason=None,
            ),
            "status",
            "StopStatus",
        ),
        (
            StopDisposition(
                status=StopStatus.REQUESTED,
                reason=cast(StopReason, "parent_cancelled"),
            ),
            "reason",
            "StopReason",
        ),
    ),
)
def test_validate_stop_disposition_rejects_invalid_field_type(
    disposition: StopDisposition,
    field: str,
    expected_type: str,
) -> None:
    """Reject incorrectly typed stop fields before pairing validation."""
    with pytest.raises(TypeError) as error:
        validate_stop_disposition(disposition)

    message = str(error.value)
    assert field in message
    assert expected_type in message
    assert "builtins.str" in message


def test_validate_stop_disposition_rejects_wrong_outer_record() -> None:
    """Reject a value that is not a StopDisposition record."""
    with pytest.raises(TypeError) as error:
        validate_stop_disposition(cast(StopDisposition, None))

    message = str(error.value)
    assert "StopDisposition" in message
    assert "NoneType" in message
    assert "None" in message


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


@pytest.mark.parametrize(
    ("status", "stop", "first_started_at", "result_artifact_id"),
    (
        (
            AttemptStatus.ALLOCATED,
            StopDisposition(status=StopStatus.NONE, reason=None),
            None,
            None,
        ),
        (
            AttemptStatus.RUNNING,
            StopDisposition(
                status=StopStatus.REQUESTED,
                reason=StopReason.PARENT_INTERRUPTED,
            ),
            1_700_000_000.0,
            None,
        ),
        (
            AttemptStatus.RESULT_READY,
            StopDisposition(
                status=StopStatus.OBSERVED,
                reason=StopReason.DUPLICATE_ACCEPTED,
            ),
            1_700_000_000.0,
            ArtifactId("artifact-a"),
        ),
        (
            AttemptStatus.STOPPED,
            StopDisposition(
                status=StopStatus.OBSERVED,
                reason=StopReason.PARENT_CANCELLED,
            ),
            None,
            None,
        ),
        (
            AttemptStatus.STOPPED,
            StopDisposition(
                status=StopStatus.OBSERVED,
                reason=StopReason.DUPLICATE_ACCEPTED,
            ),
            1_700_000_000.0,
            ArtifactId("artifact-a"),
        ),
    ),
)
def test_validate_attempt_progress_accepts_valid_observations(
    status: AttemptStatus,
    stop: StopDisposition,
    first_started_at: float | None,
    result_artifact_id: ArtifactId | None,
) -> None:
    """Accept planned attempt states without creating or changing evidence."""
    parent = _valid_parent_evaluation()
    progress = _attempt_progress(
        parent=parent,
        status=status,
        stop=stop,
        first_started_at=first_started_at,
        result_artifact_id=result_artifact_id,
    )

    validate_attempt_progress(progress, parent=parent)

    assert progress.status is status
    assert progress.stop == stop
    assert progress.first_started_at == first_started_at
    assert progress.result_artifact_id == result_artifact_id


@pytest.mark.parametrize(
    ("status", "stop", "first_started_at", "result_artifact_id", "expected"),
    (
        (
            AttemptStatus.ALLOCATED,
            StopDisposition(status=StopStatus.NONE, reason=None),
            1.0,
            None,
            "first_started_at to be None",
        ),
        (
            AttemptStatus.RUNNING,
            StopDisposition(status=StopStatus.NONE, reason=None),
            None,
            None,
            "first_started_at to be present",
        ),
        (
            AttemptStatus.RESULT_READY,
            StopDisposition(status=StopStatus.NONE, reason=None),
            1.0,
            None,
            "result_artifact_id to be present",
        ),
        (
            AttemptStatus.STOPPED,
            StopDisposition(
                status=StopStatus.REQUESTED,
                reason=StopReason.PARENT_CANCELLED,
            ),
            None,
            None,
            "stop.status to be observed",
        ),
        (
            AttemptStatus.RUNNING,
            StopDisposition(
                status=StopStatus.OBSERVED,
                reason=StopReason.PARENT_INTERRUPTED,
            ),
            1.0,
            None,
            "stop.status not to be observed",
        ),
        (
            AttemptStatus.ALLOCATED,
            StopDisposition(status=StopStatus.NONE, reason=None),
            None,
            ArtifactId("artifact-a"),
            "result_artifact_id to be None",
        ),
        (
            AttemptStatus.STOPPED,
            StopDisposition(
                status=StopStatus.OBSERVED,
                reason=StopReason.PARENT_CANCELLED,
            ),
            None,
            ArtifactId("artifact-a"),
            "first_started_at to accompany result_artifact_id",
        ),
    ),
)
def test_validate_attempt_progress_rejects_inconsistent_observation(
    status: AttemptStatus,
    stop: StopDisposition,
    first_started_at: float | None,
    result_artifact_id: ArtifactId | None,
    expected: str,
) -> None:
    """Reject inconsistent attempt-status evidence without performing a transition."""
    parent = _valid_parent_evaluation()
    progress = _attempt_progress(
        parent=parent,
        status=status,
        stop=stop,
        first_started_at=first_started_at,
        result_artifact_id=result_artifact_id,
    )

    with pytest.raises(ValueError) as error:
        validate_attempt_progress(progress, parent=parent)

    message = str(error.value)
    assert f"status: {status.value}" in message
    assert expected in message
    assert "got:" in message


@pytest.mark.parametrize(
    "first_started_at",
    (
        -1.0,
        float("inf"),
        float("-inf"),
        float("nan"),
    ),
)
def test_validate_attempt_progress_rejects_invalid_timestamp_value(
    first_started_at: float,
) -> None:
    """Reject negative and nonfinite attempt-start timestamps."""
    parent = _valid_parent_evaluation()
    progress = _attempt_progress(
        parent=parent,
        status=AttemptStatus.RUNNING,
        stop=StopDisposition(status=StopStatus.NONE, reason=None),
        first_started_at=first_started_at,
        result_artifact_id=None,
    )

    with pytest.raises(ValueError) as error:
        validate_attempt_progress(progress, parent=parent)

    message = str(error.value)
    assert "first_started_at" in message
    assert "finite, nonnegative" in message
    assert "got:" in message


def test_validate_attempt_progress_rejects_invalid_timestamp_type() -> None:
    """Reject a start timestamp that is not a floating-point value."""
    parent = _valid_parent_evaluation()
    progress = _attempt_progress(
        parent=parent,
        status=AttemptStatus.RUNNING,
        stop=StopDisposition(status=StopStatus.NONE, reason=None),
        first_started_at=cast(float, 1),
        result_artifact_id=None,
    )

    with pytest.raises(TypeError) as error:
        validate_attempt_progress(progress, parent=parent)

    message = str(error.value)
    assert "first_started_at" in message
    assert "builtins.float" in message
    assert "builtins.int" in message
    assert "1" in message


def test_validate_attempt_progress_rejects_invalid_status_type() -> None:
    """Reject a processing status outside the declared vocabulary."""
    parent = _valid_parent_evaluation()
    progress = _attempt_progress(
        parent=parent,
        status=cast(AttemptStatus, "running"),
        stop=StopDisposition(status=StopStatus.NONE, reason=None),
        first_started_at=1.0,
        result_artifact_id=None,
    )

    with pytest.raises(TypeError) as error:
        validate_attempt_progress(progress, parent=parent)

    message = str(error.value)
    assert "status" in message
    assert "AttemptStatus" in message
    assert "builtins.str" in message
    assert "'running'" in message


def test_validate_attempt_progress_rejects_empty_result_artifact_identity() -> None:
    """Reject an empty identity for a claimed completed result artifact."""
    parent = _valid_parent_evaluation()
    progress = _attempt_progress(
        parent=parent,
        status=AttemptStatus.RESULT_READY,
        stop=StopDisposition(status=StopStatus.NONE, reason=None),
        first_started_at=1.0,
        result_artifact_id=ArtifactId(""),
    )

    with pytest.raises(ValueError) as error:
        validate_attempt_progress(progress, parent=parent)

    assert str(error.value) == "result_artifact_id must be a nonempty string; got an empty string."


def test_validate_attempt_progress_rejects_invalid_routing_context() -> None:
    """Preserve routing-validation context for an attempt assigned elsewhere."""
    parent = _valid_parent_evaluation()
    progress = AttemptProgress(
        inputs=AttemptInputs(
            attempt_id=AttemptId("attempt-a"),
            parent_id=ParentId("parent-b"),
            model_id=parent.models[0].model_id,
            worker_id=WorkerId("worker-a"),
        ),
        status=AttemptStatus.ALLOCATED,
        stop=StopDisposition(status=StopStatus.NONE, reason=None),
        first_started_at=None,
        result_artifact_id=None,
    )

    with pytest.raises(ValueError) as error:
        validate_attempt_progress(progress, parent=parent)

    message = str(error.value)
    assert message.startswith("inputs is invalid:")
    assert "parent_id" in message
    assert "parent.parent_id" in message


def test_validate_attempt_progress_rejects_invalid_stop_context() -> None:
    """Preserve stop-validation context for an inconsistent stop pair."""
    parent = _valid_parent_evaluation()
    progress = _attempt_progress(
        parent=parent,
        status=AttemptStatus.ALLOCATED,
        stop=StopDisposition(
            status=StopStatus.NONE,
            reason=StopReason.PARENT_CANCELLED,
        ),
        first_started_at=None,
        result_artifact_id=None,
    )

    with pytest.raises(ValueError) as error:
        validate_attempt_progress(progress, parent=parent)

    message = str(error.value)
    assert message.startswith("stop is invalid:")
    assert "status: none" in message
    assert "reason to be None" in message


def test_validate_attempt_progress_rejects_wrong_outer_record() -> None:
    """Reject a value that is not an AttemptProgress record."""
    with pytest.raises(TypeError) as error:
        validate_attempt_progress(
            cast(AttemptProgress, None),
            parent=_valid_parent_evaluation(),
        )

    message = str(error.value)
    assert "AttemptProgress" in message
    assert "NoneType" in message
    assert "None" in message


def test_worker_availability_has_exact_string_vocabulary() -> None:
    """Expose only the planned worker-usability observations."""
    assert tuple(WorkerAvailability) == (
        WorkerAvailability.UNAVAILABLE,
        WorkerAvailability.AVAILABLE,
    )
    assert WorkerAvailability.UNAVAILABLE == "unavailable"
    assert WorkerAvailability.AVAILABLE == "available"


def test_worker_process_status_has_exact_string_vocabulary() -> None:
    """Expose only the planned worker-process observations."""
    assert tuple(WorkerProcessStatus) == (
        WorkerProcessStatus.UNKNOWN,
        WorkerProcessStatus.RUNNING,
        WorkerProcessStatus.UNAVAILABLE,
    )
    assert WorkerProcessStatus.UNKNOWN == "unknown"
    assert WorkerProcessStatus.RUNNING == "running"
    assert WorkerProcessStatus.UNAVAILABLE == "unavailable"


def test_ownership_status_has_exact_string_vocabulary() -> None:
    """Expose only the planned root-ownership observations."""
    assert tuple(OwnershipStatus) == (
        OwnershipStatus.UNKNOWN,
        OwnershipStatus.OWNED_BY_ROOT,
        OwnershipStatus.NOT_OWNED_BY_ROOT,
    )
    assert OwnershipStatus.UNKNOWN == "unknown"
    assert OwnershipStatus.OWNED_BY_ROOT == "owned_by_root"
    assert OwnershipStatus.NOT_OWNED_BY_ROOT == "not_owned_by_root"


def test_compatibility_status_has_exact_string_vocabulary() -> None:
    """Expose only the planned worker-compatibility observations."""
    assert tuple(CompatibilityStatus) == (
        CompatibilityStatus.UNCHECKED,
        CompatibilityStatus.COMPATIBLE,
        CompatibilityStatus.INCOMPATIBLE,
    )
    assert CompatibilityStatus.UNCHECKED == "unchecked"
    assert CompatibilityStatus.COMPATIBLE == "compatible"
    assert CompatibilityStatus.INCOMPATIBLE == "incompatible"


def test_parent_context_readiness_preserves_parent_context_observations() -> None:
    """Preserve one parent's independent readiness observations."""
    readiness = ParentContextReadiness(
        parent_id=ParentId("parent-a"),
        dataset_ready=True,
        evaluation_ready=False,
        partition_ready=True,
    )
    same = ParentContextReadiness(
        parent_id=ParentId("parent-a"),
        dataset_ready=True,
        evaluation_ready=False,
        partition_ready=True,
    )
    different_evaluation = ParentContextReadiness(
        parent_id=ParentId("parent-a"),
        dataset_ready=True,
        evaluation_ready=True,
        partition_ready=True,
    )

    assert readiness.parent_id == ParentId("parent-a")
    assert readiness.dataset_ready is True
    assert readiness.evaluation_ready is False
    assert readiness.partition_ready is True
    assert readiness == same
    assert readiness != different_evaluation


def test_parent_context_readiness_rejects_field_mutation() -> None:
    """Prevent reassignment of a parent-readiness observation."""
    readiness = ParentContextReadiness(
        parent_id=ParentId("parent-a"),
        dataset_ready=True,
        evaluation_ready=True,
        partition_ready=True,
    )

    with pytest.raises(FrozenInstanceError):
        readiness.__setattr__(
            "dataset_ready",
            False,
        )
