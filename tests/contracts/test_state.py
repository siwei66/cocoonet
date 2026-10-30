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
from cocoonet._contracts.failure import FailureRecord, FailureScope, TraceContext
from cocoonet._contracts.identity import (
    ArtifactId,
    AttemptId,
    DatasetId,
    EvaluationConfigId,
    FailureId,
    ModelId,
    OperationId,
    ParentId,
    PartitionId,
    WorkerId,
)
from cocoonet._contracts.state import (
    AttemptProgress,
    AttemptStatus,
    BoundedOperation,
    CompatibilityStatus,
    ExecutionStatus,
    OperationKind,
    OperationStatus,
    OwnershipStatus,
    ParentContextReadiness,
    ParentProgress,
    StopDisposition,
    StopReason,
    StopStatus,
    WorkerAvailability,
    WorkerObservation,
    WorkerProcessStatus,
    validate_attempt_progress,
    validate_bounded_operation,
    validate_parent_progress,
    validate_stop_disposition,
    validate_worker_observation,
)

_DEFAULT_OPERATION_ID = OperationId("operation-a")


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


def _worker_observation(
    *,
    availability: WorkerAvailability = WorkerAvailability.UNAVAILABLE,
    process_status: WorkerProcessStatus = WorkerProcessStatus.UNKNOWN,
    ownership: OwnershipStatus = OwnershipStatus.UNKNOWN,
    compatibility: CompatibilityStatus = CompatibilityStatus.UNCHECKED,
    parent_readiness: tuple[ParentContextReadiness, ...] = (),
    attempt_capacity: int | None = None,
    active_attempt_ids: tuple[AttemptId, ...] = (),
) -> WorkerObservation:
    """Build a worker-observation candidate with a fixed valid identity."""
    return WorkerObservation(
        worker_id=WorkerId("worker-a"),
        availability=availability,
        process_status=process_status,
        ownership=ownership,
        compatibility=compatibility,
        parent_readiness=parent_readiness,
        attempt_capacity=attempt_capacity,
        active_attempt_ids=active_attempt_ids,
    )


def _operation_context(
    *,
    operation_id: OperationId = _DEFAULT_OPERATION_ID,
    stage: str = "result_transfer",
) -> TraceContext:
    """Build an operational trace context for bounded-operation tests."""
    return TraceContext(
        stage=stage,
        operation_id=operation_id,
    )


def _operation_failure(
    *,
    context: TraceContext,
    scope: FailureScope = FailureScope.RESULT_TRANSFER,
) -> FailureRecord:
    """Build a valid operational failure for bounded-operation tests."""
    return FailureRecord(
        failure_id=FailureId("failure-a"),
        scope=scope,
        context=context,
        exception_type="OperationError",
        message="The operation attempt failed.",
        traceback="",
    )


def _bounded_operation(
    *,
    operation_id: OperationId = _DEFAULT_OPERATION_ID,
    kind: OperationKind = OperationKind.TRANSFER,
    context: TraceContext | None = None,
    attempt_limit: int = 3,
    attempts_made: int = 0,
    status: OperationStatus = OperationStatus.PENDING,
    last_failure: FailureRecord | None = None,
) -> BoundedOperation:
    """Build a bounded-operation candidate with explicit progress facts."""
    resolved_context = context if context is not None else _operation_context(operation_id=operation_id)
    return BoundedOperation(
        operation_id=operation_id,
        kind=kind,
        context=resolved_context,
        attempt_limit=attempt_limit,
        attempts_made=attempts_made,
        status=status,
        last_failure=last_failure,
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


def test_worker_observation_preserves_independent_operational_facts() -> None:
    """Preserve worker-level facts, parent readiness, capacity, and reservations."""
    readiness = ParentContextReadiness(
        parent_id=ParentId("parent-a"),
        dataset_ready=True,
        evaluation_ready=True,
        partition_ready=True,
    )
    observation = WorkerObservation(
        worker_id=WorkerId("worker-a"),
        availability=WorkerAvailability.AVAILABLE,
        process_status=WorkerProcessStatus.RUNNING,
        ownership=OwnershipStatus.OWNED_BY_ROOT,
        compatibility=CompatibilityStatus.COMPATIBLE,
        parent_readiness=(readiness,),
        attempt_capacity=2,
        active_attempt_ids=(AttemptId("attempt-a"),),
    )
    same = WorkerObservation(
        worker_id=WorkerId("worker-a"),
        availability=WorkerAvailability.AVAILABLE,
        process_status=WorkerProcessStatus.RUNNING,
        ownership=OwnershipStatus.OWNED_BY_ROOT,
        compatibility=CompatibilityStatus.COMPATIBLE,
        parent_readiness=(readiness,),
        attempt_capacity=2,
        active_attempt_ids=(AttemptId("attempt-a"),),
    )
    different_capacity = WorkerObservation(
        worker_id=WorkerId("worker-a"),
        availability=WorkerAvailability.AVAILABLE,
        process_status=WorkerProcessStatus.RUNNING,
        ownership=OwnershipStatus.OWNED_BY_ROOT,
        compatibility=CompatibilityStatus.COMPATIBLE,
        parent_readiness=(readiness,),
        attempt_capacity=3,
        active_attempt_ids=(AttemptId("attempt-a"),),
    )

    assert observation.worker_id == WorkerId("worker-a")
    assert observation.availability is WorkerAvailability.AVAILABLE
    assert observation.process_status is WorkerProcessStatus.RUNNING
    assert observation.ownership is OwnershipStatus.OWNED_BY_ROOT
    assert observation.compatibility is CompatibilityStatus.COMPATIBLE
    assert observation.parent_readiness == (readiness,)
    assert observation.attempt_capacity == 2
    assert observation.active_attempt_ids == (AttemptId("attempt-a"),)
    assert observation == same
    assert observation != different_capacity


def test_worker_observation_rejects_field_mutation() -> None:
    """Prevent reassignment of a worker observation."""
    observation = WorkerObservation(
        worker_id=WorkerId("worker-a"),
        availability=WorkerAvailability.UNAVAILABLE,
        process_status=WorkerProcessStatus.UNKNOWN,
        ownership=OwnershipStatus.UNKNOWN,
        compatibility=CompatibilityStatus.UNCHECKED,
        parent_readiness=(),
        attempt_capacity=None,
        active_attempt_ids=(),
    )

    with pytest.raises(FrozenInstanceError):
        observation.__setattr__(
            "availability",
            WorkerAvailability.AVAILABLE,
        )


@pytest.mark.parametrize(
    "active_attempt_ids",
    (
        (),
        (AttemptId("attempt-a"),),
        (AttemptId("attempt-a"), AttemptId("attempt-b")),
    ),
)
def test_validate_worker_observation_accepts_available_worker_at_each_occupancy(
    active_attempt_ids: tuple[AttemptId, ...],
) -> None:
    """Accept available workers with zero, some, or all slots occupied."""
    observation = _worker_observation(
        availability=WorkerAvailability.AVAILABLE,
        process_status=WorkerProcessStatus.RUNNING,
        ownership=OwnershipStatus.OWNED_BY_ROOT,
        compatibility=CompatibilityStatus.COMPATIBLE,
        attempt_capacity=2,
        active_attempt_ids=active_attempt_ids,
    )

    validate_worker_observation(observation)

    assert observation.active_attempt_ids == active_attempt_ids
    assert observation.attempt_capacity == 2


def test_validate_worker_observation_accepts_unavailable_worker_with_unknown_capacity() -> None:
    """Preserve assigned attempts when worker capacity is unknown."""
    observation = _worker_observation(
        parent_readiness=(
            ParentContextReadiness(
                parent_id=ParentId("parent-a"),
                dataset_ready=True,
                evaluation_ready=False,
                partition_ready=True,
            ),
        ),
        attempt_capacity=None,
        active_attempt_ids=(
            AttemptId("attempt-a"),
            AttemptId("attempt-b"),
        ),
    )

    validate_worker_observation(observation)


def test_validate_worker_observation_accepts_reservations_above_reduced_capacity() -> None:
    """Preserve earlier reservations that exceed a prospectively lower limit."""
    observation = _worker_observation(
        availability=WorkerAvailability.AVAILABLE,
        process_status=WorkerProcessStatus.RUNNING,
        ownership=OwnershipStatus.OWNED_BY_ROOT,
        compatibility=CompatibilityStatus.COMPATIBLE,
        attempt_capacity=1,
        active_attempt_ids=(
            AttemptId("attempt-a"),
            AttemptId("attempt-b"),
        ),
    )

    validate_worker_observation(observation)

    assert observation.attempt_capacity is not None
    assert len(observation.active_attempt_ids) > observation.attempt_capacity


@pytest.mark.parametrize(
    ("observation", "expected"),
    (
        (
            _worker_observation(
                availability=WorkerAvailability.AVAILABLE,
                process_status=WorkerProcessStatus.UNKNOWN,
                ownership=OwnershipStatus.OWNED_BY_ROOT,
                compatibility=CompatibilityStatus.COMPATIBLE,
                attempt_capacity=1,
            ),
            "process_status to be running",
        ),
        (
            _worker_observation(
                availability=WorkerAvailability.AVAILABLE,
                process_status=WorkerProcessStatus.RUNNING,
                ownership=OwnershipStatus.UNKNOWN,
                compatibility=CompatibilityStatus.COMPATIBLE,
                attempt_capacity=1,
            ),
            "ownership to be owned_by_root",
        ),
        (
            _worker_observation(
                availability=WorkerAvailability.AVAILABLE,
                process_status=WorkerProcessStatus.RUNNING,
                ownership=OwnershipStatus.OWNED_BY_ROOT,
                compatibility=CompatibilityStatus.UNCHECKED,
                attempt_capacity=1,
            ),
            "compatibility to be compatible",
        ),
        (
            _worker_observation(
                availability=WorkerAvailability.AVAILABLE,
                process_status=WorkerProcessStatus.RUNNING,
                ownership=OwnershipStatus.OWNED_BY_ROOT,
                compatibility=CompatibilityStatus.COMPATIBLE,
                attempt_capacity=None,
            ),
            "attempt_capacity to be known",
        ),
    ),
)
def test_validate_worker_observation_rejects_missing_availability_prerequisite(
    observation: WorkerObservation,
    expected: str,
) -> None:
    """Reject an available observation missing a required worker-level fact."""
    with pytest.raises(ValueError) as error:
        validate_worker_observation(observation)

    message = str(error.value)
    assert "availability: available" in message
    assert expected in message
    assert "got:" in message


@pytest.mark.parametrize("attempt_capacity", (0, -1))
def test_validate_worker_observation_rejects_nonpositive_capacity(
    attempt_capacity: int,
) -> None:
    """Reject a known attempt capacity that is not positive."""
    with pytest.raises(ValueError) as error:
        validate_worker_observation(_worker_observation(attempt_capacity=attempt_capacity))

    message = str(error.value)
    assert "attempt_capacity" in message
    assert "positive integer or None" in message
    assert f"got: {attempt_capacity}" in message


@pytest.mark.parametrize(
    "attempt_capacity",
    (
        cast(int, True),
        cast(int, 1.0),
    ),
)
def test_validate_worker_observation_rejects_invalid_capacity_type(
    attempt_capacity: int,
) -> None:
    """Reject boolean and floating-point attempt capacities."""
    with pytest.raises(TypeError) as error:
        validate_worker_observation(_worker_observation(attempt_capacity=attempt_capacity))

    message = str(error.value)
    assert "attempt_capacity" in message
    assert "builtins.int" in message


def test_validate_worker_observation_rejects_duplicate_active_attempt_ids() -> None:
    """Reject duplicate slot reservations for one attempt identity."""
    observation = _worker_observation(
        attempt_capacity=2,
        active_attempt_ids=(
            AttemptId("attempt-a"),
            AttemptId("attempt-a"),
        ),
    )

    with pytest.raises(ValueError) as error:
        validate_worker_observation(observation)

    message = str(error.value)
    assert "active_attempt_ids[1]" in message
    assert "unique" in message
    assert "attempt-a" in message


def test_validate_worker_observation_rejects_empty_active_attempt_id() -> None:
    """Reject an empty identity in the active reservation sequence."""
    observation = _worker_observation(
        attempt_capacity=1,
        active_attempt_ids=(AttemptId(""),),
    )

    with pytest.raises(ValueError) as error:
        validate_worker_observation(observation)

    assert "active_attempt_ids[0]" in str(error.value)
    assert "nonempty" in str(error.value)


def test_validate_worker_observation_rejects_duplicate_parent_readiness() -> None:
    """Reject competing readiness observations for the same parent."""
    readiness = ParentContextReadiness(
        parent_id=ParentId("parent-a"),
        dataset_ready=True,
        evaluation_ready=True,
        partition_ready=True,
    )
    observation = _worker_observation(
        parent_readiness=(readiness, readiness),
    )

    with pytest.raises(ValueError) as error:
        validate_worker_observation(observation)

    message = str(error.value)
    assert "parent_readiness[1].parent_id" in message
    assert "unique" in message
    assert "parent-a" in message


def test_validate_worker_observation_rejects_invalid_parent_readiness_flag() -> None:
    """Reject a non-boolean parent-context readiness observation."""
    readiness = ParentContextReadiness(
        parent_id=ParentId("parent-a"),
        dataset_ready=cast(bool, 1),
        evaluation_ready=True,
        partition_ready=True,
    )

    with pytest.raises(TypeError) as error:
        validate_worker_observation(_worker_observation(parent_readiness=(readiness,)))

    message = str(error.value)
    assert "parent_readiness[0].dataset_ready" in message
    assert "builtins.bool" in message
    assert "builtins.int" in message


def test_validate_worker_observation_rejects_wrong_outer_record() -> None:
    """Reject a value that is not a WorkerObservation record."""
    with pytest.raises(TypeError) as error:
        validate_worker_observation(cast(WorkerObservation, None))

    message = str(error.value)
    assert "WorkerObservation" in message
    assert "NoneType" in message
    assert "None" in message


def test_operation_kind_has_exact_string_vocabulary() -> None:
    """Expose only the planned bounded infrastructure operation kinds."""
    assert tuple(OperationKind) == (
        OperationKind.CONNECTION_ROUND,
        OperationKind.TRANSFER,
        OperationKind.PERSISTENCE,
    )
    assert OperationKind.CONNECTION_ROUND == "connection_round"
    assert OperationKind.TRANSFER == "transfer"
    assert OperationKind.PERSISTENCE == "persistence"


def test_operation_status_has_exact_string_vocabulary() -> None:
    """Expose only the planned bounded-operation progress states."""
    assert tuple(OperationStatus) == (
        OperationStatus.PENDING,
        OperationStatus.RUNNING,
        OperationStatus.SUCCEEDED,
        OperationStatus.FAILED,
        OperationStatus.CANCELLED,
    )
    assert OperationStatus.PENDING == "pending"
    assert OperationStatus.RUNNING == "running"
    assert OperationStatus.SUCCEEDED == "succeeded"
    assert OperationStatus.FAILED == "failed"
    assert OperationStatus.CANCELLED == "cancelled"


def test_bounded_operation_preserves_progress_and_failure() -> None:
    """Preserve operation identity, progress, context, and prior failure."""
    context = TraceContext(
        stage="result_transfer",
        operation_id=OperationId("operation-a"),
    )
    failure = FailureRecord(
        failure_id=FailureId("failure-a"),
        scope=FailureScope.RESULT_TRANSFER,
        context=context,
        exception_type="TransferError",
        message="The first transfer attempt failed.",
        traceback="",
    )
    operation = BoundedOperation(
        operation_id=OperationId("operation-a"),
        kind=OperationKind.TRANSFER,
        context=context,
        attempt_limit=3,
        attempts_made=2,
        status=OperationStatus.SUCCEEDED,
        last_failure=failure,
    )
    same = BoundedOperation(
        operation_id=OperationId("operation-a"),
        kind=OperationKind.TRANSFER,
        context=context,
        attempt_limit=3,
        attempts_made=2,
        status=OperationStatus.SUCCEEDED,
        last_failure=failure,
    )

    assert operation.operation_id == OperationId("operation-a")
    assert operation.kind is OperationKind.TRANSFER
    assert operation.context is context
    assert operation.attempt_limit == 3
    assert operation.attempts_made == 2
    assert operation.status is OperationStatus.SUCCEEDED
    assert operation.last_failure is failure
    assert operation == same


def test_bounded_operation_rejects_field_mutation() -> None:
    """Prevent reassignment of bounded-operation observations."""
    operation = BoundedOperation(
        operation_id=OperationId("operation-a"),
        kind=OperationKind.CONNECTION_ROUND,
        context=TraceContext(
            stage="connection_round",
            operation_id=OperationId("operation-a"),
        ),
        attempt_limit=3,
        attempts_made=0,
        status=OperationStatus.PENDING,
        last_failure=None,
    )

    with pytest.raises(FrozenInstanceError):
        operation.__setattr__(
            "status",
            OperationStatus.RUNNING,
        )


@pytest.mark.parametrize(
    ("status", "attempts_made", "with_failure"),
    (
        (OperationStatus.PENDING, 0, False),
        (OperationStatus.RUNNING, 1, False),
        (OperationStatus.RUNNING, 2, True),
        (OperationStatus.SUCCEEDED, 1, False),
        (OperationStatus.SUCCEEDED, 2, True),
        (OperationStatus.FAILED, 1, True),
        (OperationStatus.FAILED, 3, True),
        (OperationStatus.CANCELLED, 0, False),
        (OperationStatus.CANCELLED, 1, False),
    ),
)
def test_validate_bounded_operation_accepts_valid_status_progress(
    status: OperationStatus,
    attempts_made: int,
    with_failure: bool,
) -> None:
    """Accept valid pending, active, terminal, and exhausted observations."""
    context = _operation_context()
    failure = _operation_failure(context=context) if with_failure else None
    operation = _bounded_operation(
        context=context,
        attempts_made=attempts_made,
        status=status,
        last_failure=failure,
    )

    validate_bounded_operation(operation)

    assert operation.attempts_made == attempts_made
    assert operation.last_failure is failure


@pytest.mark.parametrize(
    ("kind", "scope"),
    (
        (OperationKind.CONNECTION_ROUND, FailureScope.WORKER_MANAGEMENT),
        (OperationKind.TRANSFER, FailureScope.DATA_SYNC),
        (OperationKind.TRANSFER, FailureScope.RESULT_TRANSFER),
        (OperationKind.PERSISTENCE, FailureScope.DATA_PREPARATION),
        (OperationKind.PERSISTENCE, FailureScope.RESULT_PERSISTENCE),
    ),
)
def test_validate_bounded_operation_accepts_kind_failure_scope(
    kind: OperationKind,
    scope: FailureScope,
) -> None:
    """Accept each operational failure scope assigned to an operation kind."""
    context = _operation_context()
    failure = _operation_failure(context=context, scope=scope)

    validate_bounded_operation(
        _bounded_operation(
            kind=kind,
            context=context,
            attempts_made=1,
            status=OperationStatus.FAILED,
            last_failure=failure,
        )
    )


@pytest.mark.parametrize(
    ("attempt_limit", "attempts_made", "expected"),
    (
        (0, 0, "attempt_limit"),
        (3, -1, "attempts_made"),
        (3, 4, "attempts_made"),
    ),
)
def test_validate_bounded_operation_rejects_invalid_count_bounds(
    attempt_limit: int,
    attempts_made: int,
    expected: str,
) -> None:
    """Reject zero limits and begun-attempt counts outside the fixed budget."""
    with pytest.raises(ValueError) as error:
        validate_bounded_operation(
            _bounded_operation(
                attempt_limit=attempt_limit,
                attempts_made=attempts_made,
            )
        )

    message = str(error.value)
    assert f"field: {expected}" in message
    assert "expected:" in message
    assert "got:" in message


@pytest.mark.parametrize(
    ("field", "attempt_limit", "attempts_made"),
    (
        ("attempt_limit", cast(int, True), 0),
        ("attempts_made", 3, cast(int, False)),
    ),
)
def test_validate_bounded_operation_rejects_boolean_count(
    field: str,
    attempt_limit: int,
    attempts_made: int,
) -> None:
    """Reject booleans where bounded-operation counts require exact integers."""
    with pytest.raises(TypeError) as error:
        validate_bounded_operation(
            _bounded_operation(
                attempt_limit=attempt_limit,
                attempts_made=attempts_made,
            )
        )

    message = str(error.value)
    assert field in message
    assert "builtins.int" in message
    assert "builtins.bool" in message


@pytest.mark.parametrize(
    "status",
    (
        OperationStatus.RUNNING,
        OperationStatus.SUCCEEDED,
        OperationStatus.FAILED,
    ),
)
def test_validate_bounded_operation_rejects_started_status_without_attempt(
    status: OperationStatus,
) -> None:
    """Reject active or completed progress before any operation attempt began."""
    with pytest.raises(ValueError) as error:
        validate_bounded_operation(_bounded_operation(status=status))

    message = str(error.value)
    assert f"status: {status.value}" in message
    assert "attempts_made to be at least 1" in message
    assert "got: 0" in message


def test_validate_bounded_operation_rejects_pending_after_attempt() -> None:
    """Reject pending progress after an operation attempt has begun."""
    with pytest.raises(ValueError) as error:
        validate_bounded_operation(_bounded_operation(attempts_made=1))

    message = str(error.value)
    assert "status: pending" in message
    assert "attempts_made to be 0" in message
    assert "got: 1" in message


@pytest.mark.parametrize(
    ("status", "attempts_made", "with_failure"),
    (
        (OperationStatus.PENDING, 0, True),
        (OperationStatus.FAILED, 1, False),
    ),
)
def test_validate_bounded_operation_rejects_invalid_failure_requirement(
    status: OperationStatus,
    attempts_made: int,
    with_failure: bool,
) -> None:
    """Reject a pending failure and a failed status without provenance."""
    context = _operation_context()
    failure = _operation_failure(context=context) if with_failure else None

    with pytest.raises(ValueError) as error:
        validate_bounded_operation(
            _bounded_operation(
                context=context,
                attempts_made=attempts_made,
                status=status,
                last_failure=failure,
            )
        )

    message = str(error.value)
    assert f"status: {status.value}" in message
    assert "expected:" in message
    assert "got:" in message


def test_validate_bounded_operation_rejects_mismatched_context_identity() -> None:
    """Reject an operation context naming a different operation."""
    operation = _bounded_operation(
        context=_operation_context(
            operation_id=OperationId("operation-b"),
        )
    )

    with pytest.raises(ValueError) as error:
        validate_bounded_operation(operation)

    message = str(error.value)
    assert "operation_id: 'operation-a'" in message
    assert "context.operation_id to equal operation_id" in message
    assert "got: 'operation-b'" in message


def test_validate_bounded_operation_rejects_mismatched_failure_context() -> None:
    """Reject a retained failure from a different operational context."""
    context = _operation_context()
    failure = _operation_failure(
        context=_operation_context(stage="data_sync"),
    )
    operation = _bounded_operation(
        context=context,
        attempts_made=1,
        status=OperationStatus.FAILED,
        last_failure=failure,
    )

    with pytest.raises(ValueError) as error:
        validate_bounded_operation(operation)

    message = str(error.value)
    assert "last_failure.context to equal context" in message
    assert "last_failure.context differs from context" in message


def test_validate_bounded_operation_rejects_failure_scope_for_kind() -> None:
    """Reject a valid operational failure assigned to the wrong kind."""
    context = _operation_context()
    failure = _operation_failure(
        context=context,
        scope=FailureScope.RESULT_TRANSFER,
    )
    operation = _bounded_operation(
        kind=OperationKind.CONNECTION_ROUND,
        context=context,
        attempts_made=1,
        status=OperationStatus.FAILED,
        last_failure=failure,
    )

    with pytest.raises(ValueError) as error:
        validate_bounded_operation(operation)

    message = str(error.value)
    assert "kind: connection_round" in message
    assert "last_failure.scope to be one of worker_management" in message
    assert "got: result_transfer" in message


@pytest.mark.parametrize(
    ("field", "operation"),
    (
        (
            "kind",
            _bounded_operation(kind=cast(OperationKind, "transfer")),
        ),
        (
            "status",
            _bounded_operation(status=cast(OperationStatus, "pending")),
        ),
    ),
)
def test_validate_bounded_operation_rejects_invalid_enum_type(
    field: str,
    operation: BoundedOperation,
) -> None:
    """Reject plain strings in operation-kind and status fields."""
    with pytest.raises(TypeError) as error:
        validate_bounded_operation(operation)

    message = str(error.value)
    assert field in message
    assert "builtins.str" in message


def test_validate_bounded_operation_rejects_wrong_outer_record() -> None:
    """Reject a value that is not a BoundedOperation record."""
    with pytest.raises(TypeError) as error:
        validate_bounded_operation(cast(BoundedOperation, None))

    message = str(error.value)
    assert "BoundedOperation" in message
    assert "NoneType" in message
    assert "None" in message
