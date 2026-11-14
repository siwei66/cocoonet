"""Tests for durable execution-state snapshot validation."""

from dataclasses import FrozenInstanceError, replace

import pytest

from cocoonet._contracts.execution import AttemptInputs
from cocoonet._contracts.failure import (
    FailureRecord,
    FailureScope,
    TraceContext,
)
from cocoonet._contracts.identity import (
    AttemptId,
    FailureId,
    ModelId,
    OperationId,
    ParentId,
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
    ParentProgress,
    PersistenceStatus,
    ReceiptStatus,
    ReceptionRecord,
    StopDisposition,
    StopStatus,
    WorkerAvailability,
    WorkerObservation,
    WorkerProcessStatus,
)
from cocoonet._storage.execution_state import (
    ExecutionStateSnapshot,
    validate_execution_state_snapshot,
)
from cocoonet._storage.metadata import NodeRole

_PARENT_ID = ParentId("a" * 16)
_MODEL_ID = ModelId("1")
_ATTEMPT_ID = AttemptId("attempt-1")
_WORKER_ID = WorkerId("worker-1")


def _attempt(
    *,
    parent_id: ParentId = _PARENT_ID,
    model_id: ModelId = _MODEL_ID,
    worker_id: WorkerId = _WORKER_ID,
) -> AttemptProgress:
    """Build one valid allocated attempt observation."""
    return AttemptProgress(
        inputs=AttemptInputs(
            attempt_id=_ATTEMPT_ID,
            parent_id=parent_id,
            model_id=model_id,
            worker_id=worker_id,
        ),
        status=AttemptStatus.ALLOCATED,
        stop=StopDisposition(
            status=StopStatus.NONE,
            reason=None,
        ),
        first_started_at=None,
        result_artifact_id=None,
    )


def _worker(
    *,
    worker_id: WorkerId = _WORKER_ID,
    active_attempt_ids: tuple[AttemptId, ...] = (),
) -> WorkerObservation:
    """Build one valid worker observation."""
    return WorkerObservation(
        worker_id=worker_id,
        availability=WorkerAvailability.UNAVAILABLE,
        process_status=WorkerProcessStatus.UNKNOWN,
        ownership=OwnershipStatus.UNKNOWN,
        compatibility=CompatibilityStatus.UNCHECKED,
        parent_readiness=(),
        attempt_capacity=None,
        active_attempt_ids=active_attempt_ids,
    )


def _snapshot(
    *,
    node_role: NodeRole = NodeRole.ROOT,
) -> ExecutionStateSnapshot:
    """Build one valid root or worker snapshot."""
    receptions: tuple[ReceptionRecord, ...] = (
        ReceptionRecord(
            parent_id=_PARENT_ID,
            model_id=_MODEL_ID,
            selected_attempt_id=None,
            receipt=ReceiptStatus.NOT_RECEIVED,
            persistence=PersistenceStatus.NOT_STARTED,
            result_artifact_id=None,
            accepted=False,
        ),
    )
    if node_role is NodeRole.WORKER:
        receptions = ()

    return ExecutionStateSnapshot(
        storage_schema_version=1,
        node_role=node_role,
        parent_id=_PARENT_ID,
        model_ids=(_MODEL_ID,),
        progress=ParentProgress(
            parent_id=_PARENT_ID,
            execution_status=ExecutionStatus.NOT_STARTED,
        ),
        attempts=(_attempt(),),
        workers=(_worker(),),
        receptions=receptions,
        operations=(),
    )


@pytest.mark.parametrize("node_role", tuple(NodeRole))
def test_validate_execution_state_snapshot_accepts_each_node_role(
    node_role: NodeRole,
) -> None:
    """Accept complete root and worker observation snapshots."""
    validate_execution_state_snapshot(_snapshot(node_role=node_role))


def test_execution_state_snapshot_is_immutable() -> None:
    """Prevent reassignment of persisted snapshot fields."""
    snapshot = _snapshot()

    with pytest.raises(FrozenInstanceError):
        snapshot.__setattr__("storage_schema_version", 2)


@pytest.mark.parametrize(
    "snapshot",
    (
        replace(_snapshot(), storage_schema_version=2),
        replace(_snapshot(), parent_id=ParentId("invalid")),
        replace(_snapshot(), model_ids=(ModelId("01"),)),
        replace(_snapshot(), model_ids=(_MODEL_ID, _MODEL_ID)),
        replace(_snapshot(), model_ids=()),
    ),
)
def test_validate_execution_state_snapshot_rejects_invalid_schema_or_ids(
    snapshot: ExecutionStateSnapshot,
) -> None:
    """Reject schema mismatch and invalid managed identity sets."""
    with pytest.raises(ValueError):
        validate_execution_state_snapshot(snapshot)


def test_validate_execution_state_snapshot_rejects_progress_parent_mismatch() -> None:
    """Require parent progress to describe the snapshot parent."""
    snapshot = replace(
        _snapshot(),
        progress=ParentProgress(
            parent_id=ParentId("b" * 16),
            execution_status=ExecutionStatus.NOT_STARTED,
        ),
    )

    with pytest.raises(ValueError, match=r"progress\.parent_id"):
        validate_execution_state_snapshot(snapshot)


def test_validate_execution_state_snapshot_rejects_invalid_nested_attempt() -> None:
    """Apply Phase 1 attempt-observation validation."""
    attempt = replace(
        _attempt(),
        status=AttemptStatus.RUNNING,
        first_started_at=None,
    )
    snapshot = replace(_snapshot(), attempts=(attempt,))

    with pytest.raises(ValueError, match=r"first_started_at"):
        validate_execution_state_snapshot(snapshot)


@pytest.mark.parametrize(
    "attempt",
    (
        _attempt(parent_id=ParentId("b" * 16)),
        _attempt(model_id=ModelId("2")),
    ),
)
def test_validate_execution_state_snapshot_rejects_attempt_routing(
    attempt: AttemptProgress,
) -> None:
    """Require every attempt to reference the fixed parent and listed unit."""
    snapshot = replace(_snapshot(), attempts=(attempt,))

    with pytest.raises(ValueError):
        validate_execution_state_snapshot(snapshot)


def test_validate_execution_state_snapshot_rejects_duplicate_attempt_id() -> None:
    """Require stored attempt identities to be unique."""
    snapshot = replace(
        _snapshot(),
        attempts=(_attempt(), _attempt()),
    )

    with pytest.raises(ValueError, match=r"attempt identity"):
        validate_execution_state_snapshot(snapshot)


def test_validate_execution_state_snapshot_rejects_invalid_active_attempt() -> None:
    """Resolve active reservations to attempts assigned to that worker."""
    snapshot = replace(
        _snapshot(),
        workers=(
            _worker(
                active_attempt_ids=(AttemptId("unknown-attempt"),),
            ),
        ),
    )

    with pytest.raises(ValueError, match=r"unknown active attempt"):
        validate_execution_state_snapshot(snapshot)


def test_validate_execution_state_snapshot_rejects_worker_root_state() -> None:
    """Exclude root reception state from worker snapshots."""
    snapshot = replace(
        _snapshot(node_role=NodeRole.WORKER),
        receptions=_snapshot().receptions,
    )

    with pytest.raises(ValueError, match=r"must not contain"):
        validate_execution_state_snapshot(snapshot)


def test_validate_execution_state_snapshot_requires_one_worker_identity() -> None:
    """Require worker snapshots to contain exactly one worker observation."""
    snapshot = replace(
        _snapshot(node_role=NodeRole.WORKER),
        workers=(),
    )

    with pytest.raises(ValueError, match=r"exactly one"):
        validate_execution_state_snapshot(snapshot)


def test_validate_execution_state_snapshot_rejects_duplicate_operation_id() -> None:
    """Require bounded operation identities to be unique."""
    operation = BoundedOperation(
        operation_id=OperationId("operation-1"),
        kind=OperationKind.TRANSFER,
        context=TraceContext(
            stage="result_transfer",
            operation_id=OperationId("operation-1"),
        ),
        attempt_limit=2,
        attempts_made=0,
        status=OperationStatus.PENDING,
        last_failure=None,
    )
    snapshot = replace(
        _snapshot(),
        operations=(operation, operation),
    )

    with pytest.raises(ValueError, match=r"operation_id must be unique"):
        validate_execution_state_snapshot(snapshot)


def test_validate_execution_state_snapshot_checks_operation_failure_links() -> None:
    """Require stored operation-failure origins to resolve."""
    operation_id = OperationId("operation-1")
    context = TraceContext(
        stage="result_transfer",
        operation_id=operation_id,
    )
    failure = FailureRecord(
        failure_id=FailureId("failure-1"),
        scope=FailureScope.RESULT_TRANSFER,
        context=context,
        exception_type="TransferError",
        message="Transfer failed.",
        traceback="",
        origin_failure_id=FailureId("missing-origin"),
    )
    operation = BoundedOperation(
        operation_id=operation_id,
        kind=OperationKind.TRANSFER,
        context=context,
        attempt_limit=2,
        attempts_made=1,
        status=OperationStatus.FAILED,
        last_failure=failure,
    )
    snapshot = replace(_snapshot(), operations=(operation,))

    with pytest.raises(ValueError, match=r"origin_failure_id"):
        validate_execution_state_snapshot(snapshot)
