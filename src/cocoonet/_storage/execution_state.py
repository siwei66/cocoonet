"""Durable execution-state snapshot contracts."""

from dataclasses import dataclass

from cocoonet._contracts.failure import validate_failure_links
from cocoonet._contracts.identity import ModelId, ParentId
from cocoonet._contracts.state import (
    AttemptProgress,
    BoundedOperation,
    ParentProgress,
    ReceptionRecord,
    WorkerObservation,
    _validate_attempt_progress_state,
    validate_bounded_operation,
    validate_parent_progress,
    validate_reception_set,
    validate_worker_observation,
)
from cocoonet._storage.metadata import NodeRole
from cocoonet._storage.paths import (
    _validate_parent_id,
    _validate_positive_decimal_identifier,
)
from cocoonet._type_validation import validate_declared_types


@dataclass(frozen=True, slots=True)
class ExecutionStateSnapshot:
    """Preserve durable execution observations for one fixed parent.

    Parameters
    ----------
    storage_schema_version
        Exact storage schema version.
    node_role
        Role of the node persisting this snapshot.
    parent_id
        Fixed parent identity.
    model_ids
        Ordered known model-unit identities.
    progress
        Parent-level execution observation.
    attempts
        Stored attempt observations.
    workers
        Stored worker observations.
    receptions
        Root-side result-reception observations.
    operations
        Bounded infrastructure-operation observations.

    Notes
    -----
    This immutable record stores observations only. It contains no scientific
    configuration and performs no allocation, selection, acceptance, retry,
    recovery, transition, or filesystem inspection.

    Node usage scope: Shared - Used by both root and worker storage.
    """

    storage_schema_version: int
    node_role: NodeRole
    parent_id: ParentId
    model_ids: tuple[ModelId, ...]
    progress: ParentProgress
    attempts: tuple[AttemptProgress, ...]
    workers: tuple[WorkerObservation, ...]
    receptions: tuple[ReceptionRecord, ...]
    operations: tuple[BoundedOperation, ...]


def validate_execution_state_snapshot(  # noqa: C901
    value: ExecutionStateSnapshot,
    /,
) -> None:
    """Validate one complete durable execution-state snapshot.

    Parameters
    ----------
    value
        Candidate snapshot and its nested Phase 1 observations.

    Raises
    ------
    TypeError
        If the snapshot, a collection, record, or field has an incorrect type.
    ValueError
        If schema, managed identities, nested observations, uniqueness,
        routing, node scope, reservation, or failure links are invalid.

    Notes
    -----
    Validation is pure and stores no derived state. It does not inspect bundle
    paths, infer transitions, perform recovery, or manufacture records.

    Node usage scope: Shared - Used by both root and worker storage.
    """
    if not isinstance(value, ExecutionStateSnapshot):
        validate_declared_types(
            value,
            expected_type=ExecutionStateSnapshot,
        )

    validate_declared_types(
        value.storage_schema_version,
        expected_type=int,
        field="storage_schema_version",
    )
    if value.storage_schema_version != 1:
        raise ValueError("storage_schema_version must be exactly 1; " f"got {value.storage_schema_version!r}.")

    validate_declared_types(
        value.node_role,
        expected_type=NodeRole,
        field="node_role",
    )
    _validate_parent_id(value.parent_id)

    validate_declared_types(
        value.model_ids,
        expected_type=tuple,
        field="model_ids",
    )
    seen_model_ids: set[ModelId] = set()
    for index, model_id in enumerate(value.model_ids):
        _validate_positive_decimal_identifier(
            model_id,
            field=f"model_ids[{index}]",
        )
        if model_id in seen_model_ids:
            raise ValueError(f"model_ids[{index}] must be unique; " f"got duplicate {model_id!r}.")
        seen_model_ids.add(model_id)

    if value.node_role is NodeRole.ROOT and not value.model_ids:
        raise ValueError("A root execution-state snapshot requires model_ids.")

    try:
        validate_parent_progress(value.progress)
    except (TypeError, ValueError) as error:
        raise type(error)(f"progress is invalid: {error}") from error
    if value.progress.parent_id != value.parent_id:
        raise ValueError(
            "progress.parent_id must equal snapshot parent_id; "
            f"expected {value.parent_id!r}, "
            f"got {value.progress.parent_id!r}."
        )

    validate_declared_types(
        value.attempts,
        expected_type=tuple,
        field="attempts",
    )
    for index, attempt in enumerate(value.attempts):
        try:
            _validate_attempt_progress_state(attempt)
        except (TypeError, ValueError) as error:
            raise type(error)(f"attempts[{index}] is invalid: {error}") from error

    validate_declared_types(
        value.receptions,
        expected_type=tuple,
        field="receptions",
    )
    attempt_inputs = tuple(attempt.inputs for attempt in value.attempts)
    try:
        validate_reception_set(
            value.receptions,
            attempts=attempt_inputs,
        )
    except (TypeError, ValueError) as error:
        raise type(error)(f"receptions are invalid: {error}") from error

    attempts_by_id = {attempt.inputs.attempt_id: attempt for attempt in value.attempts}
    for index, attempt in enumerate(value.attempts):
        if attempt.inputs.parent_id != value.parent_id:
            raise ValueError(
                f"attempts[{index}].inputs.parent_id must equal "
                f"{value.parent_id!r}; got {attempt.inputs.parent_id!r}."
            )
        if attempt.inputs.model_id not in seen_model_ids:
            raise ValueError(
                f"attempts[{index}].inputs.model_id must identify a listed " f"model; got {attempt.inputs.model_id!r}."
            )

    for index, reception in enumerate(value.receptions):
        if reception.parent_id != value.parent_id:
            raise ValueError(
                f"receptions[{index}].parent_id must equal " f"{value.parent_id!r}; got {reception.parent_id!r}."
            )
        if reception.model_id not in seen_model_ids:
            raise ValueError(
                f"receptions[{index}].model_id must identify a listed model; " f"got {reception.model_id!r}."
            )

    validate_declared_types(
        value.workers,
        expected_type=tuple,
        field="workers",
    )
    seen_worker_ids = set()
    for index, worker in enumerate(value.workers):
        try:
            validate_worker_observation(worker)
        except (TypeError, ValueError) as error:
            raise type(error)(f"workers[{index}] is invalid: {error}") from error

        if worker.worker_id in seen_worker_ids:
            raise ValueError(f"workers[{index}].worker_id must be unique; " f"got duplicate {worker.worker_id!r}.")
        seen_worker_ids.add(worker.worker_id)

        for attempt_id in worker.active_attempt_ids:
            active_attempt = attempts_by_id.get(attempt_id)
            if active_attempt is None:
                raise ValueError(f"Worker {worker.worker_id!r} references unknown active " f"attempt {attempt_id!r}.")
            if active_attempt.inputs.worker_id != worker.worker_id:
                raise ValueError(
                    f"Active attempt {attempt_id!r} must be assigned to "
                    f"worker {worker.worker_id!r}; "
                    f"got {active_attempt.inputs.worker_id!r}."
                )

    if value.node_role is NodeRole.WORKER:
        if value.receptions:
            raise ValueError("A worker execution-state snapshot must not contain " "root-side receptions.")
        if len(value.workers) != 1:
            raise ValueError(
                "A worker execution-state snapshot requires exactly one "
                f"worker observation; got {len(value.workers)}."
            )

        worker_id = value.workers[0].worker_id
        for index, attempt in enumerate(value.attempts):
            if attempt.inputs.worker_id != worker_id:
                raise ValueError(
                    f"attempts[{index}] must be assigned to the snapshot "
                    f"worker {worker_id!r}; "
                    f"got {attempt.inputs.worker_id!r}."
                )

    validate_declared_types(
        value.operations,
        expected_type=tuple,
        field="operations",
    )
    seen_operation_ids = set()
    failures = []
    for index, operation in enumerate(value.operations):
        try:
            validate_bounded_operation(operation)
        except (TypeError, ValueError) as error:
            raise type(error)(f"operations[{index}] is invalid: {error}") from error

        if operation.operation_id in seen_operation_ids:
            raise ValueError(
                f"operations[{index}].operation_id must be unique; " f"got duplicate {operation.operation_id!r}."
            )
        seen_operation_ids.add(operation.operation_id)

        if operation.last_failure is not None:
            failures.append(operation.last_failure)

    try:
        validate_failure_links(tuple(failures))
    except (TypeError, ValueError) as error:
        raise type(error)(f"operation failure links are invalid: {error}") from error
