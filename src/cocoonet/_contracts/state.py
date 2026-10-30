"""Orthogonal execution-state records for internal component contracts."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from cocoonet._contracts.execution import AttemptInputs, ParentEvaluation, validate_attempt_inputs
from cocoonet._contracts.failure import (
    FailureRecord,
    FailureScope,
    TraceContext,
    validate_failure_record,
    validate_trace_context,
)
from cocoonet._contracts.identity import ArtifactId, AttemptId, OperationId, ParentId, WorkerId, validate_identifier
from cocoonet._type_validation import validate_declared_types


class ExecutionStatus(StrEnum):
    """Identify the root-observed execution state of a parent task.

    Attributes
    ----------
    NOT_STARTED : str
        Execution handling has not begun.
    RUNNING : str
        Execution handling is active.
    INTERRUPTED : str
        The root has interrupted the parent task, while remaining attempt and
        stop handling follows its independent state.
    SETTLED : str
        Execution handling has reached its terminal state.

    Notes
    -----
    This status is separate from configuration consolidation, worker activity,
    individual attempt state, result persistence, and result acceptance.
    ``SETTLED`` describes completion of execution handling and does not imply
    that every unit result was accepted; an exhausted delivery may leave a
    unit unaccepted.

    This enum records observations only. It does not define or perform state
    transitions.

    Node usage scope: Root only - Used exclusively by the root node.
    """

    NOT_STARTED = "not_started"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    SETTLED = "settled"


@dataclass(frozen=True, kw_only=True, slots=True)
class ParentProgress:
    """Associate a parent task with its root-observed execution status.

    Parameters
    ----------
    parent_id : ParentId
        Identity of the fixed parent task whose execution is observed.
    execution_status : ExecutionStatus
        Current root-observed state of the parent's execution handling.

    Notes
    -----
    This immutable record is a parent-level observation. It does not describe
    configuration consolidation, individual attempts, worker activity,
    scientific outcomes, result persistence, or result acceptance.

    Construction preserves candidate values without validating the parent
    identity or execution-status type. It does not perform state transitions,
    persist recovery state, or infer status from detailed execution records.

    Node usage scope: Root only - Used exclusively by the root node.
    """

    parent_id: ParentId
    execution_status: ExecutionStatus


def validate_parent_progress(value: ParentProgress) -> None:
    """Validate a parent identity and its root-observed execution status.

    Parameters
    ----------
    value : ParentProgress
        Parent-level execution observation to validate.

    Raises
    ------
    TypeError
        If the record or execution status has an incorrect runtime type.
    ValueError
        If the parent identity is empty.

    Notes
    -----
    This validator checks the supplied observation only. It does not infer or
    perform execution transitions, inspect configuration consolidation,
    examine attempts or workers, or determine result acceptance.

    Node usage scope: Root only - Used exclusively by the root node.
    """
    # Validate the outer record before accessing its fields.
    if not isinstance(value, ParentProgress):
        validate_declared_types(value, expected_type=ParentProgress)

    # Validate the parent identity independently of execution-state semantics.
    validate_identifier(value.parent_id, field="parent_id")

    # Require an explicit member of the parent-execution vocabulary.
    validate_declared_types(
        value.execution_status,
        expected_type=ExecutionStatus,
        field="execution_status",
    )


class AttemptStatus(StrEnum):
    """Identify the worker-observed processing state of an execution attempt.

    Attributes
    ----------
    ALLOCATED : str
        The attempt identity, unit task, and worker are fixed, but
        attempt-specific model synchronization or processing has not begun.
    RUNNING : str
        Attempt processing has begun with model synchronization or local
        handoff and continues through reconstruction, evaluation, report
        generation, and worker-side result publication.
    RESULT_READY : str
        The complete worker-side ``AttemptResult`` artifact has been
        published and is ready for reception and persistence handling.
    STOPPED : str
        The worker has observed the requested cooperative stop for the
        attempt.

    Notes
    -----
    These states describe worker-side attempt processing. Dataset and parent
    context readiness precede allocation. Result reception, persistence,
    acceptance, and bounded resend use separate records and do not introduce
    additional attempt states.

    ``STOPPED`` may describe an attempt stopped before processing begins or
    after it has produced a result. It does not imply that the worker process
    is unavailable.

    This enum records observations only. It does not perform transitions or
    define stop-disposition invariants.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    ALLOCATED = "allocated"
    RUNNING = "running"
    RESULT_READY = "result_ready"
    STOPPED = "stopped"


class StopStatus(StrEnum):
    """Identify the cooperative-stop handling state of an execution attempt.

    Attributes
    ----------
    NONE : str
        No cooperative stop has been requested or observed for the attempt.
    REQUESTED : str
        The root has requested that the worker stop the attempt, but the
        worker has not yet applied the request at a cooperative boundary.
    OBSERVED : str
        The worker has applied the cooperative-stop request at the applicable
        boundary, ending active attempt processing or unnecessary pending
        handling.

    Notes
    -----
    Stop handling is separate from attempt processing status. A transport or
    IPC receipt alone does not make a stop observed. For an active attempt,
    observation means processing has ended at a cooperative boundary. For an
    already-terminal attempt, it means unnecessary pending handling has ended.

    An observed stop does not imply that the reusable worker process has shut
    down or that unrelated cleanup, dataset release, or public-worker release
    handling has completed.

    This enum does not identify why the stop was requested, validate its
    relationship with an attempt status, or perform state transitions. Those
    responsibilities belong to ``StopReason``, ``StopDisposition``, and their
    validators.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    NONE = "none"
    REQUESTED = "requested"
    OBSERVED = "observed"


class StopReason(StrEnum):
    """Identify why an execution attempt receives a cooperative-stop request.

    Attributes
    ----------
    PARENT_INTERRUPTED : str
        The root interrupted the parent task. The fixed parent may later be
        resumed under the established recovery rules.
    PARENT_CANCELLED : str
        The root cancelled the parent task, so its remaining attempt work is
        no longer required.
    PARENT_SUPERSEDED : str
        A later parent replaced the earlier parent, so the earlier attempt
        cannot contribute to the later parent's result set.
    DUPLICATE_ACCEPTED : str
        Another attempt of the same parent/model unit produced the result
        accepted by the root, so this duplicate is no longer required.

    Notes
    -----
    A stop reason records why cooperative stop handling was initiated.
    ``StopStatus`` separately records whether that request is pending or has
    been applied.

    Connection loss, reconnection-round exhaustion, worker-process failure,
    result retention, and cleanup are separate operational facts and are not
    cooperative-stop reasons.

    A reason does not imply worker-process termination, artifact deletion,
    dataset cleanup, or public-worker release.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    PARENT_INTERRUPTED = "parent_interrupted"
    PARENT_CANCELLED = "parent_cancelled"
    PARENT_SUPERSEDED = "parent_superseded"
    DUPLICATE_ACCEPTED = "duplicate_accepted"


@dataclass(frozen=True, kw_only=True, slots=True)
class StopDisposition:
    """Associate cooperative-stop handling state with its reason.

    Parameters
    ----------
    status : StopStatus
        Current cooperative-stop handling observation.
    reason : StopReason | None
        Cause of a requested or observed stop. A disposition with status
        ``NONE`` uses ``None``.

    Notes
    -----
    This immutable record stores two directly related stop facts. It does not
    validate their pairing, perform a transition, send or acknowledge a stop,
    terminate the worker process, or control cleanup. Pairing validity belongs
    to ``validate_stop_disposition``.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    status: StopStatus
    reason: StopReason | None


def validate_stop_disposition(value: StopDisposition) -> None:
    """Validate cooperative-stop status and reason consistency.

    Parameters
    ----------
    value : StopDisposition
        Cooperative-stop observation to validate.

    Raises
    ------
    TypeError
        If the record, status, or reason has an incorrect runtime type.
    ValueError
        If no-stop state has a reason or a requested/observed stop has none.

    Notes
    -----
    This validator checks only the stop status/reason pair. It does not inspect
    attempt-processing status, send or acknowledge a stop, perform a state
    transition, terminate a worker process, or control artifact cleanup.

    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Validate the outer record before accessing its fields.
    if not isinstance(value, StopDisposition):
        validate_declared_types(value, expected_type=StopDisposition)

    # Validate both declared field types before evaluating their relationship.
    validate_declared_types(
        value.status,
        expected_type=StopStatus,
        field="status",
    )
    validate_declared_types(
        value.reason,
        expected_type=StopReason | None,
        field="reason",
    )

    # No-stop state must not retain a reason for an event that did not occur.
    if value.status is StopStatus.NONE:
        if value.reason is not None:
            raise ValueError(
                "Invalid stop disposition:\n"
                "  status: none\n"
                "  expected: reason to be None\n"
                f"  got: reason is {value.reason.value}"
            )
        return

    # Requested and observed stops must identify why handling was initiated.
    if value.reason is None:
        raise ValueError(
            "Invalid stop disposition:\n"
            f"  status: {value.status.value}\n"
            "  expected: reason to be a StopReason\n"
            "  got: reason is None"
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class AttemptProgress:
    """Record the worker-observed processing state of one execution attempt.

    Parameters
    ----------
    inputs : AttemptInputs
        Fixed attempt, parent task, model unit, and assigned-worker identities.
    status : AttemptStatus
        Current worker-observed attempt-processing state.
    stop : StopDisposition
        Independent cooperative-stop observation for the attempt.
    first_started_at : float | None
        Finite, nonnegative UTC Unix timestamp in seconds when attempt-specific
        model synchronization or local handoff first began. It is ``None``
        before processing starts.
    result_artifact_id : ArtifactId | None
        Identity of the completed worker-side result artifact, when one has
        been published for the attempt.

    Notes
    -----
    This immutable record preserves attempt progress observations. It does not
    validate their relationships, perform a transition, acquire a timestamp,
    publish an artifact, deliver a result, or infer root-side acceptance.
    Contract validation belongs to ``validate_attempt_progress``.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    inputs: AttemptInputs
    status: AttemptStatus
    stop: StopDisposition
    first_started_at: float | None
    result_artifact_id: ArtifactId | None


def _validate_attempt_progress_fields(value: AttemptProgress) -> None:
    """Validate the direct scalar fields of an attempt-progress observation."""
    # Require an explicit member of the attempt-processing vocabulary.
    validate_declared_types(
        value.status,
        expected_type=AttemptStatus,
        field="status",
    )

    # Validate an observed start as a finite, nonnegative UTC Unix timestamp.
    if value.first_started_at is not None:
        validate_declared_types(
            value.first_started_at,
            expected_type=float,
            field="first_started_at",
        )
        if not isfinite(value.first_started_at) or value.first_started_at < 0.0:
            raise ValueError(
                "Invalid attempt progress:\n"
                "  field: first_started_at\n"
                "  expected: finite, nonnegative UTC Unix timestamp in seconds\n"
                f"  got: {value.first_started_at!r}"
            )

    # A published result artifact must have a nonempty opaque identity.
    if value.result_artifact_id is not None:
        validate_identifier(
            value.result_artifact_id,
            field="result_artifact_id",
        )


def validate_attempt_progress(  # noqa: C901
    value: AttemptProgress,
    *,
    parent: ParentEvaluation,
) -> None:
    """Validate one attempt's routing and worker-observed progress.

    Parameters
    ----------
    value : AttemptProgress
        Attempt routing, processing, stop, start, and result observations.
    parent : ParentEvaluation
        Fixed parent task against which the attempt is resolved.

    Raises
    ------
    TypeError
        If a record or field has an incorrect runtime type.
    ValueError
        If routing, timestamp, artifact, stop, or lifecycle relationships are
        invalid.

    Notes
    -----
    This validator checks one supplied observation. It does not perform state
    transitions, acquire timestamps, publish or inspect result files, select
    or accept a result, reserve worker capacity, or infer worker availability.

    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Validate the outer record before accessing its fields.
    if not isinstance(value, AttemptProgress):
        validate_declared_types(value, expected_type=AttemptProgress)

    # Validate direct scalar fields before applying lifecycle relationships.
    _validate_attempt_progress_fields(value)

    # Resolve the attempt's fixed routing against its complete parent context.
    try:
        validate_attempt_inputs(value.inputs, parent=parent)
    except (TypeError, ValueError) as error:
        raise type(error)(f"inputs is invalid: {error}") from error

    # Validate the independent stop observation before composing it with status.
    try:
        validate_stop_disposition(value.stop)
    except (TypeError, ValueError) as error:
        raise type(error)(f"stop is invalid: {error}") from error

    error_context = f"Invalid attempt progress:\n  status: {value.status.value}"

    # Allocation precedes attempt-specific model synchronization or processing.
    if value.status is AttemptStatus.ALLOCATED and value.first_started_at is not None:
        raise ValueError(
            f"{error_context}\n"
            "  expected: first_started_at to be None before processing begins\n"
            f"  got: first_started_at is {value.first_started_at!r}"
        )

    # Running and result-ready observations require an established start.
    if value.status in (AttemptStatus.RUNNING, AttemptStatus.RESULT_READY) and value.first_started_at is None:
        raise ValueError(
            f"{error_context}\n" "  expected: first_started_at to be present\n" "  got: first_started_at is None"
        )

    # Result-ready means that a completed worker-side artifact was published.
    if value.status is AttemptStatus.RESULT_READY and value.result_artifact_id is None:
        raise ValueError(
            f"{error_context}\n" "  expected: result_artifact_id to be present\n" "  got: result_artifact_id is None"
        )

    # Allocated or running attempts cannot claim a completed result artifact.
    if value.status in (AttemptStatus.ALLOCATED, AttemptStatus.RUNNING) and value.result_artifact_id is not None:
        raise ValueError(
            f"{error_context}\n"
            "  expected: result_artifact_id to be None\n"
            f"  got: result_artifact_id is {value.result_artifact_id!r}"
        )

    # Any completed result artifact requires evidence that processing started.
    if value.result_artifact_id is not None and value.first_started_at is None:
        raise ValueError(
            f"{error_context}\n"
            "  expected: first_started_at to accompany result_artifact_id\n"
            f"  got: result_artifact_id is {value.result_artifact_id!r} and first_started_at is None"
        )

    # Stopped processing requires the worker to have applied the stop request.
    if value.status is AttemptStatus.STOPPED and value.stop.status is not StopStatus.OBSERVED:
        raise ValueError(
            f"{error_context}\n"
            "  expected: stop.status to be observed\n"
            f"  got: stop.status is {value.stop.status.value}"
        )

    # Active or not-yet-started processing cannot already have observed a stop.
    if value.status in (AttemptStatus.ALLOCATED, AttemptStatus.RUNNING) and value.stop.status is StopStatus.OBSERVED:
        raise ValueError(
            f"{error_context}\n" "  expected: stop.status not to be observed\n" "  got: stop.status is observed"
        )


class WorkerAvailability(StrEnum):
    """Identify whether a configured worker is usable for new allocation.

    Attributes
    ----------
    UNAVAILABLE : str
        The worker does not currently satisfy all prerequisites for receiving
        another execution attempt.
    AVAILABLE : str
        The worker satisfies the ownership, compatibility, process, and known
        capacity prerequisites for allocation consideration.

    Notes
    -----
    Availability is separate from configured worker membership and occupied
    capacity. An available worker may have no free attempt slot, while an
    unavailable worker may retain assigned attempts and their reservations.

    This enum records an observation only. It does not calculate availability,
    inspect worker state, reserve capacity, allocate work, reconnect a worker,
    or perform a state transition.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    UNAVAILABLE = "unavailable"
    AVAILABLE = "available"


class WorkerProcessStatus(StrEnum):
    """Identify the observed operating state of a worker process.

    Attributes
    ----------
    UNKNOWN : str
        The current worker-process state cannot be established.
    RUNNING : str
        The worker process is observed to be running.
    UNAVAILABLE : str
        The worker process itself is observed to be unavailable, such as after
        a crash, machine shutdown, or external termination.

    Notes
    -----
    Process status is separate from allocation availability. A running process
    may remain unavailable for new allocation because another usability
    prerequisite is not satisfied.

    Lost contact produces ``UNKNOWN`` unless there is evidence that the process
    is unavailable. It does not by itself establish process failure.
    ``UNAVAILABLE`` describes the process itself and may require external
    intervention; it is not inferred from a model, report, synchronization,
    transfer, persistence, or connection-round failure.

    This enum records an observation only. It does not probe, monitor, restart,
    terminate, reconnect, or allocate the worker.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    UNKNOWN = "unknown"
    RUNNING = "running"
    UNAVAILABLE = "unavailable"


class OwnershipStatus(StrEnum):
    """Identify the observed root ownership of a worker.

    Attributes
    ----------
    UNKNOWN : str
        Current ownership by the observing root cannot be established.
    OWNED_BY_ROOT : str
        The worker is observed to be assigned to the observing root.
    NOT_OWNED_BY_ROOT : str
        The worker is observed not to be assigned to the observing root.

    Notes
    -----
    Ownership is separate from configured worker membership, connection
    health, process status, compatibility, readiness, and free capacity.
    Unknown ownership does not imply ownership by another root.

    For public workers, this status does not reproduce or replace the external
    service's pool membership, discovery, reservation, allocation, or release
    state. Authentication and public-allocation mechanisms are not encoded by
    ``OWNED_BY_ROOT``.

    This enum records an observation only. It does not authenticate, claim,
    release, configure, connect, disconnect, or allocate a worker.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    UNKNOWN = "unknown"
    OWNED_BY_ROOT = "owned_by_root"
    NOT_OWNED_BY_ROOT = "not_owned_by_root"


class CompatibilityStatus(StrEnum):
    """Identify the observed compatibility of a worker environment.

    Attributes
    ----------
    UNCHECKED : str
        Worker compatibility has not yet been established.
    COMPATIBLE : str
        The checked worker metadata satisfies the required compatibility
        contract.
    INCOMPATIBLE : str
        The checked worker metadata does not satisfy the required compatibility
        contract.

    Notes
    -----
    Compatibility is separate from process status, root ownership, parent
    readiness, allocation availability, and free capacity. ``UNCHECKED`` does
    not imply incompatibility.

    This enum stores the result of compatibility evaluation only. It does not
    inspect metadata, compare versions, load executable content, repair an
    environment, choose fallback components, or change worker availability.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    UNCHECKED = "unchecked"
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True, kw_only=True, slots=True)
class ParentContextReadiness:
    """Record a worker's readiness observations for one parent task.

    Parameters
    ----------
    parent_id : ParentId
        Identity of the fixed parent task whose context is observed.
    dataset_ready : bool
        Whether the parent's dataset representation is ready on the worker.
    evaluation_ready : bool
        Whether the parent's fixed evaluation configuration is ready on the
        worker.
    partition_ready : bool
        Whether the parent's realized partition context is ready on the worker.

    Notes
    -----
    Dataset, evaluation, and partition readiness remain separate because each
    artifact or context can be prepared or invalidated independently. A parent
    is ready for allocation only when all three observations are true, together
    with the worker-level prerequisites defined by ``WorkerObservation``.

    This immutable record stores observations only. It does not synchronize
    data, load configuration, transfer partitions, allocate attempts, infer
    worker availability, or perform readiness transitions. Construction does
    not validate the parent identity or flag types.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    parent_id: ParentId
    dataset_ready: bool
    evaluation_ready: bool
    partition_ready: bool


@dataclass(frozen=True, kw_only=True, slots=True)
class WorkerObservation:
    """Record the operational observations used to assess one worker.

    Parameters
    ----------
    worker_id : WorkerId
        Identity of the configured worker being observed.
    availability : WorkerAvailability
        Whether the worker currently satisfies the prerequisites for allocation
        consideration.
    process_status : WorkerProcessStatus
        Observed operating state of the worker process.
    ownership : OwnershipStatus
        Observed assignment of the worker to the root.
    compatibility : CompatibilityStatus
        Observed compatibility of the worker environment.
    parent_readiness : tuple[ParentContextReadiness, ...]
        Parent-scoped dataset, evaluation, and partition readiness observations.
    attempt_capacity : int | None
        Effective number of complete attempt slots, or ``None`` when capacity
        is unknown.
    active_attempt_ids : tuple[AttemptId, ...]
        Identities of attempts whose slot reservations remain active.

    Notes
    -----
    Availability, process state, ownership, compatibility, parent readiness,
    capacity, and reservations remain independent observations. An available
    worker may have every slot occupied, while an unavailable worker may retain
    active attempt reservations.

    Active reservations include applicable pending-delivery handling until the
    owning lifecycle releases the slot. Complete attempt progress remains in
    ``AttemptProgress`` rather than being duplicated here.

    This immutable record stores observations only. It does not probe a worker,
    calculate availability or free capacity, synchronize parent context,
    allocate attempts, release reservations, or perform state transitions.
    Construction does not validate fields or derive one field from another.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    worker_id: WorkerId
    availability: WorkerAvailability
    process_status: WorkerProcessStatus
    ownership: OwnershipStatus
    compatibility: CompatibilityStatus
    parent_readiness: tuple[ParentContextReadiness, ...]
    attempt_capacity: int | None
    active_attempt_ids: tuple[AttemptId, ...]


def _validate_parent_context_readiness(
    value: ParentContextReadiness,
    *,
    field: str,
) -> None:
    """Validate one parent-scoped worker-readiness observation."""
    # Validate the record before inspecting its parent identity or flags.
    if not isinstance(value, ParentContextReadiness):
        validate_declared_types(
            value,
            expected_type=ParentContextReadiness,
            field=field,
        )

    validate_identifier(
        value.parent_id,
        field=f"{field}.parent_id",
    )

    # Each readiness fact is independently and explicitly boolean.
    for name, ready in (
        ("dataset_ready", value.dataset_ready),
        ("evaluation_ready", value.evaluation_ready),
        ("partition_ready", value.partition_ready),
    ):
        validate_declared_types(
            ready,
            expected_type=bool,
            field=f"{field}.{name}",
        )


def _validate_worker_capacity_and_reservations(
    value: WorkerObservation,
) -> None:
    """Validate effective capacity and active attempt reservations."""
    # Unknown capacity is representable; every known capacity is positive.
    if value.attempt_capacity is not None:
        validate_declared_types(
            value.attempt_capacity,
            expected_type=int,
            field="attempt_capacity",
        )
        if value.attempt_capacity <= 0:
            raise ValueError(
                "Invalid worker observation:\n"
                "  field: attempt_capacity\n"
                "  expected: a positive integer or None\n"
                f"  got: {value.attempt_capacity!r}"
            )

    # Reservations are unique attempt identities, independent of their states.
    validate_declared_types(
        value.active_attempt_ids,
        expected_type=tuple,
        field="active_attempt_ids",
    )
    seen_attempt_ids: set[AttemptId] = set()
    for index, attempt_id in enumerate(value.active_attempt_ids):
        field = f"active_attempt_ids[{index}]"
        validate_identifier(attempt_id, field=field)
        if attempt_id in seen_attempt_ids:
            raise ValueError(
                "Invalid worker observation:\n"
                f"  field: {field}\n"
                "  expected: each active attempt identity to be unique\n"
                f"  got: duplicate identity {attempt_id!r}"
            )
        seen_attempt_ids.add(attempt_id)


def validate_worker_observation(value: WorkerObservation) -> None:
    """Validate one worker's operational and readiness observations.

    Parameters
    ----------
    value : WorkerObservation
        Worker identity, usability, process, ownership, compatibility,
        readiness, capacity, and reservation observations.

    Raises
    ------
    TypeError
        If the record or a field has an incorrect runtime type.
    ValueError
        If an identity, capacity, uniqueness constraint, or availability
        prerequisite is invalid.

    Notes
    -----
    Availability asserts worker-level usability prerequisites but does not
    imply a free slot or readiness for a particular parent. Reservations may
    equal or exceed a known capacity after a prospective capacity reduction;
    this validator preserves them and does not allocate or cancel work.

    This function does not probe a worker, calculate capacity, synchronize
    parent context, derive free slots, schedule attempts, mutate reservations,
    or perform lifecycle transitions.

    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Validate the outer record before accessing its fields.
    if not isinstance(value, WorkerObservation):
        validate_declared_types(value, expected_type=WorkerObservation)

    # Validate the worker identity and independent worker-level observations.
    validate_identifier(value.worker_id, field="worker_id")
    validate_declared_types(
        value.availability,
        expected_type=WorkerAvailability,
        field="availability",
    )
    validate_declared_types(
        value.process_status,
        expected_type=WorkerProcessStatus,
        field="process_status",
    )
    validate_declared_types(
        value.ownership,
        expected_type=OwnershipStatus,
        field="ownership",
    )
    validate_declared_types(
        value.compatibility,
        expected_type=CompatibilityStatus,
        field="compatibility",
    )

    # Validate each parent context and prohibit competing observations.
    validate_declared_types(
        value.parent_readiness,
        expected_type=tuple,
        field="parent_readiness",
    )
    seen_parent_ids: set[ParentId] = set()
    for index, readiness in enumerate(value.parent_readiness):
        field = f"parent_readiness[{index}]"
        _validate_parent_context_readiness(readiness, field=field)
        if readiness.parent_id in seen_parent_ids:
            raise ValueError(
                "Invalid worker observation:\n"
                f"  field: {field}.parent_id\n"
                "  expected: each parent readiness identity to be unique\n"
                f"  got: duplicate identity {readiness.parent_id!r}"
            )
        seen_parent_ids.add(readiness.parent_id)

    _validate_worker_capacity_and_reservations(value)

    # Available workers must satisfy every worker-level usability prerequisite.
    if value.availability is WorkerAvailability.AVAILABLE:
        prerequisites: tuple[tuple[str, StrEnum, StrEnum], ...] = (
            ("process_status", value.process_status, WorkerProcessStatus.RUNNING),
            ("ownership", value.ownership, OwnershipStatus.OWNED_BY_ROOT),
            ("compatibility", value.compatibility, CompatibilityStatus.COMPATIBLE),
        )
        for field, observed, expected in prerequisites:
            if observed is not expected:
                raise ValueError(
                    "Invalid worker observation:\n"
                    "  availability: available\n"
                    f"  expected: {field} to be {expected.value}\n"
                    f"  got: {field} is {observed.value}"
                )

        if value.attempt_capacity is None:
            raise ValueError(
                "Invalid worker observation:\n"
                "  availability: available\n"
                "  expected: attempt_capacity to be known\n"
                "  got: attempt_capacity is None"
            )


class OperationKind(StrEnum):
    """Identify a bounded infrastructure operation.

    Attributes
    ----------
    CONNECTION_ROUND : str
        One bounded round of worker connection or reconnection attempts.
    TRANSFER : str
        Bounded transfer of a dataset, model artifact, or completed result.
    PERSISTENCE : str
        Bounded persistence of a prepared artifact or received result.

    Notes
    -----
    These kinds cover only infrastructure operations with explicit attempt
    limits. They do not make model reconstruction, fitting, prediction, report
    computation, or sample-influence evaluation retryable.

    Each separately scheduled connection round has its own operation identity.
    Transfer and persistence operate on existing artifacts and do not authorize
    scientific recomputation.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    CONNECTION_ROUND = "connection_round"
    TRANSFER = "transfer"
    PERSISTENCE = "persistence"


class OperationStatus(StrEnum):
    """Identify the observed progress of a bounded operation.

    Attributes
    ----------
    PENDING : str
        No operation attempt has begun.
    RUNNING : str
        At least one operation attempt has begun and handling remains active.
    SUCCEEDED : str
        The operation completed successfully.
    FAILED : str
        The latest operation attempt failed.
    CANCELLED : str
        Operation handling was cancelled without establishing success.

    Notes
    -----
    ``FAILED`` does not independently mean that the attempt limit is exhausted.
    Exhaustion is derived when a failed operation's attempt count reaches its
    configured limit. A failed operation below that limit may still be retried
    by its later policy owner.

    ``SUCCEEDED`` may retain an earlier failure record because a later attempt
    can succeed. ``CANCELLED`` does not imply scientific failure, result
    acceptance, or worker-process unavailability.

    This enum records progress only. It does not begin, retry, cancel, or
    otherwise perform an operation.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, kw_only=True, slots=True)
class BoundedOperation:
    """Record the progress of one bounded infrastructure operation.

    Parameters
    ----------
    operation_id : OperationId
        Identity of this independently bounded operation.
    kind : OperationKind
        Infrastructure operation governed by the attempt limit.
    context : TraceContext
        Operational context associated with this operation.
    attempt_limit : int
        Total permitted attempts, including the initial attempt.
    attempts_made : int
        Number of operation attempts already begun.
    status : OperationStatus
        Current observed operation progress.
    last_failure : FailureRecord | None
        Most recent failure, or ``None`` when no attempt has failed. A
        successful operation may retain an earlier failure from a prior
        attempt.

    Notes
    -----
    Exhaustion is derived when ``status`` is ``FAILED`` and
    ``attempts_made == attempt_limit``. It is not stored as another status or
    flag.

    This immutable record preserves supplied observations only. Construction
    does not validate count bounds, status consistency, operation context, or
    failure provenance. It does not schedule, begin, retry, cancel, or reset an
    operation, and it does not perform scientific recomputation.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    operation_id: OperationId
    kind: OperationKind
    context: TraceContext
    attempt_limit: int
    attempts_made: int
    status: OperationStatus
    last_failure: FailureRecord | None


def _validate_bounded_operation_counts(value: BoundedOperation) -> None:
    """Validate a bounded operation's total limit and begun-attempt count."""
    # Validate exact integer types before applying numeric bounds.
    validate_declared_types(
        value.attempt_limit,
        expected_type=int,
        field="attempt_limit",
    )
    validate_declared_types(
        value.attempts_made,
        expected_type=int,
        field="attempts_made",
    )

    # The limit includes the initial attempt and therefore cannot be zero.
    if value.attempt_limit < 1:
        raise ValueError(
            "Invalid bounded operation:\n"
            "  field: attempt_limit\n"
            "  expected: an integer greater than or equal to 1\n"
            f"  got: {value.attempt_limit!r}"
        )

    # Begun attempts cannot be negative or exceed the fixed total limit.
    if not 0 <= value.attempts_made <= value.attempt_limit:
        raise ValueError(
            "Invalid bounded operation:\n"
            "  field: attempts_made\n"
            f"  expected: an integer from 0 through {value.attempt_limit}\n"
            f"  got: {value.attempts_made!r}"
        )


def _validate_bounded_operation_failure(value: BoundedOperation) -> None:
    """Validate an operation's optional last failure and its provenance."""
    if value.last_failure is None:
        return

    # Validate the complete failure record before comparing its provenance.
    try:
        validate_failure_record(value.last_failure)
    except (TypeError, ValueError) as error:
        raise type(error)(f"last_failure is invalid: {error}") from error

    # The retained failure must describe the same complete operation context.
    if value.last_failure.context != value.context:
        raise ValueError(
            "Invalid bounded operation:\n"
            f"  operation_id: {value.operation_id!r}\n"
            "  expected: last_failure.context to equal context\n"
            "  got: last_failure.context differs from context"
        )

    # Each operation kind permits only its architecturally assigned scopes.
    permitted_scopes: dict[OperationKind, tuple[FailureScope, ...]] = {
        OperationKind.CONNECTION_ROUND: (FailureScope.WORKER_MANAGEMENT,),
        OperationKind.TRANSFER: (
            FailureScope.DATA_SYNC,
            FailureScope.RESULT_TRANSFER,
        ),
        OperationKind.PERSISTENCE: (
            FailureScope.DATA_PREPARATION,
            FailureScope.RESULT_PERSISTENCE,
        ),
    }
    if value.last_failure.scope not in permitted_scopes[value.kind]:
        expected = ", ".join(scope.value for scope in permitted_scopes[value.kind])
        raise ValueError(
            "Invalid bounded operation:\n"
            f"  operation_id: {value.operation_id!r}\n"
            f"  kind: {value.kind.value}\n"
            f"  expected: last_failure.scope to be one of {expected}\n"
            f"  got: {value.last_failure.scope.value}"
        )


def _validate_bounded_operation_status(value: BoundedOperation) -> None:
    """Validate operation status against counts and retained failure state."""
    # Pending is the pristine state before any attempt or failure.
    if value.status is OperationStatus.PENDING:
        if value.attempts_made != 0:
            raise ValueError(
                "Invalid bounded operation:\n"
                "  status: pending\n"
                "  expected: attempts_made to be 0\n"
                f"  got: {value.attempts_made}"
            )
        if value.last_failure is not None:
            raise ValueError(
                "Invalid bounded operation:\n"
                "  status: pending\n"
                "  expected: last_failure to be None\n"
                f"  got: failure {value.last_failure.failure_id!r}"
            )
        return

    # Active, successful, and failed states require a begun attempt.
    if (
        value.status
        in (
            OperationStatus.RUNNING,
            OperationStatus.SUCCEEDED,
            OperationStatus.FAILED,
        )
        and value.attempts_made == 0
    ):
        raise ValueError(
            "Invalid bounded operation:\n"
            f"  status: {value.status.value}\n"
            "  expected: attempts_made to be at least 1\n"
            "  got: 0"
        )

    # A failed state must retain the failure that explains it.
    if value.status is OperationStatus.FAILED and value.last_failure is None:
        raise ValueError(
            "Invalid bounded operation:\n"
            "  status: failed\n"
            "  expected: last_failure to contain the latest failure\n"
            "  got: None"
        )


def validate_bounded_operation(value: BoundedOperation) -> None:
    """Validate one bounded infrastructure-operation observation.

    Parameters
    ----------
    value : BoundedOperation
        Operation identity, kind, context, limit, progress, status, and
        optional latest failure.

    Raises
    ------
    TypeError
        If the record or a field has an incorrect runtime type.
    ValueError
        If an identity, count, status relationship, context association, or
        failure provenance is invalid.

    Notes
    -----
    ``attempt_limit`` is the total number of allowed attempts, including the
    initial attempt. Exhaustion is represented by ``FAILED`` with
    ``attempts_made == attempt_limit`` and is not stored separately.

    Successful and running operations may preserve a prior failure. Cancelled
    operations may have zero or more begun attempts. Validation does not reset
    a budget, schedule a retry, create a new connection round, transfer or
    persist an artifact, or repeat scientific computation.

    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Validate the outer record and direct vocabulary fields before inspection.
    if not isinstance(value, BoundedOperation):
        validate_declared_types(value, expected_type=BoundedOperation)
    validate_identifier(value.operation_id, field="operation_id")
    validate_declared_types(
        value.kind,
        expected_type=OperationKind,
        field="kind",
    )
    validate_declared_types(
        value.status,
        expected_type=OperationStatus,
        field="status",
    )

    # Validate and align the operation's trace context with its identity.
    try:
        validate_trace_context(value.context)
    except (TypeError, ValueError) as error:
        raise type(error)(f"context is invalid: {error}") from error
    if value.context.operation_id != value.operation_id:
        raise ValueError(
            "Invalid bounded operation:\n"
            f"  operation_id: {value.operation_id!r}\n"
            "  expected: context.operation_id to equal operation_id\n"
            f"  got: {value.context.operation_id!r}"
        )

    # Validate independent counts before relationships that consume them.
    _validate_bounded_operation_counts(value)

    # Validate optional failure provenance before status requirements.
    _validate_bounded_operation_failure(value)

    # Validate the final status/count/failure relationship.
    _validate_bounded_operation_status(value)


class ReceiptStatus(StrEnum):
    """Identify root-side receipt progress for a selected result file.

    Attributes
    ----------
    NOT_RECEIVED : str
        Receipt of the selected result file has not begun.
    INCOMPLETE : str
        Receipt has begun, but the complete selected result file has not been
        received.
    COMPLETE : str
        The complete selected result file has been received.

    Notes
    -----
    Receipt progress is independent of persistence and acceptance. Complete
    receipt does not establish that the file was durably stored or accepted,
    while incomplete receipt may coexist with an incomplete persistent write.

    This enum records root-side reception progress only. It does not select an
    attempt, receive bytes, inspect a result, persist a file, acknowledge
    persistence, or accept a logical result.

    Node usage scope: Root only - Used exclusively by the root node.
    """

    NOT_RECEIVED = "not_received"
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"


class PersistenceStatus(StrEnum):
    """Identify root-side persistence progress for a selected result file.

    Attributes
    ----------
    NOT_STARTED : str
        Persistent writing of the selected result file has not begun.
    INCOMPLETE : str
        Persistent writing has begun, but the complete selected result file has
        not been durably stored.
    COMPLETE : str
        The complete selected result file has been durably stored.

    Notes
    -----
    Persistence progress is independent of receipt and acceptance. Incomplete
    persistence may coexist with incomplete receipt as bytes are written, but
    complete persistence requires complete receipt. Complete persistence does
    not independently accept the logical result.

    This enum records root-side persistence progress only. It does not receive
    bytes, inspect a result, write a file, verify durable storage, acknowledge
    persistence, or accept a logical result.

    Node usage scope: Root only - Used exclusively by the root node.
    """

    NOT_STARTED = "not_started"
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"
