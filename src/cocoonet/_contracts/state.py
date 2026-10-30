"""Orthogonal execution-state records for internal component contracts."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from cocoonet._contracts.execution import AttemptInputs, ParentEvaluation, validate_attempt_inputs
from cocoonet._contracts.identity import ArtifactId, ParentId, validate_identifier
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
