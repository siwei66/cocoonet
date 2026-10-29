"""Orthogonal execution-state records for internal component contracts."""

from dataclasses import dataclass
from enum import StrEnum

from cocoonet._contracts.execution import AttemptInputs
from cocoonet._contracts.identity import ArtifactId, ParentId


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
