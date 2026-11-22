"""Root-side prospective evaluation configuration records."""

from dataclasses import dataclass
from enum import StrEnum

from cocoonet._contracts.execution import ModelFamily, TaskKind
from cocoonet._contracts.identity import (
    ScalarLabel,
    WorkerId,
    validate_identifier,
    validate_label,
)
from cocoonet._type_validation import validate_declared_types


class WorkerKind(StrEnum):
    """Identify the configured execution topology of a worker.

    Attributes
    ----------
    ROOT_LOCAL : str
        Worker explicitly configured on the root machine.
    ONLINE : str
        Worker reached through a later online connection boundary.

    Notes
    -----
    Public/private online ownership is intentionally not represented here.
    Node usage scope: Root - Defines root-side worker topology configuration.
    """

    ROOT_LOCAL = "root-local"
    ONLINE = "online"


@dataclass(frozen=True, kw_only=True, slots=True)
class ModelRegistration:
    """Describe one opaque model input proposed for registration.

    Parameters
    ----------
    model_input : object
        In-memory model input retained without inspection or copying.
    model_label : ScalarLabel
        User-provided label that later registry operations require to be
        unique within an evaluation task.
    family : ModelFamily
        Explicitly declared model-construction family.
    task_kind : TaskKind
        Explicitly declared regression or classification task.

    Notes
    -----
    Construction records the supplied values without validating or invoking
    the opaque model input. Validation is a separate configuration boundary.
    Node usage scope: Root - Defines root-side prospective model input.
    """

    model_input: object
    model_label: ScalarLabel
    family: ModelFamily
    task_kind: TaskKind


def validate_model_registration(value: ModelRegistration, /) -> None:
    """Validate one prospective opaque model registration.

    Parameters
    ----------
    value : ModelRegistration
        Candidate model input and its explicitly declared metadata.

    Raises
    ------
    TypeError
        If the record, label, family, or task kind has an incorrect runtime
        type.
    ValueError
        If the opaque model input is ``None``.

    Notes
    -----
    Any non-``None`` model input is accepted without inspection, copying,
    serialization, method access, or family-specific validation. Registry
    uniqueness and task agreement are separate mutation-level checks.
    Node usage scope: Root - Validates root-side prospective model input.
    """
    if not isinstance(value, ModelRegistration):
        validate_declared_types(value, expected_type=ModelRegistration)

    if value.model_input is None:
        raise ValueError("model_input must not be None.")

    validate_label(value.model_label, field="model_label")
    validate_declared_types(value.family, expected_type=ModelFamily, field="family")
    validate_declared_types(value.task_kind, expected_type=TaskKind, field="task_kind")


@dataclass(frozen=True, kw_only=True, slots=True)
class WorkerConfiguration:
    """Describe one prospective root-local or online worker.

    Parameters
    ----------
    worker_label : ScalarLabel
        User-facing label used by public worker-registry operations.
    worker_id : WorkerId
        Opaque identity used by runtime routing and persistence.
    kind : WorkerKind
        Root-local or online execution topology.
    device : str, default="cpu"
        Requested device text retained without interpretation.
    n_attempt : int, default=1
        Requested positive maximum number of concurrent complete attempts.

    Notes
    -----
    This record makes no availability, compatibility, ownership, endpoint,
    or effective-capacity claim. Later CPU worker control derives
    ``C = N - 1``, ``J = max(per-attempt demand)``, and ``A = n_attempt``.
    It keeps ``A`` when ``A * J <= C``, reduces exactly once to ``A - 1``
    when permitted, and otherwise rejects without changing model or framework
    settings. GPU execution later uses one Cocoonet attempt slot. Phase 3
    performs none of these calculations.
    Node usage scope: Root - Defines root-side prospective worker configuration.
    """

    worker_label: ScalarLabel
    worker_id: WorkerId
    kind: WorkerKind
    device: str = "cpu"
    n_attempt: int = 1


def validate_worker_configuration(value: WorkerConfiguration, /) -> None:
    """Validate one prospective worker configuration.

    Parameters
    ----------
    value : WorkerConfiguration
        Candidate worker identity, topology, device, and requested capacity.

    Raises
    ------
    TypeError
        If the record or one of its fields has an incorrect runtime type.
    ValueError
        If the worker identity or device is empty, or if ``n_attempt`` is not
        positive.

    Notes
    -----
    This validation preserves requested values without probing a device,
    calculating effective capacity, resolving an endpoint, or checking
    registry-scoped uniqueness and topology limits.
    Node usage scope: Root - Validates root-side worker configuration.
    """
    if not isinstance(value, WorkerConfiguration):
        validate_declared_types(value, expected_type=WorkerConfiguration)

    validate_label(value.worker_label, field="worker_label")
    validate_identifier(value.worker_id, field="worker_id")
    validate_declared_types(value.kind, expected_type=WorkerKind, field="kind")

    validate_declared_types(value.device, expected_type=str, field="device")
    if not value.device:
        raise ValueError("device must be a nonempty string; got an empty string.")

    validate_declared_types(value.n_attempt, expected_type=int, field="n_attempt")
    if value.n_attempt < 1:
        raise ValueError(f"n_attempt must be a positive integer; got {value.n_attempt!r}.")
