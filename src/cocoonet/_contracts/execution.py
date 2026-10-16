"""Configuration and fixed-execution records for internal contracts."""

from dataclasses import dataclass
from enum import StrEnum

from cocoonet._contracts.data import PartitionReference
from cocoonet._contracts.identity import (
    AttemptId,
    EvaluationConfigId,
    ModelId,
    ParentId,
    ScalarLabel,
    UnitKey,
    WorkerId,
)


class TaskKind(StrEnum):
    """Identify the scientific prediction task performed by an evaluation.

    Attributes
    ----------
    REGRESSION : str
        Evaluation predicts a numeric target.
    CLASSIFICATION : str
        Evaluation predicts membership in a declared class vocabulary.
    """

    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class ModelFamily(StrEnum):
    """Identify the supported construction family of a model input.

    Attributes
    ----------
    SKLEARN : str
        Standard supported scikit-learn model, including standard
        scikit-learn ensemble and composition mechanisms.
    SKLEARN_LIKE : str
        Custom model implementing the required scikit-learn-like interface.
    PYTORCH : str
        PyTorch model reconstructed from its supplied training information.
    COCOONET : str
        Model constructed through Cocoonet's native model-composition tools,
        including native ensembles and other supported composite models.

    Notes
    -----
    A Cocoonet-native composition has the ``COCOONET`` family. Each constituent
    model retains its own family identity, including ``COCOONET`` for nested
    native compositions. Standard scikit-learn ensembles remain ``SKLEARN``.
    """

    SKLEARN = "sklearn"
    SKLEARN_LIKE = "sklearn-like"
    PYTORCH = "pytorch"
    COCOONET = "cocoonet"


class ConsolidationState(StrEnum):
    """Track whether evaluation configuration is committed to a parent task.

    Attributes
    ----------
    INITIALIZED : str
        A fresh working directory is ready, but the current evaluation
        configuration has not been committed to a parent task. Configuration
        edits retain this state. Successful ``start()`` creates a parent task
        and changes the state to ``CONSOLIDATED``.
    CONSOLIDATED : str
        The current evaluation configuration matches an existing parent task.
        Unchanged starts resume that parent task, and worker-only changes
        retain this state. An evaluation-configuration edit changes the state
        to ``UNCONSOLIDATED``.
    UNCONSOLIDATED : str
        The prospective evaluation configuration differs from the previous
        parent task, which remains fixed. ``start()`` is rejected. Explicit
        fresh initialization changes the state to ``INITIALIZED``.

    Notes
    -----
    Changed evaluation configuration follows ``CONSOLIDATED`` to
    ``UNCONSOLIDATED`` to ``INITIALIZED`` to ``CONSOLIDATED``. Copying an
    instance preserves its state. Execution and worker status are separate.
    """

    INITIALIZED = "initialized"
    CONSOLIDATED = "consolidated"
    UNCONSOLIDATED = "unconsolidated"


@dataclass(frozen=True, kw_only=True, slots=True)
class ModelReference:
    """Reference one fixed supplied model input.

    Parameters
    ----------
    model_id : ModelId
        Cocoonet-assigned identity of the fixed model input.
    model_label : ScalarLabel
        User-provided model label, unique within its parent task.
    family : ModelFamily
        Construction family used to interpret the model input: ``sklearn``,
        ``sklearn-like``, ``pytorch``, or ``cocoonet``.
    task_kind : TaskKind
        Regression or classification task performed by the model.

    Notes
    -----
    This record contains identifying metadata only. The owning configuration
    component retains the supplied model input and later prepares its payload.
    Construction preserves candidate values without validating them or
    inspecting, storing, or executing a model object.
    """

    model_id: ModelId
    model_label: ScalarLabel
    family: ModelFamily
    task_kind: TaskKind


@dataclass(frozen=True, kw_only=True, slots=True)
class EvaluationReference:
    """Reference one fixed task-wide evaluation configuration.

    Parameters
    ----------
    configuration_id : EvaluationConfigId
        Identity of the complete fixed evaluation-settings snapshot.
    task_kind : TaskKind
        Regression or classification task shared by every model in the
        parent task.
    class_labels : tuple[ScalarLabel, ...]
        Authoritative ordered class vocabulary. Regression uses an empty
        tuple.
    influence_requested : bool
        Whether the parent task requests optional sample-influence analysis.

    Notes
    -----
    This record identifies settings owned by the configuration component; it
    does not contain their complete implementation-specific payload. The same
    identity must not be reused for changed settings. Construction preserves
    candidate values without validating task-kind or class-label consistency.
    """

    configuration_id: EvaluationConfigId
    task_kind: TaskKind
    class_labels: tuple[ScalarLabel, ...]
    influence_requested: bool


@dataclass(frozen=True, kw_only=True, slots=True)
class ComputationSnapshot:
    """Describe the current root-side computation configuration.

    Parameters
    ----------
    root_dir : str
        User-selected Cocoonet-managed working directory.
    consolidation : ConsolidationState
        Relationship between the prospective evaluation configuration and
        its committed parent task.
    partition : PartitionReference | None
        Active dataset and realized-partition identities, or ``None`` when
        no dataset and partition are currently configured.
    evaluation : EvaluationReference | None
        Active evaluation-settings reference, or ``None`` when evaluation
        settings are not currently configured.
    models : tuple[ModelReference, ...]
        Ordered model references in the prospective configuration.
    worker_ids : tuple[WorkerId, ...]
        Ordered configured execution-target identities.
    parent_id : ParentId | None
        Existing committed parent-task identity. Initialized state uses
        ``None``; consolidated and unconsolidated states retain the existing
        parent identity.

    Notes
    -----
    This immutable record describes configuration state only. Construction
    preserves candidate values without validating paths, identities,
    uniqueness, consolidation relationships, or configuration completeness.
    It does not initialize storage, create a parent task, or own runtime
    worker resources.
    """

    root_dir: str
    consolidation: ConsolidationState
    partition: PartitionReference | None
    evaluation: EvaluationReference | None
    models: tuple[ModelReference, ...]
    worker_ids: tuple[WorkerId, ...]
    parent_id: ParentId | None


@dataclass(frozen=True, kw_only=True, slots=True)
class UnitTask:
    """Associate one logical unit task with its fixed model reference.

    Parameters
    ----------
    key : UnitKey
        Parent-task and model identities forming the logical unit identity.
    model : ModelReference
        Fixed model reference evaluated by the unit task.

    Notes
    -----
    Construction preserves candidate values without validating that
    ``key.model_id`` matches ``model.model_id``. That contextual association
    is validated as part of the owning parent evaluation. This record does
    not contain dataset, partition, evaluation-settings, worker, or attempt
    overrides.
    """

    key: UnitKey
    model: ModelReference


@dataclass(frozen=True, kw_only=True, slots=True)
class ParentEvaluation:
    """Describe the fixed scientific inputs and units of one parent task.

    Parameters
    ----------
    parent_id : ParentId
        Identity of the committed parent task.
    partition : PartitionReference
        Fixed dataset and realized-partition identities used by every unit
        and attempt in the parent task.
    evaluation : EvaluationReference
        Fixed task-wide evaluation-settings reference.
    units : tuple[UnitTask, ...]
        Ordered set of model-specific unit tasks belonging to the parent.

    Notes
    -----
    Construction preserves candidate values without validating identities,
    unit presence or uniqueness, task-kind agreement, or unit-key
    associations. Those relationships are enforced by
    ``validate_parent_evaluation``. This record does not embed partition
    memberships, model objects, worker state, or execution attempts.
    """

    parent_id: ParentId
    partition: PartitionReference
    evaluation: EvaluationReference
    units: tuple[UnitTask, ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class AttemptInputs:
    """Identify one worker execution attempt for an existing unit task.

    Parameters
    ----------
    attempt_id : AttemptId
        Identity of this execution attempt.
    unit : UnitKey
        Logical unit whose parent task supplies the fixed scientific inputs.
    worker_id : WorkerId
        Execution target assigned to the attempt.

    Notes
    -----
    This record contains attempt-routing identities only. It does not contain
    independent dataset, partition, evaluation-settings, or model overrides.
    Construction preserves candidate values without validating identities or
    confirming that the unit belongs to a supplied parent task. Those checks
    are performed by ``validate_attempt_inputs``.
    """

    attempt_id: AttemptId
    unit: UnitKey
    worker_id: WorkerId
