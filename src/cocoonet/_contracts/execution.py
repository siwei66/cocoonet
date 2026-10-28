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
    WorkerId,
    validate_identifier,
    validate_label,
)
from cocoonet._type_validation import validate_declared_types


class TaskKind(StrEnum):
    """Identify the scientific prediction task performed by an evaluation.

    Attributes
    ----------
    REGRESSION : str
        Evaluation predicts a numeric target.
    CLASSIFICATION : str
        Evaluation predicts membership in a declared class vocabulary.

    Notes
    -----
    Node usage scope: Shared - Used by both root and worker nodes.
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
    Node usage scope: Shared - Used by both root and worker nodes.
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
    Node usage scope: Root only - Used exclusively by the root node.
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
    This record contains identifying metadata only. The configuration owner
    retains the fixed model input, including supplied model/framework
    parallelism settings, and later prepares its payload. Worker attempt
    capacity is separate from this scientific input.
    Construction does not inspect, store, or execute a model object.
    Node usage scope: Shared - Used by both root and worker nodes.
    """

    model_id: ModelId
    model_label: ScalarLabel
    family: ModelFamily
    task_kind: TaskKind


def validate_model_reference(value: ModelReference) -> None:
    """Validate one model's identity, text label, family, and task kind.

    Parameters
    ----------
    value : ModelReference
        Candidate reference to a fixed supplied model input.

    Raises
    ------
    TypeError
        If the record, identity, label, family, or task kind has an
        incorrect runtime type.
    ValueError
        If the model identity is empty.

    Notes
    -----
    String values retain their representation. Family and task kind must
    be members of their declared enums. This function does not inspect,
    load, execute, or derive metadata from the referenced model.
    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Check the outer record without recursively checking its fields.
    if not isinstance(value, ModelReference):
        validate_declared_types(value, expected_type=ModelReference)

    # Delegate identity and label acceptance to their owning validators.
    validate_identifier(value.model_id, field="model_id")
    validate_label(value.model_label, field="model_label")

    # Enum membership is structural; model/framework behavior is not inspected.
    validate_declared_types(value.family, expected_type=ModelFamily, field="family")
    validate_declared_types(value.task_kind, expected_type=TaskKind, field="task_kind")


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
    Node usage scope: Shared - Used by both root and worker nodes.
    """

    configuration_id: EvaluationConfigId
    task_kind: TaskKind
    class_labels: tuple[ScalarLabel, ...]
    influence_requested: bool


def validate_evaluation_reference(value: EvaluationReference) -> None:
    """Validate fixed evaluation metadata and its ordered class vocabulary.

    Parameters
    ----------
    value : EvaluationReference
        Candidate settings identity, task kind, classes, and influence flag.

    Raises
    ------
    TypeError
        If the record or a field has an incorrect runtime type.
    ValueError
        If the settings identity is empty, class labels are duplicated,
        regression has classes, or classification has no classes.

    Notes
    -----
    Classification permits a single class. The local set checks uniqueness
    without sorting or converting the authoritative class tuple.
    Validation does not inspect the settings payload or mutate input.
    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Check the record and scalar tags without traversing class labels twice.
    if not isinstance(value, EvaluationReference):
        validate_declared_types(value, expected_type=EvaluationReference)
    validate_identifier(value.configuration_id, field="configuration_id")
    validate_declared_types(value.task_kind, expected_type=TaskKind, field="task_kind")
    validate_declared_types(value.influence_requested, expected_type=bool, field="influence_requested")
    validate_declared_types(value.class_labels, expected_type=tuple, field="class_labels")

    # Validate each class string once; bound duplicate-label diagnostics.
    seen_labels: set[ScalarLabel] = set()
    for index, label in enumerate(value.class_labels):
        validate_label(label, field=f"class_labels[{index}]")
        if label in seen_labels:
            raise ValueError(f"class_labels[{index}] must be unique; duplicate label prefix: {label!r}.")
        seen_labels.add(label)

    # Task kind determines whether an authoritative class vocabulary is required.
    if value.task_kind is TaskKind.REGRESSION:
        if value.class_labels:
            raise ValueError(f"Regression requires zero class labels; got {len(value.class_labels)}.")
    elif not value.class_labels:
        raise ValueError("Classification requires at least one class label; got zero.")


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
    Node usage scope: Root only - Used exclusively by the root node.
    """

    root_dir: str
    consolidation: ConsolidationState
    partition: PartitionReference | None
    evaluation: EvaluationReference | None
    models: tuple[ModelReference, ...]
    worker_ids: tuple[WorkerId, ...]
    parent_id: ParentId | None


def validate_computation_snapshot(value: ComputationSnapshot) -> None:  # noqa: C901
    """Validate root configuration metadata without applying start prerequisites.

    Parameters
    ----------
    value : ComputationSnapshot
        Candidate prospective configuration and consolidation state.

    Raises
    ------
    TypeError
        If a record, container, or field has an incorrect runtime type.
    ValueError
        If a path or identity is empty, a nested reference is invalid,
        model IDs/labels or worker IDs are duplicated, or parent presence
        conflicts with consolidation state.

    Notes
    -----
    Incomplete configuration remains representable. Initialized state has
    no parent; consolidated and unconsolidated states retain a parent ID.
    Local sets check configuration-scoped uniqueness without mutating input.
    This validator does not compare snapshots, enforce model/evaluation
    readiness, inspect files, calculate capacity, or perform transitions.
    Node usage scope: Root only - Used exclusively by the root node.
    """
    # Validate the outer record and its own fields without a recursive traversal.
    if not isinstance(value, ComputationSnapshot):
        validate_declared_types(value, expected_type=ComputationSnapshot)
    validate_identifier(value.root_dir, field="root_dir")
    validate_declared_types(value.consolidation, expected_type=ConsolidationState, field="consolidation")
    validate_declared_types(value.models, expected_type=tuple, field="models")
    validate_declared_types(value.worker_ids, expected_type=tuple, field="worker_ids")

    # A configured partition reference owns two opaque identities, not file access.
    if value.partition is not None:
        if not isinstance(value.partition, PartitionReference):
            validate_declared_types(value.partition, expected_type=PartitionReference, field="partition")
        validate_identifier(value.partition.partition_id, field="partition.partition_id")
        validate_identifier(value.partition.dataset_id, field="partition.dataset_id")

    # Delegate optional evaluation validity and retain its configuration field path.
    if value.evaluation is not None:
        try:
            validate_evaluation_reference(value.evaluation)
        except (TypeError, ValueError) as error:
            raise type(error)(f"evaluation is invalid: {error}") from error

    # Model references are checked once; local sets enforce scoped uniqueness.
    model_ids: set[ModelId] = set()
    model_labels: set[ScalarLabel] = set()
    for index, model in enumerate(value.models):
        try:
            validate_model_reference(model)
        except (TypeError, ValueError) as error:
            raise type(error)(f"models[{index}] is invalid: {error}") from error
        if model.model_id in model_ids:
            raise ValueError(
                f"models[{index}].model_id must be unique; duplicate ID prefix: {model.model_id!r}."
            )
        if model.model_label in model_labels:
            raise ValueError(
                f"models[{index}].model_label must be unique; duplicate label prefix: {model.model_label!r}."
            )
        model_ids.add(model.model_id)
        model_labels.add(model.model_label)

    # Configured worker identity is independent of availability and capacity.
    worker_ids: set[WorkerId] = set()
    for index, worker_id in enumerate(value.worker_ids):
        validate_identifier(worker_id, field=f"worker_ids[{index}]")
        if worker_id in worker_ids:
            raise ValueError(f"worker_ids[{index}] must be unique; duplicate ID prefix: {worker_id!r}.")
        worker_ids.add(worker_id)

    # Validate a supplied identity before checking its lifecycle relationship.
    if value.parent_id is not None:
        validate_identifier(value.parent_id, field="parent_id")
    if value.consolidation is ConsolidationState.INITIALIZED:
        if value.parent_id is not None:
            raise ValueError(
                f"Initialized snapshots require parent_id=None; got ID prefix: {value.parent_id!r}."
            )
    elif value.parent_id is None:
        raise ValueError(f"{value.consolidation.value} snapshots require a parent_id; got None.")


@dataclass(frozen=True, kw_only=True, slots=True)
class ParentEvaluation:
    """Describe one parent's fixed scientific context and ordered models.

    Parameters
    ----------
    parent_id : ParentId
        Identity of the committed parent task.
    partition : PartitionReference
        Fixed dataset and realized-partition identity pair.
    evaluation : EvaluationReference
        Fixed task-wide evaluation-settings reference.
    models : tuple[ModelReference, ...]
        Nonempty ordered model references defining the parent's logical units.

    Notes
    -----
    Each model defines one unit through the parent/model identity pair.
    Independently consumed records carry parent_id and model_id directly.
    Model inputs remain fixed, including supplied framework parallelism.
    Worker attempt limits and reservations belong to worker control.
    Construction stores candidate metadata without validation, model loading,
    partition access, or ownership of runtime resources.
    Node usage scope: Shared - Used by both root and worker nodes.
    """

    parent_id: ParentId
    partition: PartitionReference
    evaluation: EvaluationReference
    models: tuple[ModelReference, ...]


def validate_parent_evaluation(value: ParentEvaluation) -> None:
    """Validate a parent's fixed context and directly stored model references.

    Parameters
    ----------
    value : ParentEvaluation
        Candidate parent identity, partition, evaluation, and models.

    Raises
    ------
    TypeError
        If a record, container, or field has an incorrect runtime type.
    ValueError
        If an identity or reference is invalid, models are empty,
        model IDs/labels are duplicated, or a model has another task kind.

    Notes
    -----
    Evaluation and model validators own their records' validity.
    Local sets enforce parent-scoped uniqueness without mutating input.
    Model families may differ, but every model shares the evaluation task.
    Validation does not load scientific payloads, allocate attempts,
    inspect worker capacity, or change model/framework settings.
    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Check outer records without recursively traversing their fields.
    if not isinstance(value, ParentEvaluation):
        validate_declared_types(value, expected_type=ParentEvaluation)
    if not isinstance(value.partition, PartitionReference):
        validate_declared_types(value.partition, expected_type=PartitionReference, field="partition")

    # Validate the fixed identity binding without loading a partition artifact.
    validate_identifier(value.parent_id, field="parent_id")
    validate_identifier(value.partition.partition_id, field="partition.partition_id")
    validate_identifier(value.partition.dataset_id, field="partition.dataset_id")

    # Evaluation validity is owned by its dedicated validator.
    try:
        validate_evaluation_reference(value.evaluation)
    except (TypeError, ValueError) as error:
        raise type(error)(f"evaluation is invalid: {error}") from error

    # A committed parent requires models; prospective configuration may be empty.
    validate_declared_types(value.models, expected_type=tuple, field="models")
    if not value.models:
        raise ValueError("models must contain at least one model reference; got an empty tuple.")

    # Validate each model once and enforce task agreement and scoped uniqueness.
    model_ids: set[ModelId] = set()
    model_labels: set[ScalarLabel] = set()
    for index, model in enumerate(value.models):
        try:
            validate_model_reference(model)
        except (TypeError, ValueError) as error:
            raise type(error)(f"models[{index}] is invalid: {error}") from error
        if model.task_kind is not value.evaluation.task_kind:
            raise ValueError(
                f"models[{index}].task_kind must equal {value.evaluation.task_kind.value!r}; "
                f"got {model.task_kind.value!r}."
            )
        if model.model_id in model_ids:
            raise ValueError(
                f"models[{index}].model_id must be unique; duplicate ID prefix: {model.model_id!r}."
            )
        if model.model_label in model_labels:
            raise ValueError(
                f"models[{index}].model_label must be unique; duplicate label prefix: {model.model_label!r}."
            )
        model_ids.add(model.model_id)
        model_labels.add(model.model_label)


@dataclass(frozen=True, kw_only=True, slots=True)
class AttemptInputs:
    """Identify one worker execution attempt through direct routing fields.

    Parameters
    ----------
    attempt_id : AttemptId
        Identity of this execution attempt.
    parent_id : ParentId
        Fixed parent supplying the dataset, partition, and evaluation settings.
    model_id : ModelId
        Fixed model reference within that parent.
    worker_id : WorkerId
        Execution target assigned to the attempt.

    Notes
    -----
    The parent/model pair identifies the logical unit. Different attempts
    may share that pair and worker while retaining distinct attempt identities.
    This record stores no independent scientific overrides or duplicate
    composite key. Construction preserves candidate identities without
    validating membership, allocating work, or reserving worker capacity.
    Node usage scope: Shared - Used by both root and worker nodes.
    """

    attempt_id: AttemptId
    parent_id: ParentId
    model_id: ModelId
    worker_id: WorkerId


def validate_attempt_inputs(
    value: AttemptInputs,
    *,
    parent: ParentEvaluation,
) -> None:
    """Validate direct attempt identities and resolve the fixed parent/model pair.

    Parameters
    ----------
    value : AttemptInputs
        Candidate attempt, parent, model, and assigned-worker identities.
    parent : ParentEvaluation
        Fixed parent context, fully validated by this boundary.

    Raises
    ------
    TypeError
        If a record or field has an incorrect runtime type.
    ValueError
        If an identity or parent is invalid, parent_id names another parent,
        or model_id is absent from that parent's model references.

    Notes
    -----
    Parent validation occurs once in this independent boundary call.
    Different attempts may share a parent/model pair and worker. This
    function does not enforce collection-wide attempt uniqueness, inspect
    worker readiness, reserve capacity, or mutate scientific inputs.
    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Check the outer record without recursively traversing its fields.
    if not isinstance(value, AttemptInputs):
        validate_declared_types(value, expected_type=AttemptInputs)

    # Validate each routing identity directly without constructing a composite key.
    validate_identifier(value.attempt_id, field="attempt_id")
    validate_identifier(value.parent_id, field="parent_id")
    validate_identifier(value.model_id, field="model_id")
    validate_identifier(value.worker_id, field="worker_id")

    # Resolve membership only after the parent has passed its complete contract.
    try:
        validate_parent_evaluation(parent)
    except (TypeError, ValueError) as error:
        raise type(error)(f"parent is invalid: {error}") from error

    # The parent identity establishes the model-ID scope of this allocation.
    if value.parent_id != parent.parent_id:
        raise ValueError(
            "parent_id must equal parent.parent_id; "
            f"expected ID prefix: {parent.parent_id!r}; "
            f"got ID prefix: {value.parent_id!r}."
        )

    # Compare model IDs directly; one lookup needs no auxiliary key collection.
    if not any(model.model_id == value.model_id for model in parent.models):
        raise ValueError(
            "model_id must identify a parent model; "
            f"unknown ID prefix: {value.model_id!r}."
        )
