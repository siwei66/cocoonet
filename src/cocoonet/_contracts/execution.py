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
    validate_identifier,
    validate_label,
    validate_unit_key,
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


def validate_model_reference(value: ModelReference) -> None:
    """Validate identifying metadata for one fixed model input.

    Parameters
    ----------
    value : ModelReference
        Model identity, user label, construction family, and scientific task
        kind to validate.

    Raises
    ------
    TypeError
        If ``value`` is not a ModelReference or a field has an incorrect
        runtime type, including an undeclared model family or task kind.
    ValueError
        If the model identity is empty or the model label is a nonfinite
        floating-point value.

    Notes
    -----
    Runtime type validation enforces the declared ``ModelFamily`` and
    ``TaskKind`` vocabularies. Explicit value validation checks the opaque
    model identity and scalar model label. This function does not inspect,
    load, derive metadata from, or execute a model object.
    """
    validate_declared_types(
        value,
        expected_type=ModelReference,
    )

    validate_identifier(value.model_id, field="model_id")
    validate_label(value.model_label, field="model_label")


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


def validate_evaluation_reference(value: EvaluationReference) -> None:
    """Validate one fixed task-wide evaluation reference.

    Parameters
    ----------
    value : EvaluationReference
        Evaluation-settings identity, task kind, ordered class vocabulary,
        and influence request to validate.

    Raises
    ------
    TypeError
        If ``value`` is not an EvaluationReference or a field has an
        incorrect runtime type, including an undeclared task kind or a
        non-boolean influence request.
    ValueError
        If the configuration identity is empty, a class label is invalid,
        regression declares class labels, classification has no class label,
        or classification labels are not unique under Python scalar equality.

    Notes
    -----
    Regression requires an empty class-label tuple. Classification requires
    at least one unique class label; this structural layer permits a
    single-class vocabulary. Validation preserves the authoritative class
    order and does not sort, coerce, add, or remove labels. It does not inspect
    the complete evaluation-settings payload identified by
    ``configuration_id``.
    """
    validate_declared_types(
        value,
        expected_type=EvaluationReference,
    )

    validate_identifier(
        value.configuration_id,
        field="configuration_id",
    )

    seen_labels: set[ScalarLabel] = set()
    for index, label in enumerate(value.class_labels):
        validate_label(label, field=f"class_labels[{index}]")
        if label in seen_labels:
            raise ValueError(
                "class_labels must be unique under Python scalar equality; "
                f"duplicate at class_labels[{index}]: {label!r}."
            )
        seen_labels.add(label)

    if value.task_kind is TaskKind.REGRESSION:
        if value.class_labels:
            raise ValueError("Regression evaluation requires class_labels to be empty.")
    elif not value.class_labels:
        raise ValueError("Classification evaluation requires at least one class label.")


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


def validate_computation_snapshot(value: ComputationSnapshot) -> None:  # noqa: C901
    """Validate a root-side computation-configuration snapshot.

    Parameters
    ----------
    value : ComputationSnapshot
        Prospective configuration and consolidation relationship to validate.

    Raises
    ------
    TypeError
        If the snapshot or a nested field has an incorrect runtime type.
    ValueError
        If the root directory or an identity is empty, a nested reference is
        invalid, model or worker identities are duplicated, model labels
        collide, or the parent identity conflicts with consolidation state.

    Notes
    -----
    Initialized snapshots must not have a parent identity. Consolidated and
    unconsolidated snapshots retain an existing parent identity. Incomplete
    prospective configuration remains representable; this validator does not
    apply start prerequisites, compare against an older snapshot, access the
    filesystem, or perform a lifecycle transition.
    """
    # Validate the complete declared snapshot structure before semantic checks.
    validate_declared_types(
        value,
        expected_type=ComputationSnapshot,
    )

    # Validate the recorded root-directory value without filesystem access.
    validate_identifier(value.root_dir, field="root_dir")

    # Validate both identities when a dataset/partition reference is present.
    if value.partition is not None:
        validate_identifier(
            value.partition.partition_id,
            field="partition.partition_id",
        )
        validate_identifier(
            value.partition.dataset_id,
            field="partition.dataset_id",
        )

    # Validate the optional evaluation reference through its complete contract.
    if value.evaluation is not None:
        validate_evaluation_reference(value.evaluation)

    # Validate model references and enforce configuration-scoped uniqueness.
    model_ids: set[ModelId] = set()
    model_labels: set[ScalarLabel] = set()
    for index, model in enumerate(value.models):
        validate_model_reference(model)

        if model.model_id in model_ids:
            raise ValueError(
                "models must have unique model_id values; " f"duplicate at models[{index}]: {model.model_id!r}."
            )
        model_ids.add(model.model_id)

        if model.model_label in model_labels:
            raise ValueError(
                "models must have unique model_label values under Python scalar equality; "
                f"duplicate at models[{index}]: {model.model_label!r}."
            )
        model_labels.add(model.model_label)

    # Validate worker identities and enforce configuration-scoped uniqueness.
    worker_ids: set[WorkerId] = set()
    for index, worker_id in enumerate(value.worker_ids):
        validate_identifier(worker_id, field=f"worker_ids[{index}]")
        if worker_id in worker_ids:
            raise ValueError("worker_ids must be unique; " f"duplicate at worker_ids[{index}]: {worker_id!r}.")
        worker_ids.add(worker_id)

    # Validate whether parent identity presence agrees with consolidation state.
    if value.consolidation is ConsolidationState.INITIALIZED:
        if value.parent_id is not None:
            raise ValueError("Initialized snapshots must not have parent_id.")
    else:
        if value.parent_id is None:
            raise ValueError("Consolidated and unconsolidated snapshots require parent_id.")
        validate_identifier(value.parent_id, field="parent_id")


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


def validate_parent_evaluation(value: ParentEvaluation) -> None:
    """Validate the fixed scientific context and units of one parent task.

    Parameters
    ----------
    value : ParentEvaluation
        Parent identity, partition reference, evaluation reference, and known
        unit set to validate.

    Raises
    ------
    TypeError
        If the parent record or a nested field has an incorrect runtime type.
    ValueError
        If an identity or nested reference is invalid, no unit is present, a
        unit references another parent or model, a model uses another task
        kind, or model identities or labels are duplicated within the parent.

    Notes
    -----
    Every unit must resolve to this parent and its own fixed model reference.
    Every model uses the parent's task-wide evaluation kind, while model
    families may differ. Model identities and labels are unique within this
    parent only. Validation does not load partition memberships, model
    objects, or evaluation-settings payloads.
    """
    # Validate the complete declared parent structure before semantic checks.
    validate_declared_types(
        value,
        expected_type=ParentEvaluation,
    )

    # Validate the parent and dataset/partition identities.
    validate_identifier(value.parent_id, field="parent_id")
    validate_identifier(
        value.partition.partition_id,
        field="partition.partition_id",
    )
    validate_identifier(
        value.partition.dataset_id,
        field="partition.dataset_id",
    )

    # Validate the fixed task-wide evaluation reference.
    validate_evaluation_reference(value.evaluation)

    # Require at least one model-specific unit in a committed parent.
    if not value.units:
        raise ValueError("units must contain at least one unit task.")

    # Validate every unit association and parent-scoped model uniqueness.
    model_ids: set[ModelId] = set()
    model_labels: set[ScalarLabel] = set()
    for index, unit in enumerate(value.units):
        validate_unit_key(unit.key)
        validate_model_reference(unit.model)

        if unit.key.parent_id != value.parent_id:
            raise ValueError(
                f"units[{index}].key.parent_id must equal parent_id "
                f"{value.parent_id!r}; got {unit.key.parent_id!r}."
            )

        if unit.key.model_id != unit.model.model_id:
            raise ValueError(
                f"units[{index}].key.model_id must equal its model.model_id "
                f"{unit.model.model_id!r}; got {unit.key.model_id!r}."
            )

        if unit.model.task_kind is not value.evaluation.task_kind:
            raise ValueError(
                f"units[{index}].model.task_kind must equal evaluation task kind "
                f"{value.evaluation.task_kind.value!r}; "
                f"got {unit.model.task_kind.value!r}."
            )

        if unit.model.model_id in model_ids:
            raise ValueError(
                "units must have unique model_id values; " f"duplicate at units[{index}]: {unit.model.model_id!r}."
            )
        model_ids.add(unit.model.model_id)

        if unit.model.model_label in model_labels:
            raise ValueError(
                "units must have unique model_label values under Python scalar equality; "
                f"duplicate at units[{index}]: {unit.model.model_label!r}."
            )
        model_labels.add(unit.model.model_label)


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


def validate_attempt_inputs(
    value: AttemptInputs,
    *,
    parent: ParentEvaluation,
) -> None:
    """Validate one attempt's routing identities against its fixed parent.

    Parameters
    ----------
    value : AttemptInputs
        Attempt identity, logical unit key, and assigned worker to validate.
    parent : ParentEvaluation
        Fixed parent task through which the unit and scientific inputs resolve.

    Raises
    ------
    TypeError
        If the attempt, parent, or a nested field has an incorrect runtime
        type.
    ValueError
        If an identity is empty, the supplied parent is invalid, the unit
        names another parent, or the unit is not present in the parent.

    Notes
    -----
    Different attempts may reference the same unit and may use different
    workers. This validator does not require attempt or worker uniqueness
    across a collection. It does not allocate work, inspect worker state, or
    introduce independent dataset, partition, evaluation, or model inputs.
    """
    # Validate the declared attempt structure and its routing identities.
    validate_declared_types(
        value,
        expected_type=AttemptInputs,
    )
    validate_identifier(value.attempt_id, field="attempt_id")
    validate_unit_key(value.unit)
    validate_identifier(value.worker_id, field="worker_id")

    # Validate the fixed parent context before resolving unit membership.
    validate_parent_evaluation(parent)

    # Reject a unit key that explicitly identifies another parent task.
    if value.unit.parent_id != parent.parent_id:
        raise ValueError(
            f"unit.parent_id must equal parent.parent_id " f"{parent.parent_id!r}; got {value.unit.parent_id!r}."
        )

    # Resolve membership by the complete parent/model unit identity.
    known_units = {unit.key for unit in parent.units}
    if value.unit not in known_units:
        raise ValueError(
            f"unit must identify a known parent unit; got model_id "
            f"{value.unit.model_id!r} for parent_id {value.unit.parent_id!r}."
        )
