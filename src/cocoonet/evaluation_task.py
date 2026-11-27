"""Mutable root-side evaluation configuration."""

import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pandas as pd

from cocoonet._contracts.data import PartitionReference
from cocoonet._contracts.execution import (
    ConsolidationState,
    EvaluationReference,
    ModelFamily,
    ModelingSnapshot,
    ModelReference,
    ParentEvaluation,
    TaskKind,
    validate_modeling_snapshot,
    validate_parent_evaluation,
)
from cocoonet._contracts.identity import ModelId, ParentId, ScalarLabel
from cocoonet._contracts.state import ExecutionStatus
from cocoonet._storage.paths import PathInput, resolve_physical_path
from cocoonet._storage.work_directory import (
    _clone_work_directory,
    prepare_work_directory,
)
from cocoonet._type_validation import validate_args, validate_declared_types
from cocoonet.configuration import (
    ModelRegistration,
    WorkerConfiguration,
    WorkerKind,
    validate_model_registration,
    validate_worker_configuration,
)


@dataclass(frozen=True, kw_only=True, slots=True)
class _RegisteredModel:
    """Associate fixed model metadata with its opaque prospective input.

    Parameters
    ----------
    reference : ModelReference
        Authoritative model identity, label, family, and task metadata.
    model_input : object
        Opaque in-memory model input or normalized file reference.

    Notes
    -----
    The reference remains the sole stored source of model identity and label.
    Construction does not validate, copy, inspect, or execute the model input.
    Node usage scope: Root - Retains root-side prospective model registration.
    """

    reference: ModelReference
    model_input: object

    @property
    def model_id(self) -> ModelId:
        """Return the authoritative internal model identity."""
        return self.reference.model_id

    @property
    def model_label(self) -> ScalarLabel:
        """Return the authoritative user-facing model label."""
        return self.reference.model_label


@validate_args
def _normalize_model_family(value: str, /) -> ModelFamily:
    """Convert one public model-family string to its internal enum."""
    try:
        family = ModelFamily(value)
    except ValueError as error:
        raise ValueError("family must be one of 'sklearn', 'sklearn-like', " "'pytorch', or 'cocoonet'.") from error

    if family is ModelFamily.COCOONET:
        raise ValueError(
            "family='cocoonet' is reserved for Cocoonet-native models, "
            "which are currently under development and unsupported."
        )

    return family


@validate_args
def _normalize_task_kind(value: str, /) -> TaskKind:
    """Convert one public task-kind string to its internal enum."""
    try:
        return TaskKind(value)
    except ValueError as error:
        raise ValueError("task_kind must be either 'regression' or 'classification'.") from error


def _stage_model_registration(
    registration: ModelRegistration,
    /,
    *,
    model_id: ModelId,
    existing_models: list[_RegisteredModel],
    evaluation_config: EvaluationReference | None,
) -> _RegisteredModel:
    """Validate and stage one model without mutating task state.

    Parameters
    ----------
    registration : ModelRegistration
        Prospective opaque model input and declared metadata.
    model_id : ModelId
        Identity reserved by the caller's staged model counter.
    existing_models : list[_RegisteredModel]
        Current or staged registry used for collision and task checks.
    evaluation_config : EvaluationReference | None
        Current task-wide evaluation reference, when configured.

    Returns
    -------
    _RegisteredModel
        Validated immutable registration ready for a bounded commit.

    Raises
    ------
    TypeError
        If registration metadata has an incorrect runtime type.
    ValueError
        If the input path is invalid, the label is already registered, or
        the task kind conflicts with staged scientific configuration.

    Notes
    -----
    This helper performs validation and staging only. It does not mutate a
    registry, consume an ID, change lifecycle state, or read a model file.
    Node usage scope: Root - Stages root-side model registry additions.
    """
    validate_model_registration(registration)

    if any(existing.model_label == registration.model_label for existing in existing_models):
        raise ValueError(f"model_label must be unique; duplicate label: " f"{registration.model_label!r}.")

    if existing_models:
        expected_task = existing_models[0].reference.task_kind
        if registration.task_kind is not expected_task:
            raise ValueError(
                "task_kind must match existing models; "
                f"expected {expected_task.value!r}, "
                f"got {registration.task_kind.value!r}."
            )

    if evaluation_config is not None and registration.task_kind is not evaluation_config.task_kind:
        raise ValueError(
            "task_kind must match evaluation_config.task_kind; "
            f"expected {evaluation_config.task_kind.value!r}, "
            f"got {registration.task_kind.value!r}."
        )

    model_input = registration.model_input
    if isinstance(model_input, (str, Path)):
        model_path = resolve_physical_path(model_input)
        if not model_path.exists() or not model_path.is_file():
            raise ValueError("model_input path must identify an existing regular file; " f"got {str(model_path)!r}.")
        if not model_path.suffix:
            raise ValueError("model_input path must have a nonempty extension; " f"got {str(model_path)!r}.")
        model_input = model_path

    return _RegisteredModel(
        reference=ModelReference(
            model_id=model_id,
            model_label=registration.model_label,
            family=registration.family,
            task_kind=registration.task_kind,
        ),
        model_input=model_input,
    )


def _stage_worker_configuration(
    configuration: WorkerConfiguration,
    /,
    *,
    existing_workers: list[WorkerConfiguration],
) -> WorkerConfiguration:
    """Validate one worker configuration without mutating the registry."""
    validate_worker_configuration(configuration)

    if any(existing.worker_id == configuration.worker_id for existing in existing_workers):
        raise ValueError("worker_id must be unique; " f"duplicate worker ID: {configuration.worker_id!r}.")

    return configuration


_MODEL_TABLE_COLUMNS = (
    "model_id",
    "model_label",
    "family",
    "task_kind",
    "model_input",
)


def _model_input_display(model_input: object, /) -> str:
    """Return the public display value for one opaque model input."""
    if isinstance(model_input, Path):
        return str(model_input)
    return type(model_input).__name__


def _filter_registered_models(
    models: list[_RegisteredModel],
    /,
    *,
    label: str | None,
    family: str | None,
    model_input: str | None,
    use_regex: bool,
) -> tuple[_RegisteredModel, ...]:
    """Return registered models matching all supplied search criteria."""
    criteria = {
        "label": label,
        "family": family,
        "model_input": model_input,
    }

    if use_regex:
        patterns: dict[str, re.Pattern[str]] = {}
        for name, criterion in criteria.items():
            if criterion is None:
                continue
            try:
                patterns[name] = re.compile(criterion)
            except re.error as error:
                raise ValueError(f"{name} must be a valid regular expression.") from error

        def matches(name: str, value: str) -> bool:
            """Return whether one value fully matches its active pattern."""
            pattern = patterns.get(name)
            return pattern is None or pattern.fullmatch(value) is not None

    else:

        def matches(name: str, value: str) -> bool:
            """Return whether one value equals its active criterion."""
            criterion = criteria[name]
            return criterion is None or value == criterion

    return tuple(
        registered
        for registered in models
        if matches("label", str(registered.model_label))
        and matches("family", registered.reference.family.value)
        and matches("model_input", _model_input_display(registered.model_input))
    )


def _models_to_dataframe(
    models: tuple[_RegisteredModel, ...],
    /,
) -> "pd.DataFrame":
    """Build the public string-typed model registry table."""
    import pandas as pd

    rows = (
        (
            str(registered.model_id),
            str(registered.model_label),
            registered.reference.family.value,
            registered.reference.task_kind.value,
            _model_input_display(registered.model_input),
        )
        for registered in models
    )
    return pd.DataFrame(rows, columns=_MODEL_TABLE_COLUMNS, dtype="string")


_WORKER_TABLE_COLUMNS = (
    "worker_id",
    "worker_label",
    "kind",
    "device",
    "cpu_capacity",
    "n_attempt",
    "n_cpu_per_attempt",
    "memory_reserve_pct",
)


def _filter_workers(
    workers: list[WorkerConfiguration],
    /,
    *,
    worker_id: str | None,
    worker_label: str | None,
    cpu_capacity: int | str | None,
    kind: str | None,
    device: str | None,
    use_regex: bool,
) -> tuple[WorkerConfiguration, ...]:
    """Return registered workers matching all supplied criteria."""
    criteria = {
        "worker_id": worker_id,
        "worker_label": worker_label,
        "kind": kind,
        "device": device,
        "cpu_capacity": None if cpu_capacity is None else str(cpu_capacity),
    }

    if use_regex:
        patterns: dict[str, re.Pattern[str]] = {}
        for name, criterion in criteria.items():
            if criterion is None:
                continue
            try:
                patterns[name] = re.compile(criterion)
            except re.error as error:
                raise ValueError(f"{name} must be a valid regular expression.") from error

        def matches(name: str, value: str) -> bool:
            """Return whether one value fully matches its active pattern."""
            pattern = patterns.get(name)
            return pattern is None or pattern.fullmatch(value) is not None

    else:

        def matches(name: str, value: str) -> bool:
            """Return whether one value equals its active criterion."""
            criterion = criteria[name]
            return criterion is None or value == criterion

    return tuple(
        worker
        for worker in workers
        if matches("worker_id", str(worker.worker_id))
        and matches("worker_label", str(worker.worker_label))
        and matches("kind", worker.kind.value)
        and matches("device", worker.device)
        and matches(
            "cpu_capacity",
            "" if worker.cpu_capacity is None else str(worker.cpu_capacity),
        )
    )


def _workers_to_dataframe(
    workers: tuple[WorkerConfiguration, ...],
    /,
) -> "pd.DataFrame":
    """Build the public string-typed worker registry table."""
    import pandas as pd

    rows = (
        (
            str(worker.worker_id),
            str(worker.worker_label),
            worker.kind.value,
            worker.device,
            "" if worker.cpu_capacity is None else str(worker.cpu_capacity),
            str(worker.n_attempt),
            str(worker.n_cpu_per_attempt),
            str(worker.memory_reserve_pct),
        )
        for worker in workers
    )
    return pd.DataFrame(rows, columns=_WORKER_TABLE_COLUMNS, dtype="string")


def _consolidation_after_evaluation_edit(
    value: ConsolidationState,
    /,
) -> ConsolidationState:
    """Return the consolidation state after an evaluation edit.

    Parameters
    ----------
    value : ConsolidationState
        Current prospective-configuration consolidation state.

    Returns
    -------
    ConsolidationState
        ``UNCONSOLIDATED`` after editing a consolidated configuration;
        otherwise the unchanged current state.

    Notes
    -----
    Worker-only configuration changes do not use this transition.
    Node usage scope: Root - Applies root-side evaluation-edit lifecycle state.
    """
    if value is ConsolidationState.CONSOLIDATED:
        return ConsolidationState.UNCONSOLIDATED
    return value


def _require_configuration_editable(
    value: ExecutionStatus,
    /,
) -> None:
    """Reject configuration mutation during active execution.

    Parameters
    ----------
    value : ExecutionStatus
        Current root-observed execution status.

    Raises
    ------
    RuntimeError
        If execution is currently running.

    Notes
    -----
    ``NOT_STARTED``, ``INTERRUPTED``, and ``SETTLED`` permit prospective
    configuration edits. This guard does not claim that old worker activity
    has stopped after interruption.
    Node usage scope: Root - Guards root-side configuration mutation.
    """
    if value is ExecutionStatus.RUNNING:
        raise RuntimeError("Configuration cannot be changed while execution is running.")


def prepare_parent_evaluation(
    snapshot: ModelingSnapshot,
    /,
    *,
    parent_id: ParentId,
) -> ParentEvaluation:
    """Prepare one validated parent from complete prospective configuration.

    Parameters
    ----------
    snapshot : ModelingSnapshot
        Valid initialized configuration containing a partition, evaluation,
        and at least one model.
    parent_id : ParentId
        Caller-supplied identity for the prospective parent.

    Returns
    -------
    ParentEvaluation
        Validated immutable scientific context in model-registration order.

    Raises
    ------
    TypeError
        If the snapshot, parent identity, or nested record has an incorrect
        runtime type.
    ValueError
        If the snapshot is invalid or incomplete, is not initialized, already
        has a parent, contains inconsistent tasks, or the parent ID is empty.

    Notes
    -----
    This function does not generate an identity, inspect workers, persist
    files, mutate the snapshot, or commit lifecycle state.
    Node usage scope: Root - Prepares root-side parent scientific context.
    """
    validate_modeling_snapshot(snapshot)

    if snapshot.consolidation_state is not ConsolidationState.INITIALIZED:
        raise ValueError(
            "snapshot.consolidation_state must be initialized to prepare a parent; "
            f"got {snapshot.consolidation_state.value!r}."
        )
    if snapshot.data_partition is None:
        raise ValueError("snapshot.data_partition is required to prepare a parent.")
    if snapshot.evaluation_config is None:
        raise ValueError("snapshot.evaluation_config is required to prepare a parent.")
    if not snapshot.models:
        raise ValueError("snapshot.models must contain at least one model.")

    parent = ParentEvaluation(
        parent_id=parent_id,
        partition=snapshot.data_partition,
        evaluation=snapshot.evaluation_config,
        models=tuple(snapshot.models.values()),
    )
    validate_parent_evaluation(parent)
    return parent


class EvaluationTask:
    """Own reusable prospective root-side evaluation configuration.

    Parameters
    ----------
    work_dir : PathInput
        User-selected root node working directory prepared as Cocoonet-managed
        storage.
    overwrite : bool, default=False
        Whether an existing nonempty root node working directory may be
        cleared for fresh use.

    Attributes
    ----------
    _available_workers : list[WorkerConfiguration]
        Reserved snapshot of workers detected through future root-directed
        connection, heartbeat, and handshake handling. The network phase will
        initialize this list during root-node task construction and refresh it
        before attempt allocation. Phase 3 neither initializes nor reads this
        reserved state.

    Raises
    ------
    TypeError
        If the working directory or overwrite value has an incorrect runtime
        type.
    ValueError
        If the working directory is unsafe, conflicts with a configured
        dataset source, is an existing non-directory, or is nonempty without
        permission.

    Notes
    -----
    Construction always performs fresh preparation and never opens or recovers
    an existing execution. The task initially has no dataset, models, workers,
    parent, or runtime activity.
    Node usage scope: Root - Owns reusable root-side evaluation configuration.
    """

    _work_dir: Path
    _consolidation_state: ConsolidationState
    _parent_id: ParentId | None
    _execution_status: ExecutionStatus
    _data_partition: PartitionReference | None
    _evaluation_config: EvaluationReference | None
    _dataset_sources: tuple[Path, ...]
    _models: list[_RegisteredModel]
    _available_workers: list[WorkerConfiguration]
    _workers: list[WorkerConfiguration]
    _next_model_number: int

    def __init__(
        self,
        work_dir: PathInput,
        *,
        overwrite: bool = False,
    ) -> None:
        """Prepare the root node working directory and empty configuration."""
        layout = prepare_work_directory(
            work_dir,
            dataset_sources=(),
            overwrite=overwrite,
        )

        self._work_dir = layout.work_dir
        self._consolidation_state = ConsolidationState.INITIALIZED
        self._parent_id = None
        self._execution_status = ExecutionStatus.NOT_STARTED
        self._data_partition = None
        self._evaluation_config = None
        self._dataset_sources = ()
        self._models = []
        self._workers = []
        self._next_model_number = 1

    @property
    def work_dir(self) -> Path:
        """Return the canonical Cocoonet-managed root node working directory.

        Returns
        -------
        Path
            Canonical physical path returned by storage preparation.
        """
        return self._work_dir

    @property
    def state(self) -> ConsolidationState:
        """Return the prospective configuration consolidation state.

        Returns
        -------
        ConsolidationState
            Relationship between current evaluation configuration and its
            committed parent.
        """
        return self._consolidation_state

    @property
    def parent_id(self) -> ParentId | None:
        """Return the existing committed parent identity, if any.

        Returns
        -------
        ParentId | None
            Existing parent identity, or ``None`` before first commitment.
        """
        return self._parent_id

    def snapshot(self) -> ModelingSnapshot:
        """Return a validated immutable view of current configuration.

        Returns
        -------
        ModelingSnapshot
            Fresh snapshot containing immutable ordered model and worker
            mappings.

        Raises
        ------
        TypeError
            If an owned configuration value has an incorrect runtime type.
        ValueError
            If owned configuration values violate snapshot invariants.

        Notes
        -----
        Snapshot construction reads only in-memory authoritative fields. It
        does not inspect storage, models, workers, or scientific artifacts.
        """
        snapshot = ModelingSnapshot(
            work_dir=str(self._work_dir),
            consolidation_state=self._consolidation_state,
            data_partition=self._data_partition,
            evaluation_config=self._evaluation_config,
            models=MappingProxyType({registered.model_label: registered.reference for registered in self._models}),
            worker_ids=MappingProxyType({worker.worker_id: worker.worker_label for worker in self._workers}),
            parent_id=self._parent_id,
        )
        validate_modeling_snapshot(snapshot)
        return snapshot

    def __copy__(self) -> "EvaluationTask":
        """Reject shallow copying of evaluation-task ownership.

        Raises
        ------
        TypeError
            Always, because shallow copying cannot establish independent
            persistence and lifecycle ownership.

        Notes
        -----
        Use ``clone_configuration()`` for an independent execution using cloned
        configuration, or ``clone()`` to additionally clone persisted files.
        Node usage scope: Root - Prevents ambiguous duplication of root-owned
        persistence and lifecycle state.
        """
        raise TypeError(
            "EvaluationTask does not support shallow copying; use "
            "clone_configuration(new_work_dir) for a new independent execution "
            "using cloned configuration or clone(new_work_dir) for a new "
            "independent execution using cloned configuration and persisted files"
        )

    def clone_configuration(
        self,
        new_work_dir: PathInput,
    ) -> "EvaluationTask":
        """Create an independent task with the same prospective configuration.

        Parameters
        ----------
        new_work_dir : PathInput
            Missing or empty directory to prepare as fresh Cocoonet-managed
            root storage.

        Returns
        -------
        EvaluationTask
            New task retaining the dataset, evaluation, model, worker, source,
            ordering, and model-counter configuration without execution
            progress.

        Raises
        ------
        TypeError
            If ``new_work_dir`` has an incorrect runtime type.
        ValueError
            If the destination is unsafe, conflicts with a retained dataset
            source, is an existing non-directory, or is nonempty.
        StorageError
            If destination preparation fails.

        Notes
        -----
        Parent identity, consolidation state, execution status, persisted
        progress, and transient worker availability are not cloned. Registry
        containers are independent, while immutable records and opaque model
        inputs remain shared references.

        File-backed model inputs retain their original paths, including paths
        inside the source work directory, because this operation copies no
        managed files.

        Node usage scope: Root - Reuses prospective configuration in fresh
        independently owned root storage.
        """
        layout = prepare_work_directory(
            new_work_dir,
            dataset_sources=self._dataset_sources,
        )

        cloned = object.__new__(EvaluationTask)
        cloned._work_dir = layout.work_dir
        cloned._consolidation_state = ConsolidationState.INITIALIZED
        cloned._parent_id = None
        cloned._execution_status = ExecutionStatus.NOT_STARTED
        cloned._data_partition = self._data_partition
        cloned._evaluation_config = self._evaluation_config
        cloned._dataset_sources = self._dataset_sources
        cloned._models = list(self._models)
        cloned._workers = list(self._workers)
        cloned._next_model_number = self._next_model_number
        return cloned

    def clone(
        self,
        new_work_dir: PathInput,
    ) -> "EvaluationTask":
        """Clone inactive configuration and persisted execution progress.

        Parameters
        ----------
        new_work_dir : PathInput
            Missing or empty directory that will receive the complete managed
            working-directory contents.

        Returns
        -------
        EvaluationTask
            Independent task preserving configuration, parent identity,
            consolidation state, execution status, registry ordering, and
            persisted progress.

        Raises
        ------
        RuntimeError
            If execution is currently running.
        TypeError
            If ``new_work_dir`` has an incorrect runtime type.
        ValueError
            If a storage boundary is invalid, the destination is nonempty or
            nested with the source, or the source contains links.
        StorageError
            If inspection, copying, or incomplete-copy cleanup fails.

        Notes
        -----
        Transient worker availability and live connections are not cloned.
        Mutable registry containers are independent, while immutable records
        and opaque in-memory model inputs remain shared references.

        File-backed model inputs inside the original managed directory are
        rebound to corresponding copied paths. Dataset sources and other
        external model paths remain unchanged.

        Persisted files currently contain no absolute managed path fields.
        Future persistence codecs introducing such fields must implement
        schema-aware rebasing during this operation.

        Node usage scope: Root - Duplicates inactive root persistence and its
        corresponding in-memory execution progress.
        """
        if self._execution_status is ExecutionStatus.RUNNING:
            raise RuntimeError("EvaluationTask cannot be cloned while execution is running.")

        layout = _clone_work_directory(
            self._work_dir,
            new_work_dir,
            dataset_sources=self._dataset_sources,
        )

        # Rebind explicit managed paths without inspecting opaque model objects.
        cloned_models: list[_RegisteredModel] = []
        for registered in self._models:
            model_input = registered.model_input
            if isinstance(model_input, Path) and model_input.is_relative_to(self._work_dir):
                model_input = layout.work_dir / model_input.relative_to(self._work_dir)
            cloned_models.append(
                _RegisteredModel(
                    reference=registered.reference,
                    model_input=model_input,
                )
            )

        cloned = object.__new__(EvaluationTask)
        cloned._work_dir = layout.work_dir
        cloned._consolidation_state = self._consolidation_state
        cloned._parent_id = self._parent_id
        cloned._execution_status = self._execution_status
        cloned._data_partition = self._data_partition
        cloned._evaluation_config = self._evaluation_config
        cloned._dataset_sources = self._dataset_sources
        cloned._models = cloned_models
        cloned._workers = list(self._workers)
        cloned._next_model_number = self._next_model_number
        return cloned

    def initialize(
        self,
        work_dir: PathInput,
        *,
        overwrite: bool = False,
    ) -> None:
        """Prepare a fresh working directory while retaining configuration.

        Parameters
        ----------
        work_dir : PathInput
            User-selected root node working directory to prepare as fresh
            managed storage.
        overwrite : bool, default=False
            Whether an existing nonempty target working directory may be
            cleared.

        Raises
        ------
        RuntimeError
            If execution is currently running.
        TypeError
            If the working directory or overwrite value has an incorrect
            runtime type.
        ValueError
            If the target working directory is unsafe, conflicts with a
            retained dataset source, is an existing non-directory, or is
            nonempty without permission.

        Notes
        -----
        Preparation is fresh initialization, not recovery. Models, workers,
        dataset/evaluation references, source paths, registry order, and the
        next model counter remain prospective configuration.
        """
        _require_configuration_editable(self._execution_status)
        dataset_sources = tuple(self._dataset_sources)

        layout = prepare_work_directory(
            work_dir,
            dataset_sources=dataset_sources,
            overwrite=overwrite,
        )

        self._work_dir = layout.work_dir
        self._consolidation_state = ConsolidationState.INITIALIZED
        self._parent_id = None
        self._execution_status = ExecutionStatus.NOT_STARTED

    def add_model(
        self,
        model_input: object,
        model_label: str,
        family: Literal[
            "sklearn",
            "sklearn-like",
            "pytorch",
            "cocoonet",
        ],
        task_kind: Literal["regression", "classification"],
    ) -> ModelReference:
        """Add one opaque model input to prospective configuration.

        Parameters
        ----------
        model_input : object
            In-memory model instance, or a text/Path reference to an existing
            regular model file. Class definitions are not accepted. Phase 3
            retains an in-memory candidate without invoking or inspecting it
            and resolves a file reference without opening or deserializing it.
        model_label : str
            Nonempty user-facing label unique within the active model registry.
        family : {"sklearn", "sklearn-like", "pytorch", "cocoonet"}
            Declared model-construction family. ``"cocoonet"`` is reserved and
            currently rejected because Cocoonet-native models are not yet
            supported.
        task_kind : {"regression", "classification"}
            Declared prediction task.

        Returns
        -------
        ModelReference
            Immutable metadata for the newly registered model.

        Raises
        ------
        RuntimeError
            If execution is currently running.
        TypeError
            If registration metadata has an incorrect runtime type.
        ValueError
            If the input or metadata conflicts with current configuration.

        Notes
        -----
        File-backed inputs are resolved without being opened or deserialized.
        The registry and counter change only after all staging checks succeed.
        """
        _require_configuration_editable(self._execution_status)

        registered = _stage_model_registration(
            ModelRegistration(
                model_input=model_input,
                model_label=model_label,
                family=_normalize_model_family(family),
                task_kind=_normalize_task_kind(task_kind),
            ),
            model_id=ModelId(str(self._next_model_number)),
            existing_models=self._models,
            evaluation_config=self._evaluation_config,
        )

        self._models.append(registered)
        self._next_model_number += 1
        self._consolidation_state = _consolidation_after_evaluation_edit(self._consolidation_state)
        return registered.reference

    @validate_args
    def list_models(
        self,
        label: str | None = None,
        family: str | None = None,
        model_input: str | None = None,
        *,
        use_regex: bool = True,
        suppress_return: bool = False,
        tabular_return: bool = False,
        limit: int = 1000,
    ) -> "tuple[ModelReference, ...] | pd.DataFrame | None":
        """Return registered models matching all supplied criteria.

        Parameters
        ----------
        label : str | None, default=None
            Model-label pattern or exact value. ``None`` ignores this field.
        family : str | None, default=None
            Model-family pattern or exact value. ``None`` ignores this field.
        model_input : str | None, default=None
            Model-input display pattern or exact value. ``None`` ignores this
            field.
        use_regex : bool, default=True
            Whether criteria use case-sensitive full regular-expression
            matching instead of exact string equality.
        suppress_return : bool, default=False
            Whether to return no representation of matching models.
        tabular_return : bool, default=False
            Whether to return a detached string-typed table instead of
            immutable model references.
        limit : int, default=1000
            Maximum positive number of rows in a tabular return.

        Returns
        -------
        tuple[ModelReference, ...] | pandas.DataFrame | None
            Matching models in registry order. Tabular mode returns a
            string-typed DataFrame, including an empty DataFrame when no model
            matches. Reference mode returns an immutable tuple, or ``None``
            when no model matches. Suppressed returns are always ``None``.

        Raises
        ------
        TypeError
            If a criterion, matching mode, return control, or limit has an
            invalid type.
        ValueError
            If a regular expression is invalid or the limit is not positive.

        Warns
        -----
        UserWarning
            If a tabular result is truncated by ``limit``.

        Examples
        --------
        Return immutable references for exact family matches:

        >>> task.list_models(
        ...     family="sklearn",
        ...     use_regex=False,
        ...     tabular_return=False,
        ... )

        Notes
        -----
        Search reads registered attributes directly. It does not inspect model
        files or objects, mutate configuration, or derive search data from the
        returned presentation table. The tabular limit does not truncate a
        reference tuple.
        Node usage scope: Root - Presents prospective root-side model
        registrations.
        """
        if limit < 1:
            raise ValueError("limit must be positive.")

        matches = _filter_registered_models(
            self._models,
            label=label,
            family=family,
            model_input=model_input,
            use_regex=use_regex,
        )

        # Keep authoritative matches independent from the selected presentation.
        if suppress_return:
            return None
        if not tabular_return:
            if not matches:
                return None
            return tuple(model.reference for model in matches)

        if len(matches) > limit:
            warnings.warn(
                f"Model listing truncated to {limit} of {len(matches)} matching models.",
                UserWarning,
                stacklevel=2,
            )
        return _models_to_dataframe(matches[:limit])

    @validate_args
    def add_worker(
        self,
        worker_id: str | None = None,
        kind: Literal["root-local", "online"] = "root-local",
        *,
        n_attempt: int | None = None,
        n_cpu_per_attempt: int = 1,
        memory_reserve_pct: float = 20.0,
    ) -> WorkerConfiguration:
        """Add one currently available worker to task allocation.

        Parameters
        ----------
        worker_id : str | None, default=None
            Worker-generated identity to add. ``None`` selects the first
            available root-local worker for the current root machine.
        kind : {"root-local", "online"}, default="root-local"
            Required topology of the available worker.
        n_attempt : int | None, default=None
            Positive maximum parallel attempts for this task. ``None`` retains
            the available worker's current/default attempt count.
        n_cpu_per_attempt : int, default=1
            Positive logical CPU cores requested for each attempt.
        memory_reserve_pct : float, default=20.0
            Worker-memory percentage reserved when admitting another attempt.

        Returns
        -------
        WorkerConfiguration
            Immutable task configuration committed to the worker registry.

        Raises
        ------
        RuntimeError
            If execution is currently running.
        TypeError
            If an argument has an incorrect runtime type or unsupported
            literal value.
        ValueError
            If no available worker matches ``worker_id`` and ``kind``, the
            worker ID is already registered, or an execution property is
            outside its valid range.

        Notes
        -----
        This method does not discover workers, test heartbeats, establish
        handshakes, or initialize the future availability snapshot. It copies
        worker-owned identity and machine properties from the latest received
        snapshot and combines them with task-selected execution properties.
        Adding a worker changes allocation membership only and does not alter
        scientific consolidation or retained parent identity.
        Node usage scope: Root - Admits one available worker to prospective
        task allocation.
        """
        _require_configuration_editable(self._execution_status)

        available_workers: list[WorkerConfiguration] = getattr(
            self,
            "_available_workers",
            [],
        )
        matches = _filter_workers(
            available_workers,
            worker_id=worker_id,
            worker_label=None,
            cpu_capacity=None,
            kind=kind,
            device=None,
            use_regex=False,
        )
        if not matches:
            raise ValueError("No available worker matches the requested worker_id and kind.")

        available = matches[0]

        # Combine worker-owned properties with task-selected execution settings.
        configured = _stage_worker_configuration(
            WorkerConfiguration(
                worker_id=available.worker_id,
                worker_label=available.worker_label,
                kind=WorkerKind(kind),
                device=available.device,
                cpu_capacity=available.cpu_capacity,
                n_attempt=(available.n_attempt if n_attempt is None else n_attempt),
                n_cpu_per_attempt=n_cpu_per_attempt,
                memory_reserve_pct=memory_reserve_pct,
            ),
            existing_workers=self._workers,
        )

        self._workers.append(configured)
        return configured

    @validate_args
    def add_workers(  # noqa: C901
        self,
        worker_id: str | Sequence[str] | None = None,
        worker_label: str | None = None,
        cpu_capacity: int | str | None = None,
        kind: str | None = None,
        device: str | None = None,
        *,
        n_attempt: int | dict[str, int] | None = None,
        n_cpu_per_attempt: int | dict[str, int] = 1,
        memory_reserve_pct: float | dict[str, float] = 20.0,
        use_regex: bool = True,
        suppress_return: bool = False,
        tabular_return: bool = False,
        limit: int = 1000,
    ) -> "tuple[WorkerConfiguration, ...] | pd.DataFrame | None":
        """Atomically add matching available workers to task allocation.

        Parameters
        ----------
        worker_id : str | Sequence[str] | None, default=None
            Exact worker identity or identities to select. Worker IDs never
            use regular-expression matching. ``None`` ignores this field.
        worker_label : str | None, default=None
            Worker-label pattern or exact value. ``None`` ignores this field.
        cpu_capacity : int | str | None, default=None
            Logical-CPU-capacity pattern or exact display value.
        kind : str | None, default=None
            Worker-kind pattern or exact value.
        device : str | None, default=None
            Worker-device pattern or exact value.
        n_attempt : int | dict[str, int] | None, default=None
            Positive maximum parallel attempts. A dictionary configures values
            by worker ID. ``None`` retains each available worker's current or
            default value.
        n_cpu_per_attempt : int | dict[str, int], default=1
            Positive logical CPU cores requested per attempt, either shared by
            all selected workers or configured by worker ID.
        memory_reserve_pct : float | dict[str, float], default=20.0
            Reserved worker-memory percentage, either shared by all selected
            workers or configured by worker ID.
        use_regex : bool, default=True
            Whether non-ID search criteria use case-sensitive full
            regular-expression matching instead of exact equality.
        suppress_return : bool, default=False
            Whether to return no representation of added workers.
        tabular_return : bool, default=False
            Whether to return a detached string-typed table instead of an
            immutable tuple of worker configurations.
        limit : int, default=1000
            Maximum positive number of rows in a tabular return.

        Returns
        -------
        tuple[WorkerConfiguration, ...] | pandas.DataFrame | None
            Added workers in availability-snapshot order using the selected
            representation. Suppressed returns are ``None``.

        Raises
        ------
        RuntimeError
            If execution is currently running.
        TypeError
            If a search criterion, execution setting, or return control has an
            incorrect runtime type.
        ValueError
            If no available worker matches, a regular expression is invalid,
            a configuration mapping does not exactly cover the selected
            worker IDs, an execution setting is invalid, or ``limit`` is not
            positive.

        Warns
        -----
        UserWarning
            If the tabular return is truncated by ``limit``.

        Notes
        -----
        Worker IDs always use exact matching. Other worker properties follow
        ``use_regex``. Every configuration is staged before one bounded commit,
        so failure leaves task membership unchanged. This method neither
        discovers workers nor modifies the availability snapshot, scientific
        consolidation, or retained parent identity.
        Node usage scope: Root - Atomically admits matching available workers
        to prospective task allocation.
        """
        if limit < 1:
            raise ValueError("limit must be positive.")

        _require_configuration_editable(self._execution_status)

        selected_ids: set[str] | None
        if worker_id is None:
            selected_ids = None
        elif isinstance(worker_id, str):
            selected_ids = {worker_id}
        else:
            selected_ids = set(worker_id)

        available_workers: list[WorkerConfiguration] = getattr(
            self,
            "_available_workers",
            [],
        )
        matches = _filter_workers(
            available_workers,
            worker_id=None,
            worker_label=worker_label,
            cpu_capacity=cpu_capacity,
            kind=kind,
            device=device,
            use_regex=use_regex,
        )
        if selected_ids is not None:
            matches = tuple(worker for worker in matches if str(worker.worker_id) in selected_ids)
        if not matches:
            raise ValueError("No available worker matches the supplied criteria.")

        matched_ids = {str(worker.worker_id) for worker in matches}
        for field, settings in (
            ("n_attempt", n_attempt),
            ("n_cpu_per_attempt", n_cpu_per_attempt),
            ("memory_reserve_pct", memory_reserve_pct),
        ):
            if isinstance(settings, dict) and set(settings) != matched_ids:
                raise ValueError(f"{field} mapping keys must exactly match filtered " "worker IDs.")

        attempts_by_id: dict[str, int | None]
        if isinstance(n_attempt, dict):
            attempts_by_id = dict(n_attempt)
        else:
            attempts_by_id = dict.fromkeys(matched_ids, n_attempt)

        if isinstance(n_cpu_per_attempt, dict):
            cpus_by_id = dict(n_cpu_per_attempt)
        else:
            cpus_by_id = dict.fromkeys(
                matched_ids,
                n_cpu_per_attempt,
            )

        if isinstance(memory_reserve_pct, dict):
            memory_by_id = dict(memory_reserve_pct)
        else:
            memory_by_id = dict.fromkeys(
                matched_ids,
                memory_reserve_pct,
            )

        staged_workers = list(self._workers)
        added_workers: list[WorkerConfiguration] = []

        # Stage every selected worker before committing task membership.
        for available in matches:
            available_id = str(available.worker_id)
            configured_attempts = attempts_by_id[available_id]
            configured = _stage_worker_configuration(
                WorkerConfiguration(
                    worker_id=available.worker_id,
                    worker_label=available.worker_label,
                    kind=available.kind,
                    device=available.device,
                    cpu_capacity=available.cpu_capacity,
                    n_attempt=(available.n_attempt if configured_attempts is None else configured_attempts),
                    n_cpu_per_attempt=cpus_by_id[available_id],
                    memory_reserve_pct=memory_by_id[available_id],
                ),
                existing_workers=staged_workers,
            )
            staged_workers.append(configured)
            added_workers.append(configured)

        self._workers = staged_workers

        if suppress_return:
            return None
        if not tabular_return:
            return tuple(added_workers)

        if len(added_workers) > limit:
            warnings.warn(
                "Bulk worker-addition return truncated to " f"{limit} of {len(added_workers)} added workers.",
                UserWarning,
                stacklevel=2,
            )
        return _workers_to_dataframe(tuple(added_workers[:limit]))

    @validate_args
    def list_available_workers(
        self,
        worker_id: str | None = None,
        worker_label: str | None = None,
        cpu_capacity: int | str | None = None,
        kind: str | None = None,
        device: str | None = None,
        *,
        use_regex: bool = True,
        suppress_return: bool = False,
        tabular_return: bool = False,
        limit: int = 1000,
    ) -> "tuple[WorkerConfiguration, ...] | pd.DataFrame | None":
        """Return available workers matching supplied criteria.

        Parameters
        ----------
        worker_id : str | None, default=None
            Worker-ID pattern or exact value. ``None`` ignores this field.
        worker_label : str | None, default=None
            Worker-label pattern or exact value. Labels need not be unique.
        cpu_capacity : int | str | None, default=None
            Logical-CPU-capacity pattern or exact display value.
        kind : str | None, default=None
            Worker-kind pattern or exact value.
        device : str | None, default=None
            Worker-device pattern or exact value.
        use_regex : bool, default=True
            Whether criteria use case-sensitive full regular-expression
            matching instead of exact string equality.
        suppress_return : bool, default=False
            Whether to return no representation of matching workers.
        tabular_return : bool, default=False
            Whether to return a detached string-typed table instead of an
            immutable tuple of worker configurations.
        limit : int, default=1000
            Maximum positive number of rows in a tabular return.

        Returns
        -------
        tuple[WorkerConfiguration, ...] | pandas.DataFrame | None
            Matching available workers in snapshot order. Tabular mode returns
            an empty DataFrame when no worker matches. Configuration mode
            returns ``None`` when no worker matches. Suppressed returns are
            always ``None``.

        Raises
        ------
        TypeError
            If a criterion, matching mode, return control, or limit has an
            invalid type.
        ValueError
            If a regular expression is invalid or ``limit`` is not positive.

        Warns
        -----
        UserWarning
            If a tabular result is truncated by ``limit``.

        Notes
        -----
        This Phase 3 method presents the latest received availability snapshot
        without discovering workers, testing heartbeats, establishing
        handshakes, or accepting connections. It does not add a worker to task
        allocation or read the task registry. The future network phase owns
        initialization and refresh of ``_available_workers``.
        Node usage scope: Root - Presents currently available workers before
        explicit task admission.
        """
        if limit < 1:
            raise ValueError("limit must be positive.")

        available_workers: list[WorkerConfiguration] = getattr(
            self,
            "_available_workers",
            [],
        )
        matches = _filter_workers(
            available_workers,
            worker_id=worker_id,
            worker_label=worker_label,
            cpu_capacity=cpu_capacity,
            kind=kind,
            device=device,
            use_regex=use_regex,
        )

        # Keep authoritative matches independent from the selected presentation.
        if suppress_return:
            return None
        if not tabular_return:
            if not matches:
                return None
            return matches

        if len(matches) > limit:
            warnings.warn(
                "Available-worker listing truncated to " f"{limit} of {len(matches)} matching workers.",
                UserWarning,
                stacklevel=2,
            )
        return _workers_to_dataframe(matches[:limit])

    @validate_args
    def list_workers(
        self,
        worker_id: str | None = None,
        worker_label: str | None = None,
        cpu_capacity: int | str | None = None,
        kind: str | None = None,
        device: str | None = None,
        *,
        use_regex: bool = True,
        suppress_return: bool = False,
        tabular_return: bool = False,
        limit: int = 1000,
    ) -> "tuple[WorkerConfiguration, ...] | pd.DataFrame | None":
        """Return task-registered workers matching supplied criteria.

        Parameters
        ----------
        worker_id : str | None, default=None
            Worker-ID pattern or exact value. ``None`` ignores this field.
        worker_label : str | None, default=None
            Worker-label pattern or exact value. Labels need not be unique.
        cpu_capacity : int | str | None, default=None
            Logical-CPU-capacity pattern or exact display value.
        kind : str | None, default=None
            Worker-kind pattern or exact value.
        device : str | None, default=None
            Worker-device pattern or exact value.
        use_regex : bool, default=True
            Whether criteria use case-sensitive full regular-expression
            matching instead of exact string equality.
        suppress_return : bool, default=False
            Whether to return no representation of matching workers.
        tabular_return : bool, default=False
            Whether to return a detached string-typed table instead of an
            immutable tuple of worker configurations.
        limit : int, default=1000
            Maximum positive number of rows in a tabular return.

        Returns
        -------
        tuple[WorkerConfiguration, ...] | pandas.DataFrame | None
            Matching task-registered workers in registry order. Tabular mode
            returns an empty DataFrame when no worker matches. Configuration
            mode returns ``None`` when no worker matches. Suppressed returns
            are always ``None``.

        Raises
        ------
        TypeError
            If a criterion, matching mode, return control, or limit has an
            invalid type.
        ValueError
            If a regular expression is invalid or the limit is not positive.

        Warns
        -----
        UserWarning
            If a tabular result is truncated by ``limit``.

        Notes
        -----
        This method inspects only workers admitted to this task registry. It
        does not discover connected workers, test heartbeats, establish a
        handshake, or alter attempt-allocation membership. The tabular limit
        does not truncate a configuration tuple.
        Node usage scope: Root - Presents workers registered for task
        allocation.
        """
        if limit < 1:
            raise ValueError("limit must be positive.")

        matches = _filter_workers(
            self._workers,
            worker_id=worker_id,
            worker_label=worker_label,
            cpu_capacity=cpu_capacity,
            kind=kind,
            device=device,
            use_regex=use_regex,
        )

        # Keep authoritative matches independent from the selected presentation.
        if suppress_return:
            return None
        if not tabular_return:
            if not matches:
                return None
            return matches

        if len(matches) > limit:
            warnings.warn(
                "Worker listing truncated to " f"{limit} of {len(matches)} matching workers.",
                UserWarning,
                stacklevel=2,
            )
        return _workers_to_dataframe(matches[:limit])

    @validate_args
    def remove_worker(
        self,
        worker_id: str | None = None,
    ) -> WorkerConfiguration | None:
        """Remove one worker from this task's allocation registry.

        Parameters
        ----------
        worker_id : str | None, default=None
            Exact worker-generated identity to remove. ``None`` selects the
            root-local worker configured for the current root machine.

        Returns
        -------
        WorkerConfiguration | None
            Removed immutable worker configuration, or ``None`` when no
            matching worker is registered.

        Raises
        ------
        RuntimeError
            If execution is currently running.
        TypeError
            If ``worker_id`` is neither a string nor ``None``.

        Notes
        -----
        Removal changes only this task's attempt-allocation membership. It
        does not disconnect an online worker, stop a root-local worker, alter
        scientific consolidation, or invalidate the retained parent identity.
        Node usage scope: Root - Removes one worker from prospective task
        allocation.
        """
        _require_configuration_editable(self._execution_status)

        matches = _filter_workers(
            self._workers,
            worker_id=worker_id,
            worker_label=None,
            cpu_capacity=None,
            kind=WorkerKind.ROOT_LOCAL.value if worker_id is None else None,
            device=None,
            use_regex=False,
        )
        if not matches:
            print("Found 0 matched workers, no worker is removed.")
            return None

        removed = matches[0]
        self._workers = [worker for worker in self._workers if worker.worker_id != removed.worker_id]
        return removed

    @validate_args
    def remove_workers(
        self,
        worker_id: str | None = None,
        worker_label: str | None = None,
        cpu_capacity: int | str | None = None,
        kind: str | None = None,
        device: str | None = None,
        *,
        use_regex: bool = True,
        suppress_return: bool = False,
        tabular_return: bool = False,
        limit: int = 1000,
    ) -> "tuple[WorkerConfiguration, ...] | pd.DataFrame | None":
        """Atomically remove all workers matching supplied criteria.

        Parameters
        ----------
        worker_id : str | None, default=None
            Worker-ID pattern or exact value. ``None`` ignores this field.
        worker_label : str | None, default=None
            Worker-label pattern or exact value. Labels need not be unique.
        cpu_capacity : int | str | None, default=None
            Logical-CPU-capacity pattern or exact display value.
        kind : str | None, default=None
            Worker-kind pattern or exact value.
        device : str | None, default=None
            Worker-device pattern or exact value.
        use_regex : bool, default=True
            Whether criteria use case-sensitive full regular-expression
            matching instead of exact string equality.
        suppress_return : bool, default=False
            Whether to return no representation of removed workers.
        tabular_return : bool, default=False
            Whether to return a detached string-typed table instead of an
            immutable tuple of worker configurations.
        limit : int, default=1000
            Maximum positive number of rows in a tabular return.

        Returns
        -------
        tuple[WorkerConfiguration, ...] | pandas.DataFrame | None
            Removed workers in registry order, or ``None`` when return is
            suppressed or no worker matches.

        Raises
        ------
        RuntimeError
            If execution is currently running.
        TypeError
            If a criterion or control has an incorrect runtime type.
        ValueError
            If no criterion is supplied, a regular expression is invalid, or
            ``limit`` is not positive.

        Warns
        -----
        UserWarning
            If the tabular return is truncated by ``limit``.

        Notes
        -----
        Filtering and removal are not limited by the presentation-row limit.
        Successful removal preserves survivor order and does not disconnect,
        stop, or otherwise control a worker. Scientific consolidation and the
        retained parent identity remain unchanged.
        Node usage scope: Root - Removes matching workers from prospective
        task allocation.
        """
        if limit < 1:
            raise ValueError("limit must be positive.")
        if (
            worker_id,
            worker_label,
            cpu_capacity,
            kind,
            device,
        ) == (None, None, None, None, None):
            raise ValueError("At least one worker-removal criterion must be provided.")

        _require_configuration_editable(self._execution_status)

        matches = _filter_workers(
            self._workers,
            worker_id=worker_id,
            worker_label=worker_label,
            cpu_capacity=cpu_capacity,
            kind=kind,
            device=device,
            use_regex=use_regex,
        )
        if not matches:
            print("Found 0 matched workers, no worker is removed.")
            return None

        removed_ids = {worker.worker_id for worker in matches}
        self._workers = [worker for worker in self._workers if worker.worker_id not in removed_ids]

        if suppress_return:
            return None
        if not tabular_return:
            return matches

        if len(matches) > limit:
            warnings.warn(
                "Bulk worker-removal return truncated to " f"{limit} of {len(matches)} removed workers.",
                UserWarning,
                stacklevel=2,
            )
        return _workers_to_dataframe(matches[:limit])

    def remove_model(
        self,
        model_label: str,
    ) -> ModelReference | None:
        """Remove the model registered under one exact label.

        Parameters
        ----------
        model_label : str
            Exact user-facing label of the model to remove.

        Returns
        -------
        ModelReference | None
            Removed immutable model metadata, or ``None`` when no model
            matches.

        Raises
        ------
        RuntimeError
            If execution is currently running.
        TypeError
            If ``model_label`` is not a string.
        ValueError
            If ``model_label`` is empty.

        Notes
        -----
        Removal preserves survivor order and the next model-ID counter. It
        does not inspect or modify the opaque model input. A successful
        removal is a model-evaluation configuration edit.
        """
        _require_configuration_editable(self._execution_status)

        validate_declared_types(
            model_label,
            expected_type=str,
            field="model_label",
        )
        if not model_label:
            raise ValueError("model_label must be a nonempty string.")

        matches = _filter_registered_models(
            self._models,
            label=model_label,
            family=None,
            model_input=None,
            use_regex=False,
        )
        if not matches:
            print("Found 0 matched models, no model is removed.")
            return None

        registered = matches[0]
        self._models = [model for model in self._models if model is not registered]
        self._consolidation_state = _consolidation_after_evaluation_edit(self._consolidation_state)
        return registered.reference

    def add_models(  # noqa: C901
        self,
        models: dict[str, object],
        family: (
            Literal[
                "sklearn",
                "sklearn-like",
                "pytorch",
                "cocoonet",
            ]
            | dict[
                str,
                Literal[
                    "sklearn",
                    "sklearn-like",
                    "pytorch",
                    "cocoonet",
                ],
            ]
        ),
        task_kind: Literal["regression", "classification"],
        *,
        suppress_return: bool = False,
        tabular_return: bool = False,
        limit: int = 1000,
    ) -> "tuple[ModelReference, ...] | pd.DataFrame | None":
        """Atomically add multiple opaque models in supplied order.

        Parameters
        ----------
        models : dict[str, object]
            Nonempty ordered mapping from unique model labels to in-memory
            model instances or existing model-file paths.
        family : {"sklearn", "sklearn-like", "pytorch", "cocoonet"} or dict
            One family string for every model, or a mapping from every supplied
            model label to one family string. Mapping keys must exactly match
            the model keys. ``"cocoonet"`` is reserved and currently rejected.
        task_kind : {"regression", "classification"}
            Shared prediction task for every supplied model.
        suppress_return : bool, default=False
            Whether to return no representation of the added models.
        tabular_return : bool, default=False
            Whether to return a detached string-typed table instead of
            immutable model references.
        limit : int, default=1000
            Maximum rows in a tabular return.

        Returns
        -------
        tuple[ModelReference, ...] | pandas.DataFrame | None
            Newly added models only, in supplied order, using the selected
            return representation.

        Raises
        ------
        RuntimeError
            If execution is currently running.
        TypeError
            If an argument has an incorrect runtime type.
        ValueError
            If input is empty, family keys do not match model labels, a limit
            is not positive, or a registration conflicts with configuration.

        Warns
        -----
        UserWarning
            If the tabular return is truncated by ``limit``.

        Notes
        -----
        All registrations are staged before a bounded commit. Failure leaves
        the registry, model-ID counter, and lifecycle state unchanged. Model
        inputs remain opaque and file inputs are not opened or deserialized.
        """
        validate_declared_types(models, expected_type=dict, field="models")
        if not models:
            raise ValueError("models must contain at least one model.")
        validate_declared_types(
            suppress_return,
            expected_type=bool,
            field="suppress_return",
        )
        validate_declared_types(
            tabular_return,
            expected_type=bool,
            field="tabular_return",
        )
        validate_declared_types(limit, expected_type=int, field="limit")
        if limit < 1:
            raise ValueError("limit must be positive.")

        if isinstance(family, dict):
            if family.keys() != models.keys():
                raise ValueError("family mapping keys must exactly match model labels.")
            families = {model_label: _normalize_model_family(value) for model_label, value in family.items()}
        else:
            normalized_family = _normalize_model_family(family)
            families = dict.fromkeys(models, normalized_family)

        normalized_task_kind = _normalize_task_kind(task_kind)

        _require_configuration_editable(self._execution_status)

        staged_models = list(self._models)
        added_models: list[_RegisteredModel] = []
        next_model_number = self._next_model_number

        for model_label, model_input in models.items():
            if not model_label:
                raise ValueError("model_label must be a nonempty string.")

            registered = _stage_model_registration(
                ModelRegistration(
                    model_input=model_input,
                    model_label=model_label,
                    family=families[model_label],
                    task_kind=normalized_task_kind,
                ),
                model_id=ModelId(str(next_model_number)),
                existing_models=staged_models,
                evaluation_config=self._evaluation_config,
            )
            staged_models.append(registered)
            added_models.append(registered)
            next_model_number += 1

        new_state = _consolidation_after_evaluation_edit(self._consolidation_state)

        self._models = staged_models
        self._next_model_number = next_model_number
        self._consolidation_state = new_state

        if suppress_return:
            return None
        if not tabular_return:
            return tuple(model.reference for model in added_models)

        if len(added_models) > limit:
            warnings.warn(
                "Bulk model-addition return truncated to " f"{limit} of {len(added_models)} added models.",
                UserWarning,
                stacklevel=2,
            )
        return _models_to_dataframe(tuple(added_models[:limit]))

    @validate_args
    def remove_models(
        self,
        label: str | None = None,
        family: str | None = None,
        model_input: str | None = None,
        *,
        use_regex: bool = True,
        suppress_return: bool = False,
        tabular_return: bool = False,
        limit: int = 1000,
    ) -> "tuple[ModelReference, ...] | pd.DataFrame | None":
        """Atomically remove all models matching the supplied criteria.

        Parameters
        ----------
        label : str | None, default=None
            Model-label pattern or exact value. ``None`` ignores this field.
        family : str | None, default=None
            Model-family pattern or exact value. ``None`` ignores this field.
        model_input : str | None, default=None
            Model-input display pattern or exact value. ``None`` ignores this
            field.
        use_regex : bool, default=True
            Whether criteria use case-sensitive full regular-expression
            matching instead of exact string equality.
        suppress_return : bool, default=False
            Whether to return no representation of removed models.
        tabular_return : bool, default=False
            Whether to return a detached string-typed table instead of
            immutable model references.
        limit : int, default=1000
            Maximum rows in a tabular return.

        Returns
        -------
        tuple[ModelReference, ...] | pandas.DataFrame | None
            Removed models in registry order, or ``None`` when return is
            suppressed or no model matches.

        Raises
        ------
        RuntimeError
            If execution is currently running.
        TypeError
            If a criterion or control has an incorrect runtime type.
        ValueError
            If no criterion is supplied, a regular expression is invalid, or
            ``limit`` is not positive.

        Warns
        -----
        UserWarning
            If the tabular return is truncated by ``limit``.

        Notes
        -----
        Filtering reads authoritative registered attributes directly and is
        not limited by the presentation-row limit. Successful removal
        preserves survivor order and the next model-ID counter.
        """
        if limit < 1:
            raise ValueError("limit must be positive.")
        if (label, family, model_input) == (None, None, None):
            raise ValueError("At least one model-removal criterion must be provided.")

        _require_configuration_editable(self._execution_status)

        matches = _filter_registered_models(
            self._models,
            label=label,
            family=family,
            model_input=model_input,
            use_regex=use_regex,
        )
        if not matches:
            print("Found 0 matched models, no model is removed.")
            return None

        removed_ids = {model.model_id for model in matches}
        survivors = [model for model in self._models if model.model_id not in removed_ids]
        new_state = _consolidation_after_evaluation_edit(self._consolidation_state)

        self._models = survivors
        self._consolidation_state = new_state

        if suppress_return:
            return None
        if not tabular_return:
            return tuple(model.reference for model in matches)

        if len(matches) > limit:
            warnings.warn(
                "Bulk model-removal return truncated to " f"{limit} of {len(matches)} removed models.",
                UserWarning,
                stacklevel=2,
            )
        return _models_to_dataframe(matches[:limit])
