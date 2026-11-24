"""Mutable root-side evaluation configuration."""

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

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
from cocoonet._storage.work_directory import prepare_work_directory
from cocoonet.configuration import (
    ModelRegistration,
    WorkerConfiguration,
    validate_model_registration,
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
    if type(use_regex) is not bool:
        raise TypeError("use_regex must be a bool.")

    criteria = {
        "label": label,
        "family": family,
        "model_input": model_input,
    }
    for name, criterion in criteria.items():
        if criterion is not None and not isinstance(criterion, str):
            raise TypeError(f"{name} must be a string or None.")

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
) -> pd.DataFrame:
    """Build the public string-typed model registry table."""
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
    _workers: list[WorkerConfiguration]
    _next_model_number: int

    def __init__(
        self,
        work_dir: PathInput,
        /,
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
            worker_ids=MappingProxyType({worker.worker_label: worker.worker_id for worker in self._workers}),
            parent_id=self._parent_id,
        )
        validate_modeling_snapshot(snapshot)
        return snapshot

    def initialize(
        self,
        work_dir: PathInput,
        /,
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
        /,
        *,
        model_label: ScalarLabel,
        family: ModelFamily,
        task_kind: TaskKind,
    ) -> ModelReference:
        """Add one opaque model input to prospective configuration.

        Parameters
        ----------
        model_input : object
            In-memory opaque input, or an existing model-file path supplied as
            text or ``Path``.
        model_label : ScalarLabel
            User-facing label unique within the active model registry.
        family : ModelFamily
            Explicit model-construction family.
        task_kind : TaskKind
            Explicit regression or classification task.

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
                family=family,
                task_kind=task_kind,
            ),
            model_id=ModelId(str(self._next_model_number)),
            existing_models=self._models,
            evaluation_config=self._evaluation_config,
        )

        self._models.append(registered)
        self._next_model_number += 1
        self._consolidation_state = _consolidation_after_evaluation_edit(self._consolidation_state)
        return registered.reference

    def list_models(
        self,
        label: str | None = None,
        family: str | None = None,
        model_input: str | None = None,
        *,
        use_regex: bool = True,
        limit: int = 1000,
    ) -> pd.DataFrame:
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
        limit : int, default=1000
            Maximum positive number of matching rows to return.

        Returns
        -------
        pandas.DataFrame
            Matching models in registry order with string-typed columns.

        Raises
        ------
        TypeError
            If a criterion, matching mode, or limit has an invalid type.
        ValueError
            If a regular expression is invalid or the limit is not positive.

        Warns
        -----
        UserWarning
            If matching rows are truncated by ``limit``.

        Notes
        -----
        Search reads registered attributes directly. It does not inspect model
        files or objects, mutate configuration, or derive search data from the
        returned presentation table.
        """
        if not isinstance(limit, int):
            raise TypeError("limit must be an integer.")
        if limit < 1:
            raise ValueError("limit must be positive.")

        matches = _filter_registered_models(
            self._models,
            label=label,
            family=family,
            model_input=model_input,
            use_regex=use_regex,
        )
        if len(matches) > limit:
            warnings.warn(
                f"Model listing truncated to {limit} of {len(matches)} matching models.",
                UserWarning,
                stacklevel=2,
            )
        return _models_to_dataframe(matches[:limit])

    def remove_model(
        self,
        model_label: str,
        /,
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

        if not isinstance(model_label, str):
            raise TypeError("model_label must be a string.")
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
        family: ModelFamily | dict[str, ModelFamily],
        task_kind: TaskKind,
        *,
        suppress_return: bool = False,
        tabular_return: bool = True,
        limit: int = 1000,
    ) -> tuple[ModelReference, ...] | pd.DataFrame | None:
        """Atomically add multiple opaque models in supplied order.

        Parameters
        ----------
        models : dict[str, object]
            Ordered mapping from unique model labels to opaque model inputs.
        family : ModelFamily | dict[str, ModelFamily]
            One family for every model, or a mapping whose keys exactly match
            the model labels.
        task_kind : TaskKind
            Shared regression or classification task.
        suppress_return : bool, default=False
            Whether to return no representation of the added models.
        tabular_return : bool, default=True
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
        if not isinstance(models, dict):
            raise TypeError("models must be a dictionary.")
        if not models:
            raise ValueError("models must contain at least one model.")
        if type(suppress_return) is not bool:
            raise TypeError("suppress_return must be a bool.")
        if type(tabular_return) is not bool:
            raise TypeError("tabular_return must be a bool.")
        if not isinstance(limit, int):
            raise TypeError("limit must be an integer.")
        if limit < 1:
            raise ValueError("limit must be positive.")

        if isinstance(family, dict):
            if family.keys() != models.keys():
                raise ValueError("family mapping keys must exactly match model labels.")
            families = family
        elif isinstance(family, ModelFamily):
            families = dict.fromkeys(models, family)
        else:
            raise TypeError("family must be a ModelFamily or a dictionary of ModelFamily values.")

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
                    task_kind=task_kind,
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

    def remove_models(
        self,
        label: str | None = None,
        family: str | None = None,
        model_input: str | None = None,
        *,
        use_regex: bool = True,
        suppress_return: bool = False,
        tabular_return: bool = True,
        limit: int = 1000,
    ) -> tuple[ModelReference, ...] | pd.DataFrame | None:
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
        tabular_return : bool, default=True
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
        if type(suppress_return) is not bool:
            raise TypeError("suppress_return must be a bool.")
        if type(tabular_return) is not bool:
            raise TypeError("tabular_return must be a bool.")
        if not isinstance(limit, int):
            raise TypeError("limit must be an integer.")
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
