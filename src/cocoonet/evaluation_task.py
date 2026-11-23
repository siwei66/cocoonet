"""Mutable root-side evaluation configuration."""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from cocoonet._contracts.data import PartitionReference
from cocoonet._contracts.execution import (
    ConsolidationState,
    EvaluationReference,
    ModelingSnapshot,
    ModelReference,
    ParentEvaluation,
    validate_modeling_snapshot,
    validate_parent_evaluation,
)
from cocoonet._contracts.identity import ModelId, ParentId, ScalarLabel
from cocoonet._contracts.state import ExecutionStatus
from cocoonet._storage.paths import PathInput
from cocoonet._storage.root_directory import prepare_root_directory
from cocoonet.configuration import WorkerConfiguration


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
    root_dir : PathInput
        User-selected directory prepared as Cocoonet-managed root storage.
    overwrite : bool, default=False
        Whether an existing nonempty root may be cleared for fresh use.

    Raises
    ------
    TypeError
        If the root or overwrite value has an incorrect runtime type.
    ValueError
        If the root is unsafe, conflicts with a configured dataset source,
        is an existing non-directory, or is nonempty without permission.

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
        root_dir: PathInput,
        /,
        *,
        overwrite: bool = False,
    ) -> None:
        """Prepare fresh root storage and initialize empty configuration."""
        layout = prepare_root_directory(
            root_dir,
            dataset_sources=(),
            overwrite=overwrite,
        )

        self._work_dir = layout.root
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
    def root_dir(self) -> Path:
        """Return the canonical Cocoonet-managed root directory.

        Returns
        -------
        Path
            Canonical physical root path returned by storage preparation.
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
        root_dir: PathInput,
        /,
        *,
        overwrite: bool = False,
    ) -> None:
        """Prepare a fresh root while retaining prospective configuration.

        Parameters
        ----------
        root_dir : PathInput
            User-selected directory to prepare as fresh managed root storage.
        overwrite : bool, default=False
            Whether an existing nonempty target root may be cleared.

        Raises
        ------
        RuntimeError
            If execution is currently running.
        TypeError
            If the root or overwrite value has an incorrect runtime type.
        ValueError
            If the target root is unsafe, conflicts with a retained dataset
            source, is an existing non-directory, or is nonempty without
            permission.

        Notes
        -----
        Preparation is fresh initialization, not recovery. Models, workers,
        dataset/evaluation references, source paths, registry order, and the
        next model counter remain prospective configuration.
        """
        _require_configuration_editable(self._execution_status)
        dataset_sources = tuple(self._dataset_sources)

        layout = prepare_root_directory(
            root_dir,
            dataset_sources=dataset_sources,
            overwrite=overwrite,
        )

        self._work_dir = layout.root
        self._consolidation_state = ConsolidationState.INITIALIZED
        self._parent_id = None
        self._execution_status = ExecutionStatus.NOT_STARTED
