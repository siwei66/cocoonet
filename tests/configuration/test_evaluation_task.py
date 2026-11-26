"""Tests for reusable evaluation-task configuration."""

import copy
import os
from dataclasses import is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, get_type_hints

import pytest

from cocoonet._contracts.data import PartitionReference
from cocoonet._contracts.execution import (
    ConsolidationState,
    EvaluationReference,
    ModelFamily,
    ModelReference,
    TaskKind,
)
from cocoonet._contracts.identity import (
    DatasetId,
    EvaluationConfigId,
    ModelId,
    ParentId,
    PartitionId,
    WorkerId,
)
from cocoonet._contracts.state import ExecutionStatus
from cocoonet.configuration import (
    WorkerConfiguration,
    WorkerKind,
)
from cocoonet.evaluation_task import (
    EvaluationTask,
    _RegisteredModel,
)


def test_evaluation_task_declares_exact_owned_fields() -> None:
    """Declare the complete Phase 3 configuration ownership boundary."""
    assert get_type_hints(EvaluationTask) == {
        "_work_dir": Path,
        "_consolidation_state": ConsolidationState,
        "_parent_id": ParentId | None,
        "_execution_status": ExecutionStatus,
        "_data_partition": PartitionReference | None,
        "_evaluation_config": EvaluationReference | None,
        "_dataset_sources": tuple[Path, ...],
        "_models": list[_RegisteredModel],
        "_available_workers": list[WorkerConfiguration],
        "_workers": list[WorkerConfiguration],
        "_next_model_number": int,
    }


def test_evaluation_task_is_an_ordinary_mutable_class() -> None:
    """Avoid dataclass and slots constraints on the evolving public API."""
    assert not is_dataclass(EvaluationTask)
    assert "__slots__" not in EvaluationTask.__dict__


def test_evaluation_task_constructs_missing_work_dir_with_empty_configuration(
    tmp_path: Path,
) -> None:
    """Prepare a missing root node working directory and empty state."""
    work_dir = tmp_path / "work-dir"

    task = EvaluationTask(work_dir)

    assert task._work_dir == work_dir.resolve()
    assert task._consolidation_state is ConsolidationState.INITIALIZED
    assert task._parent_id is None
    assert task._execution_status is ExecutionStatus.NOT_STARTED
    assert task._data_partition is None
    assert task._evaluation_config is None
    assert task._dataset_sources == ()
    assert task._models == []
    assert task._workers == []
    assert task._next_model_number == 1
    assert {entry.name for entry in work_dir.iterdir()} == {
        "cocoonet_metadata.csv",
        "models_for_transfer",
        "attempt_results",
        "logs",
    }
    assert not (work_dir / "root_data_reference.xlsx").exists()
    assert not (work_dir / "partition_schedules.xlsx").exists()
    assert not (work_dir / "execution_configuration.csv").exists()
    assert not (work_dir / "execution_state.csv").exists()


def test_evaluation_task_work_directory_arguments_accept_keywords(
    tmp_path: Path,
) -> None:
    """Accept explicit working directories by keyword in public APIs."""
    task = EvaluationTask(work_dir=tmp_path / "first-work-dir")

    task.initialize(work_dir=tmp_path / "second-work-dir")

    assert task.work_dir == (tmp_path / "second-work-dir").resolve()


def test_evaluation_task_constructs_existing_empty_work_dir(
    tmp_path: Path,
) -> None:
    """Prepare an existing empty user-selected directory."""
    work_dir = tmp_path / "work-dir"
    work_dir.mkdir()

    task = EvaluationTask(work_dir)

    assert task._work_dir == work_dir.resolve()
    assert (work_dir / "cocoonet_metadata.csv").is_file()


def test_evaluation_task_rejects_nonempty_work_dir_without_overwrite(
    tmp_path: Path,
) -> None:
    """Reject recovery-like construction and preserve existing contents."""
    work_dir = tmp_path / "work-dir"
    work_dir.mkdir()
    retained = work_dir / "retained.txt"
    retained.write_text("retained", encoding="utf-8")

    with pytest.raises(ValueError, match=r"must be empty when overwrite=False"):
        EvaluationTask(work_dir)

    assert retained.read_text(encoding="utf-8") == "retained"
    assert {entry.name for entry in work_dir.iterdir()} == {"retained.txt"}


def test_evaluation_task_overwrite_prepares_fresh_work_dir(
    tmp_path: Path,
) -> None:
    """Clear an authorized root node working directory before construction."""
    work_dir = tmp_path / "work-dir"
    work_dir.mkdir()
    stale = work_dir / "stale.txt"
    stale.write_text("stale", encoding="utf-8")

    task = EvaluationTask(work_dir, overwrite=True)

    assert task._work_dir == work_dir.resolve()
    assert not stale.exists()
    assert {entry.name for entry in work_dir.iterdir()} == {
        "cocoonet_metadata.csv",
        "models_for_transfer",
        "attempt_results",
        "logs",
    }


def test_evaluation_task_resolves_relative_work_dir_from_console_directory(
    tmp_path: Path,
) -> None:
    """Resolve a relative work directory from the console working directory."""
    previous_directory = Path.cwd()
    os.chdir(tmp_path)
    try:
        task = EvaluationTask(Path("relative-work-dir"))
    finally:
        os.chdir(previous_directory)

    assert task._work_dir == (tmp_path / "relative-work-dir").resolve()


def test_evaluation_task_inspection_properties_return_owned_values(
    tmp_path: Path,
) -> None:
    """Expose the canonical work directory and lifecycle facts without I/O."""
    task = EvaluationTask(tmp_path / "work-dir")
    entries_before = tuple(sorted(entry.name for entry in task._work_dir.iterdir()))

    assert task.work_dir == task._work_dir
    assert task.state is ConsolidationState.INITIALIZED
    assert task.parent_id is None
    assert tuple(sorted(entry.name for entry in task._work_dir.iterdir())) == entries_before


@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("work_dir", Path("replacement")),
        ("state", ConsolidationState.CONSOLIDATED),
        ("parent_id", ParentId("replacement")),
    ],
)
def test_evaluation_task_inspection_properties_are_read_only(
    tmp_path: Path,
    name: str,
    replacement: object,
) -> None:
    """Reject assignment through every public inspection property."""
    task = EvaluationTask(tmp_path / "work-dir")

    with pytest.raises(AttributeError):
        task.__setattr__(name, replacement)

    assert task.work_dir == task._work_dir
    assert task.state is ConsolidationState.INITIALIZED
    assert task.parent_id is None


def test_evaluation_task_snapshot_returns_fresh_valid_empty_view(
    tmp_path: Path,
) -> None:
    """Return independent immutable mappings without storage changes."""
    task = EvaluationTask(tmp_path / "work-dir")
    entries_before = tuple(sorted(entry.name for entry in task.work_dir.iterdir()))

    first = task.snapshot()
    second = task.snapshot()

    assert first is not second
    assert first.work_dir == str(task.work_dir)
    assert first.consolidation_state is ConsolidationState.INITIALIZED
    assert first.data_partition is None
    assert first.evaluation_config is None
    assert first.parent_id is None
    assert type(first.models) is MappingProxyType
    assert type(first.worker_ids) is MappingProxyType
    assert first.models is not second.models
    assert first.worker_ids is not second.worker_ids
    assert not first.models
    assert not first.worker_ids
    assert tuple(sorted(entry.name for entry in task.work_dir.iterdir())) == entries_before


def test_evaluation_task_snapshot_preserves_registry_order_and_associations(
    tmp_path: Path,
) -> None:
    """Map labels to immutable identities in registration order."""
    task = EvaluationTask(tmp_path / "work-dir")
    first = _RegisteredModel(
        reference=ModelReference(
            model_id=ModelId("1"),
            model_label="first",
            family=ModelFamily.SKLEARN,
            task_kind=TaskKind.REGRESSION,
        ),
        model_input=object(),
    )
    second = _RegisteredModel(
        reference=ModelReference(
            model_id=ModelId("2"),
            model_label="second",
            family=ModelFamily.PYTORCH,
            task_kind=TaskKind.REGRESSION,
        ),
        model_input=object(),
    )
    worker = WorkerConfiguration(
        worker_id=WorkerId("abc123"),
        worker_label="local",
        kind=WorkerKind.ROOT_LOCAL,
    )
    task._models.extend((first, second))
    task._workers.append(worker)

    snapshot = task.snapshot()

    assert tuple(snapshot.models) == ("first", "second")
    assert snapshot.models["first"] is first.reference
    assert snapshot.models["second"] is second.reference
    assert tuple(snapshot.worker_ids) == (WorkerId("abc123"),)
    assert snapshot.worker_ids[WorkerId("abc123")] == "local"

    models: Any = snapshot.models
    worker_ids: Any = snapshot.worker_ids
    with pytest.raises(TypeError):
        models["replacement"] = first.reference
    with pytest.raises(TypeError):
        worker_ids[WorkerId("xyz789")] = "replacement"


def test_evaluation_task_snapshot_validates_owned_state(
    tmp_path: Path,
) -> None:
    """Reject an invalid relationship among authoritative fields."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._parent_id = ParentId("unexpected-parent")

    with pytest.raises(
        ValueError,
        match=r"Initialized snapshots require parent_id=None",
    ):
        task.snapshot()


def test_evaluation_task_initialize_retains_prospective_configuration(
    tmp_path: Path,
) -> None:
    """Prepare a new work directory and retain prospective configuration."""
    first_work_dir = tmp_path / "first-work-dir"
    second_work_dir = tmp_path / "second-work-dir"
    source = tmp_path / "dataset-source.csv"
    source.write_text("sample_label,y\nsample,1\n", encoding="utf-8")

    task = EvaluationTask(first_work_dir)
    partition = PartitionReference(
        partition_id=PartitionId("partition-a"),
        dataset_id=DatasetId("dataset-a"),
    )
    evaluation = EvaluationReference(
        configuration_id=EvaluationConfigId("evaluation-a"),
        task_kind=TaskKind.REGRESSION,
        class_labels=(),
        influence_requested=False,
    )
    registered = _RegisteredModel(
        reference=ModelReference(
            model_id=ModelId("1"),
            model_label="baseline",
            family=ModelFamily.SKLEARN,
            task_kind=TaskKind.REGRESSION,
        ),
        model_input=object(),
    )
    worker = WorkerConfiguration(
        worker_label="local",
        worker_id=WorkerId("worker-local"),
        kind=WorkerKind.ROOT_LOCAL,
    )

    task._data_partition = partition
    task._evaluation_config = evaluation
    task._dataset_sources = (source,)
    task._models.append(registered)
    task._workers.append(worker)
    task._next_model_number = 2
    task._consolidation_state = ConsolidationState.CONSOLIDATED
    task._parent_id = ParentId("existing-parent")
    task._execution_status = ExecutionStatus.SETTLED
    models = task._models
    workers = task._workers

    task.initialize(second_work_dir)

    assert task.work_dir == second_work_dir.resolve()
    assert task.state is ConsolidationState.INITIALIZED
    assert task.parent_id is None
    assert task._execution_status is ExecutionStatus.NOT_STARTED
    assert task._data_partition is partition
    assert task._evaluation_config is evaluation
    assert task._dataset_sources == (source,)
    assert task._models is models
    assert task._workers is workers
    assert task._models == [registered]
    assert task._workers == [worker]
    assert task._next_model_number == 2
    assert source.read_text(encoding="utf-8") == "sample_label,y\nsample,1\n"
    assert {entry.name for entry in second_work_dir.iterdir()} == {
        "cocoonet_metadata.csv",
        "models_for_transfer",
        "attempt_results",
        "logs",
    }


def test_evaluation_task_initialize_failure_preserves_all_fields(
    tmp_path: Path,
) -> None:
    """Leave task state unchanged when fresh preparation is rejected."""
    task = EvaluationTask(tmp_path / "current-work-dir")
    task._consolidation_state = ConsolidationState.UNCONSOLIDATED
    task._parent_id = ParentId("existing-parent")
    task._execution_status = ExecutionStatus.INTERRUPTED
    task._next_model_number = 4
    before = task.__dict__.copy()

    target = tmp_path / "nonempty-target"
    target.mkdir()
    retained = target / "retained.txt"
    retained.write_text("retained", encoding="utf-8")

    with pytest.raises(ValueError, match=r"must be empty when overwrite=False"):
        task.initialize(target)

    assert task.__dict__ == before
    assert retained.read_text(encoding="utf-8") == "retained"


def test_evaluation_task_initialize_rejects_running_before_filesystem_work(
    tmp_path: Path,
) -> None:
    """Apply the active-execution guard before preparing the target work directory."""
    task = EvaluationTask(tmp_path / "current-work-dir")
    task._execution_status = ExecutionStatus.RUNNING
    target = tmp_path / "unused-target"

    with pytest.raises(
        RuntimeError,
        match=r"Configuration cannot be changed while execution is running",
    ):
        task.initialize(target)

    assert task.work_dir == (tmp_path / "current-work-dir").resolve()
    assert not target.exists()


def test_evaluation_task_rejects_shallow_copy_without_mutation(
    tmp_path: Path,
) -> None:
    """Reject shallow copying without changing task state or storage."""
    task = EvaluationTask(tmp_path / "work-dir")
    fields_before = task.__dict__.copy()
    entries_before = tuple(sorted(task.work_dir.iterdir()))

    with pytest.raises(TypeError) as error:
        copy.copy(task)

    assert str(error.value) == (
        "EvaluationTask does not support shallow copying; use "
        "clone_configuration(new_work_dir) for a new independent execution "
        "using cloned configuration or clone(new_work_dir) for a new "
        "independent execution using cloned configuration and persisted files"
    )
    assert task.__dict__ == fields_before
    assert tuple(sorted(task.work_dir.iterdir())) == entries_before


def test_clone_configuration_reuses_configuration_in_fresh_storage(
    tmp_path: Path,
) -> None:
    """Clone prospective configuration without persisted execution progress."""
    source = tmp_path / "dataset-source.csv"
    source.write_text("sample_label,y\nsample,1\n", encoding="utf-8")
    task = EvaluationTask(tmp_path / "source-work-dir")

    managed_model = task.work_dir / "models_for_transfer" / "model.pkl"
    managed_model.write_bytes(b"model")
    partition = PartitionReference(
        partition_id=PartitionId("partition-a"),
        dataset_id=DatasetId("dataset-a"),
    )
    evaluation = EvaluationReference(
        configuration_id=EvaluationConfigId("evaluation-a"),
        task_kind=TaskKind.REGRESSION,
        class_labels=(),
        influence_requested=False,
    )
    registered = _RegisteredModel(
        reference=ModelReference(
            model_id=ModelId("1"),
            model_label="baseline",
            family=ModelFamily.SKLEARN,
            task_kind=TaskKind.REGRESSION,
        ),
        model_input=managed_model,
    )
    worker = WorkerConfiguration(
        worker_id=WorkerId("abc123"),
        worker_label="local",
        kind=WorkerKind.ROOT_LOCAL,
    )

    task._data_partition = partition
    task._evaluation_config = evaluation
    task._dataset_sources = (source,)
    task._models.append(registered)
    task._workers.append(worker)
    task._available_workers = [worker]
    task._next_model_number = 2
    task._consolidation_state = ConsolidationState.CONSOLIDATED
    task._parent_id = ParentId("parent-a")
    task._execution_status = ExecutionStatus.RUNNING

    cloned = task.clone_configuration(tmp_path / "cloned-work-dir")

    assert cloned is not task
    assert cloned.work_dir == (tmp_path / "cloned-work-dir").resolve()
    assert cloned.state is ConsolidationState.INITIALIZED
    assert cloned.parent_id is None
    assert cloned._execution_status is ExecutionStatus.NOT_STARTED
    assert cloned._data_partition is partition
    assert cloned._evaluation_config is evaluation
    assert cloned._dataset_sources == (source,)
    assert cloned._models == [registered]
    assert cloned._models is not task._models
    assert cloned._workers == [worker]
    assert cloned._workers is not task._workers
    assert cloned._next_model_number == 2
    assert cloned._models[0].model_input == managed_model
    assert cloned.list_available_workers(tabular_return=False) is None
    assert not (cloned.work_dir / "models_for_transfer" / "model.pkl").exists()
    assert {entry.name for entry in cloned.work_dir.iterdir()} == {
        "cocoonet_metadata.csv",
        "models_for_transfer",
        "attempt_results",
        "logs",
    }

    cloned._models.clear()
    cloned._workers.clear()
    assert task._models == [registered]
    assert task._workers == [worker]
    assert task._execution_status is ExecutionStatus.RUNNING
    assert managed_model.read_bytes() == b"model"


def test_clone_configuration_failure_preserves_source_and_destination(
    tmp_path: Path,
) -> None:
    """Preserve both tasks when fresh destination preparation is rejected."""
    task = EvaluationTask(tmp_path / "source-work-dir")
    source_fields = task.__dict__.copy()

    destination = tmp_path / "nonempty-destination"
    destination.mkdir()
    retained = destination / "retained.txt"
    retained.write_text("retained", encoding="utf-8")

    with pytest.raises(ValueError, match=r"must be empty when overwrite=False"):
        task.clone_configuration(destination)

    assert task.__dict__ == source_fields
    assert retained.read_text(encoding="utf-8") == "retained"
    assert {entry.name for entry in destination.iterdir()} == {"retained.txt"}


def test_clone_copies_inactive_progress_and_rebases_managed_model_paths(
    tmp_path: Path,
) -> None:
    """Copy persisted progress and rebase only root-managed model paths."""
    dataset_source = tmp_path / "dataset-source.csv"
    dataset_source.write_text("sample_label,y\nsample,1\n", encoding="utf-8")
    external_model = tmp_path / "external-model.pkl"
    external_model.write_bytes(b"external")

    task = EvaluationTask(tmp_path / "source-work-dir")
    managed_model = task.work_dir / "models_for_transfer" / "managed.pkl"
    managed_model.write_bytes(b"managed")
    execution_state = task.work_dir / "execution_state.csv"
    execution_state.write_bytes(b"execution-progress")
    partial_log = task.work_dir / "logs" / "root_running.log.partial"
    partial_log.write_bytes(b"partial-log")
    attempt_directory = task.work_dir / "attempt_results" / "attempt-1"
    attempt_directory.mkdir()
    (attempt_directory / "result.bin").write_bytes(b"result")

    managed_registration = _RegisteredModel(
        reference=ModelReference(
            model_id=ModelId("1"),
            model_label="managed",
            family=ModelFamily.SKLEARN,
            task_kind=TaskKind.REGRESSION,
        ),
        model_input=managed_model,
    )
    external_registration = _RegisteredModel(
        reference=ModelReference(
            model_id=ModelId("2"),
            model_label="external",
            family=ModelFamily.SKLEARN,
            task_kind=TaskKind.REGRESSION,
        ),
        model_input=external_model,
    )
    worker = WorkerConfiguration(
        worker_id=WorkerId("abc123"),
        worker_label="local",
        kind=WorkerKind.ROOT_LOCAL,
    )
    task._dataset_sources = (dataset_source,)
    task._models.extend((managed_registration, external_registration))
    task._workers.append(worker)
    task._available_workers = [worker]
    task._next_model_number = 3
    task._consolidation_state = ConsolidationState.CONSOLIDATED
    task._parent_id = ParentId("parent-a")
    task._execution_status = ExecutionStatus.SETTLED

    source_files = {
        path.relative_to(task.work_dir): path.read_bytes() for path in task.work_dir.rglob("*") if path.is_file()
    }
    destination = tmp_path / "cloned-work-dir"
    destination.mkdir()

    cloned = task.clone(destination)

    cloned_files = {
        path.relative_to(cloned.work_dir): path.read_bytes() for path in cloned.work_dir.rglob("*") if path.is_file()
    }
    assert cloned_files == source_files
    assert cloned.work_dir == destination.resolve()
    assert cloned.state is ConsolidationState.CONSOLIDATED
    assert cloned.parent_id == ParentId("parent-a")
    assert cloned._execution_status is ExecutionStatus.SETTLED
    assert cloned._dataset_sources == (dataset_source,)
    assert cloned._models is not task._models
    assert cloned._workers is not task._workers
    assert cloned._workers == [worker]
    assert cloned._next_model_number == 3
    assert cloned._models[0].model_input == (cloned.work_dir / "models_for_transfer" / "managed.pkl")
    assert cloned._models[1].model_input == external_model
    assert cloned.list_available_workers(tabular_return=False) is None

    cloned._models.clear()
    cloned._workers.clear()
    assert task._models == [managed_registration, external_registration]
    assert task._workers == [worker]


def test_clone_rejects_running_before_destination_mutation(
    tmp_path: Path,
) -> None:
    """Reject cloning while persisted execution state can still change."""
    task = EvaluationTask(tmp_path / "source-work-dir")
    task._execution_status = ExecutionStatus.RUNNING
    destination = tmp_path / "unused-destination"

    with pytest.raises(
        RuntimeError,
        match=r"cannot be cloned while execution is running",
    ):
        task.clone(destination)

    assert not destination.exists()
