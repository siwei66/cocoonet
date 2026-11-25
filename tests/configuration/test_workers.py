"""Tests for EvaluationTask worker-registry behavior."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from cocoonet._contracts.execution import ConsolidationState
from cocoonet._contracts.identity import ParentId, WorkerId
from cocoonet._contracts.state import ExecutionStatus
from cocoonet.configuration import WorkerConfiguration, WorkerKind
from cocoonet.evaluation_task import EvaluationTask, _stage_worker_configuration


def _worker(
    worker_id: str,
    *,
    worker_label: str = "shared-label",
    kind: WorkerKind = WorkerKind.ONLINE,
    device: str = "cpu",
    cpu_capacity: int | None = 8,
) -> WorkerConfiguration:
    """Construct one valid worker configuration for focused tests."""
    return WorkerConfiguration(
        worker_id=WorkerId(worker_id),
        worker_label=worker_label,
        kind=kind,
        device=device,
        cpu_capacity=cpu_capacity,
        n_attempt=2,
        n_cpu_per_attempt=2,
        memory_reserve_pct=20.0,
    )


def test_stage_worker_configuration_allows_duplicate_labels() -> None:
    """Stage a distinct worker ID without treating its label as an index."""
    existing = [_worker("abc123")]
    candidate = _worker("xyz789")
    before = tuple(existing)

    result = _stage_worker_configuration(
        candidate,
        existing_workers=existing,
    )

    assert result is candidate
    assert tuple(existing) == before


def test_stage_worker_configuration_rejects_duplicate_id_without_mutation() -> None:
    """Reject an existing worker identity without changing the registry."""
    existing = [_worker("abc123")]
    candidate = _worker("abc123", worker_label="replacement-label")
    before = tuple(existing)

    with pytest.raises(
        ValueError,
        match=r"^worker_id must be unique; duplicate worker ID: 'abc123'\.$",
    ):
        _stage_worker_configuration(
            candidate,
            existing_workers=existing,
        )

    assert tuple(existing) == before


def test_stage_worker_configuration_delegates_record_validation() -> None:
    """Reject an invalid worker-supplied identity without registry mutation."""
    existing = [_worker("abc123")]
    candidate = replace(_worker("xyz789"), worker_id=WorkerId("ABC123"))
    before = tuple(existing)

    with pytest.raises(
        ValueError,
        match=r"^worker_id must contain exactly six lowercase ASCII letters",
    ):
        _stage_worker_configuration(
            candidate,
            existing_workers=existing,
        )

    assert tuple(existing) == before


def test_list_workers_returns_public_table_in_registry_order(
    tmp_path: Path,
) -> None:
    """Return all registered worker fields in authoritative ID order."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._workers.extend(
        (
            _worker("abc123"),
            _worker(
                "xyz789",
                kind=WorkerKind.ROOT_LOCAL,
                device="cuda:0",
                cpu_capacity=None,
            ),
        )
    )

    result = task.list_workers()

    assert isinstance(result, pd.DataFrame)
    assert result.columns.tolist() == [
        "worker_id",
        "worker_label",
        "kind",
        "device",
        "cpu_capacity",
        "n_attempt",
        "n_cpu_per_attempt",
        "memory_reserve_pct",
    ]
    assert result["worker_id"].tolist() == ["abc123", "xyz789"]
    assert result["worker_label"].tolist() == [
        "shared-label",
        "shared-label",
    ]
    assert result["cpu_capacity"].tolist() == ["8", ""]
    assert all(dtype == pd.StringDtype() for dtype in result.dtypes)


def test_list_workers_combines_regex_criteria(
    tmp_path: Path,
) -> None:
    """Require every active regular-expression criterion to match."""
    task = EvaluationTask(tmp_path / "work-dir")
    expected = _worker(
        "abc123",
        worker_label="research-alpha",
        device="cpu",
        cpu_capacity=8,
    )
    task._workers.extend(
        (
            expected,
            _worker(
                "xyz789",
                worker_label="research-beta",
                device="cuda:0",
                cpu_capacity=16,
            ),
        )
    )

    result = task.list_workers(
        worker_id=r"abc\d{3}",
        worker_label=r"research-.*",
        cpu_capacity=r"8",
        kind=r"online",
        device=r"cpu",
        tabular_return=False,
    )

    assert result == (expected,)


def test_list_workers_reference_return_is_not_limited(
    tmp_path: Path,
) -> None:
    """Apply the presentation limit only to tabular returns."""
    task = EvaluationTask(tmp_path / "work-dir")
    expected = (_worker("abc123"), _worker("xyz789"))
    task._workers.extend(expected)

    result = task.list_workers(
        tabular_return=False,
        limit=1,
    )

    assert result == expected


def test_list_workers_handles_empty_and_suppressed_returns(
    tmp_path: Path,
) -> None:
    """Distinguish an empty table, absent tuple, and suppressed return."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._workers.append(_worker("abc123"))

    table = task.list_workers(worker_id="missing", use_regex=False)
    references = task.list_workers(
        worker_id="missing",
        use_regex=False,
        tabular_return=False,
    )
    suppressed = task.list_workers(suppress_return=True)

    assert isinstance(table, pd.DataFrame)
    assert table.empty
    assert references is None
    assert suppressed is None


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"worker_id": 1}, TypeError, "worker_id"),
        ({"worker_label": 1}, TypeError, "worker_label"),
        ({"cpu_capacity": True}, TypeError, "cpu_capacity"),
        ({"kind": 1}, TypeError, "kind"),
        ({"device": 1}, TypeError, "device"),
        ({"use_regex": 1}, TypeError, "use_regex"),
        ({"suppress_return": 1}, TypeError, "suppress_return"),
        ({"tabular_return": 1}, TypeError, "tabular_return"),
        ({"limit": True}, TypeError, "limit"),
        ({"limit": 0}, ValueError, "limit"),
        ({"worker_id": "["}, ValueError, "worker_id"),
    ],
)
def test_list_workers_rejects_invalid_arguments(
    tmp_path: Path,
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject invalid listing inputs without mutating the registry."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._workers.append(_worker("abc123"))
    before = tuple(task._workers)

    list_workers: Any = task.list_workers
    with pytest.raises(error_type, match=message):
        list_workers(**kwargs)

    assert tuple(task._workers) == before


def test_remove_worker_removes_exact_id_without_scientific_edit(
    tmp_path: Path,
) -> None:
    """Remove only the selected worker while preserving parent reuse state."""
    task = EvaluationTask(tmp_path / "work-dir")
    retained = _worker(
        "abc123",
        kind=WorkerKind.ROOT_LOCAL,
    )
    removed = _worker("xyz789", worker_label="online-worker")
    task._workers.extend((retained, removed))
    task._consolidation_state = ConsolidationState.CONSOLIDATED
    task._parent_id = ParentId("existing-parent")

    result = task.remove_worker("xyz789")

    assert result is removed
    assert task._workers == [retained]
    assert task.state is ConsolidationState.CONSOLIDATED
    assert task.parent_id == ParentId("existing-parent")


def test_remove_worker_defaults_to_root_local_worker(
    tmp_path: Path,
) -> None:
    """Use an omitted worker ID to select the current root-local worker."""
    task = EvaluationTask(tmp_path / "work-dir")
    local_worker = _worker(
        "abc123",
        kind=WorkerKind.ROOT_LOCAL,
    )
    online_worker = _worker("xyz789")
    task._workers.extend((online_worker, local_worker))

    result = task.remove_worker()

    assert result is local_worker
    assert task._workers == [online_worker]


def test_remove_worker_reports_no_match_without_mutation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Return none and preserve the registry when no worker ID matches."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._workers.append(_worker("abc123"))
    before = task.__dict__.copy()

    result = task.remove_worker("def456")

    assert result is None
    assert capsys.readouterr().out == ("Found 0 matched workers, no worker is removed.\n")
    assert task.__dict__ == before


def test_remove_worker_rejects_invalid_id_type_without_mutation(
    tmp_path: Path,
) -> None:
    """Reject an invalid selector type before changing the registry."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._workers.append(_worker("abc123"))
    before = tuple(task._workers)
    remove_worker: Any = task.remove_worker

    with pytest.raises(TypeError, match=r"worker_id"):
        remove_worker(1)

    assert tuple(task._workers) == before


def test_remove_worker_rejects_running_execution_without_mutation(
    tmp_path: Path,
) -> None:
    """Prevent allocation-membership edits during active execution."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._workers.append(_worker("abc123"))
    task._execution_status = ExecutionStatus.RUNNING
    before = tuple(task._workers)

    with pytest.raises(RuntimeError, match=r"execution is running"):
        task.remove_worker("abc123")

    assert tuple(task._workers) == before
