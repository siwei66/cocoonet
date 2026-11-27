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


def test_add_worker_adds_available_root_local_worker(
    tmp_path: Path,
) -> None:
    """Add the available root-local worker without changing scientific state."""
    task = EvaluationTask(tmp_path / "work-dir")
    available = _worker(
        "abc123",
        worker_label="root-machine",
        kind=WorkerKind.ROOT_LOCAL,
        device="cuda:0",
        cpu_capacity=16,
    )
    task._available_workers = [available]
    task._consolidation_state = ConsolidationState.CONSOLIDATED
    task._parent_id = ParentId("existing-parent")
    before_available = tuple(task._available_workers)

    result = task.add_worker()

    assert result == replace(
        available,
        n_cpu_per_attempt=1,
    )
    assert task._workers == [result]
    assert tuple(task._available_workers) == before_available
    assert task.state is ConsolidationState.CONSOLIDATED
    assert task.parent_id == ParentId("existing-parent")


def test_add_worker_adds_selected_online_worker(
    tmp_path: Path,
) -> None:
    """Combine received machine properties with selected execution settings."""
    task = EvaluationTask(tmp_path / "work-dir")
    available = _worker(
        "abc123",
        worker_label="online-machine",
        kind=WorkerKind.ONLINE,
        device="cuda:0",
        cpu_capacity=32,
    )
    task._available_workers = [available]

    result = task.add_worker(
        "abc123",
        "online",
        n_attempt=3,
        n_cpu_per_attempt=4,
        memory_reserve_pct=10.0,
    )

    assert result == replace(
        available,
        n_attempt=3,
        n_cpu_per_attempt=4,
        memory_reserve_pct=10.0,
    )
    assert task._workers == [result]


def test_add_worker_rejects_unavailable_worker_without_mutation(
    tmp_path: Path,
) -> None:
    """Reject addition before an applicable availability snapshot exists."""
    task = EvaluationTask(tmp_path / "work-dir")
    before = task.__dict__.copy()

    with pytest.raises(
        ValueError,
        match=r"No available worker matches",
    ):
        task.add_worker()

    assert task.__dict__ == before


def test_add_worker_rejects_kind_mismatch_without_mutation(
    tmp_path: Path,
) -> None:
    """Require the requested topology to match the available worker."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._available_workers = [
        _worker(
            "abc123",
            kind=WorkerKind.ONLINE,
        )
    ]
    before = tuple(task._available_workers)

    with pytest.raises(
        ValueError,
        match=r"No available worker matches",
    ):
        task.add_worker("abc123")

    assert task._workers == []
    assert tuple(task._available_workers) == before


def test_add_worker_rejects_duplicate_id_without_mutation(
    tmp_path: Path,
) -> None:
    """Reject an already registered worker identity atomically."""
    task = EvaluationTask(tmp_path / "work-dir")
    available = _worker(
        "abc123",
        kind=WorkerKind.ONLINE,
    )
    task._available_workers = [available]
    task._workers.append(available)
    before = task.__dict__.copy()

    with pytest.raises(ValueError, match=r"worker_id must be unique"):
        task.add_worker("abc123", "online")

    assert task.__dict__ == before


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"worker_id": 1}, TypeError, "worker_id"),
        ({"kind": "remote"}, TypeError, "kind"),
        ({"n_attempt": True}, TypeError, "n_attempt"),
        ({"n_cpu_per_attempt": True}, TypeError, "n_cpu_per_attempt"),
        ({"memory_reserve_pct": 20}, TypeError, "memory_reserve_pct"),
        ({"n_attempt": 0}, ValueError, "n_attempt"),
        ({"n_cpu_per_attempt": 0}, ValueError, "n_cpu_per_attempt"),
        (
            {"memory_reserve_pct": -1.0},
            ValueError,
            "memory_reserve_pct",
        ),
    ],
)
def test_add_worker_rejects_invalid_arguments_without_mutation(
    tmp_path: Path,
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject invalid arguments before committing worker membership."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._available_workers = [
        _worker(
            "abc123",
            kind=WorkerKind.ROOT_LOCAL,
        )
    ]
    before_available = tuple(task._available_workers)
    add_worker: Any = task.add_worker

    with pytest.raises(error_type, match=message):
        add_worker(**kwargs)

    assert task._workers == []
    assert tuple(task._available_workers) == before_available


def test_add_worker_rejects_running_execution_without_mutation(
    tmp_path: Path,
) -> None:
    """Prevent worker admission during active execution."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._available_workers = [
        _worker(
            "abc123",
            kind=WorkerKind.ROOT_LOCAL,
        )
    ]
    task._execution_status = ExecutionStatus.RUNNING
    before = task.__dict__.copy()

    with pytest.raises(RuntimeError, match=r"execution is running"):
        task.add_worker()

    assert task.__dict__ == before


def test_add_workers_filters_available_workers_in_snapshot_order(
    tmp_path: Path,
) -> None:
    """Filter available workers while preserving snapshot order."""
    task = EvaluationTask(tmp_path / "work-dir")
    first = _worker(
        "abc123",
        worker_label="research-one",
        device="cpu",
        cpu_capacity=8,
    )
    second = _worker(
        "xyz789",
        worker_label="research-two",
        device="cpu",
        cpu_capacity=16,
    )
    excluded = _worker(
        "def456",
        worker_label="other",
        device="cuda:0",
        cpu_capacity=32,
    )
    task._available_workers = [first, second, excluded]
    before_available = tuple(task._available_workers)

    result = task.add_workers(
        worker_id=["xyz789", "abc123"],
        worker_label=r"research-.*",
        cpu_capacity=r"8|16",
        kind=r"online",
        device=r"cpu",
    )

    expected = (
        replace(first, n_cpu_per_attempt=1),
        replace(second, n_cpu_per_attempt=1),
    )
    assert result == expected
    assert task._workers == list(expected)
    assert tuple(task._available_workers) == before_available


def test_add_workers_uses_heterogeneous_execution_settings(
    tmp_path: Path,
) -> None:
    """Configure each selected worker using worker-ID-indexed mappings."""
    task = EvaluationTask(tmp_path / "work-dir")
    first = _worker("abc123")
    second = _worker("xyz789")
    task._available_workers = [first, second]

    result = task.add_workers(
        worker_id=("abc123", "xyz789"),
        n_attempt={"abc123": 1, "xyz789": 3},
        n_cpu_per_attempt={"abc123": 2, "xyz789": 4},
        memory_reserve_pct={"abc123": 10.0, "xyz789": 25.0},
    )

    expected = (
        replace(
            first,
            n_attempt=1,
            n_cpu_per_attempt=2,
            memory_reserve_pct=10.0,
        ),
        replace(
            second,
            n_attempt=3,
            n_cpu_per_attempt=4,
            memory_reserve_pct=25.0,
        ),
    )
    assert result == expected
    assert task._workers == list(expected)


def test_add_workers_never_applies_regex_to_worker_ids(
    tmp_path: Path,
) -> None:
    """Treat every supplied worker ID as an exact identity."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._available_workers = [_worker("abc123")]
    before = task.__dict__.copy()

    with pytest.raises(ValueError, match=r"No available worker matches"):
        task.add_workers(worker_id=["abc.*"])

    assert task.__dict__ == before


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("n_attempt", {"abc123": 1}),
        ("n_cpu_per_attempt", {"abc123": 1}),
        ("memory_reserve_pct", {"abc123": 20.0}),
    ],
)
def test_add_workers_requires_exact_configuration_mapping_keys(
    tmp_path: Path,
    keyword: str,
    value: object,
) -> None:
    """Require heterogeneous settings for every filtered worker and no others."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._available_workers = [
        _worker("abc123"),
        _worker("xyz789"),
    ]
    before = task.__dict__.copy()
    add_workers: Any = task.add_workers

    with pytest.raises(
        ValueError,
        match=rf"{keyword} mapping keys must exactly match",
    ):
        add_workers(**{keyword: value})

    assert task.__dict__ == before


def test_add_workers_is_atomic_when_one_worker_is_already_registered(
    tmp_path: Path,
) -> None:
    """Avoid committing earlier staged workers when a later ID is duplicated."""
    task = EvaluationTask(tmp_path / "work-dir")
    first = _worker("abc123")
    duplicate = _worker("xyz789")
    task._available_workers = [first, duplicate]
    task._workers.append(duplicate)
    before = task.__dict__.copy()

    with pytest.raises(ValueError, match=r"worker_id must be unique"):
        task.add_workers(tabular_return=False)

    assert task.__dict__ == before


def test_add_workers_truncates_only_tabular_return(
    tmp_path: Path,
) -> None:
    """Add every match while limiting only its presentation table."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._available_workers = [
        _worker("abc123"),
        _worker("xyz789"),
    ]

    with pytest.warns(UserWarning, match=r"truncated to 1 of 2"):
        result = task.add_workers(limit=1, tabular_return=True)

    assert isinstance(result, pd.DataFrame)
    assert result["worker_id"].tolist() == ["abc123"]
    assert [worker.worker_id for worker in task._workers] == [
        "abc123",
        "xyz789",
    ]


def test_add_workers_can_suppress_return(
    tmp_path: Path,
) -> None:
    """Commit matching workers without constructing a return representation."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._available_workers = [_worker("abc123")]

    result = task.add_workers(suppress_return=True)

    assert result is None
    assert [worker.worker_id for worker in task._workers] == ["abc123"]


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"worker_id": [1]}, TypeError, "worker_id"),
        ({"worker_label": 1}, TypeError, "worker_label"),
        ({"cpu_capacity": True}, TypeError, "cpu_capacity"),
        ({"kind": 1}, TypeError, "kind"),
        ({"device": 1}, TypeError, "device"),
        ({"n_attempt": True}, TypeError, "n_attempt"),
        ({"n_cpu_per_attempt": True}, TypeError, "n_cpu_per_attempt"),
        (
            {"memory_reserve_pct": 20},
            TypeError,
            "memory_reserve_pct",
        ),
        ({"use_regex": 1}, TypeError, "use_regex"),
        ({"suppress_return": 1}, TypeError, "suppress_return"),
        ({"tabular_return": 1}, TypeError, "tabular_return"),
        ({"limit": True}, TypeError, "limit"),
        ({"limit": 0}, ValueError, "positive"),
        ({"kind": "["}, ValueError, "kind"),
        ({"n_attempt": 0}, ValueError, "n_attempt"),
        ({"n_cpu_per_attempt": 0}, ValueError, "n_cpu_per_attempt"),
        (
            {"memory_reserve_pct": -1.0},
            ValueError,
            "memory_reserve_pct",
        ),
    ],
)
def test_add_workers_rejects_invalid_arguments_without_mutation(
    tmp_path: Path,
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject invalid inputs before committing any worker."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._available_workers = [_worker("abc123")]
    before = task.__dict__.copy()
    add_workers: Any = task.add_workers

    with pytest.raises(error_type, match=message):
        add_workers(**kwargs)

    assert task.__dict__ == before


def test_add_workers_rejects_running_execution_without_mutation(
    tmp_path: Path,
) -> None:
    """Prevent bulk worker admission during active execution."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._available_workers = [_worker("abc123")]
    task._execution_status = ExecutionStatus.RUNNING
    before = task.__dict__.copy()

    with pytest.raises(RuntimeError, match=r"execution is running"):
        task.add_workers()

    assert task.__dict__ == before


def test_list_available_workers_returns_only_availability_snapshot(
    tmp_path: Path,
) -> None:
    """Present available workers without reading task membership."""
    task = EvaluationTask(tmp_path / "work-dir")
    available = (
        _worker("abc123", worker_label="available-one"),
        _worker("xyz789", worker_label="available-two"),
    )
    task._available_workers = list(available)
    task._workers.append(_worker("def456", worker_label="registered-only"))

    result = task.list_available_workers(tabular_return=True)

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
        "available-one",
        "available-two",
    ]
    assert all(dtype == pd.StringDtype() for dtype in result.dtypes)


def test_list_available_workers_combines_regex_criteria(
    tmp_path: Path,
) -> None:
    """Require every active availability criterion to match."""
    task = EvaluationTask(tmp_path / "work-dir")
    expected = _worker(
        "abc123",
        worker_label="research-alpha",
        device="cpu",
        cpu_capacity=8,
    )
    task._available_workers = [
        expected,
        _worker(
            "xyz789",
            worker_label="research-beta",
            device="cuda:0",
            cpu_capacity=16,
        ),
    ]

    result = task.list_available_workers(
        worker_id=r"abc\d{3}",
        worker_label=r"research-.*",
        cpu_capacity=r"8",
        kind=r"online",
        device=r"cpu",
    )

    assert result == (expected,)


def test_list_available_workers_default_return_is_not_limited(
    tmp_path: Path,
) -> None:
    """Return all configurations by default despite the presentation limit."""
    task = EvaluationTask(tmp_path / "work-dir")
    expected = (_worker("abc123"), _worker("xyz789"))
    task._available_workers = list(expected)

    result = task.list_available_workers(
        limit=1,
    )

    assert result == expected


def test_list_available_workers_handles_absent_and_suppressed_snapshot(
    tmp_path: Path,
) -> None:
    """Treat an uninitialized snapshot as empty and support suppression."""
    task = EvaluationTask(tmp_path / "work-dir")

    table = task.list_available_workers(tabular_return=True)
    configurations = task.list_available_workers()
    suppressed = task.list_available_workers(suppress_return=True)

    assert isinstance(table, pd.DataFrame)
    assert table.empty
    assert configurations is None
    assert suppressed is None


def test_list_available_workers_warns_when_limit_truncates(
    tmp_path: Path,
) -> None:
    """Return the snapshot prefix and warn when its table is truncated."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._available_workers = [
        _worker("abc123"),
        _worker("xyz789"),
    ]

    with pytest.warns(UserWarning, match=r"truncated to 1 of 2"):
        result = task.list_available_workers(limit=1, tabular_return=True)

    assert isinstance(result, pd.DataFrame)
    assert result["worker_id"].tolist() == ["abc123"]


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
        ({"limit": 0}, ValueError, "positive"),
        ({"worker_id": "["}, ValueError, "worker_id"),
    ],
)
def test_list_available_workers_rejects_invalid_arguments(
    tmp_path: Path,
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject invalid arguments without changing either worker collection."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._available_workers = [_worker("abc123")]
    task._workers.append(_worker("xyz789"))
    before_available = tuple(task._available_workers)
    before_registered = tuple(task._workers)
    list_available_workers: Any = task.list_available_workers

    with pytest.raises(error_type, match=message):
        list_available_workers(**kwargs)

    assert tuple(task._available_workers) == before_available
    assert tuple(task._workers) == before_registered


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

    result = task.list_workers(tabular_return=True)

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
    )

    assert result == (expected,)


def test_list_workers_default_return_is_not_limited(
    tmp_path: Path,
) -> None:
    """Return all configurations by default despite the presentation limit."""
    task = EvaluationTask(tmp_path / "work-dir")
    expected = (_worker("abc123"), _worker("xyz789"))
    task._workers.extend(expected)

    result = task.list_workers(
        limit=1,
    )

    assert result == expected


def test_list_workers_handles_empty_and_suppressed_returns(
    tmp_path: Path,
) -> None:
    """Distinguish an empty table, absent tuple, and suppressed return."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._workers.append(_worker("abc123"))

    table = task.list_workers(
        worker_id="missing",
        use_regex=False,
        tabular_return=True,
    )
    references = task.list_workers(
        worker_id="missing",
        use_regex=False,
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


def test_remove_workers_removes_all_matches_without_scientific_edit(
    tmp_path: Path,
) -> None:
    """Remove duplicate-label matches while retaining scientific state."""
    task = EvaluationTask(tmp_path / "work-dir")
    first = _worker("abc123", worker_label="batch")
    second = _worker("xyz789", worker_label="batch")
    survivor = _worker("def456", worker_label="retained")
    task._workers.extend((first, second, survivor))
    task._consolidation_state = ConsolidationState.CONSOLIDATED
    task._parent_id = ParentId("existing-parent")

    result = task.remove_workers(
        worker_label="batch",
        use_regex=False,
    )

    assert result == (first, second)
    assert task._workers == [survivor]
    assert task.state is ConsolidationState.CONSOLIDATED
    assert task.parent_id == ParentId("existing-parent")


def test_remove_workers_requires_a_criterion(
    tmp_path: Path,
) -> None:
    """Prevent accidental unfiltered removal of the complete registry."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._workers.append(_worker("abc123"))
    before = task.__dict__.copy()

    with pytest.raises(ValueError, match=r"criterion must be provided"):
        task.remove_workers()

    assert task.__dict__ == before


def test_remove_workers_reports_no_match_without_mutation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Return none and report an empty worker selection."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._workers.append(_worker("abc123"))
    before = task.__dict__.copy()

    result = task.remove_workers(worker_id="def456", use_regex=False)

    assert result is None
    assert capsys.readouterr().out == ("Found 0 matched workers, no worker is removed.\n")
    assert task.__dict__ == before


def test_remove_workers_can_suppress_return(
    tmp_path: Path,
) -> None:
    """Remove matching workers without constructing a return value."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._workers.append(_worker("abc123", worker_label="batch"))

    result = task.remove_workers(
        worker_label="batch",
        use_regex=False,
        suppress_return=True,
    )

    assert result is None
    assert task._workers == []


def test_remove_workers_truncates_only_tabular_return(
    tmp_path: Path,
) -> None:
    """Remove every match while limiting only its presentation table."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._workers.extend(
        (
            _worker("abc123", device="cpu"),
            _worker("xyz789", device="cpu"),
            _worker("def456", device="cuda:0"),
        )
    )

    with pytest.warns(UserWarning, match=r"truncated to 1 of 2"):
        result = task.remove_workers(
            device="cpu",
            use_regex=False,
            tabular_return=True,
            limit=1,
        )

    assert isinstance(result, pd.DataFrame)
    assert result["worker_id"].tolist() == ["abc123"]
    assert [worker.worker_id for worker in task._workers] == ["def456"]


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
        ({"limit": 0}, ValueError, "positive"),
        ({"worker_id": "["}, ValueError, "worker_id"),
    ],
)
def test_remove_workers_rejects_invalid_arguments_without_mutation(
    tmp_path: Path,
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject invalid criteria and controls before registry mutation."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._workers.append(_worker("abc123"))
    before = tuple(task._workers)
    remove_workers: Any = task.remove_workers

    with pytest.raises(error_type, match=message):
        remove_workers(**kwargs)

    assert tuple(task._workers) == before


def test_remove_workers_rejects_running_execution_without_mutation(
    tmp_path: Path,
) -> None:
    """Prevent bulk allocation-membership edits during active execution."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._workers.append(_worker("abc123"))
    task._execution_status = ExecutionStatus.RUNNING
    before = tuple(task._workers)

    with pytest.raises(RuntimeError, match=r"execution is running"):
        task.remove_workers(worker_id="abc123", use_regex=False)

    assert tuple(task._workers) == before
