"""Tests for EvaluationTask worker-registry behavior."""

from dataclasses import replace

import pytest

from cocoonet._contracts.identity import WorkerId
from cocoonet.configuration import WorkerConfiguration, WorkerKind
from cocoonet.evaluation_task import _stage_worker_configuration


def _worker(
    worker_id: str,
    *,
    worker_label: str = "shared-label",
) -> WorkerConfiguration:
    """Construct one valid worker configuration for focused tests."""
    return WorkerConfiguration(
        worker_id=WorkerId(worker_id),
        worker_label=worker_label,
        kind=WorkerKind.ONLINE,
        device="cpu",
        cpu_capacity=8,
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
