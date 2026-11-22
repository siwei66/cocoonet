"""Tests for root-side configuration records."""

from cocoonet.configuration import WorkerKind


def test_worker_kind_has_exact_string_vocabulary() -> None:
    """Expose only the planned root-local and online worker kinds."""
    assert tuple(WorkerKind) == (
        WorkerKind.ROOT_LOCAL,
        WorkerKind.ONLINE,
    )
    assert WorkerKind.ROOT_LOCAL == "root-local"
    assert WorkerKind.ONLINE == "online"
