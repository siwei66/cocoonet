"""Tests for focused Cocoonet storage publication support."""

from cocoonet._storage.publication import StorageError


def test_storage_error_is_plain_runtime_error() -> None:
    """Preserve ordinary exception behavior without adding retry state."""
    error = StorageError("failed to publish C:/work/result.csv")

    assert StorageError.__base__ is RuntimeError
    assert error.args == ("failed to publish C:/work/result.csv",)
