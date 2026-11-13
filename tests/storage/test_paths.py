"""Tests for immutable Cocoonet storage path records."""

from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from cocoonet._storage.paths import (
    BundlePublicationPaths,
    RootStorageLayout,
    WorkerParentStorageLayout,
    WorkerStorageLayout,
)


def test_path_records_preserve_exact_immutable_slotted_schemas(
    tmp_path: Path,
) -> None:
    """Preserve supplied paths without validation, mutation, or I/O."""
    candidate = tmp_path / "candidate"

    records: tuple[
        RootStorageLayout
        | WorkerStorageLayout
        | WorkerParentStorageLayout
        | BundlePublicationPaths,
        ...,
    ] = (
        RootStorageLayout(
            candidate,
            candidate,
            candidate,
            candidate,
            candidate,
            candidate,
            candidate,
            candidate,
            candidate,
            candidate,
            candidate,
        ),
        WorkerStorageLayout(
            candidate,
            candidate,
            candidate,
            candidate,
        ),
        WorkerParentStorageLayout(
            candidate,
            candidate,
            candidate,
            candidate,
            candidate,
            candidate,
            candidate,
        ),
        BundlePublicationPaths(
            candidate,
            candidate,
        ),
    )
    expected_fields = (
        (
            "root",
            "metadata_csv",
            "root_data_reference_xlsx",
            "partition_schedules_xlsx",
            "execution_configuration_csv",
            "execution_state_csv",
            "models_for_transfer",
            "attempt_results",
            "logs",
            "root_running_log",
            "root_error_log",
        ),
        (
            "root",
            "datasets",
            "models",
            "results",
        ),
        (
            "parent_root",
            "metadata_csv",
            "worker_data_reference_xlsx",
            "partition_schedules_xlsx",
            "execution_configuration_csv",
            "execution_state_csv",
            "attempt_results",
        ),
        (
            "staging",
            "final",
        ),
    )

    assert not candidate.exists()

    for record, field_names in zip(records, expected_fields, strict=True):
        assert tuple(field.name for field in fields(record)) == field_names
        assert all(getattr(record, field_name) == candidate for field_name in field_names)
        assert not hasattr(record, "__dict__")

        with pytest.raises(FrozenInstanceError):
            record.__setattr__(field_names[0], tmp_path / "replacement")

    assert not candidate.exists()
