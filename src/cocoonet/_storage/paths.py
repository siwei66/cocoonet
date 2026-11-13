"""Immutable path records for Cocoonet-managed storage layouts."""

import os
from dataclasses import dataclass
from pathlib import Path

type PathInput = str | os.PathLike[str]


@dataclass(frozen=True, slots=True)
class RootStorageLayout:
    """Collect the semantic paths belonging to one root storage directory.

    Notes
    -----
    This record performs no validation or filesystem operations.
    """

    root: Path
    metadata_csv: Path
    root_data_reference_xlsx: Path
    partition_schedules_xlsx: Path
    execution_configuration_csv: Path
    execution_state_csv: Path
    models_for_transfer: Path
    attempt_results: Path
    logs: Path
    root_running_log: Path
    root_error_log: Path


@dataclass(frozen=True, slots=True)
class WorkerStorageLayout:
    """Collect the top-level semantic paths belonging to one worker.

    Notes
    -----
    This record performs no validation or filesystem operations.
    """

    root: Path
    datasets: Path
    models: Path
    results: Path


@dataclass(frozen=True, slots=True)
class WorkerParentStorageLayout:
    """Collect the semantic worker paths belonging to one parent execution.

    Notes
    -----
    This record performs no validation or filesystem operations.
    """

    parent_root: Path
    metadata_csv: Path
    worker_data_reference_xlsx: Path
    partition_schedules_xlsx: Path
    execution_configuration_csv: Path
    execution_state_csv: Path
    attempt_results: Path


@dataclass(frozen=True, slots=True)
class BundlePublicationPaths:
    """Collect the staging and final paths for one result bundle.

    Notes
    -----
    This record performs no validation or filesystem operations.
    """

    staging: Path
    final: Path
