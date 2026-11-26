"""Prepare and open Cocoonet root node working directories."""

import shutil
from collections.abc import Iterable
from pathlib import Path

from cocoonet._storage.metadata import (
    CocoonetMetadata,
    NodeRole,
    current_compatibility_metadata,
    decode_cocoonet_metadata_csv,
    encode_cocoonet_metadata_csv,
)
from cocoonet._storage.paths import (
    PathInput,
    RootStorageLayout,
    _is_path_within,
    _paths_equal,
    build_root_storage_layout,
    validate_managed_work_dir,
    validate_work_dir_ownership,
)
from cocoonet._storage.publication import (
    StorageError,
    publish_staged_file,
    staged_file_path,
)
from cocoonet._type_validation import validate_declared_types


def prepare_work_directory(
    work_dir: PathInput,
    /,
    *,
    dataset_sources: Iterable[PathInput] = (),
    overwrite: bool = False,
) -> RootStorageLayout:
    """Prepare one fresh Cocoonet-managed root node working directory.

    Parameters
    ----------
    work_dir
        User-selected root node working directory.
    dataset_sources
        Complete configured collection of user-owned dataset source paths.
    overwrite
        Whether an existing nonempty working directory may have its contents
        cleared.

    Returns
    -------
    RootStorageLayout
        Canonical semantic layout for the prepared root node working
        directory.

    Raises
    ------
    TypeError
        If an argument has an incorrect runtime type.
    ValueError
        If the working directory is unsafe, conflicts with a dataset source,
        is an existing non-directory, or is nonempty without overwrite
        permission.
    StorageError
        If a required filesystem operation or metadata publication fails.

    Notes
    -----
    All ownership and protected-location checks finish before any persistent
    mutation. Fresh preparation creates only ``cocoonet_metadata.csv``,
    ``models_for_transfer/``, ``attempt_results/``, and ``logs/``.

    Clearing is confined to the canonical validated root node working
    directory. Directory symlinks and junctions are removed as entries rather
    than traversed into their targets. No rollback is promised after an
    operating-system failure during an explicitly authorized clear.

    Node usage scope: Root - Prepares fresh root-managed storage.
    """
    validate_declared_types(
        overwrite,
        expected_type=bool,
        field="overwrite",
    )

    canonical_work_dir, _ = validate_work_dir_ownership(
        work_dir,
        dataset_sources,
    )
    canonical_work_dir = validate_managed_work_dir(canonical_work_dir)
    layout = build_root_storage_layout(canonical_work_dir)

    metadata_bytes = encode_cocoonet_metadata_csv(
        CocoonetMetadata(
            storage_schema_version=1,
            node_role=NodeRole.ROOT,
            compatibility=current_compatibility_metadata(),
        )
    )

    try:
        if canonical_work_dir.exists():
            if not canonical_work_dir.is_dir():
                raise ValueError(
                    "Managed root node working directory must be a directory when it already exists; "
                    f"got {str(canonical_work_dir)!r}."
                )
            existing_entries = tuple(canonical_work_dir.iterdir())
        else:
            existing_entries = ()

        if existing_entries and not overwrite:
            raise ValueError(
                "Managed root node working directory must be empty when overwrite=False; "
                f"got nonempty directory {str(canonical_work_dir)!r}."
            )

        if overwrite:
            for entry in existing_entries:
                if entry.is_symlink():
                    entry.unlink()
                elif entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()

        canonical_work_dir.mkdir(parents=True, exist_ok=True)
        layout.models_for_transfer.mkdir()
        layout.attempt_results.mkdir()
        layout.logs.mkdir()

        metadata_staging = staged_file_path(layout.metadata_csv)
        metadata_staging.write_bytes(metadata_bytes)
    except ValueError:
        raise
    except OSError as error:
        raise StorageError(f"Failed to prepare root node working directory {str(canonical_work_dir)!r}.") from error

    publish_staged_file(layout.metadata_csv)
    return layout


def open_work_directory(
    work_dir: PathInput,
    /,
    *,
    dataset_sources: Iterable[PathInput] = (),
) -> tuple[RootStorageLayout, CocoonetMetadata]:
    """Open and validate an existing Cocoonet-managed root node working directory.

    Parameters
    ----------
    work_dir
        User-selected existing root node working directory.
    dataset_sources
        Complete configured collection of user-owned dataset source paths.

    Returns
    -------
    tuple[RootStorageLayout, CocoonetMetadata]
        Canonical root node layout and its validated persisted metadata.

    Raises
    ------
    TypeError
        If an argument has an incorrect runtime type.
    ValueError
        If the working directory is unsafe, conflicts with a dataset source,
        lacks its required Phase 2 layout, or contains invalid root metadata.
    StorageError
        If a required filesystem inspection or metadata read fails.

    Notes
    -----
    This operation is non-destructive. It does not create missing entries,
    remove stale staging, publish files, recover execution, or compare runtime
    compatibility. A stale metadata ``.partial`` entry is ignored and retained.

    Node usage scope: Root - Opens existing root-managed storage.
    """
    canonical_work_dir, _ = validate_work_dir_ownership(
        work_dir,
        dataset_sources,
    )
    canonical_work_dir = validate_managed_work_dir(canonical_work_dir)
    layout = build_root_storage_layout(canonical_work_dir)

    try:
        if not canonical_work_dir.exists():
            raise ValueError(f"Managed root node working directory must already exist: {str(canonical_work_dir)!r}.")
        if not canonical_work_dir.is_dir():
            raise ValueError(
                "Managed root node working directory must be an existing directory; "
                f"got {str(canonical_work_dir)!r}."
            )

        required_directories = (
            layout.models_for_transfer,
            layout.attempt_results,
            layout.logs,
        )
        for directory in required_directories:
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError(
                    "Existing managed root node working directory requires a real directory at " f"{str(directory)!r}."
                )

        if layout.metadata_csv.is_symlink() or not layout.metadata_csv.is_file():
            raise ValueError(
                "Existing managed root node working directory requires a final metadata file at "
                f"{str(layout.metadata_csv)!r}."
            )

        metadata_bytes = layout.metadata_csv.read_bytes()
    except ValueError:
        raise
    except OSError as error:
        raise StorageError(f"Failed to open root node working directory {str(canonical_work_dir)!r}.") from error

    metadata = decode_cocoonet_metadata_csv(metadata_bytes)
    if metadata.node_role is not NodeRole.ROOT:
        raise ValueError("Root metadata must declare node_role 'root'; " f"got {metadata.node_role.value!r}.")

    return layout, metadata


def _discard_incomplete_clone(
    work_dir: Path,
    /,
    *,
    retain_directory: bool,
) -> None:
    """Remove entries created by an unsuccessful work-directory clone."""
    if not work_dir.exists():
        return

    if not retain_directory:
        shutil.rmtree(work_dir)
        return

    for entry in work_dir.iterdir():
        if entry.is_symlink() or entry.is_junction():
            entry.unlink()
        elif entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _clone_work_directory(  # noqa: C901
    source_work_dir: Path,
    target_work_dir: PathInput,
    /,
    *,
    dataset_sources: tuple[Path, ...],
) -> RootStorageLayout:
    """Copy validated root-managed storage into a missing or empty destination."""
    source_layout, _ = open_work_directory(
        source_work_dir,
        dataset_sources=dataset_sources,
    )
    canonical_target, _ = validate_work_dir_ownership(
        target_work_dir,
        dataset_sources,
    )
    canonical_target = validate_managed_work_dir(canonical_target)

    if (
        _paths_equal(source_layout.work_dir, canonical_target)
        or _is_path_within(canonical_target, source_layout.work_dir)
        or _is_path_within(source_layout.work_dir, canonical_target)
    ):
        raise ValueError("Clone source and destination must be separate non-nested " "working directories.")

    try:
        target_preexisted = canonical_target.exists()
        if target_preexisted and (
            canonical_target.is_symlink() or canonical_target.is_junction() or not canonical_target.is_dir()
        ):
            raise ValueError(
                "Clone destination must be a missing or empty real directory; " f"got {str(canonical_target)!r}."
            )
        if target_preexisted and any(canonical_target.iterdir()):
            raise ValueError("Clone destination must be empty; " f"got nonempty directory {str(canonical_target)!r}.")

        # Links could preserve external ownership or retain the source path.
        for directory, directory_names, file_names in source_layout.work_dir.walk():
            for entry_name in (*directory_names, *file_names):
                entry = directory / entry_name
                if entry.is_symlink() or entry.is_junction():
                    raise ValueError("Clone source must not contain symlinks or junctions; " f"got {str(entry)!r}.")
    except ValueError:
        raise
    except OSError as error:
        raise StorageError("Failed to inspect work-directory clone boundaries.") from error

    try:
        shutil.copytree(
            source_layout.work_dir,
            canonical_target,
            symlinks=True,
            copy_function=shutil.copy2,
            dirs_exist_ok=target_preexisted,
        )
    except OSError as error:
        try:
            _discard_incomplete_clone(
                canonical_target,
                retain_directory=target_preexisted,
            )
        except OSError as cleanup_error:
            raise StorageError(
                "Failed to copy the work directory and remove the incomplete " f"destination {str(canonical_target)!r}."
            ) from cleanup_error
        raise StorageError(
            "Failed to clone root node working directory "
            f"{str(source_layout.work_dir)!r} to {str(canonical_target)!r}."
        ) from error

    return build_root_storage_layout(canonical_target)
