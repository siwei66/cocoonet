"""Fresh preparation of Cocoonet root storage directories."""

import shutil
from collections.abc import Iterable

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
    build_root_storage_layout,
    validate_managed_root,
    validate_root_ownership,
)
from cocoonet._storage.publication import (
    StorageError,
    publish_staged_file,
    staged_file_path,
)
from cocoonet._type_validation import validate_declared_types


def prepare_root_directory(
    root: PathInput,
    /,
    *,
    dataset_sources: Iterable[PathInput] = (),
    overwrite: bool = False,
) -> RootStorageLayout:
    """Prepare one fresh Cocoonet-managed root directory.

    Parameters
    ----------
    root
        User-selected root working directory.
    dataset_sources
        Complete configured collection of user-owned dataset source paths.
    overwrite
        Whether an existing nonempty root may have its contents cleared.

    Returns
    -------
    RootStorageLayout
        Canonical semantic layout for the prepared root.

    Raises
    ------
    TypeError
        If an argument has an incorrect runtime type.
    ValueError
        If the root is unsafe, conflicts with a dataset source, is an existing
        non-directory, or is nonempty without overwrite permission.
    StorageError
        If a required filesystem operation or metadata publication fails.

    Notes
    -----
    All ownership and protected-location checks finish before any persistent
    mutation. Fresh preparation creates only ``cocoonet_metadata.csv``,
    ``models_for_transfer/``, ``attempt_results/``, and ``logs/``.

    Clearing is confined to the canonical validated root. Directory symlinks
    and junctions are removed as entries rather than traversed into their
    targets. No rollback is promised after an operating-system failure during
    an explicitly authorized clear.

    Node usage scope: Root - Prepares fresh root-managed storage.
    """
    validate_declared_types(
        overwrite,
        expected_type=bool,
        field="overwrite",
    )

    canonical_root, _ = validate_root_ownership(
        root,
        dataset_sources,
    )
    canonical_root = validate_managed_root(canonical_root)
    layout = build_root_storage_layout(canonical_root)

    metadata_bytes = encode_cocoonet_metadata_csv(
        CocoonetMetadata(
            storage_schema_version=1,
            node_role=NodeRole.ROOT,
            compatibility=current_compatibility_metadata(),
        )
    )

    try:
        if canonical_root.exists():
            if not canonical_root.is_dir():
                raise ValueError(
                    "Managed root must be a directory when it already exists; " f"got {str(canonical_root)!r}."
                )
            existing_entries = tuple(canonical_root.iterdir())
        else:
            existing_entries = ()

        if existing_entries and not overwrite:
            raise ValueError(
                "Managed root must be empty when overwrite=False; " f"got nonempty directory {str(canonical_root)!r}."
            )

        if overwrite:
            for entry in existing_entries:
                if entry.is_symlink():
                    entry.unlink()
                elif entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()

        canonical_root.mkdir(parents=True, exist_ok=True)
        layout.models_for_transfer.mkdir()
        layout.attempt_results.mkdir()
        layout.logs.mkdir()

        metadata_staging = staged_file_path(layout.metadata_csv)
        metadata_staging.write_bytes(metadata_bytes)
    except ValueError:
        raise
    except OSError as error:
        raise StorageError(f"Failed to prepare root directory {str(canonical_root)!r}.") from error

    publish_staged_file(layout.metadata_csv)
    return layout


def open_root_directory(
    root: PathInput,
    /,
    *,
    dataset_sources: Iterable[PathInput] = (),
) -> tuple[RootStorageLayout, CocoonetMetadata]:
    """Open and validate one existing Cocoonet-managed root directory.

    Parameters
    ----------
    root
        User-selected existing root working directory.
    dataset_sources
        Complete configured collection of user-owned dataset source paths.

    Returns
    -------
    tuple[RootStorageLayout, CocoonetMetadata]
        Canonical root layout and its validated persisted metadata.

    Raises
    ------
    TypeError
        If an argument has an incorrect runtime type.
    ValueError
        If the root is unsafe, conflicts with a dataset source, lacks its
        required Phase 2 layout, or contains invalid root metadata.
    StorageError
        If a required filesystem inspection or metadata read fails.

    Notes
    -----
    This operation is non-destructive. It does not create missing entries,
    remove stale staging, publish files, recover execution, or compare runtime
    compatibility. A stale metadata ``.partial`` entry is ignored and retained.

    Node usage scope: Root - Opens existing root-managed storage.
    """
    canonical_root, _ = validate_root_ownership(
        root,
        dataset_sources,
    )
    canonical_root = validate_managed_root(canonical_root)
    layout = build_root_storage_layout(canonical_root)

    try:
        if not canonical_root.exists():
            raise ValueError(f"Managed root must already exist: {str(canonical_root)!r}.")
        if not canonical_root.is_dir():
            raise ValueError("Managed root must be an existing directory; " f"got {str(canonical_root)!r}.")

        required_directories = (
            layout.models_for_transfer,
            layout.attempt_results,
            layout.logs,
        )
        for directory in required_directories:
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("Existing managed root requires a real directory at " f"{str(directory)!r}.")

        if layout.metadata_csv.is_symlink() or not layout.metadata_csv.is_file():
            raise ValueError(
                "Existing managed root requires a final metadata file at " f"{str(layout.metadata_csv)!r}."
            )

        metadata_bytes = layout.metadata_csv.read_bytes()
    except ValueError:
        raise
    except OSError as error:
        raise StorageError(f"Failed to open root directory {str(canonical_root)!r}.") from error

    metadata = decode_cocoonet_metadata_csv(metadata_bytes)
    if metadata.node_role is not NodeRole.ROOT:
        raise ValueError("Root metadata must declare node_role 'root'; " f"got {metadata.node_role.value!r}.")

    return layout, metadata
