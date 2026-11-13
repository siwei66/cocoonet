"""Immutable path records for Cocoonet-managed storage layouts."""

import os
from dataclasses import dataclass
from pathlib import Path

from cocoonet._storage.publication import StorageError

type PathInput = str | os.PathLike[str]


def resolve_physical_path(path: PathInput, /) -> Path:
    """Return an absolute, normalized, symlink-resolved physical path.

    Parameters
    ----------
    path
        String or string-producing path-like value to resolve.

    Returns
    -------
    pathlib.Path
        Canonical physical path. A missing suffix is appended to its nearest
        physically resolved existing ancestor without creating it.

    Raises
    ------
    TypeError
        If ``path`` is not a string-producing path-like value.
    ValueError
        If the path contains an embedded NUL character.
    StorageError
        If the operating system cannot resolve the path.

    Notes
    -----
    This function does not create, modify, rename, or remove filesystem
    entries.

    Node usage scope: Shared - Used by both root and worker storage.
    """
    if not isinstance(path, (str, os.PathLike)):
        observed_type = f"{type(path).__module__}.{type(path).__qualname__}"
        raise TypeError(
            "path must be str or os.PathLike[str]; "
            f"got {observed_type}."
        )

    try:
        lexical_path = os.fspath(path)
    except TypeError as error:
        observed_type = f"{type(path).__module__}.{type(path).__qualname__}"
        raise TypeError(
            "path must be str or os.PathLike[str]; "
            f"got {observed_type}."
        ) from error

    if not isinstance(lexical_path, str):
        observed_type = (
            f"{type(lexical_path).__module__}."
            f"{type(lexical_path).__qualname__}"
        )
        raise TypeError(
            "path.__fspath__() must return str; "
            f"got {observed_type}."
        )

    if "\x00" in lexical_path:
        raise ValueError("path must not contain an embedded NUL character.")

    # ``strict=False`` resolves every existing ancestor physically and retains
    # the normalized missing suffix without creating any filesystem entry.
    try:
        return Path(lexical_path).resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise StorageError(
            f"Failed to resolve physical path {lexical_path!r}."
        ) from error


def _normalized_path_parts(path: Path, /) -> tuple[str, ...]:
    """Return platform-normalized components from a canonical path.

    Parameters
    ----------
    path
        Canonical physical path whose components are compared.

    Returns
    -------
    tuple[str, ...]
        Components normalized according to the host platform's case rules.

    Notes
    -----
    Callers must supply canonical paths returned by
    ``resolve_physical_path()``. This helper performs no filesystem access.
    """
    return tuple(os.path.normcase(part) for part in path.parts)


def _paths_equal(first: Path, second: Path, /) -> bool:
    """Return whether two canonical paths identify the same location.

    Parameters
    ----------
    first
        First canonical physical path.
    second
        Second canonical physical path.

    Returns
    -------
    bool
        Whether their platform-normalized components are equal.

    Notes
    -----
    This is a pure comparison and performs no filesystem access.
    """
    return _normalized_path_parts(first) == _normalized_path_parts(second)


def _is_path_within(path: Path, parent: Path, /) -> bool:
    """Return whether one canonical path is strictly contained in another.

    Parameters
    ----------
    path
        Candidate canonical descendant.
    parent
        Candidate canonical parent.

    Returns
    -------
    bool
        ``True`` only for strict component-wise containment. Equality is
        excluded.

    Notes
    -----
    This is a pure comparison and performs no filesystem access.
    """
    path_parts = _normalized_path_parts(path)
    parent_parts = _normalized_path_parts(parent)

    return (
        len(path_parts) > len(parent_parts)
        and path_parts[: len(parent_parts)] == parent_parts
    )


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
