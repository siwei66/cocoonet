"""Immutable path records for Cocoonet-managed storage layouts."""

import os
import site
import sys
import sysconfig
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
        raise TypeError("path must be str or os.PathLike[str]; " f"got {observed_type}.")

    try:
        lexical_path = os.fspath(path)
    except TypeError as error:
        observed_type = f"{type(path).__module__}.{type(path).__qualname__}"
        raise TypeError("path must be str or os.PathLike[str]; " f"got {observed_type}.") from error

    if not isinstance(lexical_path, str):
        observed_type = f"{type(lexical_path).__module__}." f"{type(lexical_path).__qualname__}"
        raise TypeError("path.__fspath__() must return str; " f"got {observed_type}.")

    if "\x00" in lexical_path:
        raise ValueError("path must not contain an embedded NUL character.")

    # ``strict=False`` resolves every existing ancestor physically and retains
    # the normalized missing suffix without creating any filesystem entry.
    try:
        return Path(lexical_path).resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise StorageError(f"Failed to resolve physical path {lexical_path!r}.") from error


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

    return len(path_parts) > len(parent_parts) and path_parts[: len(parent_parts)] == parent_parts


def discover_protected_locations() -> tuple[Path, ...]:
    """Discover canonical filesystem locations unsafe for managed roots.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Deterministic, deduplicated canonical installation, package,
        source-checkout, and filesystem-anchor paths.

    Raises
    ------
    StorageError
        If a discovered location cannot be physically resolved.

    Notes
    -----
    Discovery performs no filesystem mutation. A source repository is included
    only when Cocoonet is imported from this project's configured ``src``
    checkout layout.

    Node usage scope: Root - Protects user-selected managed storage.
    """
    raw_candidates: list[PathInput] = [
        sys.prefix,
        sys.exec_prefix,
        sys.base_prefix,
        sys.base_exec_prefix,
        sysconfig.get_path("purelib"),
        sysconfig.get_path("platlib"),
    ]

    # Protect both environment-wide and per-user package installation paths.
    raw_candidates.extend(
        sorted(
            site.getsitepackages(),
            key=lambda path: os.path.normcase(path),
        )
    )
    raw_candidates.append(site.getusersitepackages())

    package_directory = resolve_physical_path(Path(__file__).parent.parent)
    raw_candidates.append(package_directory)

    # The configured source layout is ``<repository>/src/cocoonet``. Protect
    # the complete repository when the imported package has that relationship.
    source_root = package_directory.parent.parent
    if (
        package_directory.parent.name == "src"
        and (source_root / "pyproject.toml").is_file()
        and _paths_equal(
            package_directory,
            resolve_physical_path(source_root / "src" / "cocoonet"),
        )
    ):
        raw_candidates.append(source_root)

    canonical_candidates = [resolve_physical_path(candidate) for candidate in raw_candidates]
    anchors = sorted(
        (resolve_physical_path(Path(candidate.anchor)) for candidate in canonical_candidates),
        key=lambda path: os.path.normcase(os.fspath(path)),
    )

    # Preserve deterministic category order while collapsing physical aliases.
    protected: list[Path] = []
    for candidate in (*anchors, *canonical_candidates):
        if not any(_paths_equal(candidate, existing) for existing in protected):
            protected.append(candidate)

    return tuple(protected)


def _validate_managed_root_against(
    root: Path,
    protected_locations: tuple[Path, ...],
    /,
) -> None:
    """Validate one canonical root against explicit protected locations.

    Parameters
    ----------
    root
        Canonical candidate managed root.
    protected_locations
        Canonical locations that the candidate must not endanger.

    Raises
    ------
    ValueError
        If the candidate equals a filesystem anchor or equals, contains, or is
        contained within another protected location.

    Notes
    -----
    This pure comparison boundary performs no filesystem access. Filesystem
    anchors reject equality only; other protected locations reject all three
    containment directions.

    Node usage scope: Root - Protects user-selected managed storage.
    """
    for protected in protected_locations:
        is_filesystem_anchor = _paths_equal(
            protected,
            Path(protected.anchor),
        )

        if is_filesystem_anchor:
            if _paths_equal(root, protected):
                raise ValueError(f"Managed root {str(root)!r} must not equal filesystem " f"anchor {str(protected)!r}.")
            continue

        if _paths_equal(root, protected) or _is_path_within(root, protected) or _is_path_within(protected, root):
            raise ValueError(
                f"Managed root {str(root)!r} must not equal, contain, or be "
                f"contained within protected location {str(protected)!r}."
            )


def validate_managed_root(root: PathInput, /) -> Path:
    """Canonicalize and validate a user-specified managed root.

    Parameters
    ----------
    root
        Absolute or relative user-specified working directory. Relative paths
        resolve from the current working directory.

    Returns
    -------
    pathlib.Path
        Canonical validated managed-root path.

    Raises
    ------
    TypeError
        If ``root`` is not a string-producing path-like value.
    ValueError
        If the path is invalid or conflicts with a protected location.
    StorageError
        If physical resolution or protected-location discovery fails.

    Notes
    -----
    This function performs no filesystem mutation. Internally selected default
    work directories do not use this user-input validation boundary.

    Node usage scope: Root - Protects user-selected managed storage.
    """
    canonical_root = resolve_physical_path(root)
    _validate_managed_root_against(
        canonical_root,
        discover_protected_locations(),
    )
    return canonical_root


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
