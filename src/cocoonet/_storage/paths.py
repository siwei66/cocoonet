"""Immutable path records for Cocoonet-managed storage layouts."""

import os
import site
import sys
import sysconfig
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from cocoonet._contracts.identity import DatasetId, ModelId, ParentId
from cocoonet._storage.publication import StorageError
from cocoonet._type_validation import validate_args

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


def _validate_canonical_dataset_source_ownership(
    root: Path,
    source: Path,
    /,
) -> None:
    """Validate one canonical source against one canonical managed root.

    Parameters
    ----------
    root
        Canonical managed-root path.
    source
        Canonical user-owned dataset-source path.

    Raises
    ------
    ValueError
        If the source equals or is contained within the managed root.

    Notes
    -----
    A managed root contained within the source is permitted. This pure helper
    performs no filesystem access or protected-location validation.

    Node usage scope: Root - Protects user-owned dataset sources.
    """
    if _paths_equal(source, root) or _is_path_within(source, root):
        raise ValueError(
            f"Dataset source {str(source)!r} must not equal or be contained " f"within managed root {str(root)!r}."
        )


def validate_dataset_source_ownership(
    root: PathInput,
    source: PathInput,
    /,
) -> tuple[Path, Path]:
    """Validate one managed-root and dataset-source ownership boundary.

    Parameters
    ----------
    root
        Absolute or relative managed-root path.
    source
        Absolute or relative user-owned dataset-source path.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Canonical root and source paths.

    Raises
    ------
    TypeError
        If either input is not a string-producing path-like value.
    ValueError
        If the source equals or is contained within the root.
    StorageError
        If either path cannot be physically resolved.

    Notes
    -----
    This function performs no filesystem mutation. Protected-root validation
    is a separate boundary.

    Node usage scope: Root - Protects user-owned dataset sources.
    """
    canonical_root = resolve_physical_path(root)
    canonical_source = resolve_physical_path(source)
    _validate_canonical_dataset_source_ownership(
        canonical_root,
        canonical_source,
    )
    return canonical_root, canonical_source


def validate_root_ownership(
    root: PathInput,
    sources: Iterable[PathInput],
    /,
) -> tuple[Path, tuple[Path, ...]]:
    """Validate a root against a complete dataset-source collection.

    Parameters
    ----------
    root
        Absolute or relative managed-root path.
    sources
        Dataset-source paths to materialize and validate as one collection.

    Returns
    -------
    tuple[pathlib.Path, tuple[pathlib.Path, ...]]
        Canonical root and ordered canonical source paths.

    Raises
    ------
    TypeError
        If the root, collection, or a source has an incorrect runtime type.
    ValueError
        If a source conflicts with the root or duplicates another canonical
        source.
    StorageError
        If a path cannot be physically resolved.

    Notes
    -----
    The complete collection is materialized and canonicalized before ownership
    checks. This function performs no filesystem mutation or protected-root
    validation.

    Node usage scope: Root - Protects user-owned dataset sources.
    """
    canonical_root = resolve_physical_path(root)

    if isinstance(sources, (str, bytes, os.PathLike)) or not isinstance(
        sources,
        Iterable,
    ):
        observed_type = f"{type(sources).__module__}.{type(sources).__qualname__}"
        raise TypeError("sources must be an iterable of path inputs; " f"got {observed_type}.")

    materialized_sources = tuple(sources)
    canonical_sources = tuple(resolve_physical_path(source) for source in materialized_sources)

    for index, source in enumerate(canonical_sources):
        for previous_index, previous in enumerate(canonical_sources[:index]):
            if _paths_equal(source, previous):
                raise ValueError(
                    f"sources[{index}] duplicates canonical source " f"sources[{previous_index}] at {str(source)!r}."
                )

        _validate_canonical_dataset_source_ownership(
            canonical_root,
            source,
        )

    return canonical_root, canonical_sources


@validate_args
def _validate_dataset_id(value: str, /) -> None:
    """Validate the exact managed-path representation of a dataset ID.

    Parameters
    ----------
    value
        Candidate dataset identifier.

    Raises
    ------
    TypeError
        If the candidate is not a string.
    ValueError
        If the candidate is not exactly 12 lowercase hexadecimal characters.

    Notes
    -----
    This function validates representation only. It does not generate an
    identifier, check collisions, or inspect the filesystem.

    Node usage scope: Shared - Used by root and worker path builders.
    """
    if len(value) != 12 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("dataset_id must be exactly 12 lowercase hexadecimal characters; " f"got {value!r}.")


@validate_args
def _validate_parent_id(value: str, /) -> None:
    """Validate the exact managed-path representation of a parent ID.

    Parameters
    ----------
    value
        Candidate parent identifier.

    Raises
    ------
    TypeError
        If the candidate is not a string.
    ValueError
        If the candidate is not exactly 16 lowercase hexadecimal characters.

    Notes
    -----
    This function validates representation only. It does not generate an
    identifier, check collisions, or inspect the filesystem.

    Node usage scope: Shared - Used by root and worker path builders.
    """
    if len(value) != 16 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("parent_id must be exactly 16 lowercase hexadecimal characters; " f"got {value!r}.")


@validate_args
def _validate_positive_decimal_identifier(
    value: str,
    *,
    field: str,
) -> None:
    """Validate canonical positive decimal text used in managed paths.

    Parameters
    ----------
    value
        Candidate model or tile identifier.
    field
        Identifier field included in diagnostics.

    Raises
    ------
    TypeError
        If either argument is not a string.
    ValueError
        If the candidate is not canonical positive decimal text.

    Notes
    -----
    Valid values begin with ``1`` through ``9`` and contain only decimal
    digits. Signs, zero, and zero padding are rejected. This function does not
    generate identifiers or inspect the filesystem.

    Node usage scope: Shared - Used by root and worker path builders.
    """
    if not value or value[0] not in "123456789" or any(character not in "0123456789" for character in value[1:]):
        raise ValueError(
            f"{field} must be canonical positive decimal text beginning at 1 "
            f"with no sign or zero padding; got {value!r}."
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


def build_root_storage_layout(root: PathInput, /) -> RootStorageLayout:
    """Build the semantic storage paths for one root directory.

    Parameters
    ----------
    root
        Absolute or relative root storage path.

    Returns
    -------
    RootStorageLayout
        Immutable collection of canonical root storage paths.

    Raises
    ------
    TypeError
        If ``root`` is not a string-producing path-like value.
    ValueError
        If the root path contains an embedded NUL character.
    StorageError
        If the root path cannot be physically resolved.

    Notes
    -----
    This function creates no filesystem entries and validates no artifact
    contents. Protected-location and dataset-source ownership validation remain
    separate caller boundaries.

    Node usage scope: Root - Defines the root persistence layout.
    """
    canonical_root = resolve_physical_path(root)
    logs = canonical_root / "logs"

    return RootStorageLayout(
        root=canonical_root,
        metadata_csv=canonical_root / "cocoonet_metadata.csv",
        root_data_reference_xlsx=(canonical_root / "root_data_reference.xlsx"),
        partition_schedules_xlsx=(canonical_root / "partition_schedules.xlsx"),
        execution_configuration_csv=(canonical_root / "execution_configuration.csv"),
        execution_state_csv=canonical_root / "execution_state.csv",
        models_for_transfer=canonical_root / "models_for_transfer",
        attempt_results=canonical_root / "attempt_results",
        logs=logs,
        root_running_log=logs / "root_running.log",
        root_error_log=logs / "root_error.log",
    )


def build_worker_storage_layout(root: PathInput, /) -> WorkerStorageLayout:
    """Build the top-level semantic storage paths for one worker.

    Parameters
    ----------
    root
        Absolute or relative worker storage path.

    Returns
    -------
    WorkerStorageLayout
        Immutable collection of canonical worker storage paths.

    Raises
    ------
    TypeError
        If ``root`` is not a string-producing path-like value.
    ValueError
        If the root path contains an embedded NUL character.
    StorageError
        If the root path cannot be physically resolved.

    Notes
    -----
    This function creates no filesystem entries and validates no artifact
    contents.

    Node usage scope: Worker - Defines the top-level worker persistence layout.
    """
    canonical_root = resolve_physical_path(root)

    return WorkerStorageLayout(
        root=canonical_root,
        datasets=canonical_root / "datasets",
        models=canonical_root / "models",
        results=canonical_root / "results",
    )


@validate_args
def build_worker_parent_storage_layout(
    layout: WorkerStorageLayout,
    parent_id: ParentId,
    /,
) -> WorkerParentStorageLayout:
    """Build one parent's semantic result paths within a worker layout.

    Parameters
    ----------
    layout
        Valid worker storage layout containing the parent result.
    parent_id
        Parent identifier represented as 16 lowercase hexadecimal characters.

    Returns
    -------
    WorkerParentStorageLayout
        Immutable collection of parent-specific worker result paths.

    Raises
    ------
    TypeError
        If an argument has an incorrect runtime type.
    ValueError
        If the parent identifier is invalid or the results directory escapes
        the worker root.
    StorageError
        If a supplied layout path cannot be physically resolved.

    Notes
    -----
    This function creates no filesystem entries and validates no artifact
    contents.

    Node usage scope: Worker - Defines one parent's worker result layout.
    """
    _validate_parent_id(parent_id)

    canonical_root = resolve_physical_path(layout.root)
    canonical_results = resolve_physical_path(layout.results)
    if not _is_path_within(canonical_results, canonical_root):
        raise ValueError(
            f"Worker results path {str(canonical_results)!r} must be contained "
            f"within worker root {str(canonical_root)!r}."
        )

    parent_root = canonical_results / f"parent_{parent_id}"

    return WorkerParentStorageLayout(
        parent_root=parent_root,
        metadata_csv=parent_root / "cocoonet_metadata.csv",
        worker_data_reference_xlsx=(parent_root / "worker_data_reference.xlsx"),
        partition_schedules_xlsx=(parent_root / "partition_schedules.xlsx"),
        execution_configuration_csv=(parent_root / "execution_configuration.csv"),
        execution_state_csv=parent_root / "execution_state.csv",
        attempt_results=parent_root / "attempt_results",
    )


@validate_args
def root_trusted_model_dill_path(
    layout: RootStorageLayout,
    model_id: ModelId,
    /,
) -> Path:
    """Build the trusted root-side model path for one model.

    Parameters
    ----------
    layout
        Root storage layout containing the trusted model.
    model_id
        Canonical positive decimal model identifier.

    Returns
    -------
    pathlib.Path
        Exact ``models_for_transfer/model_<model_id>.dill`` path.

    Raises
    ------
    TypeError
        If an argument has an incorrect runtime type.
    ValueError
        If the model identifier is invalid or the model directory escapes the
        root.
    StorageError
        If a supplied layout path cannot be physically resolved.

    Notes
    -----
    This helper locates trusted model storage only. It performs no
    serialization and does not authorize dill use for public workers.

    Node usage scope: Root - Locates trusted root-side model payloads.
    """
    _validate_positive_decimal_identifier(
        model_id,
        field="model_id",
    )

    canonical_root = resolve_physical_path(layout.root)
    canonical_models = resolve_physical_path(layout.models_for_transfer)
    if not _is_path_within(canonical_models, canonical_root):
        raise ValueError(
            f"Root model directory {str(canonical_models)!r} must be " f"contained within root {str(canonical_root)!r}."
        )

    return canonical_models / f"model_{model_id}.dill"


@validate_args
def worker_dataset_directory(
    layout: WorkerStorageLayout,
    dataset_id: DatasetId,
    /,
) -> Path:
    """Build the worker directory for one synchronized dataset.

    Parameters
    ----------
    layout
        Worker storage layout containing the dataset.
    dataset_id
        Dataset identifier represented as 12 lowercase hexadecimal characters.

    Returns
    -------
    pathlib.Path
        Exact ``datasets/dataset_<dataset_id>`` directory.

    Raises
    ------
    TypeError
        If an argument has an incorrect runtime type.
    ValueError
        If the dataset identifier is invalid or the datasets directory escapes
        the worker root.
    StorageError
        If a supplied layout path cannot be physically resolved.

    Notes
    -----
    This helper creates no directory and performs no dataset synchronization.

    Node usage scope: Worker - Locates synchronized dataset storage.
    """
    _validate_dataset_id(dataset_id)

    canonical_root = resolve_physical_path(layout.root)
    canonical_datasets = resolve_physical_path(layout.datasets)
    if not _is_path_within(canonical_datasets, canonical_root):
        raise ValueError(
            f"Worker datasets directory {str(canonical_datasets)!r} must be "
            f"contained within worker root {str(canonical_root)!r}."
        )

    return canonical_datasets / f"dataset_{dataset_id}"


@validate_args
def worker_trusted_model_dill_path(
    layout: WorkerStorageLayout,
    parent_id: ParentId,
    model_id: ModelId,
    /,
) -> Path:
    """Build the trusted worker-side model path for one parent and model.

    Parameters
    ----------
    layout
        Worker storage layout containing the trusted model.
    parent_id
        Parent identifier represented as 16 lowercase hexadecimal characters.
    model_id
        Canonical positive decimal model identifier.

    Returns
    -------
    pathlib.Path
        Exact ``models/parent_<parent_id>/model_<model_id>.dill`` path.

    Raises
    ------
    TypeError
        If an argument has an incorrect runtime type.
    ValueError
        If an identifier is invalid or the models directory escapes the worker
        root.
    StorageError
        If a supplied layout path cannot be physically resolved.

    Notes
    -----
    This helper locates trusted worker model storage only. It performs no
    serialization and does not authorize dill use for public workers.

    Node usage scope: Worker - Locates trusted worker-side model payloads.
    """
    _validate_parent_id(parent_id)
    _validate_positive_decimal_identifier(
        model_id,
        field="model_id",
    )

    canonical_root = resolve_physical_path(layout.root)
    canonical_models = resolve_physical_path(layout.models)
    if not _is_path_within(canonical_models, canonical_root):
        raise ValueError(
            f"Worker models directory {str(canonical_models)!r} must be "
            f"contained within worker root {str(canonical_root)!r}."
        )

    return canonical_models / f"parent_{parent_id}" / f"model_{model_id}.dill"


@validate_args
def root_bundle_publication_paths(
    layout: RootStorageLayout,
    model_id: ModelId,
    /,
) -> BundlePublicationPaths:
    """Build root staging and final paths for one model result bundle.

    Parameters
    ----------
    layout
        Root storage layout containing the accepted bundle.
    model_id
        Canonical positive decimal model identifier.

    Returns
    -------
    BundlePublicationPaths
        Sibling ``model_<model_id>.staging`` and ``model_<model_id>`` paths.

    Raises
    ------
    TypeError
        If an argument has an incorrect runtime type.
    ValueError
        If the model identifier is invalid or the attempt-results directory
        escapes the root.
    StorageError
        If a supplied layout path cannot be physically resolved.

    Notes
    -----
    This helper performs no publication, validation, acceptance, cleanup, or
    filesystem mutation.

    Node usage scope: Root - Locates selected bundle publication paths.
    """
    _validate_positive_decimal_identifier(
        model_id,
        field="model_id",
    )

    canonical_root = resolve_physical_path(layout.root)
    canonical_attempt_results = resolve_physical_path(layout.attempt_results)
    if not _is_path_within(canonical_attempt_results, canonical_root):
        raise ValueError(
            f"Root attempt-results directory "
            f"{str(canonical_attempt_results)!r} must be contained within root "
            f"{str(canonical_root)!r}."
        )

    model_name = f"model_{model_id}"
    return BundlePublicationPaths(
        staging=canonical_attempt_results / f"{model_name}.staging",
        final=canonical_attempt_results / model_name,
    )


@validate_args
def worker_bundle_publication_paths(
    parent_layout: WorkerParentStorageLayout,
    model_id: ModelId,
    /,
) -> BundlePublicationPaths:
    """Build worker staging and final paths for one model result bundle.

    Parameters
    ----------
    parent_layout
        Parent-specific worker layout containing the bundle.
    model_id
        Canonical positive decimal model identifier.

    Returns
    -------
    BundlePublicationPaths
        Sibling ``model_<model_id>.staging`` and ``model_<model_id>`` paths.

    Raises
    ------
    TypeError
        If an argument has an incorrect runtime type.
    ValueError
        If the model identifier is invalid or the attempt-results directory
        escapes the parent result root.
    StorageError
        If a supplied layout path cannot be physically resolved.

    Notes
    -----
    This helper performs no publication, validation, delivery, cleanup, or
    filesystem mutation.

    Node usage scope: Worker - Locates one bundle's publication paths.
    """
    _validate_positive_decimal_identifier(
        model_id,
        field="model_id",
    )

    canonical_parent_root = resolve_physical_path(parent_layout.parent_root)
    canonical_attempt_results = resolve_physical_path(parent_layout.attempt_results)
    if not _is_path_within(
        canonical_attempt_results,
        canonical_parent_root,
    ):
        raise ValueError(
            f"Worker attempt-results directory "
            f"{str(canonical_attempt_results)!r} must be contained within "
            f"parent root {str(canonical_parent_root)!r}."
        )

    model_name = f"model_{model_id}"
    return BundlePublicationPaths(
        staging=canonical_attempt_results / f"{model_name}.staging",
        final=canonical_attempt_results / model_name,
    )
