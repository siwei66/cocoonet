"""Focused atomic-publication support for Cocoonet storage."""

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from cocoonet._storage.paths import BundlePublicationPaths


class StorageError(RuntimeError):
    """Report an operational storage failure without defining retry policy.

    Notes
    -----
    Storage operations provide the operation and exact target path in the
    exception message and chain the underlying exception. This exception does
    not decide whether an operation is retried.

    Node usage scope: Shared - Used by both root and worker storage.
    """


def staged_file_path(
    final_path: str | os.PathLike[str],
    /,
) -> Path:
    """Return the exact same-directory staging path for one final file.

    Parameters
    ----------
    final_path
        Final authoritative file path.

    Returns
    -------
    pathlib.Path
        Sibling path formed by appending ``.partial`` to the final filename.

    Raises
    ------
    TypeError
        If ``final_path`` is not a string-producing path-like value.
    ValueError
        If the path contains an embedded NUL character, has no filename, or
        already ends in ``.partial``.

    Notes
    -----
    This function performs lexical path calculation only. It does not inspect,
    create, validate, publish, replace, or remove a filesystem entry.

    Node usage scope: Shared - Used by root and worker file publication.
    """
    if not isinstance(final_path, (str, os.PathLike)):
        observed_type = f"{type(final_path).__module__}." f"{type(final_path).__qualname__}"
        raise TypeError("final_path must be str or os.PathLike[str]; " f"got {observed_type}.")

    try:
        lexical_path = os.fspath(final_path)
    except TypeError as error:
        observed_type = f"{type(final_path).__module__}." f"{type(final_path).__qualname__}"
        raise TypeError("final_path must be a string-producing path-like value; " f"got {observed_type}.") from error

    if not isinstance(lexical_path, str):
        observed_type = f"{type(lexical_path).__module__}." f"{type(lexical_path).__qualname__}"
        raise TypeError("final_path.__fspath__() must return str; " f"got {observed_type}.")

    if "\x00" in lexical_path:
        raise ValueError("final_path must not contain an embedded NUL character.")

    final = Path(lexical_path)
    if final.name in ("", ".", ".."):
        raise ValueError(f"final_path must identify a file name; got {lexical_path!r}.")

    if final.name.endswith(".partial"):
        raise ValueError(f"final_path must not already end in '.partial'; " f"got {lexical_path!r}.")

    return final.with_name(f"{final.name}.partial")


def publish_staged_file(
    final_path: str | os.PathLike[str],
    /,
    *,
    replace: bool = False,
) -> None:
    """Synchronize and atomically publish one exact staging file.

    Parameters
    ----------
    final_path
        Final authoritative file path. Its staging path is obtained by
        appending ``.partial``.
    replace
        Whether to replace an existing final ``execution_state.csv``.

    Raises
    ------
    TypeError
        If ``final_path`` is not a supported path value or ``replace`` is not
        exactly a boolean.
    ValueError
        If mutable replacement is requested for another filename, a file path
        identifies a directory or symlink, or immutable final content already
        exists.
    StorageError
        If staging-file synchronization or atomic publication fails.

    Notes
    -----
    The caller must finish writing and close the staging file before calling
    this function. Immutable publication uses one same-directory rename after
    confirming that no final entry exists. Mutable replacement is restricted
    to ``execution_state.csv``.

    Node usage scope: Shared - Used by root and worker file publication.
    """
    if type(replace) is not bool:
        observed_type = f"{type(replace).__module__}." f"{type(replace).__qualname__}"
        raise TypeError(f"replace must be bool; got {observed_type}.")

    staging = staged_file_path(final_path)
    final = staging.with_name(staging.name.removesuffix(".partial"))

    if replace and final.name != "execution_state.csv":
        raise ValueError("replace=True is permitted only for execution_state.csv; " f"got {str(final)!r}.")

    if staging.is_symlink():
        raise ValueError(f"Staging file must not be a symlink: {str(staging)!r}.")
    if staging.is_dir():
        raise ValueError(f"Staging file path must not be a directory: " f"{str(staging)!r}.")

    if final.is_symlink():
        raise ValueError(f"Final file must not be a symlink: {str(final)!r}.")
    if final.is_dir():
        raise ValueError(f"Final file path must not be a directory: {str(final)!r}.")
    if final.exists() and not replace:
        raise ValueError(f"Immutable final file already exists: {str(final)!r}.")

    try:
        # The writer has already closed the staging file. Reopen it only to
        # synchronize its completed bytes before the publication rename.
        with staging.open("r+b") as staging_handle:
            os.fsync(staging_handle.fileno())

        if replace:
            os.replace(staging, final)
        else:
            os.rename(staging, final)
    except OSError as error:
        raise StorageError(f"Failed to publish staged file {str(staging)!r} " f"to {str(final)!r}.") from error


def discard_staged_file(
    final_path: str | os.PathLike[str],
    /,
) -> bool:
    """Remove the exact staging file for one final path if present.

    Parameters
    ----------
    final_path
        Final authoritative file path whose sibling ``.partial`` file may be
        removed.

    Returns
    -------
    bool
        Whether the exact staging file was removed.

    Raises
    ------
    TypeError
        If ``final_path`` is not a supported path value.
    ValueError
        If the exact staging path identifies a directory or symlink.
    StorageError
        If removing the staging file fails operationally.

    Notes
    -----
    This function derives one exact staging path through
    ``staged_file_path()``. It does not inspect or remove the final file,
    siblings, or parent directory.

    Node usage scope: Shared - Used by root and worker staging cleanup.
    """
    staging = staged_file_path(final_path)

    if staging.is_symlink():
        raise ValueError(f"Staging file must not be a symlink: {str(staging)!r}.")
    if staging.is_dir():
        raise ValueError(f"Staging file path must not be a directory: " f"{str(staging)!r}.")

    try:
        staging.unlink()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise StorageError(f"Failed to discard staged file {str(staging)!r}.") from error

    return True


def _raise_error(error: OSError) -> NoReturn:
    """Ensure a reported filesystem traversal error is not suppressed.

    Parameters
    ----------
    error
        Filesystem error supplied by an operation such as ``os.walk``.

    Raises
    ------
    OSError
        Always re-raises the supplied error so the calling storage operation
        fails instead of silently omitting inaccessible content.
    """
    raise error


def publish_staged_bundle(  # noqa: C901
    paths: "BundlePublicationPaths",
    /,
) -> None:
    """Synchronize and atomically publish one complete staged bundle.

    Parameters
    ----------
    paths
        Exact sibling staging and final model-bundle paths.

    Raises
    ------
    TypeError
        If ``paths`` is not a ``BundlePublicationPaths`` record or its fields
        are not ``Path`` values.
    ValueError
        If paths have invalid names or parents, staging is not a real
        directory, a symlink occurs anywhere in the staging tree, a
        non-regular entry occurs, or final content already exists.
    StorageError
        If tree traversal, file synchronization, or atomic publication fails.

    Notes
    -----
    This helper establishes directory publication completeness only. It does
    not validate result schemas, scientific content, manifests, acceptance, or
    lifecycle state.

    Node usage scope: Shared - Used by root and worker bundle publication.
    """
    # Import locally to avoid the module-initialization cycle created by
    # paths.py importing StorageError from this module.
    from cocoonet._storage.paths import (
        BundlePublicationPaths,
        _paths_equal,
        _validate_positive_decimal_identifier,
        resolve_physical_path,
    )

    if not isinstance(paths, BundlePublicationPaths):
        observed_type = f"{type(paths).__module__}.{type(paths).__qualname__}"
        raise TypeError("paths must be BundlePublicationPaths; " f"got {observed_type}.")
    if not isinstance(paths.staging, Path):
        observed_type = f"{type(paths.staging).__module__}." f"{type(paths.staging).__qualname__}"
        raise TypeError(f"paths.staging must be pathlib.Path; got {observed_type}.")
    if not isinstance(paths.final, Path):
        observed_type = f"{type(paths.final).__module__}." f"{type(paths.final).__qualname__}"
        raise TypeError(f"paths.final must be pathlib.Path; got {observed_type}.")

    staging = resolve_physical_path(paths.staging)
    final = resolve_physical_path(paths.final)

    if not _paths_equal(staging.parent, final.parent):
        raise ValueError(
            f"Bundle staging and final paths must share one parent; got "
            f"{str(staging.parent)!r} and {str(final.parent)!r}."
        )

    if not final.name.startswith("model_"):
        raise ValueError(f"Bundle final name must use model_<model_id>; " f"got {final.name!r}.")

    model_id = final.name.removeprefix("model_")
    _validate_positive_decimal_identifier(
        model_id,
        field="model_id",
    )

    if staging.name != f"{final.name}.staging":
        raise ValueError(f"Bundle staging name must be {final.name + '.staging'!r}; " f"got {staging.name!r}.")

    if staging.is_symlink():
        raise ValueError(f"Bundle staging directory must not be a symlink: " f"{str(staging)!r}.")
    if not staging.is_dir():
        raise ValueError(f"Bundle staging path must be an existing directory: " f"{str(staging)!r}.")

    if final.is_symlink():
        raise ValueError(f"Bundle final path must not be a symlink: {str(final)!r}.")
    if final.exists():
        raise ValueError(f"Bundle final path already exists: {str(final)!r}.")

    try:
        for directory, directory_names, file_names in os.walk(
            staging,
            topdown=True,
            onerror=_raise_error,
            followlinks=False,
        ):
            directory_names.sort()
            file_names.sort()
            current_directory = Path(directory)

            for entry_name in (*directory_names, *file_names):
                entry = current_directory / entry_name

                if entry.is_symlink():
                    raise ValueError(f"Bundle staging tree must not contain symlinks: " f"{str(entry)!r}.")

                if entry.is_dir():
                    continue
                if not entry.is_file():
                    raise ValueError(
                        f"Bundle staging tree must contain only directories " f"and regular files; got {str(entry)!r}."
                    )

                with entry.open("r+b") as file_handle:
                    os.fsync(file_handle.fileno())

        os.rename(staging, final)
    except ValueError:
        raise
    except OSError as error:
        raise StorageError(f"Failed to publish staged bundle {str(staging)!r} " f"to {str(final)!r}.") from error


def discard_staged_bundle(  # noqa: C901
    paths: "BundlePublicationPaths",
    /,
) -> bool:
    """Remove only one exact non-symlink staging bundle if present.

    Parameters
    ----------
    paths
        Exact sibling staging and final model-bundle paths.

    Returns
    -------
    bool
        Whether the exact staging directory was removed.

    Raises
    ------
    TypeError
        If ``paths`` is not a ``BundlePublicationPaths`` record or its fields
        are not ``Path`` values.
    ValueError
        If paths have invalid names or parents, staging is a symlink or
        non-directory entry, or the staging tree contains a symlink or
        unsupported entry.
    StorageError
        If tree traversal or exact staging-tree removal fails.

    Notes
    -----
    This function never removes or modifies the final bundle, sibling entries,
    or the staging directory's parent. The local complexity exemption keeps
    this single safety boundary intact instead of introducing one-use helpers.

    Node usage scope: Shared - Used by root and worker bundle cleanup.
    """
    # Import locally to avoid the module-initialization cycle created by
    # paths.py importing StorageError from this module.
    from cocoonet._storage.paths import (
        BundlePublicationPaths,
        _paths_equal,
        _validate_positive_decimal_identifier,
        resolve_physical_path,
    )

    if not isinstance(paths, BundlePublicationPaths):
        observed_type = f"{type(paths).__module__}.{type(paths).__qualname__}"
        raise TypeError("paths must be BundlePublicationPaths; " f"got {observed_type}.")
    if not isinstance(paths.staging, Path):
        observed_type = f"{type(paths.staging).__module__}." f"{type(paths.staging).__qualname__}"
        raise TypeError(f"paths.staging must be pathlib.Path; got {observed_type}.")
    if not isinstance(paths.final, Path):
        observed_type = f"{type(paths.final).__module__}." f"{type(paths.final).__qualname__}"
        raise TypeError(f"paths.final must be pathlib.Path; got {observed_type}.")

    staging = resolve_physical_path(paths.staging)
    final = resolve_physical_path(paths.final)

    if not _paths_equal(staging.parent, final.parent):
        raise ValueError(
            f"Bundle staging and final paths must share one parent; got "
            f"{str(staging.parent)!r} and {str(final.parent)!r}."
        )

    if not final.name.startswith("model_"):
        raise ValueError(f"Bundle final name must use model_<model_id>; " f"got {final.name!r}.")

    model_id = final.name.removeprefix("model_")
    _validate_positive_decimal_identifier(
        model_id,
        field="model_id",
    )

    if staging.name != f"{final.name}.staging":
        raise ValueError(f"Bundle staging name must be {final.name + '.staging'!r}; " f"got {staging.name!r}.")

    if staging.is_symlink():
        raise ValueError(f"Bundle staging directory must not be a symlink: " f"{str(staging)!r}.")
    if not staging.exists():
        return False
    if not staging.is_dir():
        raise ValueError(f"Bundle staging path must be a directory: " f"{str(staging)!r}.")

    try:
        for directory, directory_names, file_names in os.walk(
            staging,
            topdown=True,
            onerror=_raise_error,
            followlinks=False,
        ):
            directory_names.sort()
            file_names.sort()
            current_directory = Path(directory)

            for entry_name in (*directory_names, *file_names):
                entry = current_directory / entry_name

                if entry.is_symlink():
                    raise ValueError(f"Bundle staging tree must not contain symlinks: " f"{str(entry)!r}.")
                if entry.is_dir() or entry.is_file():
                    continue

                raise ValueError(
                    f"Bundle staging tree must contain only directories " f"and regular files; got {str(entry)!r}."
                )

        shutil.rmtree(staging)
    except ValueError:
        raise
    except OSError as error:
        raise StorageError(f"Failed to discard staged bundle {str(staging)!r}.") from error

    return True
