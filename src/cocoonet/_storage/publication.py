"""Focused atomic-publication support for Cocoonet storage."""

import os
from pathlib import Path


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
