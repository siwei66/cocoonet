"""Tests for focused Cocoonet storage publication support."""

from pathlib import Path

import pytest

from cocoonet._storage.publication import StorageError, staged_file_path


def test_storage_error_is_plain_runtime_error() -> None:
    """Preserve ordinary exception behavior without adding retry state."""
    error = StorageError("failed to publish C:/work/result.csv")

    assert StorageError.__base__ is RuntimeError
    assert error.args == ("failed to publish C:/work/result.csv",)


def test_staged_file_path_appends_partial_without_io(
    tmp_path: Path,
) -> None:
    """Append the exact staging suffix without creating either file."""
    final = tmp_path / "report.xlsx"

    staging = staged_file_path(final)

    assert staging == tmp_path / "report.xlsx.partial"
    assert staging.parent == final.parent
    assert not final.exists()
    assert not staging.exists()


def test_staged_file_path_preserves_relative_parent() -> None:
    """Keep a relative final path and staging path in the same directory."""
    final = Path("reports") / "execution_state.csv"

    staging = staged_file_path(final)

    assert staging == Path("reports") / "execution_state.csv.partial"
    assert staging.parent == final.parent


def test_staged_file_path_rejects_directory_like_names(
    tmp_path: Path,
) -> None:
    """Reject paths that do not identify a final filename."""
    for final_path in ("", ".", "..", tmp_path.anchor):
        with pytest.raises(
            ValueError,
            match=r"^final_path must identify a file name",
        ):
            staged_file_path(final_path)


def test_staged_file_path_rejects_existing_partial_suffix() -> None:
    """Reject a staging path supplied as though it were final."""
    with pytest.raises(
        ValueError,
        match=r"^final_path must not already end in '\.partial'",
    ):
        staged_file_path("execution_state.csv.partial")


def test_staged_file_path_rejects_embedded_nul() -> None:
    """Reject an embedded NUL before constructing a Path."""
    with pytest.raises(
        ValueError,
        match=r"^final_path must not contain an embedded NUL character\.$",
    ):
        staged_file_path("invalid\x00.csv")


def test_staged_file_path_rejects_non_path_value() -> None:
    """Reject values outside the supported path protocol."""
    with pytest.raises(
        TypeError,
        match=(r"^final_path must be str or os\.PathLike\[str\]; " r"got builtins\.int\.$"),
    ):
        staged_file_path(1)  # type: ignore[arg-type]
