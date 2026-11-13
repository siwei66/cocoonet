"""Tests for focused Cocoonet storage publication support."""

from pathlib import Path

import pytest

from cocoonet._storage.paths import BundlePublicationPaths
from cocoonet._storage.publication import (
    StorageError,
    discard_staged_bundle,
    discard_staged_file,
    publish_staged_bundle,
    publish_staged_file,
    staged_file_path,
)


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


def test_publish_staged_file_publishes_immutable_bytes(
    tmp_path: Path,
) -> None:
    """Synchronize and rename a complete immutable staging file."""
    final = tmp_path / "report.xlsx"
    staging = staged_file_path(final)
    staging.write_bytes(b"complete report")

    publish_staged_file(final)

    assert final.read_bytes() == b"complete report"
    assert not staging.exists()


def test_publish_staged_file_rejects_existing_immutable_final(
    tmp_path: Path,
) -> None:
    """Preserve existing final and staged bytes after immutable rejection."""
    final = tmp_path / "report.xlsx"
    staging = staged_file_path(final)
    final.write_bytes(b"accepted")
    staging.write_bytes(b"late")

    with pytest.raises(
        ValueError,
        match=r"^Immutable final file already exists:",
    ):
        publish_staged_file(final)

    assert final.read_bytes() == b"accepted"
    assert staging.read_bytes() == b"late"


def test_publish_staged_file_replaces_execution_state(
    tmp_path: Path,
) -> None:
    """Atomically replace the one mutable authoritative CSV."""
    final = tmp_path / "execution_state.csv"
    staging = staged_file_path(final)
    final.write_bytes(b"old state")
    staging.write_bytes(b"new state")

    publish_staged_file(final, replace=True)

    assert final.read_bytes() == b"new state"
    assert not staging.exists()


def test_publish_staged_file_rejects_replacement_for_other_name(
    tmp_path: Path,
) -> None:
    """Restrict mutable replacement to execution_state.csv."""
    final = tmp_path / "metadata.csv"
    staging = staged_file_path(final)
    staging.write_bytes(b"metadata")

    with pytest.raises(
        ValueError,
        match=r"^replace=True is permitted only for execution_state\.csv",
    ):
        publish_staged_file(final, replace=True)

    assert staging.read_bytes() == b"metadata"
    assert not final.exists()


def test_publish_staged_file_wraps_missing_staging_failure(
    tmp_path: Path,
) -> None:
    """Report a missing staging file as an operational storage failure."""
    final = tmp_path / "report.xlsx"

    with pytest.raises(
        StorageError,
        match=r"^Failed to publish staged file ",
    ) as error:
        publish_staged_file(final)

    assert isinstance(error.value.__cause__, FileNotFoundError)
    assert not final.exists()


def test_publish_staged_file_rejects_staging_directory(
    tmp_path: Path,
) -> None:
    """Reject a directory where a staging file is required."""
    final = tmp_path / "report.xlsx"
    staging = staged_file_path(final)
    staging.mkdir()

    with pytest.raises(
        ValueError,
        match=r"^Staging file path must not be a directory:",
    ):
        publish_staged_file(final)

    assert staging.is_dir()
    assert not final.exists()


def test_publish_staged_file_rejects_staging_symlink(
    tmp_path: Path,
) -> None:
    """Reject a symlink where the exact staging file is required."""
    final = tmp_path / "report.xlsx"
    staging = staged_file_path(final)
    target = tmp_path / "target.xlsx"
    target.write_bytes(b"target")

    try:
        staging.symlink_to(target)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")

    with pytest.raises(
        ValueError,
        match=r"^Staging file must not be a symlink:",
    ):
        publish_staged_file(final)

    assert target.read_bytes() == b"target"
    assert staging.is_symlink()
    assert not final.exists()


def test_publish_staged_file_rejects_non_boolean_replace(
    tmp_path: Path,
) -> None:
    """Reject truthy integers as replacement policy values."""
    final = tmp_path / "execution_state.csv"

    with pytest.raises(
        TypeError,
        match=r"^replace must be bool; got builtins\.int\.$",
    ):
        publish_staged_file(
            final,
            replace=1,  # type: ignore[arg-type]
        )


def test_discard_staged_file_returns_false_when_absent(
    tmp_path: Path,
) -> None:
    """Report that no staging file was removed when it is absent."""
    final = tmp_path / "report.xlsx"

    removed = discard_staged_file(final)

    assert removed is False
    assert not final.exists()


def test_discard_staged_file_removes_only_exact_staging_file(
    tmp_path: Path,
) -> None:
    """Remove staging while preserving final content and sibling files."""
    final = tmp_path / "report.xlsx"
    staging = staged_file_path(final)
    sibling = tmp_path / "report.xlsx.partial.backup"

    final.write_bytes(b"accepted")
    staging.write_bytes(b"stale")
    sibling.write_bytes(b"unrelated")

    removed = discard_staged_file(final)

    assert removed is True
    assert not staging.exists()
    assert final.read_bytes() == b"accepted"
    assert sibling.read_bytes() == b"unrelated"


def test_discard_staged_file_rejects_staging_directory(
    tmp_path: Path,
) -> None:
    """Reject a directory without deleting it or its contents."""
    final = tmp_path / "report.xlsx"
    staging = staged_file_path(final)
    retained = staging / "retained.txt"
    staging.mkdir()
    retained.write_bytes(b"retained")

    with pytest.raises(
        ValueError,
        match=r"^Staging file path must not be a directory:",
    ):
        discard_staged_file(final)

    assert staging.is_dir()
    assert retained.read_bytes() == b"retained"
    assert not final.exists()


def test_discard_staged_file_rejects_staging_symlink(
    tmp_path: Path,
) -> None:
    """Reject a symlink without deleting its target."""
    final = tmp_path / "report.xlsx"
    staging = staged_file_path(final)
    target = tmp_path / "target.xlsx"
    target.write_bytes(b"target")

    try:
        staging.symlink_to(target)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")

    with pytest.raises(
        ValueError,
        match=r"^Staging file must not be a symlink:",
    ):
        discard_staged_file(final)

    assert staging.is_symlink()
    assert target.read_bytes() == b"target"
    assert not final.exists()


def test_publish_staged_bundle_publishes_complete_tree(
    tmp_path: Path,
) -> None:
    """Synchronize and rename a complete nested staging tree."""
    staging = tmp_path / "model_1.staging"
    final = tmp_path / "model_1"
    nested = staging / "visualizations"
    nested.mkdir(parents=True)
    (staging / "result.xlsx").write_bytes(b"result")
    (nested / "plot.dill").write_bytes(b"plot")
    paths = BundlePublicationPaths(staging=staging, final=final)

    publish_staged_bundle(paths)

    assert not staging.exists()
    assert (final / "result.xlsx").read_bytes() == b"result"
    assert (final / "visualizations" / "plot.dill").read_bytes() == b"plot"


def test_publish_staged_bundle_rejects_existing_final_without_merge(
    tmp_path: Path,
) -> None:
    """Preserve staging and final trees when immutable final exists."""
    staging = tmp_path / "model_1.staging"
    final = tmp_path / "model_1"
    staging.mkdir()
    final.mkdir()
    (staging / "new.txt").write_bytes(b"new")
    (final / "accepted.txt").write_bytes(b"accepted")
    paths = BundlePublicationPaths(staging=staging, final=final)

    with pytest.raises(
        ValueError,
        match=r"^Bundle final path already exists:",
    ):
        publish_staged_bundle(paths)

    assert (staging / "new.txt").read_bytes() == b"new"
    assert (final / "accepted.txt").read_bytes() == b"accepted"
    assert not (final / "new.txt").exists()


def test_publish_staged_bundle_rejects_invalid_path_relationship(
    tmp_path: Path,
) -> None:
    """Reject non-sibling and incorrectly named bundle paths."""
    staging = tmp_path / "staging" / "model_1.staging"
    final = tmp_path / "final" / "model_1"

    with pytest.raises(
        ValueError,
        match=r"must share one parent",
    ):
        publish_staged_bundle(BundlePublicationPaths(staging=staging, final=final))

    with pytest.raises(
        ValueError,
        match=r"Bundle staging name must be",
    ):
        publish_staged_bundle(
            BundlePublicationPaths(
                staging=tmp_path / "wrong.staging",
                final=tmp_path / "model_1",
            )
        )


def test_publish_staged_bundle_rejects_invalid_model_name(
    tmp_path: Path,
) -> None:
    """Reject an unsafe model identifier in bundle paths."""
    with pytest.raises(
        ValueError,
        match=r"^model_id must be canonical positive decimal text",
    ):
        publish_staged_bundle(
            BundlePublicationPaths(
                staging=tmp_path / "model_01.staging",
                final=tmp_path / "model_01",
            )
        )


def test_publish_staged_bundle_requires_existing_staging_directory(
    tmp_path: Path,
) -> None:
    """Reject an absent or non-directory staging path."""
    absent = BundlePublicationPaths(
        staging=tmp_path / "model_1.staging",
        final=tmp_path / "model_1",
    )

    with pytest.raises(
        ValueError,
        match=r"Bundle staging path must be an existing directory",
    ):
        publish_staged_bundle(absent)

    absent.staging.write_bytes(b"not a directory")

    with pytest.raises(
        ValueError,
        match=r"Bundle staging path must be an existing directory",
    ):
        publish_staged_bundle(absent)


def test_publish_staged_bundle_rejects_nested_symlink(
    tmp_path: Path,
) -> None:
    """Reject a symlink anywhere in the staging tree."""
    staging = tmp_path / "model_1.staging"
    final = tmp_path / "model_1"
    staging.mkdir()
    target = tmp_path / "target.txt"
    target.write_bytes(b"target")
    link = staging / "link.txt"

    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")

    with pytest.raises(
        ValueError,
        match=r"Bundle staging tree must not contain symlinks",
    ):
        publish_staged_bundle(BundlePublicationPaths(staging=staging, final=final))

    assert link.is_symlink()
    assert target.read_bytes() == b"target"
    assert not final.exists()


def test_publish_staged_bundle_rejects_wrong_record_type() -> None:
    """Reject values outside the bundle path record boundary."""
    with pytest.raises(
        TypeError,
        match=r"^paths must be BundlePublicationPaths; got builtins\.object\.$",
    ):
        publish_staged_bundle(object())  # type: ignore[arg-type]


def test_discard_staged_bundle_returns_false_when_absent(
    tmp_path: Path,
) -> None:
    """Report that no staging bundle was removed when it is absent."""
    final = tmp_path / "model_1"
    final.mkdir()
    (final / "accepted.txt").write_bytes(b"accepted")
    paths = BundlePublicationPaths(
        staging=tmp_path / "model_1.staging",
        final=final,
    )

    removed = discard_staged_bundle(paths)

    assert removed is False
    assert (final / "accepted.txt").read_bytes() == b"accepted"


def test_discard_staged_bundle_removes_only_exact_staging_tree(
    tmp_path: Path,
) -> None:
    """Remove staging while preserving final and sibling trees."""
    staging = tmp_path / "model_1.staging"
    final = tmp_path / "model_1"
    sibling = tmp_path / "model_2.staging"
    nested = staging / "nested"

    nested.mkdir(parents=True)
    final.mkdir()
    sibling.mkdir()
    (nested / "stale.txt").write_bytes(b"stale")
    (final / "accepted.txt").write_bytes(b"accepted")
    (sibling / "retained.txt").write_bytes(b"retained")

    removed = discard_staged_bundle(BundlePublicationPaths(staging=staging, final=final))

    assert removed is True
    assert not staging.exists()
    assert (final / "accepted.txt").read_bytes() == b"accepted"
    assert (sibling / "retained.txt").read_bytes() == b"retained"


def test_discard_staged_bundle_rejects_staging_file(
    tmp_path: Path,
) -> None:
    """Reject a non-directory staging entry without removing it."""
    staging = tmp_path / "model_1.staging"
    final = tmp_path / "model_1"
    staging.write_bytes(b"not a directory")

    with pytest.raises(
        ValueError,
        match=r"^Bundle staging path must be a directory:",
    ):
        discard_staged_bundle(BundlePublicationPaths(staging=staging, final=final))

    assert staging.read_bytes() == b"not a directory"
    assert not final.exists()


def test_discard_staged_bundle_rejects_nested_symlink(
    tmp_path: Path,
) -> None:
    """Reject a nested symlink without removing staging or its target."""
    staging = tmp_path / "model_1.staging"
    final = tmp_path / "model_1"
    staging.mkdir()
    target = tmp_path / "target.txt"
    target.write_bytes(b"target")
    link = staging / "link.txt"

    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")

    with pytest.raises(
        ValueError,
        match=r"Bundle staging tree must not contain symlinks",
    ):
        discard_staged_bundle(BundlePublicationPaths(staging=staging, final=final))

    assert staging.is_dir()
    assert link.is_symlink()
    assert target.read_bytes() == b"target"
    assert not final.exists()


def test_discard_staged_bundle_rejects_invalid_relationship(
    tmp_path: Path,
) -> None:
    """Reject bundle paths that are not exact siblings."""
    with pytest.raises(
        ValueError,
        match=r"must share one parent",
    ):
        discard_staged_bundle(
            BundlePublicationPaths(
                staging=tmp_path / "staging" / "model_1.staging",
                final=tmp_path / "final" / "model_1",
            )
        )
