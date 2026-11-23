"""Tests for Cocoonet root node working directory preparation."""

from pathlib import Path
from typing import Any

import pytest

import cocoonet
from cocoonet._storage.metadata import (
    NodeRole,
    decode_cocoonet_metadata_csv,
)
from cocoonet._storage.publication import staged_file_path
from cocoonet._storage.work_directory import (
    open_work_directory,
    prepare_work_directory,
)


def test_prepare_work_directory_creates_exact_fresh_layout(
    tmp_path: Path,
) -> None:
    """Create only Phase 2 entries for a missing root node work directory."""
    work_dir = tmp_path / "nested" / "work-dir"

    layout = prepare_work_directory(work_dir)

    assert layout.work_dir == work_dir.resolve()
    assert {entry.name for entry in layout.work_dir.iterdir()} == {
        "cocoonet_metadata.csv",
        "models_for_transfer",
        "attempt_results",
        "logs",
    }
    assert layout.models_for_transfer.is_dir()
    assert layout.attempt_results.is_dir()
    assert layout.logs.is_dir()
    assert not layout.root_data_reference_xlsx.exists()
    assert not layout.partition_schedules_xlsx.exists()
    assert not layout.execution_configuration_csv.exists()
    assert not layout.execution_state_csv.exists()
    assert not layout.root_running_log.exists()
    assert not layout.root_error_log.exists()
    assert not staged_file_path(layout.metadata_csv).exists()

    metadata = decode_cocoonet_metadata_csv(layout.metadata_csv.read_bytes())
    assert metadata.storage_schema_version == 1
    assert metadata.node_role is NodeRole.ROOT


def test_prepare_work_directory_accepts_existing_empty_directory(
    tmp_path: Path,
) -> None:
    """Prepare an existing empty directory without overwrite permission."""
    work_dir = tmp_path / "work-dir"
    work_dir.mkdir()

    layout = prepare_work_directory(work_dir)

    assert layout.work_dir == work_dir.resolve()
    assert layout.metadata_csv.is_file()
    assert layout.models_for_transfer.is_dir()
    assert layout.attempt_results.is_dir()
    assert layout.logs.is_dir()


def test_prepare_work_directory_rejects_nonempty_without_mutation(
    tmp_path: Path,
) -> None:
    """Reject a nonempty work directory without explicit overwrite."""
    work_dir = tmp_path / "work-dir"
    work_dir.mkdir()
    retained = work_dir / "retained.txt"
    retained.write_bytes(b"retained")

    with pytest.raises(
        ValueError,
        match=r"must be empty when overwrite=False",
    ):
        prepare_work_directory(work_dir)

    assert retained.read_bytes() == b"retained"
    assert {entry.name for entry in work_dir.iterdir()} == {"retained.txt"}


def test_prepare_work_directory_clears_only_validated_contents(
    tmp_path: Path,
) -> None:
    """Clear an authorized work directory while preserving dataset sources."""
    work_dir = tmp_path / "work-dir"
    stale_directory = work_dir / "stale"
    stale_directory.mkdir(parents=True)
    (work_dir / "old.txt").write_bytes(b"old")
    (stale_directory / "nested.txt").write_bytes(b"nested")

    dataset_source = tmp_path / "source.csv"
    dataset_source.write_bytes(b"sample_label,y\nsample,1\n")

    layout = prepare_work_directory(
        work_dir,
        dataset_sources=(dataset_source,),
        overwrite=True,
    )

    assert dataset_source.read_bytes() == b"sample_label,y\nsample,1\n"
    assert not (work_dir / "old.txt").exists()
    assert not stale_directory.exists()
    assert {entry.name for entry in work_dir.iterdir()} == {
        "cocoonet_metadata.csv",
        "models_for_transfer",
        "attempt_results",
        "logs",
    }
    assert layout.metadata_csv.is_file()


@pytest.mark.parametrize("overwrite", (False, True))
def test_prepare_work_directory_rejects_existing_non_directory(
    tmp_path: Path,
    overwrite: bool,
) -> None:
    """Reject an existing file regardless of overwrite permission."""
    work_dir = tmp_path / "work-dir"
    work_dir.write_bytes(b"not a directory")

    with pytest.raises(
        ValueError,
        match=r"must be a directory",
    ):
        prepare_work_directory(work_dir, overwrite=overwrite)

    assert work_dir.read_bytes() == b"not a directory"


def test_prepare_work_directory_validates_all_sources_before_clearing(
    tmp_path: Path,
) -> None:
    """Reject a conflicting source collection before authorized clearing."""
    work_dir = tmp_path / "work-dir"
    work_dir.mkdir()
    retained = work_dir / "retained.txt"
    retained.write_bytes(b"retained")

    outside_source = tmp_path / "outside.csv"
    outside_source.write_bytes(b"outside")
    conflicting_source = work_dir / "source.csv"

    with pytest.raises(
        ValueError,
        match=r"must not equal or be contained within root node working directory",
    ):
        prepare_work_directory(
            work_dir,
            dataset_sources=(outside_source, conflicting_source),
            overwrite=True,
        )

    assert retained.read_bytes() == b"retained"
    assert outside_source.read_bytes() == b"outside"
    assert not conflicting_source.exists()


def test_prepare_work_directory_rejects_protected_package_root() -> None:
    """Reject the installed Cocoonet package directory without mutation."""
    package_file = cocoonet.__file__
    assert package_file is not None
    package_root = Path(package_file).resolve().parent
    entries_before = tuple(sorted(entry.name for entry in package_root.iterdir()))

    with pytest.raises(ValueError, match=r"protected location"):
        prepare_work_directory(package_root)

    assert tuple(sorted(entry.name for entry in package_root.iterdir())) == (entries_before)


def test_prepare_work_directory_resolves_safe_work_dir_symlink(
    tmp_path: Path,
) -> None:
    """Prepare the canonical safe target selected through a work-dir alias."""
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "work-dir-alias"

    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    layout = prepare_work_directory(alias)

    assert layout.work_dir == target.resolve()
    assert layout.metadata_csv == target.resolve() / "cocoonet_metadata.csv"
    assert alias.is_symlink()


def test_prepare_work_directory_does_not_follow_nested_symlink(
    tmp_path: Path,
) -> None:
    """Remove a nested link while preserving its outside target."""
    work_dir = tmp_path / "work-dir"
    work_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    retained = outside / "retained.txt"
    retained.write_bytes(b"retained")
    nested_link = work_dir / "linked-directory"

    try:
        nested_link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    prepare_work_directory(work_dir, overwrite=True)

    assert not nested_link.exists()
    assert retained.read_bytes() == b"retained"


def test_prepare_work_directory_rejects_non_boolean_overwrite(
    tmp_path: Path,
) -> None:
    """Reject integer truth values before creating the work directory."""
    work_dir = tmp_path / "work-dir"
    invalid_overwrite: Any = 1

    with pytest.raises(TypeError, match=r"overwrite"):
        prepare_work_directory(work_dir, overwrite=invalid_overwrite)

    assert not work_dir.exists()


def test_open_work_directory_returns_existing_layout_without_mutation(
    tmp_path: Path,
) -> None:
    """Return root metadata without changing working-directory entries."""
    prepared = prepare_work_directory(tmp_path / "work-dir")
    metadata_bytes = prepared.metadata_csv.read_bytes()
    entries_before = tuple(sorted(entry.name for entry in prepared.work_dir.iterdir()))

    opened, metadata = open_work_directory(prepared.work_dir)

    assert opened == prepared
    assert metadata.node_role is NodeRole.ROOT
    assert prepared.metadata_csv.read_bytes() == metadata_bytes
    assert tuple(sorted(entry.name for entry in prepared.work_dir.iterdir())) == (entries_before)


def test_open_work_directory_ignores_and_retains_stale_metadata_staging(
    tmp_path: Path,
) -> None:
    """Ignore stale staging when valid final metadata is available."""
    prepared = prepare_work_directory(tmp_path / "work-dir")
    staging = staged_file_path(prepared.metadata_csv)
    staging.write_bytes(b"stale incomplete metadata")

    opened, metadata = open_work_directory(prepared.work_dir)

    assert opened == prepared
    assert metadata.node_role is NodeRole.ROOT
    assert staging.read_bytes() == b"stale incomplete metadata"


def test_open_work_directory_rejects_missing_work_dir_without_creating_it(
    tmp_path: Path,
) -> None:
    """Reject an absent work directory without creating an entry."""
    work_dir = tmp_path / "missing"

    with pytest.raises(ValueError, match=r"must already exist"):
        open_work_directory(work_dir)

    assert not work_dir.exists()


def test_open_work_directory_rejects_non_directory(
    tmp_path: Path,
) -> None:
    """Reject an existing file without modifying its content."""
    work_dir = tmp_path / "work-dir"
    work_dir.write_bytes(b"not a directory")

    with pytest.raises(ValueError, match=r"must be an existing directory"):
        open_work_directory(work_dir)

    assert work_dir.read_bytes() == b"not a directory"


def test_open_work_directory_rejects_missing_required_directory(
    tmp_path: Path,
) -> None:
    """Reject an incomplete Phase 2 layout without repairing it."""
    prepared = prepare_work_directory(tmp_path / "work-dir")
    prepared.logs.rmdir()

    with pytest.raises(
        ValueError,
        match=r"requires a real directory",
    ):
        open_work_directory(prepared.work_dir)

    assert not prepared.logs.exists()


def test_open_work_directory_rejects_missing_final_metadata(
    tmp_path: Path,
) -> None:
    """Reject absent final metadata without creating or publishing it."""
    prepared = prepare_work_directory(tmp_path / "work-dir")
    prepared.metadata_csv.unlink()

    with pytest.raises(
        ValueError,
        match=r"requires a final metadata file",
    ):
        open_work_directory(prepared.work_dir)

    assert not prepared.metadata_csv.exists()
    assert not staged_file_path(prepared.metadata_csv).exists()


def test_open_work_directory_rejects_invalid_final_metadata(
    tmp_path: Path,
) -> None:
    """Reject malformed final metadata without replacing its bytes."""
    prepared = prepare_work_directory(tmp_path / "work-dir")
    invalid_bytes = b"field,value\nstorage_schema_version,1\n"
    prepared.metadata_csv.write_bytes(invalid_bytes)

    with pytest.raises(ValueError):
        open_work_directory(prepared.work_dir)

    assert prepared.metadata_csv.read_bytes() == invalid_bytes


def test_open_work_directory_requires_root_node_role(
    tmp_path: Path,
) -> None:
    """Reject otherwise valid metadata belonging to a worker node."""
    prepared = prepare_work_directory(tmp_path / "work-dir")
    worker_metadata = prepared.metadata_csv.read_bytes().replace(
        b"node_role,root",
        b"node_role,worker",
    )
    prepared.metadata_csv.write_bytes(worker_metadata)

    with pytest.raises(
        ValueError,
        match=r"must declare node_role 'root'",
    ):
        open_work_directory(prepared.work_dir)

    assert prepared.metadata_csv.read_bytes() == worker_metadata


def test_open_work_directory_validates_dataset_ownership_without_mutation(
    tmp_path: Path,
) -> None:
    """Reject a conflicting source without touching the work directory."""
    prepared = prepare_work_directory(tmp_path / "work-dir")
    metadata_bytes = prepared.metadata_csv.read_bytes()
    conflicting_source = prepared.work_dir / "source.csv"

    with pytest.raises(
        ValueError,
        match=r"must not equal or be contained within root node working directory",
    ):
        open_work_directory(
            prepared.work_dir,
            dataset_sources=(conflicting_source,),
        )

    assert prepared.metadata_csv.read_bytes() == metadata_bytes
    assert not conflicting_source.exists()
