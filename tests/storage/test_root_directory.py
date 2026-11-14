"""Tests for fresh Cocoonet root-directory preparation."""

from pathlib import Path
from typing import Any

import pytest

import cocoonet
from cocoonet._storage.metadata import (
    NodeRole,
    decode_cocoonet_metadata_csv,
)
from cocoonet._storage.publication import staged_file_path
from cocoonet._storage.root_directory import (
    open_root_directory,
    prepare_root_directory,
)


def test_prepare_root_directory_creates_exact_fresh_layout(
    tmp_path: Path,
) -> None:
    """Create only the Phase 2-owned entries for a missing root."""
    root = tmp_path / "nested" / "root"

    layout = prepare_root_directory(root)

    assert layout.root == root.resolve()
    assert {entry.name for entry in layout.root.iterdir()} == {
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


def test_prepare_root_directory_accepts_existing_empty_directory(
    tmp_path: Path,
) -> None:
    """Prepare an existing empty directory without overwrite permission."""
    root = tmp_path / "root"
    root.mkdir()

    layout = prepare_root_directory(root)

    assert layout.root == root.resolve()
    assert layout.metadata_csv.is_file()
    assert layout.models_for_transfer.is_dir()
    assert layout.attempt_results.is_dir()
    assert layout.logs.is_dir()


def test_prepare_root_directory_rejects_nonempty_without_mutation(
    tmp_path: Path,
) -> None:
    """Reject a nonempty root when overwrite is not explicitly enabled."""
    root = tmp_path / "root"
    root.mkdir()
    retained = root / "retained.txt"
    retained.write_bytes(b"retained")

    with pytest.raises(
        ValueError,
        match=r"must be empty when overwrite=False",
    ):
        prepare_root_directory(root)

    assert retained.read_bytes() == b"retained"
    assert {entry.name for entry in root.iterdir()} == {"retained.txt"}


def test_prepare_root_directory_clears_only_validated_root_contents(
    tmp_path: Path,
) -> None:
    """Clear an authorized root while preserving an outside dataset source."""
    root = tmp_path / "root"
    stale_directory = root / "stale"
    stale_directory.mkdir(parents=True)
    (root / "old.txt").write_bytes(b"old")
    (stale_directory / "nested.txt").write_bytes(b"nested")

    dataset_source = tmp_path / "source.csv"
    dataset_source.write_bytes(b"sample_label,y\nsample,1\n")

    layout = prepare_root_directory(
        root,
        dataset_sources=(dataset_source,),
        overwrite=True,
    )

    assert dataset_source.read_bytes() == b"sample_label,y\nsample,1\n"
    assert not (root / "old.txt").exists()
    assert not stale_directory.exists()
    assert {entry.name for entry in root.iterdir()} == {
        "cocoonet_metadata.csv",
        "models_for_transfer",
        "attempt_results",
        "logs",
    }
    assert layout.metadata_csv.is_file()


@pytest.mark.parametrize("overwrite", (False, True))
def test_prepare_root_directory_rejects_existing_non_directory(
    tmp_path: Path,
    overwrite: bool,
) -> None:
    """Reject an existing file regardless of overwrite permission."""
    root = tmp_path / "root"
    root.write_bytes(b"not a directory")

    with pytest.raises(
        ValueError,
        match=r"must be a directory",
    ):
        prepare_root_directory(root, overwrite=overwrite)

    assert root.read_bytes() == b"not a directory"


def test_prepare_root_directory_validates_all_sources_before_clearing(
    tmp_path: Path,
) -> None:
    """Reject a conflicting source collection before authorized clearing."""
    root = tmp_path / "root"
    root.mkdir()
    retained = root / "retained.txt"
    retained.write_bytes(b"retained")

    outside_source = tmp_path / "outside.csv"
    outside_source.write_bytes(b"outside")
    conflicting_source = root / "source.csv"

    with pytest.raises(
        ValueError,
        match=r"must not equal or be contained within managed root",
    ):
        prepare_root_directory(
            root,
            dataset_sources=(outside_source, conflicting_source),
            overwrite=True,
        )

    assert retained.read_bytes() == b"retained"
    assert outside_source.read_bytes() == b"outside"
    assert not conflicting_source.exists()


def test_prepare_root_directory_rejects_protected_package_root() -> None:
    """Reject the installed Cocoonet package directory without mutation."""
    package_file = cocoonet.__file__
    assert package_file is not None
    package_root = Path(package_file).resolve().parent
    entries_before = tuple(sorted(entry.name for entry in package_root.iterdir()))

    with pytest.raises(ValueError, match=r"protected location"):
        prepare_root_directory(package_root)

    assert tuple(sorted(entry.name for entry in package_root.iterdir())) == (entries_before)


def test_prepare_root_directory_resolves_safe_root_symlink(
    tmp_path: Path,
) -> None:
    """Prepare the canonical safe target selected through a root alias."""
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "root-alias"

    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    layout = prepare_root_directory(alias)

    assert layout.root == target.resolve()
    assert layout.metadata_csv == target.resolve() / "cocoonet_metadata.csv"
    assert alias.is_symlink()


def test_prepare_root_directory_does_not_follow_nested_symlink(
    tmp_path: Path,
) -> None:
    """Remove a nested link while preserving its outside target."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    retained = outside / "retained.txt"
    retained.write_bytes(b"retained")
    nested_link = root / "linked-directory"

    try:
        nested_link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    prepare_root_directory(root, overwrite=True)

    assert not nested_link.exists()
    assert retained.read_bytes() == b"retained"


def test_prepare_root_directory_rejects_non_boolean_overwrite(
    tmp_path: Path,
) -> None:
    """Reject integer truth values before creating the root."""
    root = tmp_path / "root"
    invalid_overwrite: Any = 1

    with pytest.raises(TypeError, match=r"overwrite"):
        prepare_root_directory(root, overwrite=invalid_overwrite)

    assert not root.exists()


def test_open_root_directory_returns_existing_layout_without_mutation(
    tmp_path: Path,
) -> None:
    """Return persisted root metadata without changing existing entries."""
    prepared = prepare_root_directory(tmp_path / "root")
    metadata_bytes = prepared.metadata_csv.read_bytes()
    entries_before = tuple(sorted(entry.name for entry in prepared.root.iterdir()))

    opened, metadata = open_root_directory(prepared.root)

    assert opened == prepared
    assert metadata.node_role is NodeRole.ROOT
    assert prepared.metadata_csv.read_bytes() == metadata_bytes
    assert tuple(sorted(entry.name for entry in prepared.root.iterdir())) == (entries_before)


def test_open_root_directory_ignores_and_retains_stale_metadata_staging(
    tmp_path: Path,
) -> None:
    """Ignore stale staging when valid final metadata is available."""
    prepared = prepare_root_directory(tmp_path / "root")
    staging = staged_file_path(prepared.metadata_csv)
    staging.write_bytes(b"stale incomplete metadata")

    opened, metadata = open_root_directory(prepared.root)

    assert opened == prepared
    assert metadata.node_role is NodeRole.ROOT
    assert staging.read_bytes() == b"stale incomplete metadata"


def test_open_root_directory_rejects_missing_root_without_creating_it(
    tmp_path: Path,
) -> None:
    """Reject an absent root without creating any filesystem entry."""
    root = tmp_path / "missing"

    with pytest.raises(ValueError, match=r"must already exist"):
        open_root_directory(root)

    assert not root.exists()


def test_open_root_directory_rejects_non_directory(
    tmp_path: Path,
) -> None:
    """Reject an existing file without modifying its content."""
    root = tmp_path / "root"
    root.write_bytes(b"not a directory")

    with pytest.raises(ValueError, match=r"must be an existing directory"):
        open_root_directory(root)

    assert root.read_bytes() == b"not a directory"


def test_open_root_directory_rejects_missing_required_directory(
    tmp_path: Path,
) -> None:
    """Reject an incomplete Phase 2 layout without repairing it."""
    prepared = prepare_root_directory(tmp_path / "root")
    prepared.logs.rmdir()

    with pytest.raises(
        ValueError,
        match=r"requires a real directory",
    ):
        open_root_directory(prepared.root)

    assert not prepared.logs.exists()


def test_open_root_directory_rejects_missing_final_metadata(
    tmp_path: Path,
) -> None:
    """Reject absent final metadata without creating or publishing it."""
    prepared = prepare_root_directory(tmp_path / "root")
    prepared.metadata_csv.unlink()

    with pytest.raises(
        ValueError,
        match=r"requires a final metadata file",
    ):
        open_root_directory(prepared.root)

    assert not prepared.metadata_csv.exists()
    assert not staged_file_path(prepared.metadata_csv).exists()


def test_open_root_directory_rejects_invalid_final_metadata(
    tmp_path: Path,
) -> None:
    """Reject malformed final metadata without replacing its bytes."""
    prepared = prepare_root_directory(tmp_path / "root")
    invalid_bytes = b"field,value\nstorage_schema_version,1\n"
    prepared.metadata_csv.write_bytes(invalid_bytes)

    with pytest.raises(ValueError):
        open_root_directory(prepared.root)

    assert prepared.metadata_csv.read_bytes() == invalid_bytes


def test_open_root_directory_requires_root_node_role(
    tmp_path: Path,
) -> None:
    """Reject otherwise valid metadata belonging to a worker node."""
    prepared = prepare_root_directory(tmp_path / "root")
    worker_metadata = prepared.metadata_csv.read_bytes().replace(
        b"node_role,root",
        b"node_role,worker",
    )
    prepared.metadata_csv.write_bytes(worker_metadata)

    with pytest.raises(
        ValueError,
        match=r"must declare node_role 'root'",
    ):
        open_root_directory(prepared.root)

    assert prepared.metadata_csv.read_bytes() == worker_metadata


def test_open_root_directory_validates_dataset_ownership_without_mutation(
    tmp_path: Path,
) -> None:
    """Reject a conflicting configured source without touching the root."""
    prepared = prepare_root_directory(tmp_path / "root")
    metadata_bytes = prepared.metadata_csv.read_bytes()
    conflicting_source = prepared.root / "source.csv"

    with pytest.raises(
        ValueError,
        match=r"must not equal or be contained within managed root",
    ):
        open_root_directory(
            prepared.root,
            dataset_sources=(conflicting_source,),
        )

    assert prepared.metadata_csv.read_bytes() == metadata_bytes
    assert not conflicting_source.exists()
