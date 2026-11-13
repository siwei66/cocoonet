"""Tests for immutable Cocoonet storage path records."""

import os
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from cocoonet._storage.paths import (
    BundlePublicationPaths,
    RootStorageLayout,
    WorkerParentStorageLayout,
    WorkerStorageLayout,
    _is_path_within,
    _paths_equal,
    resolve_physical_path,
)


def test_path_records_preserve_exact_immutable_slotted_schemas(
    tmp_path: Path,
) -> None:
    """Preserve supplied paths without validation, mutation, or I/O."""
    candidate = tmp_path / "candidate"

    records: tuple[
        RootStorageLayout
        | WorkerStorageLayout
        | WorkerParentStorageLayout
        | BundlePublicationPaths,
        ...,
    ] = (
        RootStorageLayout(
            candidate,
            candidate,
            candidate,
            candidate,
            candidate,
            candidate,
            candidate,
            candidate,
            candidate,
            candidate,
            candidate,
        ),
        WorkerStorageLayout(
            candidate,
            candidate,
            candidate,
            candidate,
        ),
        WorkerParentStorageLayout(
            candidate,
            candidate,
            candidate,
            candidate,
            candidate,
            candidate,
            candidate,
        ),
        BundlePublicationPaths(
            candidate,
            candidate,
        ),
    )
    expected_fields = (
        (
            "root",
            "metadata_csv",
            "root_data_reference_xlsx",
            "partition_schedules_xlsx",
            "execution_configuration_csv",
            "execution_state_csv",
            "models_for_transfer",
            "attempt_results",
            "logs",
            "root_running_log",
            "root_error_log",
        ),
        (
            "root",
            "datasets",
            "models",
            "results",
        ),
        (
            "parent_root",
            "metadata_csv",
            "worker_data_reference_xlsx",
            "partition_schedules_xlsx",
            "execution_configuration_csv",
            "execution_state_csv",
            "attempt_results",
        ),
        (
            "staging",
            "final",
        ),
    )

    assert not candidate.exists()

    for record, field_names in zip(records, expected_fields, strict=True):
        assert tuple(field.name for field in fields(record)) == field_names
        assert all(getattr(record, field_name) == candidate for field_name in field_names)
        assert not hasattr(record, "__dict__")

        with pytest.raises(FrozenInstanceError):
            record.__setattr__(field_names[0], tmp_path / "replacement")

    assert not candidate.exists()


def test_resolve_physical_path_handles_existing_and_missing_paths(
    tmp_path: Path,
) -> None:
    """Resolve existing ancestors without creating a missing suffix."""
    existing = tmp_path / "existing"
    existing.mkdir()
    missing = existing / "missing" / "leaf"

    resolved_existing = resolve_physical_path(existing)
    resolved_missing = resolve_physical_path(missing)

    assert resolved_existing == existing.resolve()
    assert resolved_existing.is_absolute()
    assert resolved_missing == existing.resolve() / "missing" / "leaf"
    assert not missing.exists()


def test_resolve_physical_path_resolves_relative_paths() -> None:
    """Resolve a relative path against the current working directory."""
    assert resolve_physical_path(Path(".")) == Path.cwd().resolve()


def test_resolve_physical_path_follows_symlinked_existing_ancestor(
    tmp_path: Path,
) -> None:
    """Resolve a real symlink before retaining a missing suffix."""
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"

    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    unresolved = link / "missing"

    assert resolve_physical_path(unresolved) == target.resolve() / "missing"
    assert not unresolved.exists()


def test_resolve_physical_path_rejects_invalid_inputs() -> None:
    """Reject incorrect argument types and embedded NUL characters."""
    with pytest.raises(
        TypeError,
        match=r"^path must be str or os\.PathLike\[str\].*builtins\.int",
    ):
        resolve_physical_path(3)  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match=r"^path must not contain an embedded NUL character\.$",
    ):
        resolve_physical_path("invalid\x00path")


def test_canonical_path_comparisons_are_component_aware(
    tmp_path: Path,
) -> None:
    """Distinguish equality, strict containment, and lexical prefixes."""
    parent = resolve_physical_path(tmp_path)
    child = parent / "child"
    lexical_prefix = parent.parent / f"{parent.name}_suffix"

    assert _paths_equal(parent, parent)
    assert not _paths_equal(child, parent)

    assert _is_path_within(child, parent)
    assert not _is_path_within(parent, parent)
    assert not _is_path_within(parent, child)
    assert not _is_path_within(lexical_prefix, parent)


def test_canonical_path_comparisons_follow_platform_case_rules(
    tmp_path: Path,
) -> None:
    """Treat case aliases according to the host filesystem convention."""
    canonical = resolve_physical_path(tmp_path)

    if os.path.normcase("A") == os.path.normcase("a"):
        case_alias = Path(str(canonical).swapcase())

        assert _paths_equal(case_alias, canonical)
        assert _is_path_within(case_alias / "child", canonical)
