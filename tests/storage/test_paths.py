"""Tests for immutable Cocoonet storage path records."""

import os
import site
import sys
import sysconfig
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import pytest

import cocoonet._storage.paths as storage_paths
from cocoonet._contracts.identity import DatasetId, ModelId, ParentId
from cocoonet._storage.paths import (
    BundlePublicationPaths,
    RootStorageLayout,
    WorkerParentStorageLayout,
    WorkerStorageLayout,
    _is_path_within,
    _paths_equal,
    _validate_dataset_id,
    _validate_managed_root_against,
    _validate_parent_id,
    _validate_positive_decimal_identifier,
    build_root_storage_layout,
    build_worker_parent_storage_layout,
    build_worker_storage_layout,
    discover_protected_locations,
    resolve_physical_path,
    root_bundle_publication_paths,
    root_trusted_model_dill_path,
    validate_dataset_source_ownership,
    validate_managed_root,
    validate_root_ownership,
    worker_bundle_publication_paths,
    worker_dataset_directory,
    worker_trusted_model_dill_path,
)


def test_path_records_preserve_exact_immutable_slotted_schemas(
    tmp_path: Path,
) -> None:
    """Preserve supplied paths without validation, mutation, or I/O."""
    candidate = tmp_path / "candidate"

    records: tuple[
        RootStorageLayout | WorkerStorageLayout | WorkerParentStorageLayout | BundlePublicationPaths,
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


def test_discover_protected_locations_includes_runtime_categories() -> None:
    """Include interpreter, package, Cocoonet, and anchor locations."""
    protected = discover_protected_locations()
    expected_runtime_paths = (
        sys.prefix,
        sys.exec_prefix,
        sys.base_prefix,
        sys.base_exec_prefix,
        sysconfig.get_path("purelib"),
        sysconfig.get_path("platlib"),
        *site.getsitepackages(),
        site.getusersitepackages(),
    )

    for expected in expected_runtime_paths:
        canonical = resolve_physical_path(expected)

        assert any(_paths_equal(canonical, protected_path) for protected_path in protected)

    module_file = storage_paths.__file__
    assert module_file is not None
    package_directory = resolve_physical_path(Path(module_file).parent.parent)

    assert any(_paths_equal(package_directory, protected_path) for protected_path in protected)

    for protected_path in protected:
        anchor = resolve_physical_path(Path(protected_path.anchor))

        assert any(_paths_equal(anchor, candidate) for candidate in protected)


def test_discover_protected_locations_includes_source_checkout() -> None:
    """Protect the complete repository when imported from its source tree."""
    module_file = storage_paths.__file__
    assert module_file is not None
    package_directory = resolve_physical_path(Path(module_file).parent.parent)
    source_root = package_directory.parent.parent

    if package_directory.parent.name != "src" or not (source_root / "pyproject.toml").is_file():
        pytest.skip("Cocoonet is not imported from a source checkout.")

    protected = discover_protected_locations()

    assert any(_paths_equal(source_root, protected_path) for protected_path in protected)


def test_discover_protected_locations_is_deterministic_and_unique() -> None:
    """Return stable canonical paths without physical duplicates."""
    first = discover_protected_locations()
    second = discover_protected_locations()

    assert first == second
    assert all(path.is_absolute() for path in first)

    for index, candidate in enumerate(first):
        assert not any(_paths_equal(candidate, previous) for previous in first[:index])


def test_managed_root_matrix_rejects_only_anchor_equality(
    tmp_path: Path,
) -> None:
    """Allow ordinary descendants while rejecting an anchor itself."""
    anchor = resolve_physical_path(Path(tmp_path.anchor))
    candidate = resolve_physical_path(tmp_path / "candidate")

    _validate_managed_root_against(candidate, (anchor,))

    with pytest.raises(
        ValueError,
        match=r"must not equal filesystem anchor",
    ):
        _validate_managed_root_against(anchor, (anchor,))


def test_managed_root_matrix_rejects_risky_containment_directions(
    tmp_path: Path,
) -> None:
    """Reject equality and both containment directions for risky paths."""
    protected = resolve_physical_path(tmp_path / "protected")
    equal = protected
    descendant = protected / "child"
    ancestor = protected.parent
    disjoint = protected.parent / "protected_elsewhere"

    for candidate in (equal, descendant, ancestor):
        with pytest.raises(
            ValueError,
            match=(r"Managed root .* must not equal, contain, or be contained " r"within protected location"),
        ):
            _validate_managed_root_against(candidate, (protected,))

    _validate_managed_root_against(disjoint, (protected,))


def test_validate_managed_root_accepts_safe_absolute_user_path(
    tmp_path: Path,
) -> None:
    """Return a canonical safe absolute user-specified path without creating it."""
    candidate = tmp_path / "safe_root"

    validated = validate_managed_root(candidate)

    assert validated == resolve_physical_path(candidate)
    assert not candidate.exists()


def test_validate_managed_root_resolves_safe_relative_user_path(
    tmp_path: Path,
) -> None:
    """Resolve a relative user path from the current working directory."""
    original_working_directory = Path.cwd()

    try:
        os.chdir(tmp_path)
        validated = validate_managed_root(Path("relative_root"))
    finally:
        os.chdir(original_working_directory)

    expected = resolve_physical_path(tmp_path / "relative_root")
    assert validated == expected
    assert not expected.exists()


def test_validate_managed_root_uses_discovered_protections() -> None:
    """Reject the loaded Cocoonet package through production discovery."""
    module_file = storage_paths.__file__
    assert module_file is not None
    package_directory = resolve_physical_path(Path(module_file).parent.parent)

    with pytest.raises(
        ValueError,
        match=r"protected location",
    ):
        validate_managed_root(package_directory)


def test_dataset_source_ownership_enforces_asymmetric_containment(
    tmp_path: Path,
) -> None:
    """Reject sources inside the root while permitting the inverse."""
    root = tmp_path / "managed_root"
    equal_source = root
    contained_source = root / "dataset.csv"
    containing_source = root.parent
    disjoint_source = root.parent / "dataset.csv"

    for source in (equal_source, contained_source):
        with pytest.raises(
            ValueError,
            match=(r"Dataset source .* must not equal or be contained within " r"managed root"),
        ):
            validate_dataset_source_ownership(root, source)

    canonical_root, canonical_containing = validate_dataset_source_ownership(root, containing_source)
    assert canonical_root == resolve_physical_path(root)
    assert canonical_containing == resolve_physical_path(containing_source)

    canonical_root, canonical_disjoint = validate_dataset_source_ownership(root, disjoint_source)
    assert canonical_root == resolve_physical_path(root)
    assert canonical_disjoint == resolve_physical_path(disjoint_source)

    assert not root.exists()
    assert not contained_source.exists()
    assert not disjoint_source.exists()


def test_validate_root_ownership_preserves_source_order(
    tmp_path: Path,
) -> None:
    """Materialize and return the complete canonical source collection."""
    root = tmp_path / "managed_root"
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    sources = (source for source in (first, second))

    canonical_root, canonical_sources = validate_root_ownership(
        root,
        sources,
    )

    assert canonical_root == resolve_physical_path(root)
    assert canonical_sources == (
        resolve_physical_path(first),
        resolve_physical_path(second),
    )
    assert not root.exists()
    assert not first.exists()
    assert not second.exists()


def test_validate_root_ownership_rejects_duplicate_canonical_sources(
    tmp_path: Path,
) -> None:
    """Reject duplicate source associations without filesystem mutation."""
    root = tmp_path / "managed_root"
    source = tmp_path / "dataset.csv"

    with pytest.raises(
        ValueError,
        match=r"sources\[1\] duplicates canonical source sources\[0\]",
    ):
        validate_root_ownership(root, (source, source))

    assert not root.exists()
    assert not source.exists()


def test_validate_root_ownership_rejects_conflict_after_materialization(
    tmp_path: Path,
) -> None:
    """Reject a conflicting collection without creating any supplied path."""
    root = tmp_path / "managed_root"
    outside = tmp_path / "outside.csv"
    conflicting = root / "inside.csv"

    with pytest.raises(
        ValueError,
        match=r"Dataset source .* managed root",
    ):
        validate_root_ownership(
            root,
            (outside, conflicting),
        )

    assert not root.exists()
    assert not outside.exists()
    assert not conflicting.exists()


def test_validate_root_ownership_rejects_path_as_source_collection(
    tmp_path: Path,
) -> None:
    """Reject one path passed where a source collection is required."""
    with pytest.raises(
        TypeError,
        match=r"^sources must be an iterable of path inputs; got ",
    ):
        validate_root_ownership(
            tmp_path / "managed_root",
            tmp_path / "dataset.csv",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "value",
    (
        "",
        "0123456789a",
        "0123456789abc",
        "0123456789AB",
        "0123456789ag",
        "dataset_0123456789ab",
        "012345/6789ab",
        ".",
        "..",
    ),
)
def test_dataset_id_rejects_invalid_managed_path_text(value: str) -> None:
    """Reject dataset IDs outside the exact lowercase hexadecimal format."""
    with pytest.raises(
        ValueError,
        match=r"^dataset_id must be exactly 12 lowercase hexadecimal",
    ):
        _validate_dataset_id(value)


def test_dataset_id_accepts_exact_lowercase_hexadecimal_text() -> None:
    """Accept exactly 12 lowercase hexadecimal characters."""
    _validate_dataset_id("0123456789ab")


def test_dataset_id_rejects_non_text_value() -> None:
    """Reject non-string dataset identifiers through shared validation."""
    with pytest.raises(TypeError) as error:
        _validate_dataset_id(1)  # type: ignore[arg-type]

    message = str(error.value)
    assert "path: value" in message
    assert "expected: builtins.str" in message
    assert "got type: builtins.int" in message


@pytest.mark.parametrize(
    "value",
    (
        "",
        "0123456789abcde",
        "0123456789abcdef0",
        "0123456789ABCDE",
        "0123456789abcdeg",
        "parent_0123456789abcdef",
        "0123456789ab/cdef",
        ".",
        "..",
    ),
)
def test_parent_id_rejects_invalid_managed_path_text(value: str) -> None:
    """Reject parent IDs outside the exact lowercase hexadecimal format."""
    with pytest.raises(
        ValueError,
        match=r"^parent_id must be exactly 16 lowercase hexadecimal",
    ):
        _validate_parent_id(value)


def test_parent_id_accepts_exact_lowercase_hexadecimal_text() -> None:
    """Accept exactly 16 lowercase hexadecimal characters."""
    _validate_parent_id("0123456789abcdef")


def test_parent_id_rejects_non_text_value() -> None:
    """Reject non-string parent identifiers through shared validation."""
    with pytest.raises(TypeError) as error:
        _validate_parent_id(1)  # type: ignore[arg-type]

    message = str(error.value)
    assert "path: value" in message
    assert "expected: builtins.str" in message
    assert "got type: builtins.int" in message


@pytest.mark.parametrize(
    "field",
    ("model_id", "tile_id"),
)
@pytest.mark.parametrize(
    "value",
    (
        "",
        "0",
        "00",
        "01",
        "+1",
        "-1",
        "1.0",
        "1/2",
        "model_1",
        ".",
        "..",
    ),
)
def test_positive_decimal_identifier_rejects_noncanonical_text(
    field: str,
    value: str,
) -> None:
    """Reject noncanonical model and tile identifier text."""
    with pytest.raises(
        ValueError,
        match=rf"^{field} must be canonical positive decimal text",
    ):
        _validate_positive_decimal_identifier(
            value,
            field=field,
        )


@pytest.mark.parametrize(
    "field",
    ("model_id", "tile_id"),
)
@pytest.mark.parametrize(
    "value",
    ("1", "9", "10", "999"),
)
def test_positive_decimal_identifier_accepts_canonical_text(
    field: str,
    value: str,
) -> None:
    """Accept canonical positive model and tile identifier text."""
    _validate_positive_decimal_identifier(
        value,
        field=field,
    )


def test_positive_decimal_identifier_rejects_boolean_value() -> None:
    """Reject booleans through the shared validation boundary."""
    with pytest.raises(TypeError) as error:
        _validate_positive_decimal_identifier(
            True,  # type: ignore[arg-type]
            field="model_id",
        )

    message = str(error.value)
    assert "path: value" in message
    assert "expected: builtins.str" in message
    assert "got type: builtins.bool" in message


def test_build_root_storage_layout_returns_exact_paths_without_io(
    tmp_path: Path,
) -> None:
    """Build every required root path without creating filesystem entries."""
    candidate = tmp_path / "managed_root"
    canonical_root = resolve_physical_path(candidate)

    layout = build_root_storage_layout(candidate)

    assert layout == RootStorageLayout(
        root=canonical_root,
        metadata_csv=canonical_root / "cocoonet_metadata.csv",
        root_data_reference_xlsx=(canonical_root / "root_data_reference.xlsx"),
        partition_schedules_xlsx=(canonical_root / "partition_schedules.xlsx"),
        execution_configuration_csv=(canonical_root / "execution_configuration.csv"),
        execution_state_csv=canonical_root / "execution_state.csv",
        models_for_transfer=canonical_root / "models_for_transfer",
        attempt_results=canonical_root / "attempt_results",
        logs=canonical_root / "logs",
        root_running_log=canonical_root / "logs" / "root_running.log",
        root_error_log=canonical_root / "logs" / "root_error.log",
    )
    assert not candidate.exists()


def test_build_worker_storage_layout_returns_exact_paths_without_io(
    tmp_path: Path,
) -> None:
    """Build every top-level worker path without creating entries."""
    candidate = tmp_path / "worker"
    canonical_root = resolve_physical_path(candidate)

    layout = build_worker_storage_layout(candidate)

    assert layout == WorkerStorageLayout(
        root=canonical_root,
        datasets=canonical_root / "datasets",
        models=canonical_root / "models",
        results=canonical_root / "results",
    )
    assert not candidate.exists()


def test_build_worker_parent_storage_layout_returns_exact_paths_without_io(
    tmp_path: Path,
) -> None:
    """Build every parent-specific worker result path without creating entries."""
    worker = build_worker_storage_layout(tmp_path / "worker")
    parent_id = ParentId("0123456789abcdef")
    parent_root = worker.results / f"parent_{parent_id}"

    layout = build_worker_parent_storage_layout(
        worker,
        parent_id,
    )

    assert layout == WorkerParentStorageLayout(
        parent_root=parent_root,
        metadata_csv=parent_root / "cocoonet_metadata.csv",
        worker_data_reference_xlsx=(parent_root / "worker_data_reference.xlsx"),
        partition_schedules_xlsx=(parent_root / "partition_schedules.xlsx"),
        execution_configuration_csv=(parent_root / "execution_configuration.csv"),
        execution_state_csv=parent_root / "execution_state.csv",
        attempt_results=parent_root / "attempt_results",
    )
    assert not parent_root.exists()


def test_build_worker_parent_storage_layout_rejects_invalid_parent_id(
    tmp_path: Path,
) -> None:
    """Reject unsafe parent text before constructing a managed path."""
    worker = build_worker_storage_layout(tmp_path / "worker")

    with pytest.raises(
        ValueError,
        match=r"^parent_id must be exactly 16 lowercase hexadecimal",
    ):
        build_worker_parent_storage_layout(
            worker,
            ParentId("../invalid"),
        )


def test_build_worker_parent_storage_layout_rejects_results_escape(
    tmp_path: Path,
) -> None:
    """Reject a results directory outside the worker root."""
    root = resolve_physical_path(tmp_path / "worker")
    worker = WorkerStorageLayout(
        root=root,
        datasets=root / "datasets",
        models=root / "models",
        results=root.parent / "outside_results",
    )

    with pytest.raises(
        ValueError,
        match=r"Worker results path .* must be contained within worker root",
    ):
        build_worker_parent_storage_layout(
            worker,
            ParentId("0123456789abcdef"),
        )

    assert not root.exists()
    assert not worker.results.exists()


def test_root_trusted_model_dill_path_returns_exact_path_without_io(
    tmp_path: Path,
) -> None:
    """Build the exact trusted root model path without creating it."""
    layout = build_root_storage_layout(tmp_path / "root")

    path = root_trusted_model_dill_path(
        layout,
        ModelId("1"),
    )

    assert path == layout.models_for_transfer / "model_1.dill"
    assert not path.exists()


def test_worker_dataset_directory_returns_exact_path_without_io(
    tmp_path: Path,
) -> None:
    """Build the exact worker dataset directory without creating it."""
    layout = build_worker_storage_layout(tmp_path / "worker")

    path = worker_dataset_directory(
        layout,
        DatasetId("0123456789ab"),
    )

    assert path == layout.datasets / "dataset_0123456789ab"
    assert not path.exists()


def test_worker_trusted_model_dill_path_returns_exact_path_without_io(
    tmp_path: Path,
) -> None:
    """Build the exact trusted worker model path without creating it."""
    layout = build_worker_storage_layout(tmp_path / "worker")

    path = worker_trusted_model_dill_path(
        layout,
        ParentId("0123456789abcdef"),
        ModelId("2"),
    )

    assert path == (layout.models / "parent_0123456789abcdef" / "model_2.dill")
    assert not path.exists()


def test_unit_paths_reject_invalid_identifiers(
    tmp_path: Path,
) -> None:
    """Reject invalid identifiers before incorporating them into paths."""
    root = build_root_storage_layout(tmp_path / "root")
    worker = build_worker_storage_layout(tmp_path / "worker")

    with pytest.raises(
        ValueError,
        match=r"^model_id must be canonical positive decimal text",
    ):
        root_trusted_model_dill_path(
            root,
            ModelId("01"),
        )

    with pytest.raises(
        ValueError,
        match=r"^dataset_id must be exactly 12 lowercase hexadecimal",
    ):
        worker_dataset_directory(
            worker,
            DatasetId("../invalid"),
        )

    with pytest.raises(
        ValueError,
        match=r"^parent_id must be exactly 16 lowercase hexadecimal",
    ):
        worker_trusted_model_dill_path(
            worker,
            ParentId("../invalid"),
            ModelId("1"),
        )


def test_unit_paths_reject_layout_escape(
    tmp_path: Path,
) -> None:
    """Reject model and dataset base directories outside their roots."""
    root_path = resolve_physical_path(tmp_path / "root")
    root = RootStorageLayout(
        root=root_path,
        metadata_csv=root_path / "cocoonet_metadata.csv",
        root_data_reference_xlsx=(root_path / "root_data_reference.xlsx"),
        partition_schedules_xlsx=(root_path / "partition_schedules.xlsx"),
        execution_configuration_csv=(root_path / "execution_configuration.csv"),
        execution_state_csv=root_path / "execution_state.csv",
        models_for_transfer=root_path.parent / "outside_models",
        attempt_results=root_path / "attempt_results",
        logs=root_path / "logs",
        root_running_log=root_path / "logs" / "root_running.log",
        root_error_log=root_path / "logs" / "root_error.log",
    )

    worker_path = resolve_physical_path(tmp_path / "worker")
    worker = WorkerStorageLayout(
        root=worker_path,
        datasets=worker_path.parent / "outside_datasets",
        models=worker_path.parent / "outside_models",
        results=worker_path / "results",
    )

    with pytest.raises(
        ValueError,
        match=r"Root model directory .* must be contained within root",
    ):
        root_trusted_model_dill_path(root, ModelId("1"))

    with pytest.raises(
        ValueError,
        match=r"Worker datasets directory .* must be contained within worker root",
    ):
        worker_dataset_directory(
            worker,
            DatasetId("0123456789ab"),
        )

    with pytest.raises(
        ValueError,
        match=r"Worker models directory .* must be contained within worker root",
    ):
        worker_trusted_model_dill_path(
            worker,
            ParentId("0123456789abcdef"),
            ModelId("1"),
        )

    assert not root_path.exists()
    assert not worker_path.exists()


def test_root_bundle_publication_paths_return_exact_siblings_without_io(
    tmp_path: Path,
) -> None:
    """Build exact root staging and final bundle siblings without creating them."""
    layout = build_root_storage_layout(tmp_path / "root")

    paths = root_bundle_publication_paths(
        layout,
        ModelId("1"),
    )

    assert paths == BundlePublicationPaths(
        staging=layout.attempt_results / "model_1.staging",
        final=layout.attempt_results / "model_1",
    )
    assert not paths.staging.exists()
    assert not paths.final.exists()


def test_worker_bundle_publication_paths_return_exact_siblings_without_io(
    tmp_path: Path,
) -> None:
    """Build exact worker staging and final bundle siblings without creating them."""
    worker = build_worker_storage_layout(tmp_path / "worker")
    parent = build_worker_parent_storage_layout(
        worker,
        ParentId("0123456789abcdef"),
    )

    paths = worker_bundle_publication_paths(
        parent,
        ModelId("2"),
    )

    assert paths == BundlePublicationPaths(
        staging=parent.attempt_results / "model_2.staging",
        final=parent.attempt_results / "model_2",
    )
    assert not paths.staging.exists()
    assert not paths.final.exists()


def test_bundle_publication_paths_reject_invalid_model_id(
    tmp_path: Path,
) -> None:
    """Reject invalid model text before constructing bundle paths."""
    root = build_root_storage_layout(tmp_path / "root")
    worker = build_worker_storage_layout(tmp_path / "worker")
    parent = build_worker_parent_storage_layout(
        worker,
        ParentId("0123456789abcdef"),
    )

    with pytest.raises(
        ValueError,
        match=r"^model_id must be canonical positive decimal text",
    ):
        root_bundle_publication_paths(
            root,
            ModelId("../invalid"),
        )

    with pytest.raises(
        ValueError,
        match=r"^model_id must be canonical positive decimal text",
    ):
        worker_bundle_publication_paths(
            parent,
            ModelId("01"),
        )


def test_bundle_publication_paths_reject_layout_escape(
    tmp_path: Path,
) -> None:
    """Reject attempt-results directories outside their owning roots."""
    root = build_root_storage_layout(tmp_path / "root")
    escaped_root = replace(
        root,
        attempt_results=root.root.parent / "outside_root_results",
    )

    worker = build_worker_storage_layout(tmp_path / "worker")
    parent = build_worker_parent_storage_layout(
        worker,
        ParentId("0123456789abcdef"),
    )
    escaped_parent = replace(
        parent,
        attempt_results=parent.parent_root.parent / "outside_worker_results",
    )

    with pytest.raises(
        ValueError,
        match=r"Root attempt-results directory .* must be contained within root",
    ):
        root_bundle_publication_paths(
            escaped_root,
            ModelId("1"),
        )

    with pytest.raises(
        ValueError,
        match=(r"Worker attempt-results directory .* must be contained within " r"parent root"),
    ):
        worker_bundle_publication_paths(
            escaped_parent,
            ModelId("1"),
        )

    assert not escaped_root.attempt_results.exists()
    assert not escaped_parent.attempt_results.exists()
