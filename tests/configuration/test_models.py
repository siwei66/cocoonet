"""Tests for prospective model registry records."""

import os
import re
from dataclasses import FrozenInstanceError, fields
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from cocoonet._contracts.execution import (
    ConsolidationState,
    EvaluationReference,
    ModelFamily,
    ModelReference,
    TaskKind,
)
from cocoonet._contracts.identity import EvaluationConfigId, ModelId, ParentId
from cocoonet._contracts.state import ExecutionStatus
from cocoonet.evaluation_task import EvaluationTask, _RegisteredModel


def test_registered_model_exposes_reference_identity_and_label() -> None:
    """Derive model identity and label from the authoritative reference."""
    reference = ModelReference(
        model_id=ModelId("1"),
        model_label="baseline",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.REGRESSION,
    )
    model_input = object()

    registered = _RegisteredModel(
        reference=reference,
        model_input=model_input,
    )

    assert registered.reference is reference
    assert registered.model_input is model_input
    assert registered.model_id == reference.model_id
    assert registered.model_label == reference.model_label
    assert tuple(field.name for field in fields(registered)) == (
        "reference",
        "model_input",
    )


def test_registered_model_is_frozen_slotted_and_keyword_only() -> None:
    """Keep the private model association immutable and explicit."""
    registered = _RegisteredModel(
        reference=ModelReference(
            model_id=ModelId("1"),
            model_label="baseline",
            family=ModelFamily.SKLEARN,
            task_kind=TaskKind.REGRESSION,
        ),
        model_input=object(),
    )

    assert not hasattr(registered, "__dict__")
    assert all(
        parameter.kind is Parameter.KEYWORD_ONLY for parameter in signature(_RegisteredModel).parameters.values()
    )

    with pytest.raises(FrozenInstanceError):
        registered.__setattr__("model_input", object())


def test_add_model_assigns_sequential_ids_and_preserves_opaque_inputs(
    tmp_path: Path,
) -> None:
    """Commit models in order without copying their opaque inputs."""
    task = EvaluationTask(tmp_path / "work-dir")
    first_input = object()
    second_input = object()

    first = task.add_model(
        first_input,
        model_label="first",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.REGRESSION,
    )
    second = task.add_model(
        second_input,
        model_label="second",
        family=ModelFamily.PYTORCH,
        task_kind=TaskKind.REGRESSION,
    )

    assert first.model_id == ModelId("1")
    assert second.model_id == ModelId("2")
    assert task._models[0].model_input is first_input
    assert task._models[1].model_input is second_input
    assert task._next_model_number == 3
    assert task.state is ConsolidationState.INITIALIZED
    assert tuple(task.snapshot().models) == ("first", "second")


def test_add_model_changes_consolidated_state_and_preserves_parent(
    tmp_path: Path,
) -> None:
    """Make committed scientific configuration unconsolidated."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._consolidation_state = ConsolidationState.CONSOLIDATED
    task._parent_id = ParentId("existing-parent")

    task.add_model(
        object(),
        model_label="baseline",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.CLASSIFICATION,
    )

    assert task.state is ConsolidationState.UNCONSOLIDATED
    assert task.parent_id == ParentId("existing-parent")


def test_add_model_rejects_duplicate_label_without_consuming_id(
    tmp_path: Path,
) -> None:
    """Leave registry, counter, and state unchanged after a collision."""
    task = EvaluationTask(tmp_path / "work-dir")
    task.add_model(
        object(),
        model_label="baseline",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.REGRESSION,
    )
    before = task.__dict__.copy()

    with pytest.raises(ValueError, match=r"model_label must be unique"):
        task.add_model(
            object(),
            model_label="baseline",
            family=ModelFamily.PYTORCH,
            task_kind=TaskKind.REGRESSION,
        )

    assert task.__dict__ == before
    assert task._next_model_number == 2


def test_add_model_rejects_mixed_task_kind_without_consuming_id(
    tmp_path: Path,
) -> None:
    """Require one task kind across the active model registry."""
    task = EvaluationTask(tmp_path / "work-dir")
    task.add_model(
        object(),
        model_label="regressor",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.REGRESSION,
    )

    with pytest.raises(ValueError, match=r"task_kind must match existing models"):
        task.add_model(
            object(),
            model_label="classifier",
            family=ModelFamily.SKLEARN,
            task_kind=TaskKind.CLASSIFICATION,
        )

    assert tuple(model.model_label for model in task._models) == ("regressor",)
    assert task._next_model_number == 2


def test_add_model_rejects_evaluation_task_mismatch(
    tmp_path: Path,
) -> None:
    """Require model task agreement with active evaluation configuration."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._evaluation_config = EvaluationReference(
        configuration_id=EvaluationConfigId("evaluation-a"),
        task_kind=TaskKind.CLASSIFICATION,
        class_labels=("negative", "positive"),
        influence_requested=False,
    )

    with pytest.raises(
        ValueError,
        match=r"task_kind must match evaluation_config\.task_kind",
    ):
        task.add_model(
            object(),
            model_label="regressor",
            family=ModelFamily.SKLEARN,
            task_kind=TaskKind.REGRESSION,
        )

    assert task._models == []
    assert task._next_model_number == 1


def test_add_model_rejects_invalid_registration_without_mutation(
    tmp_path: Path,
) -> None:
    """Delegate registration validation before committing state."""
    task = EvaluationTask(tmp_path / "work-dir")

    with pytest.raises(ValueError, match=r"model_input must not be None"):
        task.add_model(
            None,
            model_label="missing",
            family=ModelFamily.SKLEARN,
            task_kind=TaskKind.REGRESSION,
        )

    invalid_family: Any = "sklearn"
    with pytest.raises(TypeError, match="family"):
        task.add_model(
            object(),
            model_label="invalid-family",
            family=invalid_family,
            task_kind=TaskKind.REGRESSION,
        )

    assert task._models == []
    assert task._next_model_number == 1


def test_add_model_accepts_arbitrary_compound_file_extension_without_reading(
    tmp_path: Path,
) -> None:
    """Store a canonical opaque path without loader or suffix association."""
    task = EvaluationTask(tmp_path / "work-dir")
    model_path = tmp_path / "opaque.Custom.TAR.GZ"
    model_path.write_bytes(b"\x80not-a-model-payload")

    reference = task.add_model(
        model_path,
        model_label="opaque",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.REGRESSION,
    )

    assert reference.model_id == ModelId("1")
    assert task._models[0].model_input == model_path.resolve()
    assert task._models[0].model_input.name == "opaque.Custom.TAR.GZ"


def test_add_model_resolves_relative_text_path_from_console_directory(
    tmp_path: Path,
) -> None:
    """Resolve relative text paths while preserving the source filename."""
    task = EvaluationTask(tmp_path / "work-dir")
    model_path = tmp_path / "relative.unknown"
    model_path.write_bytes(b"opaque")
    previous_directory = Path.cwd()
    os.chdir(tmp_path)
    try:
        task.add_model(
            "relative.unknown",
            model_label="relative",
            family=ModelFamily.PYTORCH,
            task_kind=TaskKind.REGRESSION,
        )
    finally:
        os.chdir(previous_directory)

    assert task._models[0].model_input == model_path.resolve()


@pytest.mark.parametrize(
    "name",
    [
        "missing.model",
        "directory.model",
        "extensionless",
    ],
)
def test_add_model_rejects_invalid_file_reference(
    tmp_path: Path,
    name: str,
) -> None:
    """Reject missing, non-regular, and extensionless file references."""
    task = EvaluationTask(tmp_path / "work-dir")
    candidate = tmp_path / name
    if name == "directory.model":
        candidate.mkdir()
    elif name == "extensionless":
        candidate.write_bytes(b"opaque")

    expected = "nonempty extension" if name == "extensionless" else "existing regular file"
    with pytest.raises(ValueError, match=expected):
        task.add_model(
            candidate,
            model_label="invalid-path",
            family=ModelFamily.SKLEARN,
            task_kind=TaskKind.REGRESSION,
        )

    assert task._models == []
    assert task._next_model_number == 1


def test_add_model_rejects_running_before_staging(
    tmp_path: Path,
) -> None:
    """Reject active-execution mutation without consuming an identity."""
    task = EvaluationTask(tmp_path / "work-dir")
    task._execution_status = ExecutionStatus.RUNNING

    with pytest.raises(RuntimeError, match=r"execution is running"):
        task.add_model(
            object(),
            model_label="blocked",
            family=ModelFamily.SKLEARN,
            task_kind=TaskKind.REGRESSION,
        )

    assert task._models == []
    assert task._next_model_number == 1


def test_list_models_returns_string_table_in_registry_order(
    tmp_path: Path,
) -> None:
    """Present all registrations without mutating their stored values."""
    task = EvaluationTask(tmp_path / "work-dir")
    first_input = object()
    task.add_model(
        first_input,
        model_label="first",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.REGRESSION,
    )
    task.add_model(
        [],
        model_label="second",
        family=ModelFamily.PYTORCH,
        task_kind=TaskKind.REGRESSION,
    )

    result = task.list_models()

    assert list(result.columns) == [
        "model_id",
        "model_label",
        "family",
        "task_kind",
        "model_input",
    ]
    assert result.to_dict(orient="records") == [
        {
            "model_id": "1",
            "model_label": "first",
            "family": "sklearn",
            "task_kind": "regression",
            "model_input": "object",
        },
        {
            "model_id": "2",
            "model_label": "second",
            "family": "pytorch",
            "task_kind": "regression",
            "model_input": "list",
        },
    ]
    assert all(dtype == pd.StringDtype() for dtype in result.dtypes)
    assert result.index.equals(pd.RangeIndex(2))
    assert task._models[0].model_input is first_input


def test_list_models_combines_exact_criteria(
    tmp_path: Path,
) -> None:
    """Apply exact criteria together with logical AND semantics."""
    task = EvaluationTask(tmp_path / "work-dir")
    task.add_model(
        object(),
        model_label="first",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.REGRESSION,
    )
    task.add_model(
        object(),
        model_label="second",
        family=ModelFamily.PYTORCH,
        task_kind=TaskKind.REGRESSION,
    )

    result = task.list_models(
        label="second",
        family="pytorch",
        model_input="object",
        use_regex=False,
    )

    assert result["model_label"].tolist() == ["second"]
    assert task.list_models(label="second", family="sklearn", use_regex=False).empty


def test_list_models_uses_case_sensitive_regex_fullmatch(
    tmp_path: Path,
) -> None:
    """Use case-sensitive full matching for regular-expression searches."""
    task = EvaluationTask(tmp_path / "work-dir")
    for label in ("alpha-1", "alpha-12", "Alpha-1"):
        task.add_model(
            object(),
            model_label=label,
            family=ModelFamily.SKLEARN,
            task_kind=TaskKind.REGRESSION,
        )

    result = task.list_models(label=r"alpha-\d")

    assert result["model_label"].tolist() == ["alpha-1"]


def test_list_models_displays_and_filters_canonical_file_path(
    tmp_path: Path,
) -> None:
    """Expose the stored canonical path for file-backed inputs."""
    task = EvaluationTask(tmp_path / "work-dir")
    model_path = tmp_path / "model.custom"
    model_path.write_bytes(b"opaque")
    task.add_model(
        model_path,
        model_label="file-model",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.REGRESSION,
    )
    canonical_path = str(model_path.resolve())

    result = task.list_models(model_input=re.escape(canonical_path))

    assert result["model_input"].tolist() == [canonical_path]


def test_list_models_empty_result_retains_public_schema(
    tmp_path: Path,
) -> None:
    """Retain string columns and a zero-based index for no matches."""
    task = EvaluationTask(tmp_path / "work-dir")

    result = task.list_models(label="missing", use_regex=False)

    assert list(result.columns) == [
        "model_id",
        "model_label",
        "family",
        "task_kind",
        "model_input",
    ]
    assert all(dtype == pd.StringDtype() for dtype in result.dtypes)
    assert result.index.equals(pd.RangeIndex(0))


def test_list_models_rejects_invalid_regular_expression(
    tmp_path: Path,
) -> None:
    """Identify which search criterion contains an invalid expression."""
    task = EvaluationTask(tmp_path / "work-dir")

    with pytest.raises(ValueError, match=r"label must be a valid regular expression"):
        task.list_models(label="[")


def test_list_models_warns_when_limit_truncates_matches(
    tmp_path: Path,
) -> None:
    """Return the registry-order prefix and warn when matches are truncated."""
    task = EvaluationTask(tmp_path / "work-dir")
    for label in ("first", "second"):
        task.add_model(
            object(),
            model_label=label,
            family=ModelFamily.SKLEARN,
            task_kind=TaskKind.REGRESSION,
        )

    with pytest.warns(UserWarning, match=r"truncated to 1 of 2"):
        result = task.list_models(limit=1)

    assert result["model_label"].tolist() == ["first"]


def test_list_models_preserves_native_bool_limit_behavior(
    tmp_path: Path,
) -> None:
    """Treat true as one and reject false through positive-limit validation."""
    task = EvaluationTask(tmp_path / "work-dir")
    for label in ("first", "second"):
        task.add_model(
            object(),
            model_label=label,
            family=ModelFamily.SKLEARN,
            task_kind=TaskKind.REGRESSION,
        )

    with pytest.warns(UserWarning, match=r"truncated to True of 2"):
        result = task.list_models(limit=True)

    assert result["model_label"].tolist() == ["first"]
    with pytest.raises(ValueError, match=r"limit must be positive"):
        task.list_models(limit=False)


@pytest.mark.parametrize(
    ("keyword", "value", "error_type", "message"),
    [
        ("label", 1, TypeError, "label"),
        ("family", 1, TypeError, "family"),
        ("model_input", 1, TypeError, "model_input"),
        ("use_regex", 1, TypeError, "use_regex"),
        ("limit", 1.5, TypeError, "limit"),
        ("limit", 0, ValueError, "positive"),
    ],
)
def test_list_models_rejects_invalid_search_arguments(
    tmp_path: Path,
    keyword: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject invalid search argument types and nonpositive limits."""
    task = EvaluationTask(tmp_path / "work-dir")
    list_models: Any = task.list_models

    with pytest.raises(error_type, match=message):
        list_models(**{keyword: value})
