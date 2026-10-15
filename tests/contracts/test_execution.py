"""Tests for configuration and fixed-execution contracts."""

from dataclasses import FrozenInstanceError

import pytest

from cocoonet._contracts.execution import (
    ConsolidationState,
    EvaluationReference,
    ModelFamily,
    ModelReference,
    TaskKind,
)
from cocoonet._contracts.identity import EvaluationConfigId, ModelId


def test_task_kind_has_exact_string_vocabulary() -> None:
    """Expose only the planned regression and classification task values."""
    assert tuple(TaskKind) == (
        TaskKind.REGRESSION,
        TaskKind.CLASSIFICATION,
    )
    assert TaskKind.REGRESSION == "regression"
    assert TaskKind.CLASSIFICATION == "classification"


def test_model_family_has_exact_string_vocabulary() -> None:
    """Expose only the planned model-construction family values."""
    assert tuple(ModelFamily) == (
        ModelFamily.SKLEARN,
        ModelFamily.SKLEARN_LIKE,
        ModelFamily.PYTORCH,
        ModelFamily.COCOONET,
    )
    assert ModelFamily.SKLEARN == "sklearn"
    assert ModelFamily.SKLEARN_LIKE == "sklearn-like"
    assert ModelFamily.PYTORCH == "pytorch"
    assert ModelFamily.COCOONET == "cocoonet"


def test_consolidation_state_has_exact_string_vocabulary() -> None:
    """Expose only the planned configuration-consolidation states."""
    assert tuple(ConsolidationState) == (
        ConsolidationState.INITIALIZED,
        ConsolidationState.CONSOLIDATED,
        ConsolidationState.UNCONSOLIDATED,
    )
    assert ConsolidationState.INITIALIZED == "initialized"
    assert ConsolidationState.CONSOLIDATED == "consolidated"
    assert ConsolidationState.UNCONSOLIDATED == "unconsolidated"


def test_model_reference_preserves_fixed_model_metadata() -> None:
    """Preserve model identity, label, family, and task-kind associations."""
    reference = ModelReference(
        model_id=ModelId("model-a"),
        model_label="baseline",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.CLASSIFICATION,
    )
    same = ModelReference(
        model_id=ModelId("model-a"),
        model_label="baseline",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.CLASSIFICATION,
    )
    different_model = ModelReference(
        model_id=ModelId("model-b"),
        model_label="baseline",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.CLASSIFICATION,
    )

    assert reference.model_id == ModelId("model-a")
    assert reference.model_label == "baseline"
    assert reference.family is ModelFamily.SKLEARN
    assert reference.task_kind is TaskKind.CLASSIFICATION
    assert reference == same
    assert reference != different_model


def test_model_reference_rejects_field_mutation() -> None:
    """Prevent reassignment of fixed model-reference metadata."""
    reference = ModelReference(
        model_id=ModelId("model-a"),
        model_label="baseline",
        family=ModelFamily.COCOONET,
        task_kind=TaskKind.REGRESSION,
    )

    with pytest.raises(FrozenInstanceError):
        reference.__setattr__("model_label", "replacement")


def test_evaluation_reference_preserves_fixed_settings_metadata() -> None:
    """Preserve the settings identity, task, class order, and influence request."""
    reference = EvaluationReference(
        configuration_id=EvaluationConfigId("evaluation-a"),
        task_kind=TaskKind.CLASSIFICATION,
        class_labels=("negative", "positive"),
        influence_requested=True,
    )
    same = EvaluationReference(
        configuration_id=EvaluationConfigId("evaluation-a"),
        task_kind=TaskKind.CLASSIFICATION,
        class_labels=("negative", "positive"),
        influence_requested=True,
    )
    different_configuration = EvaluationReference(
        configuration_id=EvaluationConfigId("evaluation-b"),
        task_kind=TaskKind.CLASSIFICATION,
        class_labels=("negative", "positive"),
        influence_requested=True,
    )

    assert reference.configuration_id == EvaluationConfigId("evaluation-a")
    assert reference.task_kind is TaskKind.CLASSIFICATION
    assert reference.class_labels == ("negative", "positive")
    assert reference.influence_requested is True
    assert reference == same
    assert reference != different_configuration


def test_evaluation_reference_rejects_field_mutation() -> None:
    """Prevent reassignment of fixed evaluation-reference metadata."""
    reference = EvaluationReference(
        configuration_id=EvaluationConfigId("evaluation-a"),
        task_kind=TaskKind.REGRESSION,
        class_labels=(),
        influence_requested=False,
    )

    with pytest.raises(FrozenInstanceError):
        reference.__setattr__("influence_requested", True)
