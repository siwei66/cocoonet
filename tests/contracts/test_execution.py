"""Tests for configuration and fixed-execution contracts."""

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from typing import cast

import numpy as np
import pytest

from cocoonet._contracts.data import PartitionReference
from cocoonet._contracts.execution import (
    AttemptInputs,
    ComputationSnapshot,
    ConsolidationState,
    EvaluationReference,
    ModelFamily,
    ModelReference,
    ParentEvaluation,
    TaskKind,
    validate_attempt_inputs,
    validate_computation_snapshot,
    validate_evaluation_reference,
    validate_model_reference,
    validate_parent_evaluation,
)
from cocoonet._contracts.identity import (
    AttemptId,
    DatasetId,
    EvaluationConfigId,
    ModelId,
    ParentId,
    PartitionId,
    ScalarLabel,
    WorkerId,
)


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


def test_computation_snapshot_preserves_configuration_references() -> None:
    """Preserve the complete prospective configuration and parent association."""
    partition = PartitionReference(
        partition_id=PartitionId("partition-a"),
        dataset_id=DatasetId("dataset-a"),
    )
    evaluation = EvaluationReference(
        configuration_id=EvaluationConfigId("evaluation-a"),
        task_kind=TaskKind.CLASSIFICATION,
        class_labels=("negative", "positive"),
        influence_requested=False,
    )
    model = ModelReference(
        model_id=ModelId("model-a"),
        model_label="baseline",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.CLASSIFICATION,
    )
    snapshot = ComputationSnapshot(
        root_dir="computation-root",
        consolidation=ConsolidationState.CONSOLIDATED,
        partition=partition,
        evaluation=evaluation,
        models=(model,),
        worker_ids=(WorkerId("worker-a"),),
        parent_id=ParentId("parent-a"),
    )
    same = ComputationSnapshot(
        root_dir="computation-root",
        consolidation=ConsolidationState.CONSOLIDATED,
        partition=partition,
        evaluation=evaluation,
        models=(model,),
        worker_ids=(WorkerId("worker-a"),),
        parent_id=ParentId("parent-a"),
    )

    assert snapshot.root_dir == "computation-root"
    assert snapshot.consolidation is ConsolidationState.CONSOLIDATED
    assert snapshot.partition is partition
    assert snapshot.evaluation is evaluation
    assert snapshot.models == (model,)
    assert snapshot.worker_ids == (WorkerId("worker-a"),)
    assert snapshot.parent_id == ParentId("parent-a")
    assert snapshot == same


def test_computation_snapshot_supports_incomplete_initialized_configuration() -> None:
    """Represent an initialized computation before optional configuration is added."""
    snapshot = ComputationSnapshot(
        root_dir="computation-root",
        consolidation=ConsolidationState.INITIALIZED,
        partition=None,
        evaluation=None,
        models=(),
        worker_ids=(),
        parent_id=None,
    )

    assert snapshot.partition is None
    assert snapshot.evaluation is None
    assert snapshot.models == ()
    assert snapshot.worker_ids == ()
    assert snapshot.parent_id is None


def test_computation_snapshot_rejects_field_mutation() -> None:
    """Prevent reassignment of recorded computation-configuration fields."""
    snapshot = ComputationSnapshot(
        root_dir="computation-root",
        consolidation=ConsolidationState.INITIALIZED,
        partition=None,
        evaluation=None,
        models=(),
        worker_ids=(),
        parent_id=None,
    )

    with pytest.raises(FrozenInstanceError):
        snapshot.__setattr__("root_dir", "replacement-root")


def test_parent_evaluation_preserves_fixed_scientific_context() -> None:
    """Preserve the parent's partition, evaluation settings, and ordered models."""
    parent_id = ParentId("parent-a")
    partition = PartitionReference(
        partition_id=PartitionId("partition-a"),
        dataset_id=DatasetId("dataset-a"),
    )
    evaluation = EvaluationReference(
        configuration_id=EvaluationConfigId("evaluation-a"),
        task_kind=TaskKind.REGRESSION,
        class_labels=(),
        influence_requested=False,
    )
    unit = ModelReference(
        model_id=ModelId("model-a"),
        model_label="baseline",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.REGRESSION,
    )
    parent = ParentEvaluation(
        parent_id=parent_id,
        partition=partition,
        evaluation=evaluation,
        models=(unit,),
    )
    same = ParentEvaluation(
        parent_id=parent_id,
        partition=partition,
        evaluation=evaluation,
        models=(unit,),
    )

    assert parent.parent_id == parent_id
    assert parent.partition is partition
    assert parent.evaluation is evaluation
    assert parent.models == (unit,)
    assert parent == same


def test_parent_evaluation_rejects_field_mutation() -> None:
    """Prevent reassignment of a parent task's fixed scientific context."""
    parent = ParentEvaluation(
        parent_id=ParentId("parent-a"),
        partition=PartitionReference(
            partition_id=PartitionId("partition-a"),
            dataset_id=DatasetId("dataset-a"),
        ),
        evaluation=EvaluationReference(
            configuration_id=EvaluationConfigId("evaluation-a"),
            task_kind=TaskKind.REGRESSION,
            class_labels=(),
            influence_requested=False,
        ),
        models=(),
    )

    with pytest.raises(FrozenInstanceError):
        parent.__setattr__(
            "partition",
            PartitionReference(
                partition_id=PartitionId("partition-b"),
                dataset_id=DatasetId("dataset-a"),
            ),
        )


def test_attempt_inputs_preserves_routing_identities() -> None:
    """Preserve the attempt, logical unit, and assigned worker identities."""
    inputs = AttemptInputs(
        attempt_id=AttemptId("attempt-a"),
        parent_id=ParentId("parent-a"),
        model_id=ModelId("model-a"),
        worker_id=WorkerId("worker-a"),
    )
    same = AttemptInputs(
        attempt_id=AttemptId("attempt-a"),
        parent_id=ParentId("parent-a"),
        model_id=ModelId("model-a"),
        worker_id=WorkerId("worker-a"),
    )

    assert inputs.attempt_id == AttemptId("attempt-a")
    assert inputs.parent_id == ParentId("parent-a")
    assert inputs.model_id == ModelId("model-a")
    assert inputs.worker_id == WorkerId("worker-a")
    assert inputs == same


def test_attempt_inputs_rejects_field_mutation() -> None:
    """Prevent reassignment of an attempt's routing identities."""
    inputs = AttemptInputs(
        attempt_id=AttemptId("attempt-a"),
        parent_id=ParentId("parent-a"),
        model_id=ModelId("model-a"),
        worker_id=WorkerId("worker-a"),
    )

    with pytest.raises(FrozenInstanceError):
        inputs.__setattr__("worker_id", WorkerId("worker-b"))


@pytest.mark.parametrize("family", tuple(ModelFamily))
@pytest.mark.parametrize("task_kind", tuple(TaskKind))
def test_validate_model_reference_accepts_declared_tags(
    family: ModelFamily,
    task_kind: TaskKind,
) -> None:
    """Accept every declared model family and scientific task kind."""
    reference = ModelReference(
        model_id=ModelId("model-a"),
        model_label="baseline",
        family=family,
        task_kind=task_kind,
    )

    validate_model_reference(reference)


def test_validate_model_reference_rejects_empty_model_id() -> None:
    """Reject an empty model identity and identify the invalid field."""
    reference = ModelReference(
        model_id=ModelId(""),
        model_label="baseline",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.REGRESSION,
    )

    with pytest.raises(ValueError, match=r"^model_id.*nonempty.*empty string"):
        validate_model_reference(reference)


def test_validate_model_reference_rejects_nonfinite_model_label() -> None:
    """Reject a nonfinite floating-point model label."""
    reference = ModelReference(
        model_id=ModelId("model-a"),
        model_label=cast(ScalarLabel, float("nan")),
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.REGRESSION,
    )

    with pytest.raises(TypeError, match="path: model_label"):
        validate_model_reference(reference)


def test_validate_model_reference_rejects_undeclared_family() -> None:
    """Reject a plain string that is not a declared model-family member."""
    reference = ModelReference(
        model_id=ModelId("model-a"),
        model_label="baseline",
        family=cast(ModelFamily, "unsupported"),
        task_kind=TaskKind.REGRESSION,
    )

    with pytest.raises(TypeError) as error:
        validate_model_reference(reference)

    assert "path: family" in str(error.value)
    assert "expected: cocoonet._contracts.execution.ModelFamily" in str(error.value)


def test_validate_model_reference_rejects_undeclared_task_kind() -> None:
    """Reject a plain string that is not a declared task-kind member."""
    reference = ModelReference(
        model_id=ModelId("model-a"),
        model_label="baseline",
        family=ModelFamily.SKLEARN,
        task_kind=cast(TaskKind, "unsupported"),
    )

    with pytest.raises(TypeError) as error:
        validate_model_reference(reference)

    assert "path: task_kind" in str(error.value)
    assert "expected: cocoonet._contracts.execution.TaskKind" in str(error.value)


@pytest.mark.parametrize(
    ("task_kind", "class_labels"),
    [
        (TaskKind.REGRESSION, ()),
        (TaskKind.CLASSIFICATION, ("only-class",)),
        (TaskKind.CLASSIFICATION, ("class-b", "class-a")),
    ],
)
def test_validate_evaluation_reference_accepts_task_class_rules(
    task_kind: TaskKind,
    class_labels: tuple[ScalarLabel, ...],
) -> None:
    """Accept valid regression and ordered classification vocabularies."""
    reference = EvaluationReference(
        configuration_id=EvaluationConfigId("evaluation-a"),
        task_kind=task_kind,
        class_labels=class_labels,
        influence_requested=True,
    )

    validate_evaluation_reference(reference)

    assert reference.class_labels == class_labels


@pytest.mark.parametrize(
    ("task_kind", "class_labels", "message"),
    [
        (
            TaskKind.REGRESSION,
            ("unexpected",),
            "Regression.*zero class labels.*got 1",
        ),
        (
            TaskKind.CLASSIFICATION,
            (),
            "Classification.*at least one class label.*got zero",
        ),
    ],
)
def test_validate_evaluation_reference_rejects_invalid_task_class_rules(
    task_kind: TaskKind,
    class_labels: tuple[ScalarLabel, ...],
    message: str,
) -> None:
    """Reject class vocabularies that conflict with the scientific task."""
    reference = EvaluationReference(
        configuration_id=EvaluationConfigId("evaluation-a"),
        task_kind=task_kind,
        class_labels=class_labels,
        influence_requested=False,
    )

    with pytest.raises(ValueError, match=message):
        validate_evaluation_reference(reference)


def test_validate_evaluation_reference_rejects_equality_collision() -> None:
    """Reject class labels that collide under Python text equality."""
    reference = EvaluationReference(
        configuration_id=EvaluationConfigId("evaluation-a"),
        task_kind=TaskKind.CLASSIFICATION,
        class_labels=("duplicate", np.str_("duplicate")),
        influence_requested=False,
    )

    with pytest.raises(
        ValueError,
        match=(r"class_labels\[1\].*unique.*duplicate"),
    ):
        validate_evaluation_reference(reference)


def test_validate_evaluation_reference_rejects_invalid_class_label() -> None:
    """Reject a nonfinite class label and report its tuple position."""
    reference = EvaluationReference(
        configuration_id=EvaluationConfigId("evaluation-a"),
        task_kind=TaskKind.CLASSIFICATION,
        class_labels=("valid", cast(ScalarLabel, float("nan"))),
        influence_requested=False,
    )

    with pytest.raises(TypeError, match=r"path: class_labels\[1\]"):
        validate_evaluation_reference(reference)


def test_validate_evaluation_reference_rejects_empty_configuration_id() -> None:
    """Reject an empty evaluation-configuration identity."""
    reference = EvaluationReference(
        configuration_id=EvaluationConfigId(""),
        task_kind=TaskKind.REGRESSION,
        class_labels=(),
        influence_requested=False,
    )

    with pytest.raises(ValueError, match=r"configuration_id.*nonempty.*empty string"):
        validate_evaluation_reference(reference)


def test_validate_evaluation_reference_rejects_undeclared_task_kind() -> None:
    """Reject a plain string that is not a declared task-kind member."""
    reference = EvaluationReference(
        configuration_id=EvaluationConfigId("evaluation-a"),
        task_kind=cast(TaskKind, "unsupported"),
        class_labels=(),
        influence_requested=False,
    )

    with pytest.raises(TypeError) as error:
        validate_evaluation_reference(reference)

    assert "path: task_kind" in str(error.value)
    assert "expected: cocoonet._contracts.execution.TaskKind" in str(error.value)


def test_validate_evaluation_reference_rejects_non_boolean_influence_request() -> None:
    """Reject a non-boolean influence-analysis request."""
    reference = EvaluationReference(
        configuration_id=EvaluationConfigId("evaluation-a"),
        task_kind=TaskKind.REGRESSION,
        class_labels=(),
        influence_requested=cast(bool, 1),
    )

    with pytest.raises(TypeError) as error:
        validate_evaluation_reference(reference)

    assert "path: influence_requested" in str(error.value)
    assert "expected: builtins.bool" in str(error.value)


def test_validate_computation_snapshot_accepts_empty_initialized_snapshot() -> None:
    """Accept incomplete initialized configuration without changing it."""
    snapshot = ComputationSnapshot(
        root_dir="computation-root",
        consolidation=ConsolidationState.INITIALIZED,
        partition=None,
        evaluation=None,
        models=(),
        worker_ids=(),
        parent_id=None,
    )

    validate_computation_snapshot(snapshot)

    assert snapshot.partition is None
    assert snapshot.evaluation is None
    assert snapshot.models == ()
    assert snapshot.worker_ids == ()
    assert snapshot.parent_id is None


def test_validate_computation_snapshot_preserves_parent_after_worker_edit() -> None:
    """Allow worker-only edits while retaining a consolidated parent identity."""
    snapshot = ComputationSnapshot(
        root_dir="computation-root",
        consolidation=ConsolidationState.CONSOLIDATED,
        partition=PartitionReference(
            partition_id=PartitionId("partition-a"),
            dataset_id=DatasetId("dataset-a"),
        ),
        evaluation=EvaluationReference(
            configuration_id=EvaluationConfigId("evaluation-a"),
            task_kind=TaskKind.REGRESSION,
            class_labels=(),
            influence_requested=False,
        ),
        models=(
            ModelReference(
                model_id=ModelId("model-a"),
                model_label="baseline",
                family=ModelFamily.SKLEARN,
                task_kind=TaskKind.REGRESSION,
            ),
        ),
        worker_ids=(WorkerId("worker-a"),),
        parent_id=ParentId("parent-a"),
    )
    edited = replace(
        snapshot,
        worker_ids=(WorkerId("worker-b"),),
    )

    validate_computation_snapshot(snapshot)
    validate_computation_snapshot(edited)

    assert edited.parent_id == snapshot.parent_id
    assert edited.consolidation is ConsolidationState.CONSOLIDATED


def test_validate_computation_snapshot_accepts_incomplete_unconsolidated_snapshot() -> None:
    """Allow incomplete prospective configuration while retaining the old parent."""
    snapshot = ComputationSnapshot(
        root_dir="computation-root",
        consolidation=ConsolidationState.UNCONSOLIDATED,
        partition=None,
        evaluation=None,
        models=(),
        worker_ids=(),
        parent_id=ParentId("parent-a"),
    )

    validate_computation_snapshot(snapshot)


@pytest.mark.parametrize(
    ("consolidation", "parent_id", "message"),
    [
        (
            ConsolidationState.INITIALIZED,
            ParentId("parent-a"),
            "Initialized.*parent_id=None.*parent-a",
        ),
        (
            ConsolidationState.CONSOLIDATED,
            None,
            "consolidated.*parent_id.*got None",
        ),
        (
            ConsolidationState.UNCONSOLIDATED,
            None,
            "consolidated.*parent_id.*got None",
        ),
    ],
)
def test_validate_computation_snapshot_rejects_invalid_parent_relationship(
    consolidation: ConsolidationState,
    parent_id: ParentId | None,
    message: str,
) -> None:
    """Reject parent identities that conflict with consolidation state."""
    snapshot = ComputationSnapshot(
        root_dir="computation-root",
        consolidation=consolidation,
        partition=None,
        evaluation=None,
        models=(),
        worker_ids=(),
        parent_id=parent_id,
    )

    with pytest.raises(ValueError, match=message):
        validate_computation_snapshot(snapshot)


def test_validate_computation_snapshot_rejects_empty_root_dir() -> None:
    """Reject an empty recorded root-directory value."""
    snapshot = ComputationSnapshot(
        root_dir="",
        consolidation=ConsolidationState.INITIALIZED,
        partition=None,
        evaluation=None,
        models=(),
        worker_ids=(),
        parent_id=None,
    )

    with pytest.raises(ValueError, match=r"^root_dir.*nonempty.*empty string"):
        validate_computation_snapshot(snapshot)


@pytest.mark.parametrize(
    ("partition_id", "dataset_id", "field"),
    [
        (PartitionId(""), DatasetId("dataset-a"), "partition.partition_id"),
        (PartitionId("partition-a"), DatasetId(""), "partition.dataset_id"),
    ],
)
def test_validate_computation_snapshot_rejects_empty_partition_identity(
    partition_id: PartitionId,
    dataset_id: DatasetId,
    field: str,
) -> None:
    """Reject an empty partition or associated dataset identity."""
    snapshot = ComputationSnapshot(
        root_dir="computation-root",
        consolidation=ConsolidationState.INITIALIZED,
        partition=PartitionReference(
            partition_id=partition_id,
            dataset_id=dataset_id,
        ),
        evaluation=None,
        models=(),
        worker_ids=(),
        parent_id=None,
    )

    with pytest.raises(ValueError, match=rf"^{field}.*nonempty.*empty string"):
        validate_computation_snapshot(snapshot)


def test_validate_computation_snapshot_rejects_invalid_evaluation() -> None:
    """Validate an included evaluation reference through its own contract."""
    snapshot = ComputationSnapshot(
        root_dir="computation-root",
        consolidation=ConsolidationState.INITIALIZED,
        partition=None,
        evaluation=EvaluationReference(
            configuration_id=EvaluationConfigId(""),
            task_kind=TaskKind.REGRESSION,
            class_labels=(),
            influence_requested=False,
        ),
        models=(),
        worker_ids=(),
        parent_id=None,
    )

    with pytest.raises(
        ValueError,
        match=r"configuration_id.*nonempty.*empty string",
    ):
        validate_computation_snapshot(snapshot)


def test_validate_computation_snapshot_rejects_duplicate_model_id() -> None:
    """Reject repeated model identities in the prospective configuration."""
    models = (
        ModelReference(
            model_id=ModelId("model-a"),
            model_label="first",
            family=ModelFamily.SKLEARN,
            task_kind=TaskKind.REGRESSION,
        ),
        ModelReference(
            model_id=ModelId("model-a"),
            model_label="second",
            family=ModelFamily.PYTORCH,
            task_kind=TaskKind.REGRESSION,
        ),
    )
    snapshot = ComputationSnapshot(
        root_dir="computation-root",
        consolidation=ConsolidationState.INITIALIZED,
        partition=None,
        evaluation=None,
        models=models,
        worker_ids=(),
        parent_id=None,
    )

    with pytest.raises(
        ValueError,
        match=r"models\[1\].model_id.*unique.*model-a",
    ):
        validate_computation_snapshot(snapshot)


def test_validate_computation_snapshot_rejects_model_label_collision() -> None:
    """Reject model labels that collide under Python text equality."""
    models = (
        ModelReference(
            model_id=ModelId("model-a"),
            model_label="duplicate",
            family=ModelFamily.SKLEARN,
            task_kind=TaskKind.REGRESSION,
        ),
        ModelReference(
            model_id=ModelId("model-b"),
            model_label=np.str_("duplicate"),
            family=ModelFamily.PYTORCH,
            task_kind=TaskKind.REGRESSION,
        ),
    )
    snapshot = ComputationSnapshot(
        root_dir="computation-root",
        consolidation=ConsolidationState.INITIALIZED,
        partition=None,
        evaluation=None,
        models=models,
        worker_ids=(),
        parent_id=None,
    )

    with pytest.raises(
        ValueError,
        match=(r"models\[1\].model_label.*unique.*duplicate"),
    ):
        validate_computation_snapshot(snapshot)


def test_validate_computation_snapshot_rejects_duplicate_worker_id() -> None:
    """Reject repeated configured worker identities."""
    snapshot = ComputationSnapshot(
        root_dir="computation-root",
        consolidation=ConsolidationState.INITIALIZED,
        partition=None,
        evaluation=None,
        models=(),
        worker_ids=(WorkerId("worker-a"), WorkerId("worker-a")),
        parent_id=None,
    )

    with pytest.raises(
        ValueError,
        match=r"worker_ids\[1\].*unique.*worker-a",
    ):
        validate_computation_snapshot(snapshot)


def _valid_parent_evaluation(
    *,
    parent_id: ParentId | None = None,
    model_id: ModelId | None = None,
    model_label: ScalarLabel = "baseline",
) -> ParentEvaluation:
    """Construct a valid regression parent for focused validator tests."""
    resolved_parent_id = ParentId("parent-a") if parent_id is None else parent_id
    resolved_model_id = ModelId("model-a") if model_id is None else model_id

    partition = PartitionReference(
        partition_id=PartitionId("partition-a"),
        dataset_id=DatasetId("dataset-a"),
    )

    evaluation = EvaluationReference(
        configuration_id=EvaluationConfigId("evaluation-a"),
        task_kind=TaskKind.REGRESSION,
        class_labels=(),
        influence_requested=False,
    )

    model = ModelReference(
        model_id=resolved_model_id,
        model_label=model_label,
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.REGRESSION,
    )

    unit = model

    return ParentEvaluation(
        parent_id=resolved_parent_id,
        partition=partition,
        evaluation=evaluation,
        models=(unit,),
    )


def test_validate_parent_evaluation_accepts_same_label_in_different_parents() -> None:
    """Keep model-label uniqueness scoped independently to each parent."""
    first = _valid_parent_evaluation()
    second = _valid_parent_evaluation(parent_id=ParentId("parent-b"))

    validate_parent_evaluation(first)
    validate_parent_evaluation(second)

    assert first.models[0].model_label == second.models[0].model_label


def test_validate_parent_evaluation_accepts_mixed_model_families() -> None:
    """Allow different model families when every model shares the task kind."""
    parent = _valid_parent_evaluation()

    second_model = ModelReference(
        model_id=ModelId("model-b"),
        model_label="neural",
        family=ModelFamily.PYTORCH,
        task_kind=TaskKind.REGRESSION,
    )
    second_unit = second_model

    parent = replace(parent, models=(*parent.models, second_unit))

    validate_parent_evaluation(parent)


def test_validate_parent_evaluation_rejects_empty_parent_id() -> None:
    """Reject an empty committed parent identity."""
    parent = _valid_parent_evaluation(parent_id=ParentId(""))

    with pytest.raises(ValueError, match=r"^parent_id.*nonempty.*empty string"):
        validate_parent_evaluation(parent)


@pytest.mark.parametrize(
    ("partition_id", "dataset_id", "field"),
    [
        (PartitionId(""), DatasetId("dataset-a"), "partition.partition_id"),
        (PartitionId("partition-a"), DatasetId(""), "partition.dataset_id"),
    ],
)
def test_validate_parent_evaluation_rejects_empty_partition_identity(
    partition_id: PartitionId,
    dataset_id: DatasetId,
    field: str,
) -> None:
    """Reject an empty partition or associated dataset identity."""
    partition = PartitionReference(
        partition_id=partition_id,
        dataset_id=dataset_id,
    )
    parent = replace(
        _valid_parent_evaluation(),
        partition=partition,
    )

    with pytest.raises(ValueError, match=rf"^{field}.*nonempty.*empty string"):
        validate_parent_evaluation(parent)


def test_validate_parent_evaluation_rejects_empty_units() -> None:
    """Reject a committed parent without a model-specific unit."""
    parent = replace(
        _valid_parent_evaluation(),
        models=(),
    )

    with pytest.raises(ValueError, match=r"models.*at least one.*empty tuple"):
        validate_parent_evaluation(parent)


def test_validate_parent_evaluation_rejects_mixed_task_kinds() -> None:
    """Reject a model whose task kind differs from the parent evaluation."""
    parent = _valid_parent_evaluation()
    original_unit = parent.models[0]

    mismatched_model = replace(
        original_unit,
        task_kind=TaskKind.CLASSIFICATION,
    )
    mismatched_unit = mismatched_model
    parent = replace(
        parent,
        models=(mismatched_unit,),
    )

    with pytest.raises(
        ValueError,
        match=(r"models\[0\].task_kind.*regression.*classification"),
    ):
        validate_parent_evaluation(parent)


def test_validate_parent_evaluation_rejects_duplicate_model_id() -> None:
    """Reject repeated model identities within one parent."""
    parent = _valid_parent_evaluation()

    duplicate_model = ModelReference(
        model_id=ModelId("model-a"),
        model_label="second",
        family=ModelFamily.PYTORCH,
        task_kind=TaskKind.REGRESSION,
    )
    duplicate_unit = duplicate_model
    parent = replace(
        parent,
        models=(*parent.models, duplicate_unit),
    )

    with pytest.raises(
        ValueError,
        match=r"models\[1\].model_id.*unique.*model-a",
    ):
        validate_parent_evaluation(parent)


def test_validate_parent_evaluation_rejects_model_label_collision() -> None:
    """Reject parent-scoped model labels that collide under Python equality."""
    parent = _valid_parent_evaluation(model_label="duplicate")

    duplicate_model = ModelReference(
        model_id=ModelId("model-b"),
        model_label=np.str_("duplicate"),
        family=ModelFamily.PYTORCH,
        task_kind=TaskKind.REGRESSION,
    )
    duplicate_unit = duplicate_model
    parent = replace(
        parent,
        models=(*parent.models, duplicate_unit),
    )

    with pytest.raises(
        ValueError,
        match=(r"models\[1\].model_label.*unique.*duplicate"),
    ):
        validate_parent_evaluation(parent)


def test_validate_attempt_inputs_accepts_known_parent_unit() -> None:
    """Accept an attempt routed to a unit contained by the supplied parent."""
    parent = _valid_parent_evaluation()
    inputs = AttemptInputs(
        attempt_id=AttemptId("attempt-a"),
        parent_id=parent.parent_id,
        model_id=parent.models[0].model_id,
        worker_id=WorkerId("worker-a"),
    )

    validate_attempt_inputs(inputs, parent=parent)


def test_validate_attempt_inputs_accepts_duplicate_attempts_for_same_unit() -> None:
    """Allow distinct attempts and workers to execute the same logical unit."""
    parent = _valid_parent_evaluation()
    first = AttemptInputs(
        attempt_id=AttemptId("attempt-a"),
        parent_id=parent.parent_id,
        model_id=parent.models[0].model_id,
        worker_id=WorkerId("worker-a"),
    )
    duplicate = AttemptInputs(
        attempt_id=AttemptId("attempt-b"),
        parent_id=parent.parent_id,
        model_id=parent.models[0].model_id,
        worker_id=WorkerId("worker-b"),
    )

    validate_attempt_inputs(first, parent=parent)
    validate_attempt_inputs(duplicate, parent=parent)

    assert (first.parent_id, first.model_id) == (duplicate.parent_id, duplicate.model_id)
    assert first.attempt_id != duplicate.attempt_id
    assert first.worker_id != duplicate.worker_id


@pytest.mark.parametrize(
    ("attempt_id", "worker_id", "field"),
    [
        (AttemptId(""), WorkerId("worker-a"), "attempt_id"),
        (AttemptId("attempt-a"), WorkerId(""), "worker_id"),
    ],
)
def test_validate_attempt_inputs_rejects_empty_routing_identity(
    attempt_id: AttemptId,
    worker_id: WorkerId,
    field: str,
) -> None:
    """Reject an empty attempt or worker identity."""
    parent = _valid_parent_evaluation()
    inputs = AttemptInputs(
        attempt_id=attempt_id,
        parent_id=parent.parent_id,
        model_id=parent.models[0].model_id,
        worker_id=worker_id,
    )

    with pytest.raises(ValueError, match=rf"^{field}.*nonempty.*empty string"):
        validate_attempt_inputs(inputs, parent=parent)


def test_validate_attempt_inputs_rejects_invalid_unit_identity() -> None:
    """Reject an empty model identity in the supplied parent/model association."""
    parent = _valid_parent_evaluation()
    inputs = AttemptInputs(
        attempt_id=AttemptId("attempt-a"),
        parent_id=parent.parent_id,
        model_id=ModelId(""),
        worker_id=WorkerId("worker-a"),
    )

    with pytest.raises(ValueError, match=r"^model_id.*nonempty.*empty string"):
        validate_attempt_inputs(inputs, parent=parent)


def test_validate_attempt_inputs_rejects_cross_parent_unit() -> None:
    """Reject a parent/model association that explicitly identifies another parent task."""
    parent = _valid_parent_evaluation()
    inputs = AttemptInputs(
        attempt_id=AttemptId("attempt-a"),
        parent_id=ParentId("parent-b"),
        model_id=parent.models[0].model_id,
        worker_id=WorkerId("worker-a"),
    )

    with pytest.raises(
        ValueError,
        match=(r"parent_id.*parent.parent_id.*parent-a.*parent-b"),
    ):
        validate_attempt_inputs(inputs, parent=parent)


def test_validate_attempt_inputs_rejects_unknown_parent_unit() -> None:
    """Reject an unknown model unit within the correct parent task."""
    parent = _valid_parent_evaluation()
    inputs = AttemptInputs(
        attempt_id=AttemptId("attempt-a"),
        parent_id=parent.parent_id,
        model_id=ModelId("model-b"),
        worker_id=WorkerId("worker-a"),
    )

    with pytest.raises(
        ValueError,
        match=(r"model_id.*parent model.*model-b"),
    ):
        validate_attempt_inputs(inputs, parent=parent)


def test_validate_attempt_inputs_rejects_invalid_parent_context() -> None:
    """Reject routing against a parent that has no known unit set."""
    parent = replace(
        _valid_parent_evaluation(),
        models=(),
    )
    inputs = AttemptInputs(
        attempt_id=AttemptId("attempt-a"),
        parent_id=parent.parent_id,
        model_id=ModelId("model-a"),
        worker_id=WorkerId("worker-a"),
    )

    with pytest.raises(ValueError, match=r"models.*at least one.*empty tuple"):
        validate_attempt_inputs(inputs, parent=parent)


@pytest.mark.parametrize(
    "validator,record_type",
    [
        (validate_model_reference, ModelReference),
        (validate_evaluation_reference, EvaluationReference),
        (validate_computation_snapshot, ComputationSnapshot),
        (validate_parent_evaluation, ParentEvaluation),
    ],
)
def test_execution_boundaries_reject_wrong_records(validator: object, record_type: type) -> None:
    """Reject missing records before reading any fixed execution fields."""
    with pytest.raises(TypeError) as error:
        cast(Callable[[object], None], validator)(None)
    assert record_type.__name__ in str(error.value)
    assert "got type: builtins.NoneType" in str(error.value)


@pytest.mark.parametrize(
    "field,invalid", [("partition", None), ("evaluation", None), ("models", []), ("models", (None,))]
)
def test_parent_rejects_invalid_components(field: str, invalid: object) -> None:
    """Check required context records and model containers at the parent boundary."""
    # Widen only test construction so runtime validators receive the invalid candidate.
    parent = cast(Callable[..., ParentEvaluation], replace)(_valid_parent_evaluation(), **{field: invalid})
    with pytest.raises(TypeError) as error:
        validate_parent_evaluation(parent)
    assert field in str(error.value)
    assert "expected:" in str(error.value) and "got type:" in str(error.value)


@pytest.mark.parametrize(
    "field,invalid",
    [
        ("partition", "partition"),
        ("evaluation", "evaluation"),
        ("models", []),
        ("models", (None,)),
        ("worker_ids", []),
        ("worker_ids", (1,)),
        ("root_dir", 1),
        ("consolidation", "initialized"),
    ],
)
def test_snapshot_rejects_invalid_component_types(field: str, invalid: object) -> None:
    """Validate provided configuration without requiring startup completeness."""
    snapshot = ComputationSnapshot(
        root_dir="root",
        consolidation=ConsolidationState.INITIALIZED,
        partition=None,
        evaluation=None,
        models=(),
        worker_ids=(),
        parent_id=None,
    )
    with pytest.raises(TypeError) as error:
        validate_computation_snapshot(cast(Callable[..., ComputationSnapshot], replace)(snapshot, **{field: invalid}))
    assert field in str(error.value)


@pytest.mark.parametrize("field", ["parent_id", "model_id", "attempt_id", "worker_id"])
@pytest.mark.parametrize("invalid,error_type", [("", ValueError), (None, TypeError), (1, TypeError)])
def test_attempt_checks_each_direct_identity(field: str, invalid: object, error_type: type[Exception]) -> None:
    """Validate every routing identity without a composite unit-key record."""
    parent = _valid_parent_evaluation()
    inputs = AttemptInputs(
        parent_id=parent.parent_id,
        model_id=parent.models[0].model_id,
        attempt_id=AttemptId("attempt-a"),
        worker_id=WorkerId("worker-a"),
    )
    with pytest.raises(error_type) as error:
        validate_attempt_inputs(cast(Callable[..., AttemptInputs], replace)(inputs, **{field: invalid}), parent=parent)
    assert field in str(error.value)


def test_attempt_rejects_wrong_outer_record() -> None:
    """Reject an allocation payload that is not an AttemptInputs record."""
    with pytest.raises(TypeError, match="AttemptInputs"):
        validate_attempt_inputs(cast(AttemptInputs, None), parent=_valid_parent_evaluation())


def test_duplicate_attempts_on_same_worker_preserve_fixed_context() -> None:
    """Permit distinct attempts for a parent/model pair without worker overrides."""
    parent = _valid_parent_evaluation()
    first = AttemptInputs(
        parent_id=parent.parent_id,
        model_id=parent.models[0].model_id,
        attempt_id=AttemptId("attempt-a"),
        worker_id=WorkerId("worker-a"),
    )
    second = replace(first, attempt_id=AttemptId("attempt-b"))
    validate_attempt_inputs(first, parent=parent)
    validate_attempt_inputs(second, parent=parent)
    assert first.worker_id == second.worker_id
    assert parent == _valid_parent_evaluation()


@pytest.mark.parametrize("label", [0, True, np.int64(1), np.float32(0.5)])
def test_model_labels_reject_unstandardized_scalars(label: object) -> None:
    """Apply the common text-label domain to model labels."""
    model = replace(_valid_parent_evaluation().models[0], model_label=cast(ScalarLabel, label))
    with pytest.raises(TypeError, match="model_label"):
        validate_model_reference(model)


def test_attempt_rejects_invalid_parent_record() -> None:
    """Keep contextual TypeError when an allocation lacks a valid parent record."""
    inputs = AttemptInputs(
        parent_id=ParentId("parent-a"),
        model_id=ModelId("model-a"),
        attempt_id=AttemptId("attempt-a"),
        worker_id=WorkerId("worker-a"),
    )
    with pytest.raises(TypeError) as error:
        validate_attempt_inputs(inputs, parent=cast(ParentEvaluation, None))
    assert "parent is invalid" in str(error.value) and "ParentEvaluation" in str(error.value)


def test_duplicate_model_label_diagnostic_keeps_full_text() -> None:
    """Identify complete long labels while reporting their parent-scoped collision."""
    label = "model-" + "a" * 120 + "-distinct-ending"
    parent = _valid_parent_evaluation(model_label=label)
    second = replace(parent.models[0], model_id=ModelId("model-b"), model_label=np.str_(label))
    with pytest.raises(ValueError) as error:
        validate_parent_evaluation(replace(parent, models=(*parent.models, second)))
    assert "models[1].model_label" in str(error.value)
    assert label in str(error.value)
