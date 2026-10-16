"""Tests for configuration and fixed-execution contracts."""

from dataclasses import FrozenInstanceError

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
    UnitTask,
)
from cocoonet._contracts.identity import (
    AttemptId,
    DatasetId,
    EvaluationConfigId,
    ModelId,
    ParentId,
    PartitionId,
    UnitKey,
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


def test_unit_task_preserves_key_and_model_association() -> None:
    """Preserve the logical unit identity and its fixed model reference."""
    key = UnitKey(
        parent_id=ParentId("parent-a"),
        model_id=ModelId("model-a"),
    )
    model = ModelReference(
        model_id=ModelId("model-a"),
        model_label="baseline",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.REGRESSION,
    )
    unit = UnitTask(
        key=key,
        model=model,
    )
    same = UnitTask(
        key=key,
        model=model,
    )

    assert unit.key is key
    assert unit.model is model
    assert unit == same


def test_unit_task_rejects_field_mutation() -> None:
    """Prevent reassignment of a unit task's identity or model reference."""
    unit = UnitTask(
        key=UnitKey(
            parent_id=ParentId("parent-a"),
            model_id=ModelId("model-a"),
        ),
        model=ModelReference(
            model_id=ModelId("model-a"),
            model_label="baseline",
            family=ModelFamily.SKLEARN,
            task_kind=TaskKind.REGRESSION,
        ),
    )

    with pytest.raises(FrozenInstanceError):
        unit.__setattr__(
            "key",
            UnitKey(
                parent_id=ParentId("parent-b"),
                model_id=ModelId("model-a"),
            ),
        )


def test_parent_evaluation_preserves_fixed_scientific_context() -> None:
    """Preserve the parent's partition, evaluation settings, and ordered units."""
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
    unit = UnitTask(
        key=UnitKey(
            parent_id=parent_id,
            model_id=ModelId("model-a"),
        ),
        model=ModelReference(
            model_id=ModelId("model-a"),
            model_label="baseline",
            family=ModelFamily.SKLEARN,
            task_kind=TaskKind.REGRESSION,
        ),
    )
    parent = ParentEvaluation(
        parent_id=parent_id,
        partition=partition,
        evaluation=evaluation,
        units=(unit,),
    )
    same = ParentEvaluation(
        parent_id=parent_id,
        partition=partition,
        evaluation=evaluation,
        units=(unit,),
    )

    assert parent.parent_id == parent_id
    assert parent.partition is partition
    assert parent.evaluation is evaluation
    assert parent.units == (unit,)
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
        units=(),
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
    unit = UnitKey(
        parent_id=ParentId("parent-a"),
        model_id=ModelId("model-a"),
    )
    inputs = AttemptInputs(
        attempt_id=AttemptId("attempt-a"),
        unit=unit,
        worker_id=WorkerId("worker-a"),
    )
    same = AttemptInputs(
        attempt_id=AttemptId("attempt-a"),
        unit=unit,
        worker_id=WorkerId("worker-a"),
    )

    assert inputs.attempt_id == AttemptId("attempt-a")
    assert inputs.unit is unit
    assert inputs.worker_id == WorkerId("worker-a")
    assert inputs == same


def test_attempt_inputs_rejects_field_mutation() -> None:
    """Prevent reassignment of an attempt's routing identities."""
    inputs = AttemptInputs(
        attempt_id=AttemptId("attempt-a"),
        unit=UnitKey(
            parent_id=ParentId("parent-a"),
            model_id=ModelId("model-a"),
        ),
        worker_id=WorkerId("worker-a"),
    )

    with pytest.raises(FrozenInstanceError):
        inputs.__setattr__("worker_id", WorkerId("worker-b"))
