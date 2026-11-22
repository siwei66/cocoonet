"""Tests for evaluation-task lifecycle helpers."""

from dataclasses import replace
from types import MappingProxyType
from typing import Any

import pytest

from cocoonet._contracts.data import PartitionReference
from cocoonet._contracts.execution import (
    ConsolidationState,
    EvaluationReference,
    ModelFamily,
    ModelingSnapshot,
    ModelReference,
    TaskKind,
)
from cocoonet._contracts.identity import (
    DatasetId,
    EvaluationConfigId,
    ModelId,
    ParentId,
    PartitionId,
)
from cocoonet._contracts.state import ExecutionStatus
from cocoonet.evaluation_task import (
    _consolidation_after_evaluation_edit,
    _require_configuration_editable,
    prepare_parent_evaluation,
)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (
            ConsolidationState.INITIALIZED,
            ConsolidationState.INITIALIZED,
        ),
        (
            ConsolidationState.CONSOLIDATED,
            ConsolidationState.UNCONSOLIDATED,
        ),
        (
            ConsolidationState.UNCONSOLIDATED,
            ConsolidationState.UNCONSOLIDATED,
        ),
    ],
)
def test_consolidation_after_evaluation_edit(
    before: ConsolidationState,
    after: ConsolidationState,
) -> None:
    """Apply the exact lifecycle transition for evaluation edits."""
    assert _consolidation_after_evaluation_edit(before) is after


def test_require_configuration_editable_rejects_running() -> None:
    """Reject prospective edits while execution is active."""
    with pytest.raises(
        RuntimeError,
        match=r"^Configuration cannot be changed while execution is running\.$",
    ):
        _require_configuration_editable(ExecutionStatus.RUNNING)


@pytest.mark.parametrize(
    "status",
    [
        ExecutionStatus.NOT_STARTED,
        ExecutionStatus.INTERRUPTED,
        ExecutionStatus.SETTLED,
    ],
)
def test_require_configuration_editable_permits_inactive_observations(
    status: ExecutionStatus,
) -> None:
    """Permit prospective edits for every non-running observation."""
    _require_configuration_editable(status)


def _complete_snapshot() -> ModelingSnapshot:
    """Construct a complete initialized snapshot for parent preparation."""
    model = ModelReference(
        model_id=ModelId("1"),
        model_label="baseline",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.REGRESSION,
    )
    return ModelingSnapshot(
        work_dir="managed-root",
        consolidation_state=ConsolidationState.INITIALIZED,
        data_partition=PartitionReference(
            partition_id=PartitionId("partition-a"),
            dataset_id=DatasetId("dataset-a"),
        ),
        evaluation_config=EvaluationReference(
            configuration_id=EvaluationConfigId("evaluation-a"),
            task_kind=TaskKind.REGRESSION,
            class_labels=(),
            influence_requested=False,
        ),
        models=MappingProxyType({"baseline": model}),
        worker_ids=MappingProxyType({}),
        parent_id=None,
    )


def test_prepare_parent_evaluation_builds_complete_ordered_parent() -> None:
    """Build a parent without requiring a worker or mutating the snapshot."""
    snapshot = _complete_snapshot()

    parent = prepare_parent_evaluation(
        snapshot,
        parent_id=ParentId("parent-a"),
    )

    assert parent.parent_id == ParentId("parent-a")
    assert parent.partition is snapshot.data_partition
    assert parent.evaluation is snapshot.evaluation_config
    assert parent.models == tuple(snapshot.models.values())
    assert snapshot.consolidation_state is ConsolidationState.INITIALIZED
    assert snapshot.parent_id is None
    assert not snapshot.worker_ids


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("data_partition", None, "data_partition"),
        ("evaluation_config", None, "evaluation_config"),
        ("models", MappingProxyType({}), "models"),
    ],
)
def test_prepare_parent_evaluation_rejects_incomplete_snapshot(
    field: str,
    invalid: Any,
    message: str,
) -> None:
    """Reject every missing scientific prerequisite."""
    snapshot = replace(_complete_snapshot(), **{field: invalid})

    with pytest.raises(ValueError, match=message):
        prepare_parent_evaluation(snapshot, parent_id=ParentId("parent-a"))


def test_prepare_parent_evaluation_rejects_non_initialized_snapshot() -> None:
    """Reject preparation from an existing consolidated execution."""
    snapshot = replace(
        _complete_snapshot(),
        consolidation_state=ConsolidationState.CONSOLIDATED,
        parent_id=ParentId("existing-parent"),
    )

    with pytest.raises(ValueError, match=r"consolidation_state.*initialized"):
        prepare_parent_evaluation(snapshot, parent_id=ParentId("parent-a"))


def test_prepare_parent_evaluation_rejects_already_parented_snapshot() -> None:
    """Reject an initialized snapshot that already names a parent."""
    snapshot = replace(
        _complete_snapshot(),
        parent_id=ParentId("existing-parent"),
    )

    with pytest.raises(ValueError, match=r"Initialized snapshots require parent_id=None"):
        prepare_parent_evaluation(snapshot, parent_id=ParentId("parent-a"))


def test_prepare_parent_evaluation_rejects_mixed_task_kinds() -> None:
    """Reject a model whose task differs from the evaluation task."""
    snapshot = _complete_snapshot()
    mismatched_model = replace(
        next(iter(snapshot.models.values())),
        task_kind=TaskKind.CLASSIFICATION,
    )
    snapshot = replace(
        snapshot,
        models=MappingProxyType({"baseline": mismatched_model}),
    )

    with pytest.raises(ValueError, match=r"models\[0\]\.task_kind"):
        prepare_parent_evaluation(snapshot, parent_id=ParentId("parent-a"))


def test_prepare_parent_evaluation_rejects_invalid_parent_id() -> None:
    """Validate the caller-supplied identity before returning the parent."""
    with pytest.raises(ValueError, match=r"parent_id.*nonempty"):
        prepare_parent_evaluation(
            _complete_snapshot(),
            parent_id=ParentId(""),
        )
