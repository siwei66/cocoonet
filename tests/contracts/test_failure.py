"""Tests for traceable failure contracts."""

from dataclasses import FrozenInstanceError

import pytest

from cocoonet._contracts.failure import FailureRecord, FailureScope, TraceContext
from cocoonet._contracts.identity import (
    AttemptId,
    FailureId,
    FoldId,
    ModelId,
    OperationId,
    ParentId,
    ReportName,
    SampleId,
    WorkerId,
)


def test_failure_scope_has_exact_string_vocabulary() -> None:
    """Expose only the planned failure-scope values."""
    assert tuple(FailureScope) == (
        FailureScope.FIT_PREDICTION,
        FailureScope.REPORT,
        FailureScope.MEASURE,
        FailureScope.SAMPLE_INFLUENCE,
        FailureScope.DATA_PREPARATION,
        FailureScope.DATA_SYNC,
        FailureScope.WORKER_MANAGEMENT,
        FailureScope.RESULT_TRANSFER,
        FailureScope.RESULT_PERSISTENCE,
        FailureScope.WORKER_PROCESS,
    )
    assert FailureScope.FIT_PREDICTION == "fit_prediction"
    assert FailureScope.REPORT == "report"
    assert FailureScope.MEASURE == "measure"
    assert FailureScope.SAMPLE_INFLUENCE == "sample_influence"
    assert FailureScope.DATA_PREPARATION == "data_preparation"
    assert FailureScope.DATA_SYNC == "data_sync"
    assert FailureScope.WORKER_MANAGEMENT == "worker_management"
    assert FailureScope.RESULT_TRANSFER == "result_transfer"
    assert FailureScope.RESULT_PERSISTENCE == "result_persistence"
    assert FailureScope.WORKER_PROCESS == "worker_process"


def test_trace_context_defaults_optional_fields_to_none() -> None:
    """Represent a minimal stage without inventing optional identities."""
    context = TraceContext(stage="data_preparation")

    assert context.stage == "data_preparation"
    assert context.parent_id is None
    assert context.model_id is None
    assert context.attempt_id is None
    assert context.worker_id is None
    assert context.fold_id is None
    assert context.sample_id is None
    assert context.class_label is None
    assert context.report_name is None
    assert context.operation_id is None


def test_trace_context_preserves_complete_failure_coordinates() -> None:
    """Preserve all supplied scientific and operational failure coordinates."""
    context = TraceContext(
        stage="prediction",
        parent_id=ParentId("parent-a"),
        model_id=ModelId("model-a"),
        attempt_id=AttemptId("attempt-a"),
        worker_id=WorkerId("worker-a"),
        fold_id=FoldId("fold-a"),
        sample_id=SampleId("sample-a"),
        class_label="positive",
        report_name=ReportName.VALIDATION,
        operation_id=OperationId("operation-a"),
    )
    same = TraceContext(
        stage="prediction",
        parent_id=ParentId("parent-a"),
        model_id=ModelId("model-a"),
        attempt_id=AttemptId("attempt-a"),
        worker_id=WorkerId("worker-a"),
        fold_id=FoldId("fold-a"),
        sample_id=SampleId("sample-a"),
        class_label="positive",
        report_name=ReportName.VALIDATION,
        operation_id=OperationId("operation-a"),
    )

    assert context.stage == "prediction"
    assert context.parent_id == ParentId("parent-a")
    assert context.model_id == ModelId("model-a")
    assert context.attempt_id == AttemptId("attempt-a")
    assert context.worker_id == WorkerId("worker-a")
    assert context.fold_id == FoldId("fold-a")
    assert context.sample_id == SampleId("sample-a")
    assert context.class_label == "positive"
    assert context.report_name is ReportName.VALIDATION
    assert context.operation_id == OperationId("operation-a")
    assert context == same


def test_trace_context_rejects_field_mutation() -> None:
    """Prevent reassignment of recorded failure coordinates."""
    context = TraceContext(stage="prediction")

    with pytest.raises(FrozenInstanceError):
        context.__setattr__("stage", "replacement")


def test_failure_record_defaults_origin_to_none() -> None:
    """Represent an independent failure without inventing an origin link."""
    context = TraceContext(
        stage="data_preparation",
        operation_id=OperationId("operation-a"),
    )
    failure = FailureRecord(
        failure_id=FailureId("failure-a"),
        scope=FailureScope.DATA_PREPARATION,
        context=context,
        exception_type="DatasetPreparationError",
        message="Partition persistence failed.",
        traceback="",
    )

    assert failure.failure_id == FailureId("failure-a")
    assert failure.scope is FailureScope.DATA_PREPARATION
    assert failure.context is context
    assert failure.exception_type == "DatasetPreparationError"
    assert failure.message == "Partition persistence failed."
    assert failure.traceback == ""
    assert failure.origin_failure_id is None


def test_failure_record_preserves_origin_and_diagnostics() -> None:
    """Preserve failure context, diagnostic text, and an explicit origin link."""
    context = TraceContext(
        stage="prediction",
        parent_id=ParentId("parent-a"),
        model_id=ModelId("model-a"),
        attempt_id=AttemptId("attempt-a"),
        worker_id=WorkerId("worker-a"),
        fold_id=FoldId("fold-a"),
        sample_id=SampleId("sample-a"),
        class_label="positive",
        report_name=ReportName.VALIDATION,
    )
    failure = FailureRecord(
        failure_id=FailureId("failure-b"),
        scope=FailureScope.FIT_PREDICTION,
        context=context,
        exception_type="builtins.RuntimeError",
        message="Prediction failed.",
        traceback="Traceback: prediction failed.",
        origin_failure_id=FailureId("failure-a"),
    )
    same = FailureRecord(
        failure_id=FailureId("failure-b"),
        scope=FailureScope.FIT_PREDICTION,
        context=context,
        exception_type="builtins.RuntimeError",
        message="Prediction failed.",
        traceback="Traceback: prediction failed.",
        origin_failure_id=FailureId("failure-a"),
    )

    assert failure.failure_id == FailureId("failure-b")
    assert failure.scope is FailureScope.FIT_PREDICTION
    assert failure.context is context
    assert failure.exception_type == "builtins.RuntimeError"
    assert failure.message == "Prediction failed."
    assert failure.traceback == "Traceback: prediction failed."
    assert failure.origin_failure_id == FailureId("failure-a")
    assert failure == same


def test_failure_record_rejects_field_mutation() -> None:
    """Prevent reassignment of recorded failure information."""
    failure = FailureRecord(
        failure_id=FailureId("failure-a"),
        scope=FailureScope.REPORT,
        context=TraceContext(
            stage="validation_report",
            report_name=ReportName.VALIDATION,
        ),
        exception_type="ReportError",
        message="Report generation failed.",
        traceback="",
    )

    with pytest.raises(FrozenInstanceError):
        failure.__setattr__("message", "replacement")
