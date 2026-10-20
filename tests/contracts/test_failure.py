"""Tests for traceable failure contracts."""

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from cocoonet._contracts.failure import (
    FailureRecord,
    FailureScope,
    TraceContext,
    validate_failure_links,
    validate_failure_record,
    validate_trace_context,
)
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


def test_validate_trace_context_accepts_minimal_stage() -> None:
    """Accept a nonempty stage without inventing optional coordinates."""
    validate_trace_context(TraceContext(stage="data_preparation"))


def test_validate_trace_context_accepts_complete_context_without_mutation() -> None:
    """Accept and preserve complete scientific and operational coordinates."""
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
    original = TraceContext(
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

    validate_trace_context(context)

    assert context == original


def test_validate_trace_context_rejects_empty_stage() -> None:
    """Reject an empty descriptive failure stage."""
    with pytest.raises(ValueError, match=r"^stage must be a nonempty string\.$"):
        validate_trace_context(TraceContext(stage=""))


def test_validate_trace_context_rejects_non_string_stage() -> None:
    """Reject an incorrectly typed stage through declared-type validation."""
    context = TraceContext(stage=cast(str, 1))

    with pytest.raises(TypeError) as error:
        validate_trace_context(context)

    message = str(error.value)
    assert "path: value.stage" in message
    assert "expected: builtins.str" in message
    assert "got type: builtins.int" in message
    assert "got value: 1" in message


@pytest.mark.parametrize(
    ("context", "field"),
    [
        (
            TraceContext(stage="prediction", parent_id=ParentId("")),
            "parent_id",
        ),
        (
            TraceContext(
                stage="prediction",
                parent_id=ParentId("parent-a"),
                model_id=ModelId(""),
            ),
            "model_id",
        ),
        (
            TraceContext(
                stage="prediction",
                parent_id=ParentId("parent-a"),
                model_id=ModelId("model-a"),
                attempt_id=AttemptId(""),
                worker_id=WorkerId("worker-a"),
            ),
            "attempt_id",
        ),
        (
            TraceContext(stage="worker", worker_id=WorkerId("")),
            "worker_id",
        ),
        (
            TraceContext(
                stage="prediction",
                parent_id=ParentId("parent-a"),
                fold_id=FoldId(""),
            ),
            "fold_id",
        ),
        (
            TraceContext(
                stage="prediction",
                parent_id=ParentId("parent-a"),
                sample_id=SampleId(""),
            ),
            "sample_id",
        ),
        (
            TraceContext(
                stage="data_preparation",
                operation_id=OperationId(""),
            ),
            "operation_id",
        ),
    ],
)
def test_validate_trace_context_rejects_empty_supplied_identity(
    context: TraceContext,
    field: str,
) -> None:
    """Validate every supplied opaque context identity."""
    with pytest.raises(
        ValueError,
        match=rf"^{field} must be a nonempty string\.$",
    ):
        validate_trace_context(context)


def test_validate_trace_context_rejects_nonfinite_class_label() -> None:
    """Reject a nonfinite floating-point class coordinate."""
    context = TraceContext(
        stage="class_measure",
        class_label=float("nan"),
    )

    with pytest.raises(
        ValueError,
        match=r"^class_label must be finite; got nan\.$",
    ):
        validate_trace_context(context)


def test_validate_trace_context_rejects_undeclared_report_name() -> None:
    """Reject a plain string that is not a ReportName member."""
    context = TraceContext(
        stage="report",
        report_name=cast(ReportName, "validation"),
    )

    with pytest.raises(TypeError) as error:
        validate_trace_context(context)

    message = str(error.value)
    assert "path: value.report_name" in message
    assert "expected: cocoonet._contracts.identity.ReportName" in message


@pytest.mark.parametrize(
    ("context", "message"),
    [
        (
            TraceContext(
                stage="model",
                model_id=ModelId("model-a"),
            ),
            "model_id requires parent_id.",
        ),
        (
            TraceContext(
                stage="prediction",
                model_id=ModelId("model-a"),
                attempt_id=AttemptId("attempt-a"),
                worker_id=WorkerId("worker-a"),
            ),
            "attempt_id requires parent_id.",
        ),
        (
            TraceContext(
                stage="prediction",
                parent_id=ParentId("parent-a"),
                attempt_id=AttemptId("attempt-a"),
                worker_id=WorkerId("worker-a"),
            ),
            "attempt_id requires model_id.",
        ),
        (
            TraceContext(
                stage="prediction",
                parent_id=ParentId("parent-a"),
                model_id=ModelId("model-a"),
                attempt_id=AttemptId("attempt-a"),
            ),
            "attempt_id requires worker_id.",
        ),
        (
            TraceContext(
                stage="prediction",
                fold_id=FoldId("fold-a"),
            ),
            "fold_id requires parent_id.",
        ),
        (
            TraceContext(
                stage="prediction",
                sample_id=SampleId("sample-a"),
            ),
            "sample_id requires parent_id.",
        ),
    ],
)
def test_validate_trace_context_rejects_orphan_coordinate(
    context: TraceContext,
    message: str,
) -> None:
    """Reject scientific coordinates lacking their required broader context."""
    with pytest.raises(ValueError, match=rf"^{message}$"):
        validate_trace_context(context)


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


@pytest.mark.parametrize(
    ("scope", "context"),
    [
        (
            FailureScope.FIT_PREDICTION,
            TraceContext(
                stage="prediction",
                parent_id=ParentId("parent-a"),
                model_id=ModelId("model-a"),
                attempt_id=AttemptId("attempt-a"),
                worker_id=WorkerId("worker-a"),
            ),
        ),
        (
            FailureScope.REPORT,
            TraceContext(
                stage="validation_report",
                report_name=ReportName.VALIDATION,
            ),
        ),
        (
            FailureScope.MEASURE,
            TraceContext(
                stage="accuracy",
                report_name=ReportName.PERFORMANCE,
            ),
        ),
        (
            FailureScope.SAMPLE_INFLUENCE,
            TraceContext(
                stage="removal_prediction",
                parent_id=ParentId("parent-a"),
                model_id=ModelId("model-a"),
                attempt_id=AttemptId("attempt-a"),
                worker_id=WorkerId("worker-a"),
                fold_id=FoldId("fold-a"),
                sample_id=SampleId("sample-a"),
                report_name=ReportName.INFLUENCE,
            ),
        ),
        (
            FailureScope.DATA_PREPARATION,
            TraceContext(
                stage="partition_persistence",
                operation_id=OperationId("operation-a"),
            ),
        ),
        (
            FailureScope.DATA_SYNC,
            TraceContext(
                stage="dataset_transfer",
                operation_id=OperationId("operation-a"),
            ),
        ),
        (
            FailureScope.WORKER_MANAGEMENT,
            TraceContext(
                stage="worker_release",
                operation_id=OperationId("operation-a"),
            ),
        ),
        (
            FailureScope.RESULT_TRANSFER,
            TraceContext(
                stage="result_transfer",
                operation_id=OperationId("operation-a"),
            ),
        ),
        (
            FailureScope.RESULT_PERSISTENCE,
            TraceContext(
                stage="result_persistence",
                operation_id=OperationId("operation-a"),
            ),
        ),
        (
            FailureScope.WORKER_PROCESS,
            TraceContext(
                stage="process_exit",
                worker_id=WorkerId("worker-a"),
            ),
        ),
    ],
)
def test_validate_failure_record_accepts_each_scope(
    scope: FailureScope,
    context: TraceContext,
) -> None:
    """Accept every failure scope with its minimum required context."""
    failure = FailureRecord(
        failure_id=FailureId("failure-a"),
        scope=scope,
        context=context,
        exception_type="builtins.RuntimeError",
        message="The operation failed.",
        traceback="",
    )

    validate_failure_record(failure)


def test_validate_failure_record_preserves_captured_diagnostics() -> None:
    """Preserve supplied exception, message, and traceback text exactly."""
    failure = FailureRecord(
        failure_id=FailureId("failure-b"),
        scope=FailureScope.REPORT,
        context=TraceContext(
            stage="validation_report",
            report_name=ReportName.VALIDATION,
        ),
        exception_type="package.ReportError",
        message="  Report generation failed.\nAdditional detail.  ",
        traceback="Traceback (most recent call last):\n  report failed\n",
        origin_failure_id=FailureId("failure-a"),
    )

    validate_failure_record(failure)

    assert failure.exception_type == "package.ReportError"
    assert failure.message == "  Report generation failed.\nAdditional detail.  "
    assert failure.traceback == "Traceback (most recent call last):\n  report failed\n"
    assert failure.origin_failure_id == FailureId("failure-a")


def test_validate_failure_record_rejects_empty_failure_id() -> None:
    """Reject an empty failure identity."""
    failure = FailureRecord(
        failure_id=FailureId(""),
        scope=FailureScope.REPORT,
        context=TraceContext(
            stage="validation_report",
            report_name=ReportName.VALIDATION,
        ),
        exception_type="ReportError",
        message="Report generation failed.",
        traceback="",
    )

    with pytest.raises(
        ValueError,
        match=r"^failure_id must be a nonempty string\.$",
    ):
        validate_failure_record(failure)


@pytest.mark.parametrize(
    ("exception_type", "message", "expected"),
    [
        ("", "Failure message.", "exception_type"),
        ("RuntimeError", "", "message"),
    ],
)
def test_validate_failure_record_rejects_empty_diagnostic_field(
    exception_type: str,
    message: str,
    expected: str,
) -> None:
    """Require nonempty exception-type and message fields."""
    failure = FailureRecord(
        failure_id=FailureId("failure-a"),
        scope=FailureScope.REPORT,
        context=TraceContext(
            stage="validation_report",
            report_name=ReportName.VALIDATION,
        ),
        exception_type=exception_type,
        message=message,
        traceback="",
    )

    with pytest.raises(
        ValueError,
        match=rf"^{expected} must be a nonempty string\.$",
    ):
        validate_failure_record(failure)


def test_validate_failure_record_rejects_empty_origin_id() -> None:
    """Reject an explicitly supplied empty origin identity."""
    failure = FailureRecord(
        failure_id=FailureId("failure-b"),
        scope=FailureScope.REPORT,
        context=TraceContext(
            stage="validation_report",
            report_name=ReportName.VALIDATION,
        ),
        exception_type="ReportError",
        message="Report generation failed.",
        traceback="",
        origin_failure_id=FailureId(""),
    )

    with pytest.raises(
        ValueError,
        match=r"^origin_failure_id must be a nonempty string\.$",
    ):
        validate_failure_record(failure)


def test_validate_failure_record_rejects_self_origin() -> None:
    """Reject a failure that identifies itself as its origin."""
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
        origin_failure_id=FailureId("failure-a"),
    )

    with pytest.raises(
        ValueError,
        match=r"^origin_failure_id must not equal failure_id\.$",
    ):
        validate_failure_record(failure)


@pytest.mark.parametrize(
    "scope",
    [
        FailureScope.REPORT,
        FailureScope.MEASURE,
    ],
)
def test_validate_failure_record_rejects_report_scope_without_name(
    scope: FailureScope,
) -> None:
    """Require report identity for report and measure failures."""
    failure = FailureRecord(
        failure_id=FailureId("failure-a"),
        scope=scope,
        context=TraceContext(stage="report"),
        exception_type="ReportError",
        message="Report generation failed.",
        traceback="",
    )

    with pytest.raises(
        ValueError,
        match=rf"^{scope.value} failure requires report_name\.$",
    ):
        validate_failure_record(failure)


@pytest.mark.parametrize(
    ("context", "field"),
    [
        (
            TraceContext(
                stage="removal_fit",
                parent_id=ParentId("parent-a"),
                model_id=ModelId("model-a"),
                fold_id=FoldId("fold-a"),
                sample_id=SampleId("sample-a"),
                report_name=ReportName.INFLUENCE,
            ),
            "attempt_id",
        ),
        (
            TraceContext(
                stage="removal_fit",
                parent_id=ParentId("parent-a"),
                model_id=ModelId("model-a"),
                attempt_id=AttemptId("attempt-a"),
                worker_id=WorkerId("worker-a"),
                sample_id=SampleId("sample-a"),
                report_name=ReportName.INFLUENCE,
            ),
            "fold_id",
        ),
        (
            TraceContext(
                stage="removal_fit",
                parent_id=ParentId("parent-a"),
                model_id=ModelId("model-a"),
                attempt_id=AttemptId("attempt-a"),
                worker_id=WorkerId("worker-a"),
                fold_id=FoldId("fold-a"),
                report_name=ReportName.INFLUENCE,
            ),
            "sample_id",
        ),
        (
            TraceContext(
                stage="removal_fit",
                parent_id=ParentId("parent-a"),
                model_id=ModelId("model-a"),
                attempt_id=AttemptId("attempt-a"),
                worker_id=WorkerId("worker-a"),
                fold_id=FoldId("fold-a"),
                sample_id=SampleId("sample-a"),
            ),
            "report_name",
        ),
    ],
)
def test_validate_failure_record_rejects_incomplete_influence_context(
    context: TraceContext,
    field: str,
) -> None:
    """Require complete attempt, fold, sample, and report influence context."""
    failure = FailureRecord(
        failure_id=FailureId("failure-a"),
        scope=FailureScope.SAMPLE_INFLUENCE,
        context=context,
        exception_type="InfluenceError",
        message="Removal evaluation failed.",
        traceback="",
    )

    with pytest.raises(
        ValueError,
        match=rf"^sample_influence failure requires {field}\.$",
    ):
        validate_failure_record(failure)


def test_validate_failure_record_rejects_wrong_influence_report() -> None:
    """Require the influence report category for sample-influence failures."""
    failure = FailureRecord(
        failure_id=FailureId("failure-a"),
        scope=FailureScope.SAMPLE_INFLUENCE,
        context=TraceContext(
            stage="removal_fit",
            parent_id=ParentId("parent-a"),
            model_id=ModelId("model-a"),
            attempt_id=AttemptId("attempt-a"),
            worker_id=WorkerId("worker-a"),
            fold_id=FoldId("fold-a"),
            sample_id=SampleId("sample-a"),
            report_name=ReportName.VALIDATION,
        ),
        exception_type="InfluenceError",
        message="Removal evaluation failed.",
        traceback="",
    )

    with pytest.raises(
        ValueError,
        match=r"^sample_influence failure requires report_name='influence'\.$",
    ):
        validate_failure_record(failure)


@pytest.mark.parametrize(
    "scope",
    [
        FailureScope.DATA_PREPARATION,
        FailureScope.DATA_SYNC,
        FailureScope.WORKER_MANAGEMENT,
        FailureScope.RESULT_TRANSFER,
        FailureScope.RESULT_PERSISTENCE,
    ],
)
def test_validate_failure_record_rejects_operation_scope_without_operation_id(
    scope: FailureScope,
) -> None:
    """Require operation identity for every bounded operational failure."""
    failure = FailureRecord(
        failure_id=FailureId("failure-a"),
        scope=scope,
        context=TraceContext(stage="operation"),
        exception_type="OperationError",
        message="The operation failed.",
        traceback="",
    )

    with pytest.raises(
        ValueError,
        match=rf"^{scope.value} failure requires operation_id\.$",
    ):
        validate_failure_record(failure)


def test_validate_failure_record_rejects_fit_prediction_without_attempt() -> None:
    """Require attempt context for a fitting or prediction failure."""
    failure = FailureRecord(
        failure_id=FailureId("failure-a"),
        scope=FailureScope.FIT_PREDICTION,
        context=TraceContext(stage="prediction"),
        exception_type="PredictionError",
        message="Prediction failed.",
        traceback="",
    )

    with pytest.raises(
        ValueError,
        match=r"^fit_prediction failure requires attempt_id\.$",
    ):
        validate_failure_record(failure)


def test_validate_failure_record_rejects_worker_process_without_worker() -> None:
    """Require worker identity for a worker-process failure."""
    failure = FailureRecord(
        failure_id=FailureId("failure-a"),
        scope=FailureScope.WORKER_PROCESS,
        context=TraceContext(stage="process_exit"),
        exception_type="WorkerProcessError",
        message="The worker process exited.",
        traceback="",
    )

    with pytest.raises(
        ValueError,
        match=r"^worker_process failure requires worker_id\.$",
    ):
        validate_failure_record(failure)


def _linked_report_failure(
    failure_id: str,
    *,
    origin_failure_id: str | None = None,
    parent_id: str = "parent-a",
    model_id: str = "model-a",
    attempt_id: str = "attempt-a",
    worker_id: str = "worker-a",
    sample_id: str | None = None,
) -> FailureRecord:
    """Build a valid report failure for origin-link validation tests."""
    context = TraceContext(
        stage="validation_report",
        parent_id=ParentId(parent_id),
        model_id=ModelId(model_id),
        attempt_id=AttemptId(attempt_id),
        worker_id=WorkerId(worker_id),
        sample_id=SampleId(sample_id) if sample_id is not None else None,
        report_name=ReportName.VALIDATION,
    )

    return FailureRecord(
        failure_id=FailureId(failure_id),
        scope=FailureScope.REPORT,
        context=context,
        exception_type="ReportError",
        message="Report generation failed.",
        traceback="",
        origin_failure_id=(FailureId(origin_failure_id) if origin_failure_id is not None else None),
    )


def test_validate_failure_links_accepts_empty_collection() -> None:
    """Accept an empty failure collection without inventing failures."""
    validate_failure_links(())


def test_validate_failure_links_accepts_shared_origin() -> None:
    """Allow multiple narrower failures to share one compatible origin."""
    origin = _linked_report_failure("failure-origin")
    first = _linked_report_failure(
        "failure-first",
        origin_failure_id="failure-origin",
        sample_id="sample-a",
    )
    second = _linked_report_failure(
        "failure-second",
        origin_failure_id="failure-origin",
        sample_id="sample-b",
    )

    validate_failure_links((origin, first, second))


def test_validate_failure_links_rejects_duplicate_failure_id() -> None:
    """Reject failure collections with ambiguous duplicate identities."""
    first = _linked_report_failure("failure-a")
    duplicate = _linked_report_failure("failure-a")

    with pytest.raises(
        ValueError,
        match=r"^duplicate failure_id: 'failure-a'\.$",
    ):
        validate_failure_links((first, duplicate))


def test_validate_failure_links_rejects_dangling_origin() -> None:
    """Reject an origin identity absent from the supplied collection."""
    failure = _linked_report_failure(
        "failure-a",
        origin_failure_id="failure-missing",
    )

    with pytest.raises(
        ValueError,
        match=r"^origin_failure_id 'failure-missing' does not resolve\.$",
    ):
        validate_failure_links((failure,))


def test_validate_failure_links_rejects_self_origin() -> None:
    """Reject a self-originating record through individual validation."""
    failure = _linked_report_failure(
        "failure-a",
        origin_failure_id="failure-a",
    )

    with pytest.raises(
        ValueError,
        match=r"^origin_failure_id must not equal failure_id\.$",
    ):
        validate_failure_links((failure,))


def test_validate_failure_links_rejects_origin_cycle() -> None:
    """Reject a cycle spanning multiple otherwise valid failure records."""
    first = _linked_report_failure(
        "failure-a",
        origin_failure_id="failure-b",
    )
    second = _linked_report_failure(
        "failure-b",
        origin_failure_id="failure-c",
    )
    third = _linked_report_failure(
        "failure-c",
        origin_failure_id="failure-a",
    )

    with pytest.raises(
        ValueError,
        match=r"^failure origin links must be acyclic\.$",
    ):
        validate_failure_links((first, second, third))


@pytest.mark.parametrize(
    ("dependent", "field"),
    [
        (
            _linked_report_failure(
                "failure-dependent",
                origin_failure_id="failure-origin",
                parent_id="parent-b",
            ),
            "parent_id",
        ),
        (
            _linked_report_failure(
                "failure-dependent",
                origin_failure_id="failure-origin",
                model_id="model-b",
            ),
            "model_id",
        ),
        (
            _linked_report_failure(
                "failure-dependent",
                origin_failure_id="failure-origin",
                attempt_id="attempt-b",
            ),
            "attempt_id",
        ),
        (
            _linked_report_failure(
                "failure-dependent",
                origin_failure_id="failure-origin",
                worker_id="worker-b",
            ),
            "worker_id",
        ),
    ],
)
def test_validate_failure_links_rejects_conflicting_shared_context(
    dependent: FailureRecord,
    field: str,
) -> None:
    """Reject linked failures with conflicting shared execution identities."""
    origin = _linked_report_failure("failure-origin")

    with pytest.raises(
        ValueError,
        match=(rf"^failure 'failure-dependent' and origin " rf"'failure-origin' have conflicting {field}\.$"),
    ):
        validate_failure_links((origin, dependent))
