"""Tests for standardized result-envelope contracts."""

from dataclasses import FrozenInstanceError, dataclass, replace
from typing import cast

import pytest

from cocoonet._contracts.data import (
    FoldRecord,
    PartitionReference,
    PartitionSchedule,
    SampleRecord,
    SourceReference,
)
from cocoonet._contracts.execution import (
    AttemptInputs,
    EvaluationReference,
    ModelFamily,
    ModelReference,
    ParentEvaluation,
    TaskKind,
    UnitTask,
)
from cocoonet._contracts.failure import (
    FailureRecord,
    FailureScope,
    TraceContext,
)
from cocoonet._contracts.identity import (
    AttemptId,
    DatasetId,
    EvaluationConfigId,
    FailureId,
    FoldId,
    ModelId,
    OperationId,
    ParentId,
    PartitionId,
    ReportName,
    SampleId,
    UnitKey,
    WorkerId,
)
from cocoonet._contracts.prediction import (
    ClassificationPredictions,
    RegressionPredictions,
)
from cocoonet._contracts.result import (
    AttemptResult,
    EvaluationOutcome,
    ReportOutcome,
    ReportStatus,
    validate_attempt_result,
    validate_report_outcome,
)


@dataclass(frozen=True, slots=True)
class _ExampleReportContent:
    """Provide small immutable report content for declaration tests."""

    value: float | None


def _attempt_inputs() -> AttemptInputs:
    """Build stable attempt-routing identities for result declaration tests."""
    return AttemptInputs(
        attempt_id=AttemptId("attempt-a"),
        unit=UnitKey(
            parent_id=ParentId("parent-a"),
            model_id=ModelId("model-a"),
        ),
        worker_id=WorkerId("worker-a"),
    )


def test_report_status_has_exact_string_vocabulary() -> None:
    """Expose only the planned report-outcome states."""
    assert tuple(ReportStatus) == (
        ReportStatus.COMPLETE,
        ReportStatus.FAILED,
        ReportStatus.BLOCKED,
        ReportStatus.NOT_REQUESTED,
    )
    assert ReportStatus.COMPLETE == "complete"
    assert ReportStatus.FAILED == "failed"
    assert ReportStatus.BLOCKED == "blocked"
    assert ReportStatus.NOT_REQUESTED == "not_requested"


def test_evaluation_outcome_has_exact_staged_vocabulary() -> None:
    """Expose only the planned completed-attempt outcomes."""
    assert tuple(EvaluationOutcome) == (
        EvaluationOutcome.COMPLETED,
        EvaluationOutcome.COMPLETED_WITH_ERRORS,
        EvaluationOutcome.VALIDATION_FAILED,
        EvaluationOutcome.RECONSTRUCTION_FAILED,
    )
    assert EvaluationOutcome.COMPLETED == "completed"
    assert EvaluationOutcome.COMPLETED_WITH_ERRORS == "completed_with_errors"
    assert EvaluationOutcome.VALIDATION_FAILED == "validation_failed"
    assert EvaluationOutcome.RECONSTRUCTION_FAILED == "reconstruction_failed"


def test_report_outcome_preserves_typed_report_content_and_failures() -> None:
    """Preserve typed report content and its report-level failures."""
    outcome = ReportOutcome[_ExampleReportContent](
        name=ReportName.PERFORMANCE,
        status=ReportStatus.FAILED,
        payload=_ExampleReportContent(value=1.25),
        failure_ids=(FailureId("failure-a"),),
    )
    same = ReportOutcome[_ExampleReportContent](
        name=ReportName.PERFORMANCE,
        status=ReportStatus.FAILED,
        payload=_ExampleReportContent(value=1.25),
        failure_ids=(FailureId("failure-a"),),
    )

    assert outcome.name is ReportName.PERFORMANCE
    assert outcome.status is ReportStatus.FAILED
    assert outcome.payload == _ExampleReportContent(value=1.25)
    assert outcome.failure_ids == (FailureId("failure-a"),)
    assert outcome == same


def test_report_outcome_accepts_absent_report_content() -> None:
    """Represent a report outcome without retained report content."""
    outcome = ReportOutcome[_ExampleReportContent](
        name=ReportName.RESIDUAL,
        status=ReportStatus.BLOCKED,
        payload=None,
        failure_ids=(FailureId("prerequisite-failure"),),
    )

    assert outcome.payload is None


def test_report_outcome_rejects_field_mutation() -> None:
    """Prevent reassignment of report-outcome fields."""
    outcome = ReportOutcome[_ExampleReportContent](
        name=ReportName.VALIDATION,
        status=ReportStatus.COMPLETE,
        payload=_ExampleReportContent(value=1.25),
        failure_ids=(),
    )

    with pytest.raises(FrozenInstanceError):
        outcome.__setattr__("status", ReportStatus.FAILED)


def test_attempt_result_preserves_complete_typed_result() -> None:
    """Preserve predictions and each typed report slot without conversion."""
    predictions = (
        RegressionPredictions(
            fold_id=FoldId("fold-a"),
            sample_ids=(),
            values=(),
            failure_ids=(),
        ),
    )
    validation = ReportOutcome[_ExampleReportContent](
        name=ReportName.VALIDATION,
        status=ReportStatus.COMPLETE,
        payload=_ExampleReportContent(value=1.0),
        failure_ids=(),
    )
    performance = ReportOutcome[_ExampleReportContent](
        name=ReportName.PERFORMANCE,
        status=ReportStatus.COMPLETE,
        payload=_ExampleReportContent(value=2.0),
        failure_ids=(),
    )
    residual = ReportOutcome[_ExampleReportContent](
        name=ReportName.RESIDUAL,
        status=ReportStatus.COMPLETE,
        payload=_ExampleReportContent(value=3.0),
        failure_ids=(),
    )
    visualization = ReportOutcome[_ExampleReportContent](
        name=ReportName.VISUALIZATION,
        status=ReportStatus.COMPLETE,
        payload=_ExampleReportContent(value=4.0),
        failure_ids=(),
    )
    influence = ReportOutcome[_ExampleReportContent](
        name=ReportName.INFLUENCE,
        status=ReportStatus.NOT_REQUESTED,
        payload=None,
        failure_ids=(),
    )

    result = AttemptResult[
        _ExampleReportContent,
        _ExampleReportContent,
        _ExampleReportContent,
        _ExampleReportContent,
        _ExampleReportContent,
    ](
        inputs=_attempt_inputs(),
        outcome=EvaluationOutcome.COMPLETED,
        predictions=predictions,
        validation=validation,
        performance=performance,
        residual=residual,
        visualization=visualization,
        influence=influence,
        failures=(),
    )

    assert result.inputs == _attempt_inputs()
    assert result.outcome is EvaluationOutcome.COMPLETED
    assert result.predictions == predictions
    assert result.validation == validation
    assert result.performance == performance
    assert result.residual == residual
    assert result.visualization == visualization
    assert result.influence == influence
    assert result.failures == ()


def test_attempt_result_represents_reconstruction_failure_without_reports() -> None:
    """Represent a reconstruction failure without predictions or reports."""
    result = AttemptResult[
        _ExampleReportContent,
        _ExampleReportContent,
        _ExampleReportContent,
        _ExampleReportContent,
        _ExampleReportContent,
    ](
        inputs=_attempt_inputs(),
        outcome=EvaluationOutcome.RECONSTRUCTION_FAILED,
        predictions=(),
        validation=None,
        performance=None,
        residual=None,
        visualization=None,
        influence=None,
        failures=(),
    )

    assert result.predictions == ()
    assert result.validation is None
    assert result.performance is None
    assert result.residual is None
    assert result.visualization is None
    assert result.influence is None


def test_attempt_result_rejects_field_mutation() -> None:
    """Prevent reassignment of complete attempt-result fields."""
    result = AttemptResult[
        _ExampleReportContent,
        _ExampleReportContent,
        _ExampleReportContent,
        _ExampleReportContent,
        _ExampleReportContent,
    ](
        inputs=_attempt_inputs(),
        outcome=EvaluationOutcome.RECONSTRUCTION_FAILED,
        predictions=(),
        validation=None,
        performance=None,
        residual=None,
        visualization=None,
        influence=None,
        failures=(),
    )

    with pytest.raises(FrozenInstanceError):
        result.__setattr__("outcome", EvaluationOutcome.COMPLETED)


@pytest.mark.parametrize(
    ("status", "payload", "failure_ids"),
    [
        pytest.param(
            ReportStatus.COMPLETE,
            _ExampleReportContent(value=1.0),
            (),
            id="complete",
        ),
        pytest.param(
            ReportStatus.FAILED,
            _ExampleReportContent(value=1.0),
            (FailureId("failure-a"),),
            id="failed-with-content",
        ),
        pytest.param(
            ReportStatus.FAILED,
            None,
            (FailureId("failure-a"),),
            id="failed-without-content",
        ),
        pytest.param(
            ReportStatus.BLOCKED,
            _ExampleReportContent(value=None),
            (FailureId("failure-a"),),
            id="blocked-with-content",
        ),
        pytest.param(
            ReportStatus.BLOCKED,
            None,
            (FailureId("failure-a"),),
            id="blocked-without-content",
        ),
        pytest.param(
            ReportStatus.NOT_REQUESTED,
            None,
            (),
            id="not-requested",
        ),
    ],
)
def test_validate_report_outcome_accepts_valid_status_relationships(
    status: ReportStatus,
    payload: _ExampleReportContent | None,
    failure_ids: tuple[FailureId, ...],
) -> None:
    """Accept every valid generic report status relationship."""
    outcome = ReportOutcome[_ExampleReportContent](
        name=ReportName.PERFORMANCE,
        status=status,
        payload=payload,
        failure_ids=failure_ids,
    )

    validate_report_outcome(outcome)

    assert outcome.payload is payload
    assert outcome.failure_ids == failure_ids


@pytest.mark.parametrize(
    "scientific_value",
    [
        pytest.param(float("nan"), id="undefined"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
        pytest.param(None, id="schema-defined-none"),
    ],
)
def test_validate_report_outcome_preserves_scientific_content(
    scientific_value: float | None,
) -> None:
    """Leave scientific value interpretation to the concrete payload schema."""
    payload = _ExampleReportContent(value=scientific_value)
    outcome = ReportOutcome[_ExampleReportContent](
        name=ReportName.PERFORMANCE,
        status=ReportStatus.COMPLETE,
        payload=payload,
        failure_ids=(),
    )

    validate_report_outcome(outcome)

    assert outcome.payload is payload


@pytest.mark.parametrize(
    ("status", "payload", "failure_ids", "message"),
    [
        pytest.param(
            ReportStatus.COMPLETE,
            None,
            (),
            (
                "Invalid report outcome:\n"
                "  report: performance\n"
                "  status: complete\n"
                "  expected: payload to be present\n"
                "  got: payload is None"
            ),
            id="complete-without-content",
        ),
        pytest.param(
            ReportStatus.COMPLETE,
            _ExampleReportContent(value=1.0),
            (FailureId("failure-a"),),
            (
                "Invalid report outcome:\n"
                "  report: performance\n"
                "  status: complete\n"
                "  expected: failure_ids to be empty\n"
                "  got: failure_ids count is 1"
            ),
            id="complete-with-failure",
        ),
        pytest.param(
            ReportStatus.FAILED,
            _ExampleReportContent(value=1.0),
            (),
            (
                "Invalid report outcome:\n"
                "  report: performance\n"
                "  status: failed\n"
                "  expected: failure_ids to contain at least one reference\n"
                "  got: failure_ids is empty"
            ),
            id="failed-without-failure",
        ),
        pytest.param(
            ReportStatus.BLOCKED,
            None,
            (),
            (
                "Invalid report outcome:\n"
                "  report: performance\n"
                "  status: blocked\n"
                "  expected: failure_ids to contain at least one reference\n"
                "  got: failure_ids is empty"
            ),
            id="blocked-without-failure",
        ),
        pytest.param(
            ReportStatus.NOT_REQUESTED,
            _ExampleReportContent(value=1.0),
            (),
            (
                "Invalid report outcome:\n"
                "  report: performance\n"
                "  status: not_requested\n"
                "  expected: payload to be None\n"
                "  got: payload is present"
            ),
            id="unrequested-with-content",
        ),
        pytest.param(
            ReportStatus.NOT_REQUESTED,
            None,
            (FailureId("failure-a"),),
            (
                "Invalid report outcome:\n"
                "  report: performance\n"
                "  status: not_requested\n"
                "  expected: failure_ids to be empty\n"
                "  got: failure_ids count is 1"
            ),
            id="unrequested-with-failure",
        ),
    ],
)
def test_validate_report_outcome_rejects_invalid_status_relationships(
    status: ReportStatus,
    payload: _ExampleReportContent | None,
    failure_ids: tuple[FailureId, ...],
    message: str,
) -> None:
    """Reject content and failure combinations inconsistent with the status."""
    outcome = ReportOutcome[_ExampleReportContent](
        name=ReportName.PERFORMANCE,
        status=status,
        payload=payload,
        failure_ids=failure_ids,
    )

    with pytest.raises(ValueError) as error:
        validate_report_outcome(outcome)

    assert str(error.value) == message


def test_validate_report_outcome_rejects_empty_failure_identity() -> None:
    """Identify an empty report-level failure reference precisely."""
    outcome = ReportOutcome[_ExampleReportContent](
        name=ReportName.PERFORMANCE,
        status=ReportStatus.FAILED,
        payload=None,
        failure_ids=(FailureId(""),),
    )

    with pytest.raises(ValueError) as error:
        validate_report_outcome(outcome)

    assert str(error.value) == (
        "Invalid report outcome:\n"
        "  report: performance\n"
        "  status: failed\n"
        "  field: failure_ids[0]\n"
        "  expected: a nonempty failure identity\n"
        "  got: an empty string"
    )


def test_validate_report_outcome_rejects_wrong_status_type() -> None:
    """Preserve the unified type diagnostic for an invalid status type."""
    outcome = ReportOutcome[_ExampleReportContent](
        name=ReportName.PERFORMANCE,
        status=cast(ReportStatus, "complete"),
        payload=_ExampleReportContent(value=1.0),
        failure_ids=(),
    )

    with pytest.raises(TypeError) as error:
        validate_report_outcome(outcome)

    assert str(error.value) == (
        "Type validation failed\n"
        "  path: value.status\n"
        "  expected: cocoonet._contracts.result.ReportStatus\n"
        "  got type: builtins.str\n"
        "  got value: 'complete'"
    )


type _ExampleAttemptResult = AttemptResult[
    _ExampleReportContent,
    _ExampleReportContent,
    _ExampleReportContent,
    _ExampleReportContent,
    _ExampleReportContent,
]


def _sample(number: int) -> SampleRecord:
    """Build one normalized sample for attempt-result validation tests."""
    return SampleRecord(
        sample_id=SampleId(f"sample-{number}"),
        sample_label=f"Sample {number}",
        group=f"group-{number}",
        train_mask=1,
        test_mask=1,
        target=float(number),
        sources=(SourceReference(relative_path=f"samples/{number}.csv"),),
    )


def _schedule() -> PartitionSchedule:
    """Build a two-fold schedule with distinct training and testing groups."""
    return PartitionSchedule(
        reference=PartitionReference(
            partition_id=PartitionId("partition-a"),
            dataset_id=DatasetId("dataset-a"),
        ),
        samples=tuple(_sample(number) for number in range(1, 5)),
        folds=(
            FoldRecord(
                fold_id=FoldId("fold-a"),
                training_sample_ids=(SampleId("sample-1"), SampleId("sample-2")),
                testing_sample_ids=(SampleId("sample-3"), SampleId("sample-4")),
            ),
            FoldRecord(
                fold_id=FoldId("fold-b"),
                training_sample_ids=(SampleId("sample-3"), SampleId("sample-4")),
                testing_sample_ids=(SampleId("sample-1"), SampleId("sample-2")),
            ),
        ),
    )


def _parent(
    *,
    task_kind: TaskKind = TaskKind.REGRESSION,
    influence_requested: bool = False,
) -> ParentEvaluation:
    """Build the fixed parent context used by attempt-result tests."""
    class_labels = () if task_kind is TaskKind.REGRESSION else ("negative", "positive")
    model = ModelReference(
        model_id=ModelId("model-a"),
        model_label="Model A",
        family=ModelFamily.SKLEARN,
        task_kind=task_kind,
    )
    unit = UnitTask(
        key=UnitKey(parent_id=ParentId("parent-a"), model_id=model.model_id),
        model=model,
    )
    return ParentEvaluation(
        parent_id=ParentId("parent-a"),
        partition=_schedule().reference,
        evaluation=EvaluationReference(
            configuration_id=EvaluationConfigId("evaluation-a"),
            task_kind=task_kind,
            class_labels=class_labels,
            influence_requested=influence_requested,
        ),
        units=(unit,),
    )


def _report(
    name: ReportName,
    *,
    status: ReportStatus = ReportStatus.COMPLETE,
    failure_ids: tuple[FailureId, ...] = (),
    value: float | None = 1.0,
) -> ReportOutcome[_ExampleReportContent]:
    """Build one report outcome with content appropriate to its status."""
    payload = None if status is ReportStatus.NOT_REQUESTED else _ExampleReportContent(value=value)
    return ReportOutcome(
        name=name,
        status=status,
        payload=payload,
        failure_ids=failure_ids,
    )


def _failure(
    failure_id: str,
    *,
    scope: FailureScope = FailureScope.FIT_PREDICTION,
    stage: str = "prediction",
    fold_id: str | None = "fold-a",
    sample_id: str | None = "sample-3",
    class_label: str | int | None = None,
    report_name: ReportName | None = None,
    attempt_id: str | None = "attempt-a",
    worker_id: str | None = "worker-a",
    parent_id: str | None = "parent-a",
    model_id: str | None = "model-a",
    origin_failure_id: FailureId | None = None,
) -> FailureRecord:
    """Build one scientific failure with configurable result coordinates."""
    return FailureRecord(
        failure_id=FailureId(failure_id),
        scope=scope,
        context=TraceContext(
            stage=stage,
            parent_id=ParentId(parent_id) if parent_id is not None else None,
            model_id=ModelId(model_id) if model_id is not None else None,
            attempt_id=AttemptId(attempt_id) if attempt_id is not None else None,
            worker_id=WorkerId(worker_id) if worker_id is not None else None,
            fold_id=FoldId(fold_id) if fold_id is not None else None,
            sample_id=SampleId(sample_id) if sample_id is not None else None,
            class_label=class_label,
            report_name=report_name,
        ),
        exception_type="RuntimeError",
        message="evaluation failed",
        traceback="traceback",
        origin_failure_id=origin_failure_id,
    )


def _regression_predictions() -> tuple[RegressionPredictions, ...]:
    """Build complete regression predictions for both scheduled folds."""
    return (
        RegressionPredictions(
            fold_id=FoldId("fold-a"),
            sample_ids=(SampleId("sample-3"), SampleId("sample-4")),
            values=(3.0, 4.0),
            failure_ids=((), ()),
        ),
        RegressionPredictions(
            fold_id=FoldId("fold-b"),
            sample_ids=(SampleId("sample-1"), SampleId("sample-2")),
            values=(1.0, 2.0),
            failure_ids=((), ()),
        ),
    )


def _classification_predictions() -> tuple[ClassificationPredictions, ...]:
    """Build complete classification predictions for both scheduled folds."""
    return (
        ClassificationPredictions(
            fold_id=FoldId("fold-a"),
            sample_ids=(SampleId("sample-3"), SampleId("sample-4")),
            class_labels=("negative", "positive"),
            predicted_labels=("positive", "negative"),
            label_failure_ids=((), ()),
            probabilities=((0.2, 0.8), (0.7, 0.3)),
            probability_failure_ids=(((), ()), ((), ())),
        ),
        ClassificationPredictions(
            fold_id=FoldId("fold-b"),
            sample_ids=(SampleId("sample-1"), SampleId("sample-2")),
            class_labels=("negative", "positive"),
            predicted_labels=("negative", "positive"),
            label_failure_ids=((), ()),
            probabilities=((0.9, 0.1), (0.4, 0.6)),
            probability_failure_ids=(((), ()), ((), ())),
        ),
    )


def _complete_result(
    *,
    predictions: tuple[RegressionPredictions | ClassificationPredictions, ...] | None = None,
    influence_requested: bool = False,
) -> _ExampleAttemptResult:
    """Build a fully successful attempt result for either task kind."""
    if predictions is None:
        predictions = _regression_predictions()
    influence = (
        _report(ReportName.INFLUENCE)
        if influence_requested
        else _report(ReportName.INFLUENCE, status=ReportStatus.NOT_REQUESTED)
    )
    return AttemptResult(
        inputs=_attempt_inputs(),
        outcome=EvaluationOutcome.COMPLETED,
        predictions=predictions,
        validation=_report(ReportName.VALIDATION),
        performance=_report(ReportName.PERFORMANCE),
        residual=_report(ReportName.RESIDUAL),
        visualization=_report(ReportName.VISUALIZATION),
        influence=influence,
        failures=(),
    )


def _validation_failed_result(*, influence_requested: bool = False) -> _ExampleAttemptResult:
    """Build a failed primary validation retaining one later successful fold."""
    failure_id = FailureId("prediction-failure")
    predictions = (
        RegressionPredictions(
            fold_id=FoldId("fold-a"),
            sample_ids=(SampleId("sample-3"), SampleId("sample-4")),
            values=(None, 4.0),
            failure_ids=((failure_id,), ()),
        ),
        _regression_predictions()[1],
    )
    influence = (
        _report(
            ReportName.INFLUENCE,
            status=ReportStatus.BLOCKED,
            failure_ids=(failure_id,),
        )
        if influence_requested
        else _report(ReportName.INFLUENCE, status=ReportStatus.NOT_REQUESTED)
    )
    return AttemptResult(
        inputs=_attempt_inputs(),
        outcome=EvaluationOutcome.VALIDATION_FAILED,
        predictions=predictions,
        validation=_report(
            ReportName.VALIDATION,
            status=ReportStatus.FAILED,
            failure_ids=(failure_id,),
        ),
        performance=_report(
            ReportName.PERFORMANCE,
            status=ReportStatus.BLOCKED,
            failure_ids=(failure_id,),
        ),
        residual=_report(
            ReportName.RESIDUAL,
            status=ReportStatus.BLOCKED,
            failure_ids=(failure_id,),
        ),
        visualization=_report(
            ReportName.VISUALIZATION,
            status=ReportStatus.BLOCKED,
            failure_ids=(failure_id,),
        ),
        influence=influence,
        failures=(_failure("prediction-failure"),),
    )


def test_validate_attempt_result_accepts_completed_regression() -> None:
    """Accept a complete regression result aligned with its parent and folds."""
    result = _complete_result()

    validate_attempt_result(result, parent=_parent(), schedule=_schedule())

    assert result.outcome is EvaluationOutcome.COMPLETED


def test_validate_attempt_result_accepts_completed_classification() -> None:
    """Accept complete classification predictions in parent class order."""
    result = _complete_result(predictions=_classification_predictions())

    validate_attempt_result(
        result,
        parent=_parent(task_kind=TaskKind.CLASSIFICATION),
        schedule=_schedule(),
    )

    assert result.outcome is EvaluationOutcome.COMPLETED


def test_validate_attempt_result_preserves_formula_defined_nan() -> None:
    """Preserve a formula-defined NaN inside a concrete report payload."""
    result = replace(
        _complete_result(),
        performance=_report(ReportName.PERFORMANCE, value=float("nan")),
    )

    validate_attempt_result(result, parent=_parent(), schedule=_schedule())

    assert result.performance is not None
    assert result.performance.payload is not None
    assert result.performance.payload.value != result.performance.payload.value


def test_validate_attempt_result_accepts_reconstruction_failure() -> None:
    """Accept the failure-only envelope produced by reconstruction failure."""
    failure = _failure(
        "reconstruction-failure",
        stage="reconstruction",
        fold_id=None,
        sample_id=None,
    )
    result: _ExampleAttemptResult = AttemptResult(
        inputs=_attempt_inputs(),
        outcome=EvaluationOutcome.RECONSTRUCTION_FAILED,
        predictions=(),
        validation=None,
        performance=None,
        residual=None,
        visualization=None,
        influence=None,
        failures=(failure,),
    )

    validate_attempt_result(result, parent=_parent(), schedule=_schedule())

    assert result.failures == (failure,)


def test_validate_attempt_result_preserves_later_successful_fold() -> None:
    """Accept later successful fold results after an earlier fold failure."""
    result = _validation_failed_result()

    validate_attempt_result(result, parent=_parent(), schedule=_schedule())

    assert result.predictions[1] == _regression_predictions()[1]


def test_validate_attempt_result_accepts_blocked_retained_reports() -> None:
    """Accept retained dependent report forms and blocked requested influence."""
    result = _validation_failed_result(influence_requested=True)

    validate_attempt_result(
        result,
        parent=_parent(influence_requested=True),
        schedule=_schedule(),
    )

    assert result.influence is not None
    assert result.influence.status is ReportStatus.BLOCKED


def test_validate_attempt_result_accepts_ordinary_report_failure() -> None:
    """Derive completion with errors from a failed ordinary report."""
    failure_id = FailureId("performance-failure")
    failure = _failure(
        "performance-failure",
        scope=FailureScope.REPORT,
        stage="performance",
        fold_id=None,
        sample_id=None,
        report_name=ReportName.PERFORMANCE,
    )
    result = replace(
        _complete_result(),
        outcome=EvaluationOutcome.COMPLETED_WITH_ERRORS,
        performance=_report(
            ReportName.PERFORMANCE,
            status=ReportStatus.FAILED,
            failure_ids=(failure_id,),
        ),
        failures=(failure,),
    )

    validate_attempt_result(result, parent=_parent(), schedule=_schedule())

    assert result.outcome is EvaluationOutcome.COMPLETED_WITH_ERRORS


def test_validate_attempt_result_accepts_sample_influence_failure() -> None:
    """Derive completion with errors from an independent influence failure."""
    failure_id = FailureId("influence-failure")
    failure = _failure(
        "influence-failure",
        scope=FailureScope.SAMPLE_INFLUENCE,
        stage="sample-removal-validation",
        fold_id="fold-a",
        sample_id="sample-1",
        report_name=ReportName.INFLUENCE,
    )
    result = replace(
        _complete_result(influence_requested=True),
        outcome=EvaluationOutcome.COMPLETED_WITH_ERRORS,
        influence=_report(
            ReportName.INFLUENCE,
            status=ReportStatus.FAILED,
            failure_ids=(failure_id,),
        ),
        failures=(failure,),
    )

    validate_attempt_result(
        result,
        parent=_parent(influence_requested=True),
        schedule=_schedule(),
    )

    assert result.outcome is EvaluationOutcome.COMPLETED_WITH_ERRORS


def test_validate_attempt_result_rejects_mismatched_schedule() -> None:
    """Reject a schedule that is not the parent's fixed partition."""
    result = _complete_result()
    schedule = replace(
        _schedule(),
        reference=replace(
            _schedule().reference,
            partition_id=PartitionId("partition-b"),
        ),
    )

    with pytest.raises(ValueError) as error:
        validate_attempt_result(result, parent=_parent(), schedule=schedule)

    assert str(error.value) == (
        "Invalid attempt result:\n"
        "  parent_id: parent-a\n"
        "  model_id: model-a\n"
        "  attempt_id: attempt-a\n"
        "  field: schedule.reference\n"
        "  expected: PartitionReference(partition_id='partition-a', dataset_id='dataset-a')\n"
        "  got: PartitionReference(partition_id='partition-b', dataset_id='dataset-a')"
    )


def test_validate_attempt_result_rejects_wrong_prediction_type() -> None:
    """Reject classification predictions for a regression parent."""
    result = _complete_result(predictions=_classification_predictions())

    with pytest.raises(ValueError) as error:
        validate_attempt_result(result, parent=_parent(), schedule=_schedule())

    assert str(error.value) == (
        "Invalid attempt result:\n"
        "  parent_id: parent-a\n"
        "  model_id: model-a\n"
        "  attempt_id: attempt-a\n"
        "  field: predictions[0]\n"
        "  expected: RegressionPredictions\n"
        "  got: ClassificationPredictions"
    )


def test_validate_attempt_result_rejects_unresolved_failure_reference() -> None:
    """Reject a report reference absent from the attempt failure collection."""
    unresolved_id = FailureId("missing-failure")
    result = replace(
        _complete_result(),
        outcome=EvaluationOutcome.COMPLETED_WITH_ERRORS,
        performance=_report(
            ReportName.PERFORMANCE,
            status=ReportStatus.FAILED,
            failure_ids=(unresolved_id,),
        ),
    )

    with pytest.raises(ValueError) as error:
        validate_attempt_result(result, parent=_parent(), schedule=_schedule())

    assert str(error.value) == (
        "Invalid attempt result:\n"
        "  parent_id: parent-a\n"
        "  model_id: model-a\n"
        "  attempt_id: attempt-a\n"
        "  field: performance.failure_ids[0]\n"
        "  expected: a failure_id present in failures\n"
        "  got: unresolved failure_id 'missing-failure'"
    )


def test_validate_attempt_result_rejects_operational_failure() -> None:
    """Reject an operational failure from a scientific result envelope."""
    failure = FailureRecord(
        failure_id=FailureId("sync-failure"),
        scope=FailureScope.DATA_SYNC,
        context=TraceContext(
            stage="sync",
            operation_id=OperationId("operation-a"),
        ),
        exception_type="RuntimeError",
        message="sync failed",
        traceback="traceback",
    )
    result = replace(_complete_result(), failures=(failure,))

    with pytest.raises(ValueError) as error:
        validate_attempt_result(result, parent=_parent(), schedule=_schedule())

    assert str(error.value) == (
        "Invalid attempt result:\n"
        "  parent_id: parent-a\n"
        "  model_id: model-a\n"
        "  attempt_id: attempt-a\n"
        "  field: failures[0].scope\n"
        "  expected: a scientific attempt-result failure scope\n"
        "  got: data_sync"
    )


def test_validate_attempt_result_rejects_cross_attempt_failure() -> None:
    """Reject failure provenance naming another execution attempt."""
    failure = _failure("prediction-failure", attempt_id="attempt-b")
    result = replace(_validation_failed_result(), failures=(failure,))

    with pytest.raises(ValueError) as error:
        validate_attempt_result(result, parent=_parent(), schedule=_schedule())

    assert str(error.value) == (
        "Invalid attempt result:\n"
        "  parent_id: parent-a\n"
        "  model_id: model-a\n"
        "  attempt_id: attempt-a\n"
        "  field: failures[0].context.attempt_id\n"
        "  expected: 'attempt-a'\n"
        "  got: 'attempt-b'"
    )


def test_validate_attempt_result_rejects_misaligned_fold_reference() -> None:
    """Reject a value-level failure tied to a different known fold."""
    failure = _failure(
        "prediction-failure",
        fold_id="fold-b",
        sample_id="sample-1",
    )
    result = replace(_validation_failed_result(), failures=(failure,))

    with pytest.raises(ValueError) as error:
        validate_attempt_result(result, parent=_parent(), schedule=_schedule())

    assert str(error.value) == (
        "Invalid attempt result:\n"
        "  parent_id: parent-a\n"
        "  model_id: model-a\n"
        "  attempt_id: attempt-a\n"
        "  field: predictions[0].failure_ids[0][0]\n"
        "  expected: failure context for fold 'fold-a'\n"
        "  got: failure context for fold 'fold-b'"
    )


def test_validate_attempt_result_rejects_misaligned_class_reference() -> None:
    """Reject a probability failure tied to another class column."""
    failure_id = FailureId("class-failure")
    failure = _failure(
        "class-failure",
        fold_id="fold-a",
        sample_id="sample-3",
        class_label="negative",
    )
    first_batch = replace(
        _classification_predictions()[0],
        probabilities=((0.2, None), (0.7, 0.3)),
        probability_failure_ids=(((), (failure_id,)), ((), ())),
    )
    base = _complete_result(
        predictions=(first_batch, _classification_predictions()[1]),
    )
    result = replace(
        base,
        outcome=EvaluationOutcome.VALIDATION_FAILED,
        validation=_report(
            ReportName.VALIDATION,
            status=ReportStatus.FAILED,
            failure_ids=(failure_id,),
        ),
        performance=_report(
            ReportName.PERFORMANCE,
            status=ReportStatus.BLOCKED,
            failure_ids=(failure_id,),
        ),
        residual=_report(
            ReportName.RESIDUAL,
            status=ReportStatus.BLOCKED,
            failure_ids=(failure_id,),
        ),
        visualization=_report(
            ReportName.VISUALIZATION,
            status=ReportStatus.BLOCKED,
            failure_ids=(failure_id,),
        ),
        failures=(failure,),
    )

    with pytest.raises(ValueError) as error:
        validate_attempt_result(
            result,
            parent=_parent(task_kind=TaskKind.CLASSIFICATION),
            schedule=_schedule(),
        )

    assert str(error.value) == (
        "Invalid attempt result:\n"
        "  parent_id: parent-a\n"
        "  model_id: model-a\n"
        "  attempt_id: attempt-a\n"
        "  field: predictions[0].probability_failure_ids[0][1][0]\n"
        "  expected: failure context for class 'positive'\n"
        "  got: failure context for class 'negative'"
    )


def test_validate_attempt_result_rejects_nontraining_influence_sample() -> None:
    """Reject influence provenance for a sample outside the fold training set."""
    failure_id = FailureId("influence-failure")
    failure = _failure(
        "influence-failure",
        scope=FailureScope.SAMPLE_INFLUENCE,
        stage="sample-removal-validation",
        fold_id="fold-a",
        sample_id="sample-3",
        report_name=ReportName.INFLUENCE,
    )
    result = replace(
        _complete_result(influence_requested=True),
        outcome=EvaluationOutcome.COMPLETED_WITH_ERRORS,
        influence=_report(
            ReportName.INFLUENCE,
            status=ReportStatus.FAILED,
            failure_ids=(failure_id,),
        ),
        failures=(failure,),
    )

    with pytest.raises(ValueError) as error:
        validate_attempt_result(
            result,
            parent=_parent(influence_requested=True),
            schedule=_schedule(),
        )

    assert str(error.value) == (
        "Invalid attempt result:\n"
        "  parent_id: parent-a\n"
        "  model_id: model-a\n"
        "  attempt_id: attempt-a\n"
        "  field: failures[0].context.sample_id\n"
        "  expected: a training member of the referenced influence fold\n"
        "  got: 'sample-3'"
    )


def test_validate_attempt_result_rejects_complete_failure_owner() -> None:
    """Reject a report failure when its owning report remains complete."""
    failure = _failure(
        "performance-failure",
        scope=FailureScope.REPORT,
        stage="performance",
        fold_id=None,
        sample_id=None,
        report_name=ReportName.PERFORMANCE,
    )
    result = replace(
        _complete_result(),
        validation=_report(
            ReportName.VALIDATION,
            status=ReportStatus.FAILED,
            failure_ids=(FailureId("performance-failure"),),
        ),
        failures=(failure,),
    )

    with pytest.raises(ValueError) as error:
        validate_attempt_result(result, parent=_parent(), schedule=_schedule())

    assert str(error.value) == (
        "Invalid attempt result:\n"
        "  parent_id: parent-a\n"
        "  model_id: model-a\n"
        "  attempt_id: attempt-a\n"
        "  field: failures[0].context.report_name\n"
        "  expected: report 'performance' to have failed status\n"
        "  got: complete"
    )


def test_validate_attempt_result_rejects_unreferenced_failure() -> None:
    """Reject a failure unrelated to any retained result or origin chain."""
    failure = _failure(
        "performance-failure",
        scope=FailureScope.REPORT,
        stage="performance",
        fold_id=None,
        sample_id=None,
        report_name=ReportName.PERFORMANCE,
    )
    result = replace(
        _complete_result(),
        outcome=EvaluationOutcome.COMPLETED_WITH_ERRORS,
        performance=_report(
            ReportName.PERFORMANCE,
            status=ReportStatus.FAILED,
            failure_ids=(FailureId("other-failure"),),
        ),
        failures=(
            failure,
            _failure(
                "other-failure",
                scope=FailureScope.REPORT,
                stage="performance",
                fold_id=None,
                sample_id=None,
                report_name=ReportName.PERFORMANCE,
            ),
        ),
    )

    with pytest.raises(ValueError) as error:
        validate_attempt_result(result, parent=_parent(), schedule=_schedule())

    assert str(error.value) == (
        "Invalid attempt result:\n"
        "  parent_id: parent-a\n"
        "  model_id: model-a\n"
        "  attempt_id: attempt-a\n"
        "  field: failures\n"
        "  expected: every failure to be referenced by a report or prediction, "
        "or be an origin of a referenced failure\n"
        "  got: unreferenced failure_id 'performance-failure'"
    )


@pytest.mark.parametrize(
    ("result", "parent", "expected"),
    [
        pytest.param(
            replace(
                _complete_result(),
                failures=(
                    _failure(
                        "reconstruction-failure",
                        stage="reconstruction",
                        fold_id=None,
                        sample_id=None,
                    ),
                ),
            ),
            _parent(),
            "reconstruction_failed because reconstruction failed",
            id="reconstruction-precedence",
        ),
        pytest.param(
            replace(
                _validation_failed_result(),
                outcome=EvaluationOutcome.COMPLETED_WITH_ERRORS,
            ),
            _parent(),
            "validation_failed because primary validation contains a failure",
            id="validation-precedence",
        ),
        pytest.param(
            replace(
                _complete_result(),
                outcome=EvaluationOutcome.COMPLETED,
                performance=_report(
                    ReportName.PERFORMANCE,
                    status=ReportStatus.FAILED,
                    failure_ids=(FailureId("performance-failure"),),
                ),
                failures=(
                    _failure(
                        "performance-failure",
                        scope=FailureScope.REPORT,
                        stage="performance",
                        fold_id=None,
                        sample_id=None,
                        report_name=ReportName.PERFORMANCE,
                    ),
                ),
            ),
            _parent(),
            "completed_with_errors",
            id="downstream-precedence",
        ),
    ],
)
def test_validate_attempt_result_rejects_wrong_outcome_precedence(
    result: _ExampleAttemptResult,
    parent: ParentEvaluation,
    expected: str,
) -> None:
    """Reject an attempt outcome that ignores the highest-priority failure."""
    with pytest.raises(ValueError) as error:
        validate_attempt_result(result, parent=parent, schedule=_schedule())

    assert f"  expected: {expected}" in str(error.value)


def test_validate_attempt_result_rejects_wrong_report_name() -> None:
    """Reject a report stored in a slot for another report identity."""
    result = replace(
        _complete_result(),
        performance=_report(ReportName.RESIDUAL),
    )

    with pytest.raises(ValueError) as error:
        validate_attempt_result(result, parent=_parent(), schedule=_schedule())

    assert str(error.value) == (
        "Invalid attempt result:\n"
        "  parent_id: parent-a\n"
        "  model_id: model-a\n"
        "  attempt_id: attempt-a\n"
        "  field: performance.name\n"
        "  expected: performance\n"
        "  got: residual"
    )
