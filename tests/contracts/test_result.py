"""Tests for standardized result-envelope contracts."""

from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass, fields, replace
from typing import cast

import numpy as np
import pytest

import cocoonet._contracts.result as result_contracts
from cocoonet._contracts.data import (
    FoldRecord,
    PartitionReference,
    PartitionSchedule,
    SampleRecord,
)
from cocoonet._contracts.execution import (
    EvaluationReference,
    ModelFamily,
    ModelReference,
    ParentEvaluation,
    TaskKind,
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
    derive_evaluation_outcome,
    validate_attempt_result,
    validate_report_outcome,
)


@dataclass(frozen=True, slots=True)
class _ExampleReportContent:
    """Provide small immutable report content for declaration tests."""

    value: float | None


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
        status=ReportStatus.FAILED,
        payload=_ExampleReportContent(value=1.25),
        failure_ids=(FailureId("failure-a"),),
    )
    same = ReportOutcome[_ExampleReportContent](
        status=ReportStatus.FAILED,
        payload=_ExampleReportContent(value=1.25),
        failure_ids=(FailureId("failure-a"),),
    )

    assert outcome.status is ReportStatus.FAILED
    assert outcome.payload == _ExampleReportContent(value=1.25)
    assert outcome.failure_ids == (FailureId("failure-a"),)
    assert outcome == same


def test_report_outcome_accepts_absent_report_content() -> None:
    """Represent a report outcome without retained report content."""
    outcome = ReportOutcome[_ExampleReportContent](
        status=ReportStatus.BLOCKED,
        payload=None,
        failure_ids=(FailureId("prerequisite-failure"),),
    )

    assert outcome.payload is None


def test_report_outcome_rejects_field_mutation() -> None:
    """Prevent reassignment of report-outcome fields."""
    outcome = ReportOutcome[_ExampleReportContent](
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
        status=ReportStatus.COMPLETE,
        payload=_ExampleReportContent(value=1.0),
        failure_ids=(),
    )
    performance = ReportOutcome[_ExampleReportContent](
        status=ReportStatus.COMPLETE,
        payload=_ExampleReportContent(value=2.0),
        failure_ids=(),
    )
    residual = ReportOutcome[_ExampleReportContent](
        status=ReportStatus.COMPLETE,
        payload=_ExampleReportContent(value=3.0),
        failure_ids=(),
    )
    visualization = ReportOutcome[_ExampleReportContent](
        status=ReportStatus.COMPLETE,
        payload=_ExampleReportContent(value=4.0),
        failure_ids=(),
    )
    influence = ReportOutcome[_ExampleReportContent](
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
        parent_id=ParentId("parent-a"),
        model_id=ModelId("model-a"),
        attempt_id=AttemptId("attempt-a"),
        worker_id=WorkerId("worker-a"),
        predictions=predictions,
        validation=validation,
        performance=performance,
        residual=residual,
        visualization=visualization,
        influence=influence,
        failures=(),
    )

    assert (result.parent_id, result.model_id, result.attempt_id, result.worker_id) == (
        "parent-a",
        "model-a",
        "attempt-a",
        "worker-a",
    )
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
        parent_id=ParentId("parent-a"),
        model_id=ModelId("model-a"),
        attempt_id=AttemptId("attempt-a"),
        worker_id=WorkerId("worker-a"),
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
        parent_id=ParentId("parent-a"),
        model_id=ModelId("model-a"),
        attempt_id=AttemptId("attempt-a"),
        worker_id=WorkerId("worker-a"),
        predictions=(),
        validation=None,
        performance=None,
        residual=None,
        visualization=None,
        influence=None,
        failures=(),
    )

    with pytest.raises(FrozenInstanceError):
        result.__setattr__("attempt_id", AttemptId("replacement"))


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
        status=status,
        payload=payload,
        failure_ids=failure_ids,
    )

    with pytest.raises(ValueError) as error:
        validate_report_outcome(outcome)

    expected_details = message
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


def test_validate_report_outcome_rejects_empty_failure_identity() -> None:
    """Identify an empty report-level failure reference precisely."""
    outcome = ReportOutcome[_ExampleReportContent](
        status=ReportStatus.FAILED,
        payload=None,
        failure_ids=(FailureId(""),),
    )

    with pytest.raises(ValueError) as error:
        validate_report_outcome(outcome)

    assert "failure_ids[0]" in str(error.value)
    assert "nonempty" in str(error.value) and "empty string" in str(error.value)


def test_validate_report_outcome_rejects_wrong_status_type() -> None:
    """Preserve the unified type diagnostic for an invalid status type."""
    outcome = ReportOutcome[_ExampleReportContent](
        status=cast(ReportStatus, "complete"),
        payload=_ExampleReportContent(value=1.0),
        failure_ids=(),
    )

    with pytest.raises(TypeError) as error:
        validate_report_outcome(outcome)

    expected_details = (
        "Type validation failed\n"
        "  path: status\n"
        "  expected: cocoonet._contracts.result.ReportStatus\n"
        "  got type: builtins.str\n"
        "  got value: 'complete'"
    )
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


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
        sources=(f"samples/{number}.csv",),
    )


def _schedule(*, task_kind: TaskKind = TaskKind.REGRESSION) -> PartitionSchedule:
    """Build a two-fold schedule with distinct training and testing groups."""
    return PartitionSchedule(
        reference=PartitionReference(
            partition_id=PartitionId("partition-a"),
            dataset_id=DatasetId("dataset-a"),
        ),
        samples=tuple(
            (
                replace(_sample(number), target=("negative" if number % 2 else "positive"))
                if task_kind is TaskKind.CLASSIFICATION
                else _sample(number)
            )
            for number in range(1, 5)
        ),
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
    return ParentEvaluation(
        parent_id=ParentId("parent-a"),
        partition=_schedule().reference,
        evaluation=EvaluationReference(
            configuration_id=EvaluationConfigId("evaluation-a"),
            task_kind=task_kind,
            class_labels=class_labels,
            influence_requested=influence_requested,
        ),
        models=(model,),
    )


def _report(
    *,
    status: ReportStatus = ReportStatus.COMPLETE,
    failure_ids: tuple[FailureId, ...] = (),
    value: float | None = 1.0,
) -> ReportOutcome[_ExampleReportContent]:
    """Build one report outcome with content appropriate to its status."""
    payload = None if status is ReportStatus.NOT_REQUESTED else _ExampleReportContent(value=value)
    return ReportOutcome(
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
    class_label: str | None = None,
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
    influence = _report() if influence_requested else _report(status=ReportStatus.NOT_REQUESTED)
    return AttemptResult(
        parent_id=ParentId("parent-a"),
        model_id=ModelId("model-a"),
        attempt_id=AttemptId("attempt-a"),
        worker_id=WorkerId("worker-a"),
        predictions=predictions,
        validation=_report(),
        performance=_report(),
        residual=_report(),
        visualization=_report(),
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
            status=ReportStatus.BLOCKED,
            failure_ids=(failure_id,),
        )
        if influence_requested
        else _report(status=ReportStatus.NOT_REQUESTED)
    )
    return AttemptResult(
        parent_id=ParentId("parent-a"),
        model_id=ModelId("model-a"),
        attempt_id=AttemptId("attempt-a"),
        worker_id=WorkerId("worker-a"),
        predictions=predictions,
        validation=_report(
            status=ReportStatus.FAILED,
            failure_ids=(failure_id,),
        ),
        performance=_report(
            status=ReportStatus.BLOCKED,
            failure_ids=(failure_id,),
        ),
        residual=_report(
            status=ReportStatus.BLOCKED,
            failure_ids=(failure_id,),
        ),
        visualization=_report(
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

    assert derive_evaluation_outcome(result) is EvaluationOutcome.COMPLETED


def test_validate_attempt_result_accepts_completed_classification() -> None:
    """Accept complete classification predictions in parent class order."""
    result = _complete_result(predictions=_classification_predictions())

    validate_attempt_result(
        result,
        parent=_parent(task_kind=TaskKind.CLASSIFICATION),
        schedule=_schedule(task_kind=TaskKind.CLASSIFICATION),
    )

    assert derive_evaluation_outcome(result) is EvaluationOutcome.COMPLETED


def test_validate_attempt_result_preserves_formula_defined_nan() -> None:
    """Preserve a formula-defined NaN inside a concrete report payload."""
    result = replace(
        _complete_result(),
        performance=_report(value=float("nan")),
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
        parent_id=ParentId("parent-a"),
        model_id=ModelId("model-a"),
        attempt_id=AttemptId("attempt-a"),
        worker_id=WorkerId("worker-a"),
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
        performance=_report(
            status=ReportStatus.FAILED,
            failure_ids=(failure_id,),
        ),
        failures=(failure,),
    )

    validate_attempt_result(result, parent=_parent(), schedule=_schedule())

    assert derive_evaluation_outcome(result) is EvaluationOutcome.COMPLETED_WITH_ERRORS


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
        influence=_report(
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

    assert derive_evaluation_outcome(result) is EvaluationOutcome.COMPLETED_WITH_ERRORS


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

    expected_details = (
        "Invalid attempt result:\n"
        "  parent_id: parent-a\n"
        "  model_id: model-a\n"
        "  attempt_id: attempt-a\n"
        "  worker_id: worker-a\n"
        "  field: schedule.reference.partition_id\n"
        "  expected: 'partition-a'\n"
        "  got: 'partition-b'"
    )
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


def test_validate_attempt_result_rejects_wrong_prediction_type() -> None:
    """Reject classification predictions for a regression parent."""
    result = _complete_result(predictions=_classification_predictions())

    with pytest.raises(ValueError) as error:
        validate_attempt_result(result, parent=_parent(), schedule=_schedule())

    expected_details = (
        "Invalid attempt result:\n"
        "  parent_id: parent-a\n"
        "  model_id: model-a\n"
        "  attempt_id: attempt-a\n"
        "  worker_id: worker-a\n"
        "  field: predictions[0]\n"
        "  expected: RegressionPredictions\n"
        "  got: ClassificationPredictions"
    )
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


def test_validate_attempt_result_rejects_unresolved_failure_reference() -> None:
    """Reject a report reference absent from the attempt failure collection."""
    unresolved_id = FailureId("missing-failure")
    result = replace(
        _complete_result(),
        performance=_report(
            status=ReportStatus.FAILED,
            failure_ids=(unresolved_id,),
        ),
    )

    with pytest.raises(ValueError) as error:
        validate_attempt_result(result, parent=_parent(), schedule=_schedule())

    expected_details = (
        "Invalid attempt result:\n"
        "  parent_id: parent-a\n"
        "  model_id: model-a\n"
        "  attempt_id: attempt-a\n"
        "  worker_id: worker-a\n"
        "  field: performance.failure_ids[0]\n"
        "  expected: a failure_id present in failures\n"
        "  got: unresolved failure_id 'missing-failure'"
    )
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


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

    expected_details = (
        "Invalid attempt result:\n"
        "  parent_id: parent-a\n"
        "  model_id: model-a\n"
        "  attempt_id: attempt-a\n"
        "  worker_id: worker-a\n"
        "  field: failures[0].scope\n"
        "  expected: a scientific attempt-result failure scope\n"
        "  got: data_sync"
    )
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


def test_validate_attempt_result_rejects_cross_attempt_failure() -> None:
    """Reject failure provenance naming another execution attempt."""
    failure = _failure("prediction-failure", attempt_id="attempt-b")
    result = replace(_validation_failed_result(), failures=(failure,))

    with pytest.raises(ValueError) as error:
        validate_attempt_result(result, parent=_parent(), schedule=_schedule())

    expected_details = (
        "Invalid attempt result:\n"
        "  parent_id: parent-a\n"
        "  model_id: model-a\n"
        "  attempt_id: attempt-a\n"
        "  worker_id: worker-a\n"
        "  field: failures[0].context.attempt_id\n"
        "  expected: 'attempt-a'\n"
        "  got: 'attempt-b'"
    )
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


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

    expected_details = (
        "Invalid attempt result:\n"
        "  parent_id: parent-a\n"
        "  model_id: model-a\n"
        "  attempt_id: attempt-a\n"
        "  worker_id: worker-a\n"
        "  field: predictions[0].failure_ids[0][0]\n"
        "  expected: failure context for fold 'fold-a'\n"
        "  got: failure context for fold 'fold-b'"
    )
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


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
        validation=_report(
            status=ReportStatus.FAILED,
            failure_ids=(failure_id,),
        ),
        performance=_report(
            status=ReportStatus.BLOCKED,
            failure_ids=(failure_id,),
        ),
        residual=_report(
            status=ReportStatus.BLOCKED,
            failure_ids=(failure_id,),
        ),
        visualization=_report(
            status=ReportStatus.BLOCKED,
            failure_ids=(failure_id,),
        ),
        failures=(failure,),
    )

    with pytest.raises(ValueError) as error:
        validate_attempt_result(
            result,
            parent=_parent(task_kind=TaskKind.CLASSIFICATION),
            schedule=_schedule(task_kind=TaskKind.CLASSIFICATION),
        )

    expected_details = (
        "Invalid attempt result:\n"
        "  parent_id: parent-a\n"
        "  model_id: model-a\n"
        "  attempt_id: attempt-a\n"
        "  worker_id: worker-a\n"
        "  field: predictions[0].probability_failure_ids[0][1][0]\n"
        "  expected: failure context for class 'positive'\n"
        "  got: failure context for class 'negative'"
    )
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


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
        influence=_report(
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

    expected_details = (
        "Invalid attempt result:\n"
        "  parent_id: parent-a\n"
        "  model_id: model-a\n"
        "  attempt_id: attempt-a\n"
        "  worker_id: worker-a\n"
        "  field: failures[0].context.sample_id\n"
        "  expected: a training member of the referenced influence fold\n"
        "  got: 'sample-3'"
    )
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


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
            status=ReportStatus.FAILED,
            failure_ids=(FailureId("performance-failure"),),
        ),
        failures=(failure,),
    )

    with pytest.raises(ValueError) as error:
        validate_attempt_result(result, parent=_parent(), schedule=_schedule())

    expected_details = (
        "Invalid attempt result:\n"
        "  parent_id: parent-a\n"
        "  model_id: model-a\n"
        "  attempt_id: attempt-a\n"
        "  worker_id: worker-a\n"
        "  field: failures[0].context.report_name\n"
        "  expected: report 'performance' to have failed status\n"
        "  got: complete"
    )
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


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
        performance=_report(
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

    expected_details = (
        "Invalid attempt result:\n"
        "  parent_id: parent-a\n"
        "  model_id: model-a\n"
        "  attempt_id: attempt-a\n"
        "  worker_id: worker-a\n"
        "  field: failures\n"
        "  expected: every failure to be referenced by a report or prediction, "
        "or be an origin of a referenced failure\n"
        "  got: unreferenced failure_id 'performance-failure'"
    )
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


def _reconstruction_failed_result() -> _ExampleAttemptResult:
    """Build a complete failure-only result with reconstruction provenance."""
    return replace(
        _complete_result(),
        predictions=(),
        validation=None,
        performance=None,
        residual=None,
        visualization=None,
        influence=None,
        failures=(_failure("reconstruction-failure", stage="reconstruction", fold_id=None, sample_id=None),),
    )


@pytest.mark.parametrize(
    "result,expected",
    [
        (_complete_result(), EvaluationOutcome.COMPLETED),
        (_reconstruction_failed_result(), EvaluationOutcome.RECONSTRUCTION_FAILED),
        (_validation_failed_result(), EvaluationOutcome.VALIDATION_FAILED),
        (
            replace(
                _complete_result(),
                performance=_report(status=ReportStatus.FAILED, failure_ids=(FailureId("report-failure"),)),
                failures=(
                    _failure(
                        "report-failure",
                        scope=FailureScope.REPORT,
                        stage="performance",
                        fold_id=None,
                        sample_id=None,
                        report_name=ReportName.PERFORMANCE,
                    ),
                ),
            ),
            EvaluationOutcome.COMPLETED_WITH_ERRORS,
        ),
    ],
)
def test_derive_outcome_from_validated_facts(result: _ExampleAttemptResult, expected: EvaluationOutcome) -> None:
    """Derive each outcome, preserving primary-failure precedence over blocked reports."""
    validate_attempt_result(result, parent=_parent(), schedule=_schedule())
    before = replace(result)
    assert derive_evaluation_outcome(result) is expected
    assert derive_evaluation_outcome(result) is expected
    assert result == before


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), None])
def test_payload_edge_values_do_not_imply_failure(value: float | None) -> None:
    """Leave formula-defined and non-applicable payload values to the owning schema."""
    result = replace(_complete_result(), performance=_report(value=value))
    validate_attempt_result(result, parent=_parent(), schedule=_schedule())
    assert derive_evaluation_outcome(result) is EvaluationOutcome.COMPLETED
    assert result.performance is not None and result.performance.payload is not None
    assert result.performance.payload.value is value


def test_outcome_derivation_does_not_revalidate_or_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Summarize validated records without invoking the consuming boundary again."""
    completed = _complete_result()
    failed = _validation_failed_result()
    validate_attempt_result(completed, parent=_parent(), schedule=_schedule())
    validate_attempt_result(failed, parent=_parent(), schedule=_schedule())

    def unexpected_validation(*args: object, **kwargs: object) -> None:
        """Fail if derivation invokes recursive result validation."""
        pytest.fail("Outcome derivation must use its validated-input precondition")

    monkeypatch.setattr(result_contracts, "validate_attempt_result", unexpected_validation)
    assert derive_evaluation_outcome(completed) is EvaluationOutcome.COMPLETED
    assert derive_evaluation_outcome(failed) is EvaluationOutcome.VALIDATION_FAILED
    assert derive_evaluation_outcome(completed) is EvaluationOutcome.COMPLETED
    assert {field.name for field in fields(completed)} == {
        "parent_id",
        "model_id",
        "attempt_id",
        "worker_id",
        "predictions",
        "validation",
        "performance",
        "residual",
        "visualization",
        "influence",
        "failures",
    }
    assert {field.name for field in fields(_report())} == {"status", "payload", "failure_ids"}


@pytest.mark.parametrize(
    "task_kind,target",
    [
        (TaskKind.REGRESSION, np.float16(0.5)),
        (TaskKind.REGRESSION, np.float32(0.5)),
        (TaskKind.REGRESSION, np.longdouble("1.00000000000000001")),
        (TaskKind.REGRESSION, np.finfo(np.longdouble).max),
        (TaskKind.CLASSIFICATION, np.str_("positive")),
    ],
)
def test_attempt_preserves_task_compatible_target_scalars(task_kind: TaskKind, target: object) -> None:
    """Validate task context without converting standardized target values."""
    schedule = _schedule(task_kind=task_kind)
    # Widen only test construction so runtime validators receive the invalid candidate.
    first = cast(Callable[..., SampleRecord], replace)(schedule.samples[0], **{"target": target})
    schedule = replace(schedule, samples=(first, *schedule.samples[1:]))
    validate_attempt_result(_reconstruction_failed_result(), parent=_parent(task_kind=task_kind), schedule=schedule)
    assert schedule.samples[0].target is target


@pytest.mark.parametrize("reconstruction_failed", [False, True])
@pytest.mark.parametrize(
    "task_kind,target,error_type",
    [
        (TaskKind.REGRESSION, "positive", TypeError),
        (TaskKind.CLASSIFICATION, 1.5, TypeError),
        (TaskKind.CLASSIFICATION, "unknown", ValueError),
        (TaskKind.REGRESSION, float("nan"), ValueError),
        (TaskKind.REGRESSION, True, TypeError),
        (TaskKind.CLASSIFICATION, 1, TypeError),
    ],
)
def test_attempt_validates_masked_targets_before_stage_shortcut(
    reconstruction_failed: bool,
    task_kind: TaskKind,
    target: object,
    error_type: type[Exception],
) -> None:
    """Check every dataset target, including samples excluded from all folds."""
    schedule = _schedule(task_kind=task_kind)
    masked = cast(Callable[..., SampleRecord], replace)(_sample(5), train_mask=0, test_mask=0, **{"target": target})
    schedule = replace(schedule, samples=(*schedule.samples, masked))
    result = (
        _reconstruction_failed_result()
        if reconstruction_failed
        else _complete_result(
            predictions=_regression_predictions() if task_kind is TaskKind.REGRESSION else _classification_predictions()
        )
    )
    with pytest.raises(error_type) as error:
        validate_attempt_result(result, parent=_parent(task_kind=task_kind), schedule=schedule)
    assert "target" in str(error.value)
    assert "samples[4]" in str(error.value)


@pytest.mark.parametrize("field", ["parent_id", "model_id", "attempt_id", "worker_id"])
@pytest.mark.parametrize("invalid,error_type", [("", ValueError), (None, TypeError), (1, TypeError)])
def test_result_validates_direct_routing_fields(field: str, invalid: object, error_type: type[Exception]) -> None:
    """Reject invalid direct identities before interpreting scientific results."""
    with pytest.raises(error_type) as error:
        validate_attempt_result(
            cast(Callable[..., _ExampleAttemptResult], replace)(_complete_result(), **{field: invalid}),
            parent=_parent(),
            schedule=_schedule(),
        )
    assert field in str(error.value)


@pytest.mark.parametrize("field,identity", [("parent_id", "other-parent"), ("model_id", "unknown-model")])
def test_result_resolves_parent_and_model(field: str, identity: str) -> None:
    """Reject cross-parent or unknown-model routing even with valid ID spelling."""
    with pytest.raises(ValueError) as error:
        validate_attempt_result(
            cast(Callable[..., _ExampleAttemptResult], replace)(_complete_result(), **{field: identity}),
            parent=_parent(),
            schedule=_schedule(),
        )
    assert field in str(error.value) and identity in str(error.value)


def test_result_rejects_wrong_dataset_binding() -> None:
    """Require both members of the parent's partition association to match."""
    schedule = _schedule()
    schedule = replace(schedule, reference=replace(schedule.reference, dataset_id=DatasetId("other-dataset")))
    with pytest.raises(ValueError) as error:
        validate_attempt_result(_complete_result(), parent=_parent(), schedule=schedule)
    assert "schedule.reference.dataset_id" in str(error.value)
    assert "dataset-a" in str(error.value) and "other-dataset" in str(error.value)


def test_result_boundaries_reject_wrong_outer_records() -> None:
    """Reject non-record envelopes and invalid parent/schedule context."""
    with pytest.raises(TypeError, match="ReportOutcome"):
        validate_report_outcome(cast(ReportOutcome[object], None))
    with pytest.raises(TypeError, match="AttemptResult"):
        validate_attempt_result(cast(_ExampleAttemptResult, None), parent=_parent(), schedule=_schedule())
    with pytest.raises(TypeError, match="ParentEvaluation"):
        validate_attempt_result(_complete_result(), parent=cast(ParentEvaluation, None), schedule=_schedule())
    with pytest.raises(TypeError, match="PartitionSchedule"):
        validate_attempt_result(_complete_result(), parent=_parent(), schedule=cast(PartitionSchedule, None))


@pytest.mark.parametrize("field,invalid", [("predictions", []), ("failures", []), ("failures", (None,))])
def test_result_rejects_malformed_collections(field: str, invalid: object) -> None:
    """Require immutable result collections and valid failure records."""
    with pytest.raises(TypeError):
        validate_attempt_result(
            cast(Callable[..., _ExampleAttemptResult], replace)(_complete_result(), **{field: invalid}),
            parent=_parent(),
            schedule=_schedule(),
        )


@pytest.mark.parametrize("field", ["validation", "performance", "residual", "visualization", "influence"])
@pytest.mark.parametrize(
    "report,error_type",
    [
        (None, ValueError),
        ("report", TypeError),
        (ReportOutcome(status=ReportStatus.COMPLETE, payload=None, failure_ids=()), ValueError),
    ],
)
def test_result_requires_valid_report_in_each_slot(field: str, report: object, error_type: type[Exception]) -> None:
    """Reject absent or invalid post-reconstruction reports with their slot context."""
    with pytest.raises(error_type) as error:
        validate_attempt_result(
            cast(Callable[..., _ExampleAttemptResult], replace)(_complete_result(), **{field: report}),
            parent=_parent(),
            schedule=_schedule(),
        )
    assert f"field: {field}" in str(error.value)


@pytest.mark.parametrize("field", ["validation", "performance", "residual", "visualization"])
def test_mandatory_report_cannot_be_unrequested(field: str) -> None:
    """Reserve not_requested for optional influence alone."""
    with pytest.raises(ValueError) as error:
        validate_attempt_result(
            cast(Callable[..., _ExampleAttemptResult], replace)(
                _complete_result(), **{field: _report(status=ReportStatus.NOT_REQUESTED)}
            ),
            parent=_parent(),
            schedule=_schedule(),
        )
    assert f"field: {field}.status" in str(error.value) and "not_requested" in str(error.value)


@pytest.mark.parametrize("requested", [False, True])
def test_influence_status_matches_parent_request(requested: bool) -> None:
    """Require influence reporting to follow the fixed evaluation setting."""
    with pytest.raises(ValueError) as error:
        validate_attempt_result(
            _complete_result(influence_requested=not requested),
            parent=_parent(influence_requested=requested),
            schedule=_schedule(),
        )
    assert "influence.status" in str(error.value)


@pytest.mark.parametrize("field", ["validation", "performance", "residual", "visualization", "influence"])
def test_reconstruction_failure_has_no_report_records(field: str) -> None:
    """Reject even an unrequested report record before successful reconstruction."""
    with pytest.raises(ValueError) as error:
        validate_attempt_result(
            cast(Callable[..., _ExampleAttemptResult], replace)(_reconstruction_failed_result(), **{field: _report()}),
            parent=_parent(),
            schedule=_schedule(),
        )
    assert f"field: {field}" in str(error.value) and "None after reconstruction failure" in str(error.value)


def test_reconstruction_failure_cannot_include_later_stage_work() -> None:
    """Reject predictions and mixed-stage failures instead of masking them through precedence."""
    failed = _reconstruction_failed_result()
    with pytest.raises(ValueError, match="empty tuple after reconstruction failure"):
        validate_attempt_result(
            replace(failed, predictions=_regression_predictions()), parent=_parent(), schedule=_schedule()
        )
    with pytest.raises(ValueError, match="only reconstruction-stage"):
        validate_attempt_result(
            replace(failed, failures=(*failed.failures, _failure("later-fold"))), parent=_parent(), schedule=_schedule()
        )


def test_failure_only_result_requires_reconstruction_provenance() -> None:
    """Reject an empty candidate envelope lacking the only allowed failure-only cause."""
    with pytest.raises(ValueError, match="report outcome after successful reconstruction"):
        validate_attempt_result(
            replace(_reconstruction_failed_result(), failures=()), parent=_parent(), schedule=_schedule()
        )


@pytest.mark.parametrize("predictions", [(), _regression_predictions()[:1], _regression_predictions()[::-1]])
def test_result_requires_every_fold_in_saved_order(predictions: tuple[RegressionPredictions, ...]) -> None:
    """Reject missing or reordered folds without discarding later successful results."""
    with pytest.raises(ValueError) as error:
        validate_attempt_result(
            replace(_complete_result(), predictions=predictions), parent=_parent(), schedule=_schedule()
        )
    assert ("batches in schedule order" if len(predictions) != 2 else "fold_id") in str(error.value)


def test_classification_parent_rejects_regression_batches() -> None:
    """Enforce task-specific prediction type in both task directions."""
    with pytest.raises(ValueError, match="ClassificationPredictions"):
        validate_attempt_result(
            _complete_result(),
            parent=_parent(task_kind=TaskKind.CLASSIFICATION),
            schedule=_schedule(task_kind=TaskKind.CLASSIFICATION),
        )


@pytest.mark.parametrize(
    "field,report",
    [
        ("validation", _report()),
        (
            "validation",
            ReportOutcome(status=ReportStatus.FAILED, payload=None, failure_ids=(FailureId("prediction-failure"),)),
        ),
        ("performance", _report()),
        ("residual", _report()),
        ("visualization", _report()),
        ("influence", _report()),
    ],
)
def test_primary_failure_requires_consistent_dependent_reports(field: str, report: object) -> None:
    """Preserve failed validation content and block dependent influence computation."""
    with pytest.raises(ValueError) as error:
        validate_attempt_result(
            cast(Callable[..., _ExampleAttemptResult], replace)(
                _validation_failed_result(influence_requested=True), **{field: report}
            ),
            parent=_parent(influence_requested=True),
            schedule=_schedule(),
        )
    assert f"field: {field}." in str(error.value)


def test_successful_primary_validation_requires_complete_validation_report() -> None:
    """Reject a validation-report status unsupported by primary-fold failures."""
    failure = _failure(
        "validation-report",
        scope=FailureScope.REPORT,
        stage="report",
        fold_id=None,
        sample_id=None,
        report_name=ReportName.VALIDATION,
    )
    result = replace(
        _complete_result(),
        failures=(failure,),
        validation=_report(status=ReportStatus.FAILED, failure_ids=(failure.failure_id,)),
    )
    with pytest.raises(ValueError, match="complete after successful primary validation"):
        validate_attempt_result(result, parent=_parent(), schedule=_schedule())


@pytest.mark.parametrize(
    "failure,field",
    [
        (_failure("prediction-failure", fold_id=None, sample_id=None), "fold_id"),
        (_failure("prediction-failure", fold_id="unknown-fold"), "fold_id"),
        (_failure("prediction-failure", sample_id="unknown-sample"), "sample_id"),
        (_failure("prediction-failure", sample_id="sample-1"), "sample_id"),
        (_failure("prediction-failure", class_label="positive"), "class_label"),
        (_failure("prediction-failure", parent_id="other-parent"), "parent_id"),
        (_failure("prediction-failure", model_id="other-model"), "model_id"),
        (_failure("prediction-failure", worker_id="other-worker"), "worker_id"),
    ],
)
def test_result_rejects_invalid_failure_coordinates(failure: FailureRecord, field: str) -> None:
    """Resolve scientific provenance within the fixed parent, fold, and dataset."""
    with pytest.raises(ValueError) as error:
        validate_attempt_result(
            replace(_validation_failed_result(), failures=(failure,)), parent=_parent(), schedule=_schedule()
        )
    assert f"failures[0].context.{field}" in str(error.value)


def test_classification_failure_requires_known_class() -> None:
    """Reject failure coordinates outside the authoritative class vocabulary."""
    result = replace(
        _complete_result(predictions=_classification_predictions()),
        failures=(_failure("unknown-class", class_label="unknown"),),
    )
    with pytest.raises(ValueError, match="class in the parent vocabulary"):
        validate_attempt_result(
            result,
            parent=_parent(task_kind=TaskKind.CLASSIFICATION),
            schedule=_schedule(task_kind=TaskKind.CLASSIFICATION),
        )


def test_prediction_failure_reference_requires_matching_sample() -> None:
    """Reject provenance for a different held-out sample in the same fold."""
    result = replace(_validation_failed_result(), failures=(_failure("prediction-failure", sample_id="sample-4"),))
    with pytest.raises(ValueError) as error:
        validate_attempt_result(result, parent=_parent(), schedule=_schedule())
    assert "sample 'sample-3'" in str(error.value) and "sample 'sample-4'" in str(error.value)


def test_prediction_unavailability_cannot_reference_report_failure() -> None:
    """Keep model prediction failures distinct from later report computation failures."""
    failure = _failure(
        "prediction-failure", scope=FailureScope.REPORT, stage="performance", report_name=ReportName.PERFORMANCE
    )
    result = replace(
        _validation_failed_result(),
        failures=(failure,),
        performance=_report(status=ReportStatus.FAILED, failure_ids=(failure.failure_id,)),
    )
    with pytest.raises(ValueError, match="expected: a fit_prediction failure"):
        validate_attempt_result(result, parent=_parent(), schedule=_schedule())


def test_result_preserves_origin_only_failures() -> None:
    """Retain an originating failure reachable only through dependent provenance."""
    origin = _failure("fit-origin", stage="fit", sample_id=None)
    dependent = _failure("prediction-failure", origin_failure_id=origin.failure_id)
    result = replace(_validation_failed_result(), failures=(dependent, origin))
    validate_attempt_result(result, parent=_parent(), schedule=_schedule())
    assert result.failures == (dependent, origin)
    assert derive_evaluation_outcome(result) is EvaluationOutcome.VALIDATION_FAILED


@pytest.mark.parametrize("unavailable_label", [True, False])
def test_classification_result_resolves_independent_unavailability(unavailable_label: bool) -> None:
    """Resolve label or probability failures while preserving independent successful values."""
    failure_id = FailureId("prediction-failure")
    first = _classification_predictions()[0]
    first = (
        replace(first, predicted_labels=(None, "negative"), label_failure_ids=((failure_id,), ()))
        if unavailable_label
        else replace(
            first, probabilities=((0.2, None), (0.7, 0.3)), probability_failure_ids=(((), (failure_id,)), ((), ()))
        )
    )
    failure = _failure("prediction-failure", class_label=None if unavailable_label else "positive")
    result = replace(
        _validation_failed_result(), predictions=(first, _classification_predictions()[1]), failures=(failure,)
    )
    validate_attempt_result(
        result, parent=_parent(task_kind=TaskKind.CLASSIFICATION), schedule=_schedule(task_kind=TaskKind.CLASSIFICATION)
    )
    assert result.predictions[1] == _classification_predictions()[1]
    assert derive_evaluation_outcome(result) is EvaluationOutcome.VALIDATION_FAILED


@pytest.mark.parametrize("scope", [FailureScope.REPORT, FailureScope.MEASURE])
def test_blocked_ordinary_report_derives_completion_with_errors(scope: FailureScope) -> None:
    """Preserve downstream blocking after successful baseline validation."""
    failure = _failure(
        "residual-failure", scope=scope, stage="residual", fold_id=None, sample_id=None, report_name=ReportName.RESIDUAL
    )
    result = replace(
        _complete_result(),
        failures=(failure,),
        residual=_report(status=ReportStatus.FAILED, failure_ids=(failure.failure_id,)),
        performance=_report(status=ReportStatus.BLOCKED, failure_ids=(failure.failure_id,)),
    )
    validate_attempt_result(result, parent=_parent(), schedule=_schedule())
    assert derive_evaluation_outcome(result) is EvaluationOutcome.COMPLETED_WITH_ERRORS
    assert result.validation == _report()


@pytest.mark.parametrize("stage", ["fold_initialization", "fit", "prediction"])
def test_primary_failure_with_empty_testing_population(stage: str) -> None:
    """Derive primary failure from provenance even without unavailable prediction positions."""
    schedule = _schedule()
    empty_fold = replace(schedule.folds[0], testing_sample_ids=())
    schedule = replace(schedule, folds=(empty_fold, schedule.folds[1]))
    empty_batch = RegressionPredictions(fold_id=empty_fold.fold_id, sample_ids=(), values=(), failure_ids=())
    result = replace(
        _validation_failed_result(),
        predictions=(empty_batch, _regression_predictions()[1]),
        failures=(_failure("prediction-failure", stage=stage, sample_id=None),),
    )
    validate_attempt_result(result, parent=_parent(), schedule=schedule)
    assert derive_evaluation_outcome(result) is EvaluationOutcome.VALIDATION_FAILED


def test_ordinary_failure_does_not_block_successful_requested_influence() -> None:
    """Keep independent report and influence branches after complete primary validation."""
    failure = _failure(
        "performance-failure",
        scope=FailureScope.REPORT,
        stage="performance",
        fold_id=None,
        sample_id=None,
        report_name=ReportName.PERFORMANCE,
    )
    result = replace(
        _complete_result(influence_requested=True),
        failures=(failure,),
        performance=_report(status=ReportStatus.FAILED, failure_ids=(failure.failure_id,)),
    )
    validate_attempt_result(result, parent=_parent(influence_requested=True), schedule=_schedule())
    assert result.influence == _report()
    assert derive_evaluation_outcome(result) is EvaluationOutcome.COMPLETED_WITH_ERRORS


@pytest.mark.parametrize("invalid", [[], (1,), (None,)])
def test_report_failure_references_require_tuple_of_identifiers(invalid: object) -> None:
    """Validate report provenance independently of generic payload contents."""
    report = ReportOutcome(
        status=ReportStatus.FAILED,
        payload=_ExampleReportContent(value=1.0),
        failure_ids=cast(tuple[FailureId, ...], invalid),
    )
    with pytest.raises(TypeError) as error:
        validate_report_outcome(report)
    assert "failure_ids" in str(error.value)
    assert "expected:" in str(error.value) and "got type:" in str(error.value)


@pytest.mark.parametrize(
    "scope",
    [
        FailureScope.DATA_PREPARATION,
        FailureScope.DATA_SYNC,
        FailureScope.WORKER_MANAGEMENT,
        FailureScope.RESULT_TRANSFER,
        FailureScope.RESULT_PERSISTENCE,
        FailureScope.WORKER_PROCESS,
    ],
)
def test_result_excludes_all_operational_failure_scopes(scope: FailureScope) -> None:
    """Keep operational incidents outside a scientific attempt-result envelope."""
    failure = FailureRecord(
        failure_id=FailureId("operational"),
        scope=scope,
        context=TraceContext(
            stage="operation", worker_id=WorkerId("worker-a"), operation_id=OperationId("operation-a")
        ),
        exception_type="RuntimeError",
        message="Operation failed",
        traceback="",
    )
    with pytest.raises(ValueError) as error:
        validate_attempt_result(
            replace(_complete_result(), failures=(failure,)), parent=_parent(), schedule=_schedule()
        )
    assert "failures[0].scope" in str(error.value)
    assert "scientific" in str(error.value) and scope.value in str(error.value)
