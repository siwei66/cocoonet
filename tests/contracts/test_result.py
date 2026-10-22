"""Tests for standardized result-envelope contracts."""

from dataclasses import FrozenInstanceError, dataclass
from typing import cast

import pytest

from cocoonet._contracts.execution import AttemptInputs
from cocoonet._contracts.identity import (
    AttemptId,
    FailureId,
    FoldId,
    ModelId,
    ParentId,
    ReportName,
    UnitKey,
    WorkerId,
)
from cocoonet._contracts.prediction import RegressionPredictions
from cocoonet._contracts.result import (
    AttemptResult,
    EvaluationOutcome,
    ReportOutcome,
    ReportStatus,
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
