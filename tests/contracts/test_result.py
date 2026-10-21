"""Tests for standardized result-envelope contracts."""

from dataclasses import FrozenInstanceError, dataclass

import pytest

from cocoonet._contracts.identity import FailureId, ReportName
from cocoonet._contracts.result import ReportOutcome, ReportStatus


@dataclass(frozen=True, slots=True)
class _ExampleReportContent:
    """Provide small immutable report content for declaration tests."""

    value: float


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
