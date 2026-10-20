"""Tests for standardized result-envelope contracts."""

from cocoonet._contracts.result import ReportStatus


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
