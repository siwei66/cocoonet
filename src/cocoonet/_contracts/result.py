"""Standardized result envelopes for internal component contracts."""

from enum import StrEnum


class ReportStatus(StrEnum):
    """Identify the outcome state of one requested report.

    Attributes
    ----------
    COMPLETE : str
        Report payload is present without operational failure references.
    FAILED : str
        Report has one or more computational failures. Its payload may preserve
        successful and identified unavailable results or may be absent.
    BLOCKED : str
        Report generation was prevented by an originating prerequisite
        failure referenced by the outcome.
    NOT_REQUESTED : str
        Optional report was not requested and has no payload or failure
        references.

    Notes
    -----
    Any computational failure within a report gives the report ``failed``
    status, including when its payload preserves usable results. There is no
    separate partial-report status.

    This enum records report-level state only. Payload and failure-reference
    relationships are enforced later by ``validate_report_outcome``.

    Formula-defined ``NaN``, mathematically permitted infinity, and
    schema-defined non-applicable ``None`` values do not independently change
    a report's status.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    COMPLETE = "complete"
    FAILED = "failed"
    BLOCKED = "blocked"
    NOT_REQUESTED = "not_requested"
