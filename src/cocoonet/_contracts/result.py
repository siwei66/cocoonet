"""Standardized result envelopes for internal component contracts."""

from dataclasses import dataclass
from enum import StrEnum

from cocoonet._contracts.identity import FailureId, ReportName


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


@dataclass(frozen=True, kw_only=True, slots=True)
class ReportOutcome[T]:
    """Record one report's generated content, overall status, and failures.

    Parameters
    ----------
    name : ReportName
        Category of report represented by this record.
    status : ReportStatus
        Overall state of the report.
    payload : T | None
        Entire structured scientific content generated for this report,
        excluding its report-level identity, status, and failure references.
        For example, a performance ``ReportOutcome`` is the complete report
        record; its payload is the performance-results structure; and
        accuracy, precision, and recall are individual named values inside
        that payload. ``None`` means that no report content was retained, not
        that an individual named value is unavailable.
    failure_ids : tuple[FailureId, ...]
        Failures that affected or prevented generation of this report.

    Notes
    -----
    ``payload`` is the general field name for the report's generated content;
    it does not mean serialized bytes or an unstructured dictionary. When
    present, it must be an immutable instance of the concrete report-content
    type defined by that report's implementation phase.

    ``status`` describes the report as a whole. It does not replace or change
    the scientific-value and value-level failure rules defined by the report
    content.

    Report content may define finite values, mathematically permitted
    infinity, mathematically undefined ``NaN``, computationally unavailable
    ``None`` with value-level failures, or schema-defined non-applicable
    ``None`` without failures. Their human-readable forms remain numeric
    values, ``Inf``, ``-Inf``, ``NaN``, and ``NA`` as defined by that report.

    A ``payload`` of ``None`` means that no identified report content was
    retained. It is not a scientific value and must not be displayed as a
    value-level ``NA``.

    Construction preserves candidate values without validating relationships
    between the status, report content, and failure references. Those rules
    belong to ``validate_report_outcome``.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    name: ReportName
    status: ReportStatus
    payload: T | None
    failure_ids: tuple[FailureId, ...]
