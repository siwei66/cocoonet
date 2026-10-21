"""Standardized result envelopes for internal component contracts."""

from dataclasses import dataclass
from enum import StrEnum

from cocoonet._contracts.execution import AttemptInputs
from cocoonet._contracts.failure import FailureRecord
from cocoonet._contracts.identity import FailureId, ReportName
from cocoonet._contracts.prediction import PredictionBatch


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


class EvaluationOutcome(StrEnum):
    """Identify the final scientific outcome of one completed attempt.

    Attributes
    ----------
    COMPLETED : str
        Initial reconstruction, primary validation, and every requested
        report completed without a computational failure.
    COMPLETED_WITH_ERRORS : str
        Initial reconstruction and primary validation succeeded, but at least
        one ordinary or requested influence report failed or was blocked.
    VALIDATION_FAILED : str
        Initial reconstruction succeeded, but model initialization, fitting,
        or prediction failed in at least one primary-validation fold.
    RECONSTRUCTION_FAILED : str
        Initial model reconstruction failed before primary validation began.

    Notes
    -----
    This enum describes a completed attempt result. It does not represent
    execution progress, interruption, or cancellation.

    A validation failure may preserve successful earlier or later fold
    results. A reconstruction failure has no predictions or reports.
    Report-level failures after successful primary validation produce
    ``completed_with_errors`` and do not create another attempt outcome.

    Outcome precedence is ``reconstruction_failed``, then
    ``validation_failed``, then ``completed_with_errors``, then ``completed``.
    The declaration order does not define that precedence. Relationships
    between an outcome and its predictions, reports, and failures are enforced
    later by ``validate_attempt_result``.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    VALIDATION_FAILED = "validation_failed"
    RECONSTRUCTION_FAILED = "reconstruction_failed"


@dataclass(frozen=True, kw_only=True, slots=True)
class AttemptResult[V, P, R, Z, I]:
    """Record the complete scientific result of one execution attempt.

    Parameters
    ----------
    inputs : AttemptInputs
        Attempt, unit-task, and worker identities associated with the result.
    outcome : EvaluationOutcome
        Final scientific outcome of the completed attempt.
    predictions : tuple[PredictionBatch, ...]
        Primary-validation predictions in realized schedule order.
    validation : ReportOutcome[V] | None
        Validation report, or no report record when reconstruction failed.
    performance : ReportOutcome[P] | None
        Performance report, or no report record when reconstruction failed.
    residual : ReportOutcome[R] | None
        Residual-analysis report, or no report record when reconstruction
        failed.
    visualization : ReportOutcome[Z] | None
        Visualization-data report, or no report record when reconstruction
        failed.
    influence : ReportOutcome[I] | None
        Optional influence report, its not-requested outcome, or no report
        record when reconstruction failed.
    failures : tuple[FailureRecord, ...]
        Scientific failures recorded during this execution attempt.

    Notes
    -----
    This record is the complete in-memory result envelope for one attempt. It
    is not serialized file content and does not control result delivery,
    persistence, acceptance, or retries.

    A report field of ``None`` means that no report record exists because
    initial reconstruction failed. It is distinct from a ``ReportOutcome``
    whose payload is ``None`` and from an unavailable named scientific value
    inside a retained payload.

    Generated scientific values remain inside the report payloads. Finite
    values, permitted infinity, formula-defined ``NaN``, and unavailable or
    non-applicable ``None`` retain the meanings and failure-reference rules
    defined by their owning report schemas.

    Construction preserves candidate fields without validating schedule
    order, report names, outcome relationships, failure links, or scientific
    context. Those checks belong to ``validate_attempt_result``.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    inputs: AttemptInputs
    outcome: EvaluationOutcome
    predictions: tuple[PredictionBatch, ...]
    validation: ReportOutcome[V] | None
    performance: ReportOutcome[P] | None
    residual: ReportOutcome[R] | None
    visualization: ReportOutcome[Z] | None
    influence: ReportOutcome[I] | None
    failures: tuple[FailureRecord, ...]
