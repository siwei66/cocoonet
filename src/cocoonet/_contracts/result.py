"""Standardized result envelopes for internal component contracts."""

from dataclasses import dataclass
from enum import StrEnum

from cocoonet._contracts.data import (
    FoldRecord,
    PartitionSchedule,
    validate_partition_schedule,
)
from cocoonet._contracts.execution import (
    AttemptInputs,
    ParentEvaluation,
    TaskKind,
    validate_attempt_inputs,
)
from cocoonet._contracts.failure import (
    FailureRecord,
    FailureScope,
    validate_failure_links,
)
from cocoonet._contracts.identity import (
    FailureId,
    FoldId,
    ReportName,
    SampleId,
    ScalarLabel,
)
from cocoonet._contracts.prediction import (
    ClassificationPredictions,
    PredictionBatch,
    RegressionPredictions,
    validate_classification_predictions,
    validate_regression_predictions,
)
from cocoonet._type_validation import validate_declared_types


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


def validate_report_outcome[T](value: ReportOutcome[T]) -> None:
    """Validate one generic report outcome and its failure relationship.

    Parameters
    ----------
    value : ReportOutcome[T]
        Report identity, status, content, and failure references to validate.

    Raises
    ------
    TypeError
        If the report outcome or one of its declared fields has an incorrect
        runtime type.
    ValueError
        If a failure identity is empty or the status conflicts with the
        presence of report content or failure references.

    Notes
    -----
    A complete report requires content and no failures. Failed and blocked
    reports require at least one failure reference and may retain content.
    A not-requested report has neither content nor failure references.

    This generic validator does not inspect the report payload, resolve failure
    references, reject duplicate references, or decide whether
    ``not_requested`` is permitted for a particular report slot. Concrete
    payload validators and ``validate_attempt_result`` own those checks.

    Validation preserves the report content, including permitted infinity,
    formula-defined ``NaN``, and schema-defined ``None`` values inside it.
    Error diagnostics never render the payload.

    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Validate the complete declared envelope before applying status rules.
    validate_declared_types(
        value,
        expected_type=ReportOutcome,
    )

    error_context = "Invalid report outcome:\n" f"  report: {value.name.value}\n" f"  status: {value.status.value}"

    # Validate every supplied failure identity without exposing unrelated
    # report content in the diagnostic.
    for index, failure_id in enumerate(value.failure_ids):
        if not failure_id:
            raise ValueError(
                f"{error_context}\n"
                f"  field: failure_ids[{index}]\n"
                "  expected: a nonempty failure identity\n"
                "  got: an empty string"
            )

    # Require content and prohibit operational failures for complete reports.
    if value.status is ReportStatus.COMPLETE:
        if value.payload is None:
            raise ValueError(f"{error_context}\n" "  expected: payload to be present\n" "  got: payload is None")
        if value.failure_ids:
            raise ValueError(
                f"{error_context}\n"
                "  expected: failure_ids to be empty\n"
                f"  got: failure_ids count is {len(value.failure_ids)}"
            )

    # Require failure provenance while permitting retained or absent content.
    elif value.status in (ReportStatus.FAILED, ReportStatus.BLOCKED):
        if not value.failure_ids:
            raise ValueError(
                f"{error_context}\n"
                "  expected: failure_ids to contain at least one reference\n"
                "  got: failure_ids is empty"
            )

    # Represent an unrequested optional report without content or failures.
    else:
        if value.payload is not None:
            raise ValueError(f"{error_context}\n" "  expected: payload to be None\n" "  got: payload is present")
        if value.failure_ids:
            raise ValueError(
                f"{error_context}\n"
                "  expected: failure_ids to be empty\n"
                f"  got: failure_ids count is {len(value.failure_ids)}"
            )


type _AttemptReports[V, P, R, Z, I] = tuple[
    ReportOutcome[V],
    ReportOutcome[P],
    ReportOutcome[R],
    ReportOutcome[Z],
    ReportOutcome[I],
]


def _validate_report_slot[T](
    report: ReportOutcome[T] | None,
    *,
    field: str,
    expected_name: ReportName,
    error_context: str,
) -> ReportOutcome[T]:
    """Validate and return one required report slot.

    Parameters
    ----------
    report : ReportOutcome[T] | None
        Candidate report stored in the slot.
    field : str
        Attempt-result field containing the report.
    expected_name : ReportName
        Report identity required by that field.
    error_context : str
        Attempt identity prefix used in diagnostics.

    Returns
    -------
    ReportOutcome[T]
        Validated report with its original payload type preserved.

    Raises
    ------
    TypeError
        If the report or one of its declared fields has an incorrect runtime
        type.
    ValueError
        If the required report is absent or has the wrong report identity.
    """
    # Every report slot exists after initial reconstruction succeeds.
    if report is None:
        raise ValueError(
            f"{error_context}\n"
            f"  field: {field}\n"
            "  expected: a report outcome after successful reconstruction\n"
            "  got: None"
        )

    # Validate the generic report contract without interpreting its payload.
    validate_report_outcome(report)

    # Bind the report identity to its fixed result-envelope slot.
    if report.name is not expected_name:
        raise ValueError(
            f"{error_context}\n"
            f"  field: {field}.name\n"
            f"  expected: {expected_name.value}\n"
            f"  got: {report.name.value}"
        )

    return report


def _validate_failure_fold(
    failure: FailureRecord,
    *,
    failure_path: str,
    folds_by_id: dict[FoldId, FoldRecord],
    error_context: str,
) -> FoldRecord | None:
    """Resolve one failure's applicable fold against the saved schedule."""
    context = failure.context

    # Every post-reconstruction fit/prediction failure belongs to one fold.
    if failure.scope is FailureScope.FIT_PREDICTION and context.stage != "reconstruction" and context.fold_id is None:
        raise ValueError(
            f"{error_context}\n"
            f"  field: {failure_path}.context.fold_id\n"
            "  expected: a fold identity for primary-validation failure\n"
            "  got: None"
        )

    # Absence remains valid for task-wide and report-level failures.
    if context.fold_id is None:
        return None

    fold = folds_by_id.get(context.fold_id)
    if fold is None:
        raise ValueError(
            f"{error_context}\n"
            f"  field: {failure_path}.context.fold_id\n"
            "  expected: a fold in the supplied schedule\n"
            f"  got: {context.fold_id!r}"
        )
    return fold


def _validate_failure_sample(
    failure: FailureRecord,
    *,
    failure_path: str,
    fold: FoldRecord | None,
    schedule_sample_ids: set[SampleId],
    error_context: str,
) -> None:
    """Validate one failure's optional sample and population membership."""
    sample_id = failure.context.sample_id
    if sample_id is None:
        return

    # Every supplied sample must belong to the fixed dataset representation.
    if sample_id not in schedule_sample_ids:
        raise ValueError(
            f"{error_context}\n"
            f"  field: {failure_path}.context.sample_id\n"
            "  expected: a sample in the supplied schedule\n"
            f"  got: {sample_id!r}"
        )

    # Influence removes only samples belonging to the fold's training set.
    if failure.scope is FailureScope.SAMPLE_INFLUENCE:
        if fold is None or sample_id not in fold.training_sample_ids:
            raise ValueError(
                f"{error_context}\n"
                f"  field: {failure_path}.context.sample_id\n"
                "  expected: a training member of the referenced influence fold\n"
                f"  got: {sample_id!r}"
            )

    # A sample-specific primary prediction error identifies a testing sample.
    elif (
        failure.scope is FailureScope.FIT_PREDICTION
        and failure.context.stage == "prediction"
        and (fold is None or sample_id not in fold.testing_sample_ids)
    ):
        raise ValueError(
            f"{error_context}\n"
            f"  field: {failure_path}.context.sample_id\n"
            "  expected: a testing member of the referenced prediction fold\n"
            f"  got: {sample_id!r}"
        )


def _validate_failure_class(
    failure: FailureRecord,
    *,
    failure_path: str,
    parent: ParentEvaluation,
    error_context: str,
) -> None:
    """Validate one failure's optional class against the parent vocabulary."""
    class_label = failure.context.class_label
    if class_label is None:
        return

    # Regression results have no class-specific failure coordinates.
    if parent.evaluation.task_kind is TaskKind.REGRESSION:
        raise ValueError(
            f"{error_context}\n"
            f"  field: {failure_path}.context.class_label\n"
            "  expected: no class label for regression\n"
            f"  got: {class_label!r}"
        )

    # Classification coordinates use the exact parent-defined vocabulary.
    if class_label not in parent.evaluation.class_labels:
        raise ValueError(
            f"{error_context}\n"
            f"  field: {failure_path}.context.class_label\n"
            "  expected: a class in the parent vocabulary\n"
            f"  got: {class_label!r}"
        )


def _validate_attempt_failure_context(
    failure: FailureRecord,
    *,
    failure_index: int,
    inputs: AttemptInputs,
    parent: ParentEvaluation,
    folds_by_id: dict[FoldId, FoldRecord],
    schedule_sample_ids: set[SampleId],
    error_context: str,
) -> None:
    """Validate one failure's scope and attempt-specific coordinates."""
    failure_path = f"failures[{failure_index}]"
    context = failure.context

    # The result envelope contains scientific failures only.
    allowed_scopes = {
        FailureScope.FIT_PREDICTION,
        FailureScope.REPORT,
        FailureScope.MEASURE,
        FailureScope.SAMPLE_INFLUENCE,
    }
    if failure.scope not in allowed_scopes:
        raise ValueError(
            f"{error_context}\n"
            f"  field: {failure_path}.scope\n"
            "  expected: a scientific attempt-result failure scope\n"
            f"  got: {failure.scope.value}"
        )

    # Every supplied execution coordinate must identify this attempt.
    expected_coordinates = (
        ("parent_id", context.parent_id, inputs.unit.parent_id),
        ("model_id", context.model_id, inputs.unit.model_id),
        ("attempt_id", context.attempt_id, inputs.attempt_id),
        ("worker_id", context.worker_id, inputs.worker_id),
    )
    for field, actual, expected in expected_coordinates:
        if actual is not None and actual != expected:
            raise ValueError(
                f"{error_context}\n"
                f"  field: {failure_path}.context.{field}\n"
                f"  expected: {expected!r}\n"
                f"  got: {actual!r}"
            )

    # Resolve the narrower scientific coordinates independently.
    fold = _validate_failure_fold(
        failure,
        failure_path=failure_path,
        folds_by_id=folds_by_id,
        error_context=error_context,
    )
    _validate_failure_sample(
        failure,
        failure_path=failure_path,
        fold=fold,
        schedule_sample_ids=schedule_sample_ids,
        error_context=error_context,
    )
    _validate_failure_class(
        failure,
        failure_path=failure_path,
        parent=parent,
        error_context=error_context,
    )


def _validate_attempt_failures[V, P, R, Z, I](
    value: AttemptResult[V, P, R, Z, I],
    *,
    parent: ParentEvaluation,
    schedule: PartitionSchedule,
    error_context: str,
) -> dict[FailureId, FailureRecord]:
    """Validate all failure records and return their identity lookup."""
    # Validate individual records, IDs, origin links, and origin cycles.
    validate_failure_links(value.failures)

    # Reuse schedule lookups across every failure record.
    folds_by_id = {fold.fold_id: fold for fold in schedule.folds}
    schedule_sample_ids = {sample.sample_id for sample in schedule.samples}

    # Apply the result-envelope context requirements to every failure.
    for failure_index, failure in enumerate(value.failures):
        _validate_attempt_failure_context(
            failure,
            failure_index=failure_index,
            inputs=value.inputs,
            parent=parent,
            folds_by_id=folds_by_id,
            schedule_sample_ids=schedule_sample_ids,
            error_context=error_context,
        )

    return {failure.failure_id: failure for failure in value.failures}


def _validate_reconstruction_outcome[V, P, R, Z, I](
    value: AttemptResult[V, P, R, Z, I],
    *,
    error_context: str,
) -> bool:
    """Validate reconstruction precedence and its failure-only envelope."""
    reconstruction_failures = tuple(
        failure
        for failure in value.failures
        if (failure.scope is FailureScope.FIT_PREDICTION and failure.context.stage == "reconstruction")
    )

    # A recorded reconstruction failure determines the attempt outcome.
    if value.outcome is not EvaluationOutcome.RECONSTRUCTION_FAILED:
        if reconstruction_failures:
            raise ValueError(
                f"{error_context}\n"
                "  field: outcome\n"
                "  expected: reconstruction_failed because reconstruction failed\n"
                f"  got: {value.outcome.value}"
            )
        return False

    # The failure-only result requires explicit reconstruction provenance.
    if not reconstruction_failures:
        raise ValueError(
            f"{error_context}\n"
            "  field: failures\n"
            "  expected: at least one fit_prediction failure at the reconstruction stage\n"
            "  got: no reconstruction failure"
        )
    if len(reconstruction_failures) != len(value.failures):
        raise ValueError(
            f"{error_context}\n"
            "  field: failures\n"
            "  expected: only reconstruction-stage fit_prediction failures\n"
            "  got: failures from another scientific stage or scope"
        )

    # Reconstruction failure occurs before predictions or reports exist.
    if value.predictions:
        raise ValueError(
            f"{error_context}\n"
            "  field: predictions\n"
            "  expected: an empty tuple after reconstruction failure\n"
            f"  got: {len(value.predictions)} prediction batches"
        )

    report_fields = (
        ("validation", value.validation),
        ("performance", value.performance),
        ("residual", value.residual),
        ("visualization", value.visualization),
        ("influence", value.influence),
    )
    for field, report in report_fields:
        if report is not None:
            raise ValueError(
                f"{error_context}\n"
                f"  field: {field}\n"
                "  expected: None after reconstruction failure\n"
                "  got: a report outcome"
            )

    return True


def _resolve_attempt_failure_reference(
    failure_id: FailureId,
    *,
    path: str,
    failures_by_id: dict[FailureId, FailureRecord],
    referenced_failure_ids: set[FailureId],
    error_context: str,
    required_scope: FailureScope | None = None,
    fold_id: str | None = None,
    sample_id: str | None = None,
    class_label: ScalarLabel | None = None,
) -> FailureRecord:
    """Resolve one failure reference and validate supplied value coordinates."""
    failure = failures_by_id.get(failure_id)
    if failure is None:
        raise ValueError(
            f"{error_context}\n"
            f"  field: {path}\n"
            "  expected: a failure_id present in failures\n"
            f"  got: unresolved failure_id {failure_id!r}"
        )

    # Prediction unavailability must originate in fit/prediction work.
    if required_scope is not None and failure.scope is not required_scope:
        raise ValueError(
            f"{error_context}\n"
            f"  field: {path}\n"
            f"  expected: a {required_scope.value} failure\n"
            f"  got: a {failure.scope.value} failure"
        )

    # A shared fold-level failure may omit sample and class coordinates.
    expected_coordinates = (
        ("fold", fold_id, failure.context.fold_id),
        ("sample", sample_id, failure.context.sample_id),
        ("class", class_label, failure.context.class_label),
    )
    for coordinate, expected, actual in expected_coordinates:
        if expected is not None and actual is not None and actual != expected:
            raise ValueError(
                f"{error_context}\n"
                f"  field: {path}\n"
                f"  expected: failure context for {coordinate} {expected!r}\n"
                f"  got: failure context for {coordinate} {actual!r}"
            )

    referenced_failure_ids.add(failure_id)
    return failure


def _validate_report_failure_ownership[V, P, R, Z, I](
    failures: tuple[FailureRecord, ...],
    *,
    reports: _AttemptReports[V, P, R, Z, I],
    error_context: str,
) -> None:
    """Require each report computation failure to fail its owning report."""
    validation, performance, residual, visualization, influence = reports
    status_by_name = {
        ReportName.VALIDATION: validation.status,
        ReportName.PERFORMANCE: performance.status,
        ReportName.RESIDUAL: residual.status,
        ReportName.VISUALIZATION: visualization.status,
        ReportName.INFLUENCE: influence.status,
    }

    # Report and measure scopes identify the report where failure occurred.
    for failure_index, failure in enumerate(failures):
        if failure.scope not in (
            FailureScope.REPORT,
            FailureScope.MEASURE,
            FailureScope.SAMPLE_INFLUENCE,
        ):
            continue

        report_name = failure.context.report_name
        if report_name is None:
            continue

        owner_status = status_by_name[report_name]
        if owner_status is not ReportStatus.FAILED:
            raise ValueError(
                f"{error_context}\n"
                f"  field: failures[{failure_index}].context.report_name\n"
                f"  expected: report {report_name.value!r} to have failed status\n"
                f"  got: {owner_status.value}"
            )


def _validate_attempt_reports[V, P, R, Z, I](
    value: AttemptResult[V, P, R, Z, I],
    *,
    parent: ParentEvaluation,
    failures_by_id: dict[FailureId, FailureRecord],
    referenced_failure_ids: set[FailureId],
    error_context: str,
) -> _AttemptReports[V, P, R, Z, I]:
    """Validate report slots, names, request state, and failure references."""
    # Validate every required report while preserving its payload type.
    validation = _validate_report_slot(
        value.validation,
        field="validation",
        expected_name=ReportName.VALIDATION,
        error_context=error_context,
    )
    performance = _validate_report_slot(
        value.performance,
        field="performance",
        expected_name=ReportName.PERFORMANCE,
        error_context=error_context,
    )
    residual = _validate_report_slot(
        value.residual,
        field="residual",
        expected_name=ReportName.RESIDUAL,
        error_context=error_context,
    )
    visualization = _validate_report_slot(
        value.visualization,
        field="visualization",
        expected_name=ReportName.VISUALIZATION,
        error_context=error_context,
    )
    influence = _validate_report_slot(
        value.influence,
        field="influence",
        expected_name=ReportName.INFLUENCE,
        error_context=error_context,
    )

    reports: _AttemptReports[V, P, R, Z, I] = (
        validation,
        performance,
        residual,
        visualization,
        influence,
    )

    # Mandatory reports cannot use the optional not-requested state.
    mandatory_statuses = (
        ("validation", validation.status),
        ("performance", performance.status),
        ("residual", residual.status),
        ("visualization", visualization.status),
    )
    for field, status in mandatory_statuses:
        if status is ReportStatus.NOT_REQUESTED:
            raise ValueError(
                f"{error_context}\n"
                f"  field: {field}.status\n"
                "  expected: complete, failed, or blocked\n"
                "  got: not_requested"
            )

    # Match influence state to the fixed parent-level request.
    if parent.evaluation.influence_requested and influence.status is ReportStatus.NOT_REQUESTED:
        raise ValueError(
            f"{error_context}\n"
            "  field: influence.status\n"
            "  expected: an outcome for requested influence analysis\n"
            "  got: not_requested"
        )
    if not parent.evaluation.influence_requested and influence.status is not ReportStatus.NOT_REQUESTED:
        raise ValueError(
            f"{error_context}\n"
            "  field: influence.status\n"
            "  expected: not_requested because influence was not requested\n"
            f"  got: {influence.status.value}"
        )

    # Resolve report references without changing their originating context.
    report_failure_ids = (
        ("validation", validation.failure_ids),
        ("performance", performance.failure_ids),
        ("residual", residual.failure_ids),
        ("visualization", visualization.failure_ids),
        ("influence", influence.failure_ids),
    )
    for field, failure_ids in report_failure_ids:
        for failure_index, failure_id in enumerate(failure_ids):
            _resolve_attempt_failure_reference(
                failure_id,
                path=f"{field}.failure_ids[{failure_index}]",
                failures_by_id=failures_by_id,
                referenced_failure_ids=referenced_failure_ids,
                error_context=error_context,
            )

    # A failure occurring inside a report marks that owning report failed.
    _validate_report_failure_ownership(
        value.failures,
        reports=reports,
        error_context=error_context,
    )

    return reports


def _validate_regression_prediction_batch(
    prediction: RegressionPredictions,
    *,
    fold: FoldRecord,
    fold_index: int,
    failures_by_id: dict[FailureId, FailureRecord],
    referenced_failure_ids: set[FailureId],
    error_context: str,
) -> bool:
    """Validate one regression batch and resolve unavailable-value failures."""
    validate_regression_predictions(prediction, fold=fold)
    unavailable = False

    # Resolve every value-level failure against its exact fold and sample.
    for sample_index, (prediction_value, failure_ids) in enumerate(
        zip(prediction.values, prediction.failure_ids, strict=True)
    ):
        unavailable |= prediction_value is None
        for failure_index, failure_id in enumerate(failure_ids):
            _resolve_attempt_failure_reference(
                failure_id,
                path=f"predictions[{fold_index}].failure_ids[{sample_index}][{failure_index}]",
                failures_by_id=failures_by_id,
                referenced_failure_ids=referenced_failure_ids,
                error_context=error_context,
                required_scope=FailureScope.FIT_PREDICTION,
                fold_id=prediction.fold_id,
                sample_id=prediction.sample_ids[sample_index],
            )

    return unavailable


def _validate_classification_prediction_batch(
    prediction: ClassificationPredictions,
    *,
    fold: FoldRecord,
    fold_index: int,
    class_labels: tuple[ScalarLabel, ...],
    failures_by_id: dict[FailureId, FailureRecord],
    referenced_failure_ids: set[FailureId],
    error_context: str,
) -> bool:
    """Validate one classification batch and resolve value-level failures."""
    validate_classification_predictions(
        prediction,
        fold=fold,
        class_labels=class_labels,
    )
    unavailable = False

    # Resolve predicted-label failures against their fold and sample.
    for sample_index, failure_ids in enumerate(prediction.label_failure_ids):
        unavailable |= prediction.predicted_labels[sample_index] is None
        for failure_index, failure_id in enumerate(failure_ids):
            _resolve_attempt_failure_reference(
                failure_id,
                path=f"predictions[{fold_index}].label_failure_ids[{sample_index}][{failure_index}]",
                failures_by_id=failures_by_id,
                referenced_failure_ids=referenced_failure_ids,
                error_context=error_context,
                required_scope=FailureScope.FIT_PREDICTION,
                fold_id=prediction.fold_id,
                sample_id=prediction.sample_ids[sample_index],
            )

    # Resolve probability failures against fold, sample, and class column.
    for sample_index, failure_row in enumerate(prediction.probability_failure_ids):
        for class_index, failure_ids in enumerate(failure_row):
            unavailable |= prediction.probabilities[sample_index][class_index] is None
            for failure_index, failure_id in enumerate(failure_ids):
                _resolve_attempt_failure_reference(
                    failure_id,
                    path=(
                        f"predictions[{fold_index}].probability_failure_ids"
                        f"[{sample_index}][{class_index}][{failure_index}]"
                    ),
                    failures_by_id=failures_by_id,
                    referenced_failure_ids=referenced_failure_ids,
                    error_context=error_context,
                    required_scope=FailureScope.FIT_PREDICTION,
                    fold_id=prediction.fold_id,
                    sample_id=prediction.sample_ids[sample_index],
                    class_label=prediction.class_labels[class_index],
                )

    return unavailable


def _validate_attempt_predictions[V, P, R, Z, I](
    value: AttemptResult[V, P, R, Z, I],
    *,
    parent: ParentEvaluation,
    schedule: PartitionSchedule,
    failures_by_id: dict[FailureId, FailureRecord],
    referenced_failure_ids: set[FailureId],
    error_context: str,
) -> bool:
    """Validate prediction type, fold order, values, and failure references."""
    # Require one prediction batch for each authoritative schedule fold.
    if len(value.predictions) != len(schedule.folds):
        raise ValueError(
            f"{error_context}\n"
            "  field: predictions\n"
            f"  expected: {len(schedule.folds)} batches in schedule order\n"
            f"  got: {len(value.predictions)} batches"
        )

    unavailable = False

    # Validate each batch using the parent task kind and matching saved fold.
    for fold_index, (prediction, fold) in enumerate(zip(value.predictions, schedule.folds, strict=True)):
        if parent.evaluation.task_kind is TaskKind.REGRESSION:
            if not isinstance(prediction, RegressionPredictions):
                raise ValueError(
                    f"{error_context}\n"
                    f"  field: predictions[{fold_index}]\n"
                    "  expected: RegressionPredictions\n"
                    f"  got: {type(prediction).__name__}"
                )
            unavailable |= _validate_regression_prediction_batch(
                prediction,
                fold=fold,
                fold_index=fold_index,
                failures_by_id=failures_by_id,
                referenced_failure_ids=referenced_failure_ids,
                error_context=error_context,
            )
        else:
            if not isinstance(prediction, ClassificationPredictions):
                raise ValueError(
                    f"{error_context}\n"
                    f"  field: predictions[{fold_index}]\n"
                    "  expected: ClassificationPredictions\n"
                    f"  got: {type(prediction).__name__}"
                )
            unavailable |= _validate_classification_prediction_batch(
                prediction,
                fold=fold,
                fold_index=fold_index,
                class_labels=parent.evaluation.class_labels,
                failures_by_id=failures_by_id,
                referenced_failure_ids=referenced_failure_ids,
                error_context=error_context,
            )

    return unavailable


def _validate_failure_coverage(
    *,
    failures_by_id: dict[FailureId, FailureRecord],
    referenced_failure_ids: set[FailureId],
    error_context: str,
) -> None:
    """Require every non-reconstruction failure to support retained results."""
    reachable_ids = set(referenced_failure_ids)
    pending_ids = list(referenced_failure_ids)

    # Preserve origin-only records by following every referenced origin chain.
    while pending_ids:
        failure_id = pending_ids.pop()
        origin_id = failures_by_id[failure_id].origin_failure_id
        if origin_id is not None and origin_id not in reachable_ids:
            reachable_ids.add(origin_id)
            pending_ids.append(origin_id)

    # Reject unrelated records that cannot be reached from any result value.
    for failure_id in failures_by_id:
        if failure_id not in reachable_ids:
            raise ValueError(
                f"{error_context}\n"
                "  field: failures\n"
                "  expected: every failure to be referenced by a report or prediction, "
                "or be an origin of a referenced failure\n"
                f"  got: unreferenced failure_id {failure_id!r}"
            )


def _validate_primary_failure_outcome[V, P, R, Z, I](
    value: AttemptResult[V, P, R, Z, I],
    *,
    parent: ParentEvaluation,
    reports: _AttemptReports[V, P, R, Z, I],
    primary_validation_failed: bool,
    error_context: str,
) -> bool:
    """Validate attempt and report states after primary-validation failure."""
    if not primary_validation_failed:
        return False

    validation, performance, residual, visualization, influence = reports

    # Primary failure has precedence over all report-level outcomes.
    if value.outcome is not EvaluationOutcome.VALIDATION_FAILED:
        raise ValueError(
            f"{error_context}\n"
            "  field: outcome\n"
            "  expected: validation_failed because primary validation contains a failure\n"
            f"  got: {value.outcome.value}"
        )

    # The validation report retains its available fold-level content.
    if validation.status is not ReportStatus.FAILED:
        raise ValueError(
            f"{error_context}\n"
            "  field: validation.status\n"
            "  expected: failed after primary-validation failure\n"
            f"  got: {validation.status.value}"
        )
    if validation.payload is None:
        raise ValueError(
            f"{error_context}\n"
            "  field: validation.payload\n"
            "  expected: retained available fold content\n"
            "  got: None"
        )

    # Dependent reports may fail or be blocked, with or without retained form.
    for field, report in (
        ("performance", performance),
        ("residual", residual),
        ("visualization", visualization),
    ):
        if report.status not in (ReportStatus.FAILED, ReportStatus.BLOCKED):
            raise ValueError(
                f"{error_context}\n"
                f"  field: {field}.status\n"
                "  expected: failed or blocked after primary-validation failure\n"
                f"  got: {report.status.value}"
            )

    # Requested influence cannot start without complete baseline predictions.
    expected_influence_status = (
        ReportStatus.BLOCKED if parent.evaluation.influence_requested else ReportStatus.NOT_REQUESTED
    )
    if influence.status is not expected_influence_status:
        raise ValueError(
            f"{error_context}\n"
            "  field: influence.status\n"
            f"  expected: {expected_influence_status.value} after primary-validation failure\n"
            f"  got: {influence.status.value}"
        )

    return True


def _validate_successful_primary_outcome[V, P, R, Z, I](
    value: AttemptResult[V, P, R, Z, I],
    *,
    parent: ParentEvaluation,
    reports: _AttemptReports[V, P, R, Z, I],
    error_context: str,
) -> None:
    """Derive final outcome after completely successful primary validation."""
    validation, performance, residual, visualization, influence = reports

    # Successful primary predictions require a complete validation report.
    if validation.status is not ReportStatus.COMPLETE:
        raise ValueError(
            f"{error_context}\n"
            "  field: validation.status\n"
            "  expected: complete after successful primary validation\n"
            f"  got: {validation.status.value}"
        )

    # Only requested report branches contribute to downstream-error state.
    downstream_statuses: tuple[ReportStatus, ...] = (
        performance.status,
        residual.status,
        visualization.status,
    )
    if parent.evaluation.influence_requested:
        downstream_statuses += (influence.status,)

    downstream_error = any(status in (ReportStatus.FAILED, ReportStatus.BLOCKED) for status in downstream_statuses)
    expected_outcome = EvaluationOutcome.COMPLETED_WITH_ERRORS if downstream_error else EvaluationOutcome.COMPLETED

    if value.outcome is not expected_outcome:
        raise ValueError(
            f"{error_context}\n"
            "  field: outcome\n"
            f"  expected: {expected_outcome.value}\n"
            f"  got: {value.outcome.value}"
        )


def validate_attempt_result[V, P, R, Z, I](
    value: AttemptResult[V, P, R, Z, I],
    *,
    parent: ParentEvaluation,
    schedule: PartitionSchedule,
) -> None:
    """Validate one complete attempt result against its parent and schedule.

    Parameters
    ----------
    value : AttemptResult[V, P, R, Z, I]
        Complete scientific result envelope to validate.
    parent : ParentEvaluation
        Fixed parent task supplying the unit, task kind, class vocabulary,
        and partition association.
    schedule : PartitionSchedule
        Authoritative realized folds and sample memberships used by the
        attempt.

    Raises
    ------
    TypeError
        If the result, parent, schedule, predictions, reports, failures, or
        their nested fields have incorrect runtime types.
    ValueError
        If identities or associations conflict, predictions do not match the
        task or schedule, failure provenance does not resolve, report and
        failure states conflict, or the attempt outcome is inconsistent.

    Notes
    -----
    Initial reconstruction failure is the only result form without predictions
    or report records. After successful reconstruction, one prediction batch
    corresponds to every saved fold in schedule order and every report slot
    contains a report outcome.

    Any primary-fold initialization, fitting, or prediction failure produces
    ``validation_failed``. Later independent folds remain represented.
    Report-only or sample-influence failures after successful primary
    validation produce ``completed_with_errors``.

    This validator resolves envelope relationships only. It does not inspect
    concrete report payloads, execute models, regenerate folds, retry work,
    persist results, or render reports.

    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Validate the outer result type without expanding unresolved payload
    # variables through nested generic report fields.
    if not isinstance(value, AttemptResult):
        validate_declared_types(
            value,
            expected_type=AttemptResult,
        )

    # Validate concrete envelope fields whose types do not depend on report
    # payload parameters. Report payloads remain owned by report validators.
    validate_declared_types(
        value.outcome,
        expected_type=EvaluationOutcome,
        field="value.outcome",
    )
    validate_declared_types(
        value.predictions,
        expected_type=tuple[PredictionBatch, ...],
        field="value.predictions",
    )
    for field, report in (
        ("validation", value.validation),
        ("performance", value.performance),
        ("residual", value.residual),
        ("visualization", value.visualization),
        ("influence", value.influence),
    ):
        validate_declared_types(
            report,
            expected_type=ReportOutcome | None,
            field=f"value.{field}",
        )

    # Validate the authoritative schedule and attempt routing context.
    validate_partition_schedule(schedule)
    validate_attempt_inputs(value.inputs, parent=parent)

    error_context = (
        "Invalid attempt result:\n"
        f"  parent_id: {value.inputs.unit.parent_id}\n"
        f"  model_id: {value.inputs.unit.model_id}\n"
        f"  attempt_id: {value.inputs.attempt_id}"
    )

    # Bind the schedule to the exact partition fixed by the parent.
    if parent.partition != schedule.reference:
        raise ValueError(
            f"{error_context}\n"
            "  field: schedule.reference\n"
            f"  expected: {parent.partition!r}\n"
            f"  got: {schedule.reference!r}"
        )

    # Validate all scientific failure records before resolving their use.
    failures_by_id = _validate_attempt_failures(
        value,
        parent=parent,
        schedule=schedule,
        error_context=error_context,
    )

    # Reconstruction failure is a complete failure-only result.
    if _validate_reconstruction_outcome(value, error_context=error_context):
        return

    referenced_failure_ids: set[FailureId] = set()

    # Validate report slots and their report-level failure references.
    reports = _validate_attempt_reports(
        value,
        parent=parent,
        failures_by_id=failures_by_id,
        referenced_failure_ids=referenced_failure_ids,
        error_context=error_context,
    )

    # Validate fold predictions and their value-level failure references.
    unavailable_prediction = _validate_attempt_predictions(
        value,
        parent=parent,
        schedule=schedule,
        failures_by_id=failures_by_id,
        referenced_failure_ids=referenced_failure_ids,
        error_context=error_context,
    )

    # Every retained failure must support a result or its provenance chain.
    _validate_failure_coverage(
        failures_by_id=failures_by_id,
        referenced_failure_ids=referenced_failure_ids,
        error_context=error_context,
    )

    # Any post-reconstruction fit/prediction failure affects validation.
    primary_validation_failed = unavailable_prediction or any(
        failure.scope is FailureScope.FIT_PREDICTION for failure in value.failures
    )
    if _validate_primary_failure_outcome(
        value,
        parent=parent,
        reports=reports,
        primary_validation_failed=primary_validation_failed,
        error_context=error_context,
    ):
        return

    # Otherwise derive success or downstream-error completion from reports.
    _validate_successful_primary_outcome(
        value,
        parent=parent,
        reports=reports,
        error_context=error_context,
    )
