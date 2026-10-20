"""Failure tracing records for internal component contracts."""

from dataclasses import dataclass
from enum import StrEnum

from cocoonet._contracts.identity import (
    AttemptId,
    FailureId,
    FoldId,
    ModelId,
    OperationId,
    ParentId,
    ReportName,
    SampleId,
    ScalarLabel,
    WorkerId,
    validate_identifier,
    validate_label,
)
from cocoonet._type_validation import validate_declared_types


class FailureScope(StrEnum):
    """Identify the boundary affected by a recorded failure.

    Attributes
    ----------
    FIT_PREDICTION : str
        Failure during model reconstruction, fitting or training, prediction,
        or generation of shared validation predictions.
    REPORT : str
        Failure affecting one report.
    MEASURE : str
        Failure affecting one measure within a report.
    SAMPLE_INFLUENCE : str
        Failure affecting one sample-removal influence evaluation.
    DATA_PREPARATION : str
        Failure to prepare or persist dataset or partition artifacts required
        for execution.
    DATA_SYNC : str
        Failure to synchronize datasets or required artifacts between root
        and worker.
    WORKER_MANAGEMENT : str
        Failure of worker allocation, authorization, stopping, disconnection,
        or release.
    RESULT_TRANSFER : str
        Failure to transfer a completed evaluation result from worker to root.
    RESULT_PERSISTENCE : str
        Failure to persist a received evaluation result in root storage.
    WORKER_PROCESS : str
        Failure indicating that the worker process is unavailable because it
        crashed, terminated unexpectedly, or encountered an equivalent
        process-level runtime or infrastructure failure.

    Notes
    -----
    Configuration and startup errors are immediate root-side errors and are
    not represented by this enum. Distinct stages and result states separate
    fatal reconstruction or baseline-fit failures from isolated prediction
    failures. No other scope implies that the worker process is unavailable.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    FIT_PREDICTION = "fit_prediction"
    REPORT = "report"
    MEASURE = "measure"
    SAMPLE_INFLUENCE = "sample_influence"
    DATA_PREPARATION = "data_preparation"
    DATA_SYNC = "data_sync"
    WORKER_MANAGEMENT = "worker_management"
    RESULT_TRANSFER = "result_transfer"
    RESULT_PERSISTENCE = "result_persistence"
    WORKER_PROCESS = "worker_process"


@dataclass(frozen=True, kw_only=True, slots=True)
class TraceContext:
    """Record the precise identities associated with a failure.

    Parameters
    ----------
    stage : str
        Descriptive computational or operational stage where the failure
        occurred.
    parent_id : ParentId | None
        Parent-task identity when known and applicable.
    model_id : ModelId | None
        Model identity when known and applicable.
    attempt_id : AttemptId | None
        Execution-attempt identity when known and applicable.
    worker_id : WorkerId | None
        Worker identity when known and applicable.
    fold_id : FoldId | None
        Fold identity when known and applicable.
    sample_id : SampleId | None
        Sample identity when known and applicable.
    class_label : ScalarLabel | None
        Class identity when the failure is class-specific.
    report_name : ReportName | None
        Report category when the failure affects a report or measure.
    operation_id : OperationId | None
        Operational identity for data, worker-management, transfer, or
        persistence failures.

    Notes
    -----
    Construction preserves candidate values without validating identities or
    contextual dependencies. ``None`` means that a field is not applicable or
    is not known; it does not create an anonymous replacement identity.
    General context dependencies are enforced by ``validate_trace_context``;
    scope-specific requirements belong to ``validate_failure_record``.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    stage: str
    parent_id: ParentId | None = None
    model_id: ModelId | None = None
    attempt_id: AttemptId | None = None
    worker_id: WorkerId | None = None
    fold_id: FoldId | None = None
    sample_id: SampleId | None = None
    class_label: ScalarLabel | None = None
    report_name: ReportName | None = None
    operation_id: OperationId | None = None


def validate_trace_context(value: TraceContext) -> None:  # noqa: C901
    """Validate general identity and dependency rules for failure context.

    Parameters
    ----------
    value : TraceContext
        Failure stage and applicable scientific or operational coordinates to
        validate.

    Raises
    ------
    TypeError
        If ``value`` is not a TraceContext or a field has an incorrect runtime
        type, including an undeclared report name.
    ValueError
        If the stage or a supplied identity is empty, a class label is
        nonfinite, or a model, attempt, fold, or sample coordinate lacks its
        required broader context.

    Notes
    -----
    An attempt requires parent, model, and worker coordinates. A model, fold,
    or sample coordinate requires a parent. Worker, report, and operation
    coordinates have no additional general dependency because their
    scope-specific requirements belong to ``validate_failure_record``.

    This validator does not infer a failure scope, decide whether an optional
    coordinate should have been supplied, capture exceptions, or modify the
    context.

    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Validate the complete declared context structure before semantic checks.
    validate_declared_types(
        value,
        expected_type=TraceContext,
    )

    # Validate the descriptive stage without normalizing its spelling.
    if not value.stage:
        raise ValueError("stage must be a nonempty string.")

    # Validate every supplied opaque identity without inventing absent values.
    optional_identifiers: tuple[tuple[str, str | None], ...] = (
        ("parent_id", value.parent_id),
        ("model_id", value.model_id),
        ("attempt_id", value.attempt_id),
        ("worker_id", value.worker_id),
        ("fold_id", value.fold_id),
        ("sample_id", value.sample_id),
        ("operation_id", value.operation_id),
    )
    for field, identifier in optional_identifiers:
        if identifier is not None:
            validate_identifier(identifier, field=field)

    # Validate an applicable class coordinate without coercing the label.
    if value.class_label is not None:
        validate_label(value.class_label, field="class_label")

    # Validate the complete routing context required by an execution attempt.
    if value.attempt_id is not None:
        if value.parent_id is None:
            raise ValueError("attempt_id requires parent_id.")
        if value.model_id is None:
            raise ValueError("attempt_id requires model_id.")
        if value.worker_id is None:
            raise ValueError("attempt_id requires worker_id.")

    # Validate broader parent context for narrower scientific coordinates.
    if value.model_id is not None and value.parent_id is None:
        raise ValueError("model_id requires parent_id.")
    if value.fold_id is not None and value.parent_id is None:
        raise ValueError("fold_id requires parent_id.")
    if value.sample_id is not None and value.parent_id is None:
        raise ValueError("sample_id requires parent_id.")


@dataclass(frozen=True, kw_only=True, slots=True)
class FailureRecord:
    """Record one traceable failure and its optional originating failure.

    Parameters
    ----------
    failure_id : FailureId
        Identity of this failure record.
    scope : FailureScope
        Boundary affected by the failure.
    context : TraceContext
        Stage and applicable scientific or operational identities.
    exception_type : str
        Qualified or descriptive exception type.
    message : str
        Descriptive failure message.
    traceback : str
        Complete captured traceback, or an empty string when the failure was
        detected without catching an exception.
    origin_failure_id : FailureId | None
        Earlier failure responsible for this dependent failure, when
        applicable.

    Notes
    -----
    Construction preserves supplied diagnostic text and context without
    validating identities, scope-specific requirements, or origin links.
    Origin resolution, context agreement, and cycle prevention belong to the
    planned failure validators. This record does not infer worker-process
    failure or replace unavailable scientific values.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    failure_id: FailureId
    scope: FailureScope
    context: TraceContext
    exception_type: str
    message: str
    traceback: str
    origin_failure_id: FailureId | None = None


def validate_failure_record(value: FailureRecord) -> None:
    """Validate one failure record and its scope-specific context.

    Parameters
    ----------
    value : FailureRecord
        Failure identity, scope, context, diagnostics, and optional origin to
        validate.

    Raises
    ------
    TypeError
        If ``value`` or one of its fields has an incorrect runtime type.
    ValueError
        If an identity or required diagnostic string is empty, the origin is
        self-referential, the context is invalid, or the selected scope lacks
        required context.

    Notes
    -----
    Fit/prediction failures require attempt context. Report and measure
    failures require a report name. Sample-influence failures require attempt,
    fold, sample, and influence-report context. Operational failures require
    an operation identity, and worker-process failures require a worker
    identity.

    Origin resolution, context agreement between linked failures, and cycle
    detection belong to ``validate_failure_links``. This validator preserves
    diagnostic text and permits an empty traceback for a detected failure that
    did not catch an exception.

    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Validate the complete declared record structure before semantic checks.
    validate_declared_types(
        value,
        expected_type=FailureRecord,
    )

    # Validate the failure and optional origin identities independently.
    validate_identifier(value.failure_id, field="failure_id")
    if value.origin_failure_id is not None:
        validate_identifier(value.origin_failure_id, field="origin_failure_id")
        if value.origin_failure_id == value.failure_id:
            raise ValueError("origin_failure_id must not equal failure_id.")

    # Validate general context identities and dependencies before scope rules.
    validate_trace_context(value.context)

    # Preserve diagnostic text while requiring its identifying fields.
    if not value.exception_type:
        raise ValueError("exception_type must be a nonempty string.")
    if not value.message:
        raise ValueError("message must be a nonempty string.")

    # Define the minimum context coordinates required by every failure scope.
    required_context: dict[
        FailureScope,
        tuple[tuple[str, object | None], ...],
    ] = {
        FailureScope.FIT_PREDICTION: (("attempt_id", value.context.attempt_id),),
        FailureScope.REPORT: (("report_name", value.context.report_name),),
        FailureScope.MEASURE: (("report_name", value.context.report_name),),
        FailureScope.SAMPLE_INFLUENCE: (
            ("attempt_id", value.context.attempt_id),
            ("fold_id", value.context.fold_id),
            ("sample_id", value.context.sample_id),
            ("report_name", value.context.report_name),
        ),
        FailureScope.DATA_PREPARATION: (("operation_id", value.context.operation_id),),
        FailureScope.DATA_SYNC: (("operation_id", value.context.operation_id),),
        FailureScope.WORKER_MANAGEMENT: (("operation_id", value.context.operation_id),),
        FailureScope.RESULT_TRANSFER: (("operation_id", value.context.operation_id),),
        FailureScope.RESULT_PERSISTENCE: (("operation_id", value.context.operation_id),),
        FailureScope.WORKER_PROCESS: (("worker_id", value.context.worker_id),),
    }
    for field, context_value in required_context[value.scope]:
        if context_value is None:
            raise ValueError(f"{value.scope.value} failure requires {field}.")

    # Require the designated report category for sample-influence failures.
    if value.scope is FailureScope.SAMPLE_INFLUENCE and value.context.report_name is not ReportName.INFLUENCE:
        raise ValueError("sample_influence failure requires report_name='influence'.")


def validate_failure_links(values: tuple[FailureRecord, ...]) -> None:  # noqa: C901
    """Validate identities and dependency links across failure records.

    Parameters
    ----------
    values : tuple[FailureRecord, ...]
        Complete failure collection whose origin links must resolve internally.

    Raises
    ------
    TypeError
        If ``values`` is not a tuple of FailureRecord instances or a record
        contains an incorrectly typed field.
    ValueError
        If a record is invalid, failure identities are duplicated, an origin
        does not resolve, linked records have conflicting execution context,
        or the origin graph contains a cycle.

    Notes
    -----
    Parent, model, attempt, and worker identities must agree wherever both a
    dependent failure and its origin supply them. An origin may omit narrower
    fold, sample, class, stage, or report context that belongs only to its
    dependent failure.

    Empty collections are valid. Validation uses ID-indexed lookup and
    linear-time origin traversal without recursive graph processing.

    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Validate the declared collection structure before traversing its records.
    validate_declared_types(
        values,
        expected_type=tuple[FailureRecord, ...],
    )

    # Validate individual records and build one unambiguous identity index.
    failures_by_id: dict[FailureId, FailureRecord] = {}
    for failure in values:
        validate_failure_record(failure)
        if failure.failure_id in failures_by_id:
            raise ValueError(f"duplicate failure_id: {failure.failure_id!r}.")
        failures_by_id[failure.failure_id] = failure

    # Resolve every origin and compare the shared execution coordinates.
    for failure in values:
        if failure.origin_failure_id is None:
            continue

        origin = failures_by_id.get(failure.origin_failure_id)
        if origin is None:
            raise ValueError(f"origin_failure_id {failure.origin_failure_id!r} does not resolve.")

        shared_coordinates = (
            (
                "parent_id",
                failure.context.parent_id,
                origin.context.parent_id,
            ),
            (
                "model_id",
                failure.context.model_id,
                origin.context.model_id,
            ),
            (
                "attempt_id",
                failure.context.attempt_id,
                origin.context.attempt_id,
            ),
            (
                "worker_id",
                failure.context.worker_id,
                origin.context.worker_id,
            ),
        )
        for field, dependent_value, origin_value in shared_coordinates:
            if dependent_value is not None and origin_value is not None and dependent_value != origin_value:
                raise ValueError(
                    f"failure {failure.failure_id!r} and origin " f"{origin.failure_id!r} have conflicting {field}."
                )

    # Traverse each single-origin chain and reject the first detected cycle.
    validated_ids: set[FailureId] = set()
    for failure_id in failures_by_id:
        chain_ids: set[FailureId] = set()
        current_id: FailureId | None = failure_id

        while current_id is not None and current_id not in validated_ids:
            if current_id in chain_ids:
                raise ValueError("failure origin links must be acyclic.")

            chain_ids.add(current_id)
            current_id = failures_by_id[current_id].origin_failure_id

        validated_ids.update(chain_ids)
