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
        Failure during initial model reconstruction or primary-fold model
        initialization, fitting/training, or prediction.
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
    not represented by this enum. Initial reconstruction failure ends evaluation
    before primary validation. After reconstruction, a primary-fold model
    initialization, fitting/training, or prediction failure produces
    validation_failed; later independent folds continue when usable.

    Removal-model failures belong to sample_influence. These failures do not
    imply worker-process failure or authorize retries.

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
        Standardized text class label when the failure is class-specific.
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
    """Validate failure coordinates and their general context dependencies.

    Parameters
    ----------
    value : TraceContext
        Candidate stage and applicable scientific or operational coordinates.

    Raises
    ------
    TypeError
        If the record or a field has an incorrect runtime type, including
        a non-string class label or an undeclared report name.
    ValueError
        If the stage or an identity is empty, or a coordinate lacks its
        required parent, model, or worker context.

    Notes
    -----
    An attempt requires parent, model, and worker identities. Model, fold,
    and sample identities require a parent. A class label alone establishes
    no additional general dependency. Scope-specific requirements belong
    to validate_failure_record.

    Identifiers and labels retain their representation. This function does
    not infer missing coordinates, resolve dataset membership, capture
    exceptions, or mutate the record.

    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Validate the context record type before accessing its fields.
    if not isinstance(value, TraceContext):
        validate_declared_types(value, expected_type=TraceContext)

    # Validate the descriptive stage without normalizing its spelling.
    validate_declared_types(value.stage, expected_type=str, field="stage")
    if not value.stage:
        raise ValueError("stage must be a nonempty string; got an empty string.")

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

    # Validate applicable class and report coordinates without coercing their values.
    if value.class_label is not None:
        validate_label(value.class_label, field="class_label")
    if value.report_name is not None:
        validate_declared_types(value.report_name, expected_type=ReportName, field="report_name")

    # Validate the complete routing context required by an execution attempt.
    if value.attempt_id is not None:
        for field, identifier in (
            ("parent_id", value.parent_id),
            ("model_id", value.model_id),
            ("worker_id", value.worker_id),
        ):
            if identifier is None:
                raise ValueError(f"attempt_id {value.attempt_id!r} requires {field}; got None.")

    # Validate broader parent context for narrower scientific coordinates.
    for field, identifier in (
        ("model_id", value.model_id),
        ("fold_id", value.fold_id),
        ("sample_id", value.sample_id),
    ):
        if identifier is not None and value.parent_id is None:
            raise ValueError(f"{field} {identifier!r} requires parent_id; got None.")


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
    validation. validate_failure_record owns record and scope checks;
    validate_failure_links owns origin resolution, shared-context agreement,
    and cycle detection. Context is shared with scientific and operational
    boundaries; it carries no lifecycle behavior. This record neither implies
    worker-process failure nor replaces unavailable scientific values.

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
    """Validate one failure's fields, context, and scope requirements.

    Parameters
    ----------
    value : FailureRecord
        Candidate failure metadata, diagnostics, and optional origin identity.

    Raises
    ------
    TypeError
        If the record or a field has an incorrect runtime type.
    ValueError
        If an identity or required diagnostic string is empty, the origin
        is self-referential, or the context violates general or scope rules.

    Notes
    -----
    Context validity is delegated once to validate_trace_context. The local
    scope table lists required coordinates; it does not retain runtime state.
    Diagnostic text is preserved, including a complete supplied traceback.
    An empty traceback is permitted for a detected failure, but this validator
    cannot determine whether the producer omitted a captured traceback.
    Origin resolution and cycle detection belong to validate_failure_links.
    This function neither captures exceptions nor mutates inputs.

    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Validate the record type, scope, and diagnostic field types before semantic checks.
    if not isinstance(value, FailureRecord):
        validate_declared_types(value, expected_type=FailureRecord)
    validate_declared_types(value.scope, expected_type=FailureScope, field="scope")
    validate_declared_types(value.exception_type, expected_type=str, field="exception_type")
    validate_declared_types(value.message, expected_type=str, field="message")
    validate_declared_types(value.traceback, expected_type=str, field="traceback")

    # Validate the failure and optional origin identities independently.
    validate_identifier(value.failure_id, field="failure_id")
    if value.origin_failure_id is not None:
        validate_identifier(value.origin_failure_id, field="origin_failure_id")
        if value.origin_failure_id == value.failure_id:
            raise ValueError(f"origin_failure_id must differ from failure_id; both are {value.failure_id!r}.")

    # Validate general context identities and dependencies before scope rules.
    try:
        validate_trace_context(value.context)
    except (TypeError, ValueError) as error:
        raise type(error)(f"context is invalid: {error}") from error

    # Preserve diagnostic text while requiring its identifying fields.
    if not value.exception_type:
        raise ValueError("exception_type must be a nonempty string; got an empty string.")
    if not value.message:
        raise ValueError("message must be a nonempty string; got an empty string.")

    # Define the minimum context coordinates required by every failure scope.
    required_context: dict[FailureScope, tuple[tuple[str, object | None], ...]] = {
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
            raise ValueError(
                f"failure {value.failure_id!r} with scope {value.scope.value!r} "
                f"requires context.{field}; got None."
            )

    # Require the designated report category for sample-influence failures.
    if value.scope is FailureScope.SAMPLE_INFLUENCE and value.context.report_name is not ReportName.INFLUENCE:
        raise ValueError(
            f"failure {value.failure_id!r} with scope 'sample_influence' "
            f"requires context.report_name='influence'; got {value.context.report_name!r}."
        )


def validate_failure_links(values: tuple[FailureRecord, ...]) -> None:  # noqa: C901
    """Validate a complete failure collection and its origin relationships.

    Parameters
    ----------
    values : tuple[FailureRecord, ...]
        Candidate records whose origin IDs must resolve within this collection.

    Raises
    ------
    TypeError
        If the collection, a record, or a field has an incorrect runtime type.
    ValueError
        If a record is invalid, failure IDs are duplicated, an origin is
        missing, shared execution identities conflict, or origins form a cycle.

    Notes
    -----
    Each record and its context are validated once in this boundary call.
    Linked parent, model, attempt, and worker identities must agree wherever
    both records supply them. Narrower fold, sample, class, stage, and report
    coordinates may differ; an origin need not carry its dependents' detail.
    Empty collections are valid. The lookup and visited sets are temporary
    validation data. Iterative traversal is linear in the number of records,
    excluding the cost of validating their fields. No input is mutated and
    no failure is copied, repaired, or discarded.

    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Validate the collection's tuple type before traversing its records.
    validate_declared_types(values, expected_type=tuple, field="values")

    # Validate individual records and build one unambiguous identity index.
    failures_by_id: dict[FailureId, FailureRecord] = {}
    for index, failure in enumerate(values):
        try:
            validate_failure_record(failure)
        except (TypeError, ValueError) as error:
            raise type(error)(f"values[{index}] is invalid: {error}") from error
        if failure.failure_id in failures_by_id:
            raise ValueError(f"values[{index}].failure_id must be unique; got duplicate {failure.failure_id!r}.")
        failures_by_id[failure.failure_id] = failure

    # Resolve every origin and compare the shared execution coordinates.
    for failure in values:
        if failure.origin_failure_id is None:
            continue
        origin = failures_by_id.get(failure.origin_failure_id)
        if origin is None:
            raise ValueError(
                f"failure {failure.failure_id!r} requires origin_failure_id to resolve "
                f"within values; got unknown {failure.origin_failure_id!r}."
            )
        shared_coordinates = (
            ("parent_id", failure.context.parent_id, origin.context.parent_id),
            ("model_id", failure.context.model_id, origin.context.model_id),
            ("attempt_id", failure.context.attempt_id, origin.context.attempt_id),
            ("worker_id", failure.context.worker_id, origin.context.worker_id),
        )
        for field, dependent_value, origin_value in shared_coordinates:
            if dependent_value is not None and origin_value is not None and dependent_value != origin_value:
                raise ValueError(
                    f"failure {failure.failure_id!r} and origin {origin.failure_id!r} "
                    f"require matching context.{field}; got {dependent_value!r} and {origin_value!r}."
                )

    # Traverse each single-origin chain and reject the first detected cycle.
    validated_ids: set[FailureId] = set()
    for failure_id in failures_by_id:
        chain_ids: set[FailureId] = set()
        current_id: FailureId | None = failure_id
        while current_id is not None and current_id not in validated_ids:
            if current_id in chain_ids:
                raise ValueError(
                    f"failure {failure_id!r} requires acyclic origin links; "
                    f"encountered {current_id!r} again in its origin chain."
                )
            chain_ids.add(current_id)
            current_id = failures_by_id[current_id].origin_failure_id
        validated_ids.update(chain_ids)
