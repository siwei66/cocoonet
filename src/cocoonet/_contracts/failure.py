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
)


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
