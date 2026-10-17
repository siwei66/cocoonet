"""Failure tracing records for internal component contracts."""

from enum import StrEnum


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
