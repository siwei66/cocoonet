"""Tests for traceable failure contracts."""

from cocoonet._contracts.failure import FailureScope


def test_failure_scope_has_exact_string_vocabulary() -> None:
    """Expose only the planned failure-scope values."""
    assert tuple(FailureScope) == (
        FailureScope.FIT_PREDICTION,
        FailureScope.REPORT,
        FailureScope.MEASURE,
        FailureScope.SAMPLE_INFLUENCE,
        FailureScope.DATA_PREPARATION,
        FailureScope.DATA_SYNC,
        FailureScope.WORKER_MANAGEMENT,
        FailureScope.RESULT_TRANSFER,
        FailureScope.RESULT_PERSISTENCE,
        FailureScope.WORKER_PROCESS,
    )
    assert FailureScope.FIT_PREDICTION == "fit_prediction"
    assert FailureScope.REPORT == "report"
    assert FailureScope.MEASURE == "measure"
    assert FailureScope.SAMPLE_INFLUENCE == "sample_influence"
    assert FailureScope.DATA_PREPARATION == "data_preparation"
    assert FailureScope.DATA_SYNC == "data_sync"
    assert FailureScope.WORKER_MANAGEMENT == "worker_management"
    assert FailureScope.RESULT_TRANSFER == "result_transfer"
    assert FailureScope.RESULT_PERSISTENCE == "result_persistence"
    assert FailureScope.WORKER_PROCESS == "worker_process"
