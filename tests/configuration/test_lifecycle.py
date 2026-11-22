"""Tests for evaluation-task lifecycle helpers."""

import pytest

from cocoonet._contracts.execution import ConsolidationState
from cocoonet._contracts.state import ExecutionStatus
from cocoonet.evaluation_task import (
    _consolidation_after_evaluation_edit,
    _require_configuration_editable,
)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (
            ConsolidationState.INITIALIZED,
            ConsolidationState.INITIALIZED,
        ),
        (
            ConsolidationState.CONSOLIDATED,
            ConsolidationState.UNCONSOLIDATED,
        ),
        (
            ConsolidationState.UNCONSOLIDATED,
            ConsolidationState.UNCONSOLIDATED,
        ),
    ],
)
def test_consolidation_after_evaluation_edit(
    before: ConsolidationState,
    after: ConsolidationState,
) -> None:
    """Apply the exact lifecycle transition for evaluation edits."""
    assert _consolidation_after_evaluation_edit(before) is after


def test_require_configuration_editable_rejects_running() -> None:
    """Reject prospective edits while execution is active."""
    with pytest.raises(
        RuntimeError,
        match=r"^Configuration cannot be changed while execution is running\.$",
    ):
        _require_configuration_editable(ExecutionStatus.RUNNING)


@pytest.mark.parametrize(
    "status",
    [
        ExecutionStatus.NOT_STARTED,
        ExecutionStatus.INTERRUPTED,
        ExecutionStatus.SETTLED,
    ],
)
def test_require_configuration_editable_permits_inactive_observations(
    status: ExecutionStatus,
) -> None:
    """Permit prospective edits for every non-running observation."""
    _require_configuration_editable(status)
