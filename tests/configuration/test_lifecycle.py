"""Tests for evaluation-task lifecycle helpers."""

import pytest

from cocoonet._contracts.execution import ConsolidationState
from cocoonet.evaluation_task import _consolidation_after_evaluation_edit


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
