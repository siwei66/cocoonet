"""Mutable root-side evaluation configuration."""

from dataclasses import dataclass

from cocoonet._contracts.execution import ConsolidationState, ModelReference
from cocoonet._contracts.identity import ModelId, ScalarLabel
from cocoonet._contracts.state import ExecutionStatus


@dataclass(frozen=True, kw_only=True, slots=True)
class _RegisteredModel:
    """Associate fixed model metadata with its opaque prospective input.

    Parameters
    ----------
    reference : ModelReference
        Authoritative model identity, label, family, and task metadata.
    model_input : object
        Opaque in-memory model input or normalized file reference.

    Notes
    -----
    The reference remains the sole stored source of model identity and label.
    Construction does not validate, copy, inspect, or execute the model input.
    Node usage scope: Root - Retains root-side prospective model registration.
    """

    reference: ModelReference
    model_input: object

    @property
    def model_id(self) -> ModelId:
        """Return the authoritative internal model identity."""
        return self.reference.model_id

    @property
    def model_label(self) -> ScalarLabel:
        """Return the authoritative user-facing model label."""
        return self.reference.model_label


def _consolidation_after_evaluation_edit(
    value: ConsolidationState,
    /,
) -> ConsolidationState:
    """Return the consolidation state after an evaluation edit.

    Parameters
    ----------
    value : ConsolidationState
        Current prospective-configuration consolidation state.

    Returns
    -------
    ConsolidationState
        ``UNCONSOLIDATED`` after editing a consolidated configuration;
        otherwise the unchanged current state.

    Notes
    -----
    Worker-only configuration changes do not use this transition.
    Node usage scope: Root - Applies root-side evaluation-edit lifecycle state.
    """
    if value is ConsolidationState.CONSOLIDATED:
        return ConsolidationState.UNCONSOLIDATED
    return value


def _require_configuration_editable(
    value: ExecutionStatus,
    /,
) -> None:
    """Reject configuration mutation during active execution.

    Parameters
    ----------
    value : ExecutionStatus
        Current root-observed execution status.

    Raises
    ------
    RuntimeError
        If execution is currently running.

    Notes
    -----
    ``NOT_STARTED``, ``INTERRUPTED``, and ``SETTLED`` permit prospective
    configuration edits. This guard does not claim that old worker activity
    has stopped after interruption.
    Node usage scope: Root - Guards root-side configuration mutation.
    """
    if value is ExecutionStatus.RUNNING:
        raise RuntimeError("Configuration cannot be changed while execution is running.")
