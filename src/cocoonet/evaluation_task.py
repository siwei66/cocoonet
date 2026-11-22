"""Mutable root-side evaluation configuration."""

from dataclasses import dataclass

from cocoonet._contracts.execution import ModelReference
from cocoonet._contracts.identity import ModelId, ScalarLabel


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
