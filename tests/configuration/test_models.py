"""Tests for prospective model registry records."""

from dataclasses import FrozenInstanceError, fields
from inspect import Parameter, signature

import pytest

from cocoonet._contracts.execution import (
    ModelFamily,
    ModelReference,
    TaskKind,
)
from cocoonet._contracts.identity import ModelId
from cocoonet.evaluation_task import _RegisteredModel


def test_registered_model_exposes_reference_identity_and_label() -> None:
    """Derive model identity and label from the authoritative reference."""
    reference = ModelReference(
        model_id=ModelId("1"),
        model_label="baseline",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.REGRESSION,
    )
    model_input = object()

    registered = _RegisteredModel(
        reference=reference,
        model_input=model_input,
    )

    assert registered.reference is reference
    assert registered.model_input is model_input
    assert registered.model_id == reference.model_id
    assert registered.model_label == reference.model_label
    assert tuple(field.name for field in fields(registered)) == (
        "reference",
        "model_input",
    )


def test_registered_model_is_frozen_slotted_and_keyword_only() -> None:
    """Keep the private model association immutable and explicit."""
    registered = _RegisteredModel(
        reference=ModelReference(
            model_id=ModelId("1"),
            model_label="baseline",
            family=ModelFamily.SKLEARN,
            task_kind=TaskKind.REGRESSION,
        ),
        model_input=object(),
    )

    assert not hasattr(registered, "__dict__")
    assert all(
        parameter.kind is Parameter.KEYWORD_ONLY for parameter in signature(_RegisteredModel).parameters.values()
    )

    with pytest.raises(FrozenInstanceError):
        registered.__setattr__("model_input", object())
