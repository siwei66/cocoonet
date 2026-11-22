"""Tests for root-side configuration records."""

from dataclasses import FrozenInstanceError, fields
from inspect import Parameter, signature

import pytest

from cocoonet._contracts.execution import ModelFamily, TaskKind
from cocoonet.configuration import ModelRegistration, WorkerKind


def test_worker_kind_has_exact_string_vocabulary() -> None:
    """Expose only the planned root-local and online worker kinds."""
    assert tuple(WorkerKind) == (
        WorkerKind.ROOT_LOCAL,
        WorkerKind.ONLINE,
    )
    assert WorkerKind.ROOT_LOCAL == "root-local"
    assert WorkerKind.ONLINE == "online"


def test_model_registration_preserves_opaque_input_and_metadata() -> None:
    """Retain an opaque input and its explicitly declared metadata."""
    model_input = object()

    registration = ModelRegistration(
        model_input=model_input,
        model_label="baseline",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.REGRESSION,
    )

    assert registration.model_input is model_input
    assert registration.model_label == "baseline"
    assert registration.family is ModelFamily.SKLEARN
    assert registration.task_kind is TaskKind.REGRESSION
    assert tuple(field.name for field in fields(registration)) == (
        "model_input",
        "model_label",
        "family",
        "task_kind",
    )


def test_model_registration_is_frozen_slotted_and_keyword_only() -> None:
    """Keep registration records immutable, compact, and explicit."""
    registration = ModelRegistration(
        model_input=object(),
        model_label="baseline",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.REGRESSION,
    )

    assert not hasattr(registration, "__dict__")
    assert all(
        parameter.kind is Parameter.KEYWORD_ONLY for parameter in signature(ModelRegistration).parameters.values()
    )

    with pytest.raises(FrozenInstanceError):
        registration.__setattr__("model_label", "replacement")
