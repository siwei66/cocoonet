"""Tests for root-side configuration records."""

from dataclasses import FrozenInstanceError, fields, replace
from inspect import Parameter, signature
from typing import Any

import pytest

from cocoonet._contracts.execution import ModelFamily, TaskKind
from cocoonet._contracts.identity import WorkerId
from cocoonet.configuration import (
    ModelRegistration,
    WorkerConfiguration,
    WorkerKind,
    validate_model_registration,
)


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


def test_validate_model_registration_accepts_opaque_input() -> None:
    """Accept a non-None input without changing or inspecting it."""
    model_input = object()
    registration = ModelRegistration(
        model_input=model_input,
        model_label="baseline",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.REGRESSION,
    )

    validate_model_registration(registration)

    assert registration.model_input is model_input


def test_validate_model_registration_rejects_none_input() -> None:
    """Reject the sole disallowed opaque model-input value."""
    registration = ModelRegistration(
        model_input=None,
        model_label="baseline",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.REGRESSION,
    )

    with pytest.raises(ValueError, match=r"^model_input must not be None\.$"):
        validate_model_registration(registration)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("model_label", 1),
        ("family", "sklearn"),
        ("task_kind", "regression"),
    ],
)
def test_validate_model_registration_rejects_invalid_metadata(
    field: str,
    invalid: Any,
) -> None:
    """Reject metadata outside its declared runtime type."""
    registration = ModelRegistration(
        model_input=object(),
        model_label="baseline",
        family=ModelFamily.SKLEARN,
        task_kind=TaskKind.REGRESSION,
    )
    invalid_registration = replace(registration, **{field: invalid})

    with pytest.raises(TypeError, match=field):
        validate_model_registration(invalid_registration)


@pytest.mark.parametrize("invalid", [None, object()])
def test_validate_model_registration_rejects_wrong_outer_record(invalid: Any) -> None:
    """Reject values that are not model-registration records."""
    with pytest.raises(TypeError, match="ModelRegistration"):
        validate_model_registration(invalid)


def test_worker_configuration_preserves_local_and_online_requests() -> None:
    """Retain worker identity, topology, device, and requested capacity."""
    local = WorkerConfiguration(
        worker_label="local",
        worker_id=WorkerId("worker-local"),
        kind=WorkerKind.ROOT_LOCAL,
    )
    online = WorkerConfiguration(
        worker_label="remote",
        worker_id=WorkerId("worker-remote"),
        kind=WorkerKind.ONLINE,
        device="cuda:0",
        n_attempt=3,
    )

    assert local.worker_label == "local"
    assert local.worker_id == WorkerId("worker-local")
    assert local.kind is WorkerKind.ROOT_LOCAL
    assert local.device == "cpu"
    assert local.n_attempt == 1

    assert online.worker_label == "remote"
    assert online.worker_id == WorkerId("worker-remote")
    assert online.kind is WorkerKind.ONLINE
    assert online.device == "cuda:0"
    assert online.n_attempt == 3


def test_worker_configuration_is_frozen_slotted_and_keyword_only() -> None:
    """Keep worker configuration immutable, compact, and explicit."""
    configuration = WorkerConfiguration(
        worker_label="local",
        worker_id=WorkerId("worker-local"),
        kind=WorkerKind.ROOT_LOCAL,
    )

    assert not hasattr(configuration, "__dict__")
    assert all(
        parameter.kind is Parameter.KEYWORD_ONLY for parameter in signature(WorkerConfiguration).parameters.values()
    )

    with pytest.raises(FrozenInstanceError):
        configuration.__setattr__("device", "cuda:0")


def test_worker_configuration_has_only_requested_capacity() -> None:
    """Avoid claiming calculated capacity or observed device availability."""
    configuration = WorkerConfiguration(
        worker_label="local",
        worker_id=WorkerId("worker-local"),
        kind=WorkerKind.ROOT_LOCAL,
    )

    assert tuple(field.name for field in fields(configuration)) == (
        "worker_label",
        "worker_id",
        "kind",
        "device",
        "n_attempt",
    )
    assert not hasattr(configuration, "effective_n_attempt")
    assert not hasattr(configuration, "device_available")
