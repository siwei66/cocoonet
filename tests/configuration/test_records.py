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
    validate_worker_configuration,
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


def test_worker_configuration_preserves_machine_and_execution_properties() -> None:
    """Retain worker identity, machine properties, and execution settings."""
    local = WorkerConfiguration(
        worker_id=WorkerId("abc123"),
        worker_label="shared-label",
        kind=WorkerKind.ROOT_LOCAL,
    )
    online = WorkerConfiguration(
        worker_id=WorkerId("xyz789"),
        worker_label="shared-label",
        kind=WorkerKind.ONLINE,
        device="cuda:0",
        cpu_capacity=16,
        n_attempt=3,
        n_cpu_per_attempt=4,
        memory_reserve_pct=25.0,
    )

    assert local.worker_id == WorkerId("abc123")
    assert local.worker_label == "shared-label"
    assert local.kind is WorkerKind.ROOT_LOCAL
    assert local.device == "cpu"
    assert local.cpu_capacity is None
    assert local.n_attempt == 1
    assert local.n_cpu_per_attempt == 1
    assert local.memory_reserve_pct == 20.0

    assert online.worker_id == WorkerId("xyz789")
    assert online.worker_label == "shared-label"
    assert online.kind is WorkerKind.ONLINE
    assert online.device == "cuda:0"
    assert online.cpu_capacity == 16
    assert online.n_attempt == 3
    assert online.n_cpu_per_attempt == 4
    assert online.memory_reserve_pct == 25.0


def test_worker_configuration_is_frozen_slotted_and_keyword_only() -> None:
    """Keep worker configuration immutable, compact, and explicit."""
    configuration = WorkerConfiguration(
        worker_id=WorkerId("abc123"),
        worker_label="local",
        kind=WorkerKind.ROOT_LOCAL,
    )

    assert not hasattr(configuration, "__dict__")
    assert all(
        parameter.kind is Parameter.KEYWORD_ONLY for parameter in signature(WorkerConfiguration).parameters.values()
    )

    with pytest.raises(FrozenInstanceError):
        configuration.__setattr__("device", "cuda:0")


def test_worker_configuration_matches_public_field_order() -> None:
    """Match the public worker configuration declaration exactly."""
    configuration = WorkerConfiguration(
        worker_id=WorkerId("abc123"),
        worker_label="local",
        kind=WorkerKind.ROOT_LOCAL,
    )

    assert tuple(field.name for field in fields(configuration)) == (
        "worker_id",
        "worker_label",
        "kind",
        "device",
        "cpu_capacity",
        "n_attempt",
        "n_cpu_per_attempt",
        "memory_reserve_pct",
    )
    assert not hasattr(configuration, "effective_n_attempt")
    assert not hasattr(configuration, "device_available")


def test_validate_worker_configuration_accepts_valid_properties() -> None:
    """Accept valid worker properties and execution configurations."""
    local = WorkerConfiguration(
        worker_id=WorkerId("abc123"),
        worker_label="shared-label",
        kind=WorkerKind.ROOT_LOCAL,
    )
    online = WorkerConfiguration(
        worker_id=WorkerId("xyz789"),
        worker_label="shared-label",
        kind=WorkerKind.ONLINE,
        device="cuda:0",
        cpu_capacity=16,
        n_attempt=3,
        n_cpu_per_attempt=4,
        memory_reserve_pct=0.0,
    )
    fully_reserved = replace(online, memory_reserve_pct=100.0)

    validate_worker_configuration(local)
    validate_worker_configuration(online)
    validate_worker_configuration(fully_reserved)


@pytest.mark.parametrize(
    ("field", "invalid", "error_type"),
    [
        ("worker_id", 1, TypeError),
        ("worker_id", WorkerId(""), ValueError),
        ("worker_id", WorkerId("abc12"), ValueError),
        ("worker_id", WorkerId("abc1234"), ValueError),
        ("worker_id", WorkerId("ABC123"), ValueError),
        ("worker_id", WorkerId("abc-12"), ValueError),
        ("worker_label", 1, TypeError),
        ("kind", "root-local", TypeError),
        ("device", 1, TypeError),
        ("device", "", ValueError),
        ("cpu_capacity", 0, ValueError),
        ("cpu_capacity", -1, ValueError),
        ("cpu_capacity", True, TypeError),
        ("cpu_capacity", 1.0, TypeError),
        ("n_attempt", 0, ValueError),
        ("n_attempt", -1, ValueError),
        ("n_attempt", True, TypeError),
        ("n_attempt", 1.0, TypeError),
        ("n_attempt", "1", TypeError),
        ("n_cpu_per_attempt", 0, ValueError),
        ("n_cpu_per_attempt", -1, ValueError),
        ("n_cpu_per_attempt", True, TypeError),
        ("n_cpu_per_attempt", 1.0, TypeError),
        ("memory_reserve_pct", -0.1, ValueError),
        ("memory_reserve_pct", 100.1, ValueError),
        ("memory_reserve_pct", 20, TypeError),
        ("memory_reserve_pct", "20.0", TypeError),
    ],
)
def test_validate_worker_configuration_rejects_invalid_fields(
    field: str,
    invalid: Any,
    error_type: type[Exception],
) -> None:
    """Reject invalid worker fields without interpreting valid requests."""
    configuration = WorkerConfiguration(
        worker_id=WorkerId("abc123"),
        worker_label="local",
        kind=WorkerKind.ROOT_LOCAL,
    )
    invalid_configuration = replace(configuration, **{field: invalid})

    with pytest.raises(error_type, match=field):
        validate_worker_configuration(invalid_configuration)


@pytest.mark.parametrize("invalid", [None, object()])
def test_validate_worker_configuration_rejects_wrong_outer_record(
    invalid: Any,
) -> None:
    """Reject values that are not worker-configuration records."""
    with pytest.raises(TypeError, match="WorkerConfiguration"):
        validate_worker_configuration(invalid)
