"""Tests for shared identity contracts."""

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from cocoonet._contracts.identity import (
    DatasetId,
    ModelId,
    ParentId,
    ReportName,
    ScalarLabel,
    UnitKey,
    validate_identifier,
    validate_label,
    validate_unit_key,
)


@pytest.mark.parametrize("report_name", tuple(ReportName))
def test_report_name_string_round_trip(report_name: ReportName) -> None:
    """Preserve report identity through its string representation.

    Parameters
    ----------
    report_name : ReportName
        Report category to round-trip.
    """
    assert ReportName(str(report_name)) is report_name


@pytest.mark.parametrize(
    "value",
    ["", "accuracy", "Validation", " validation ", "baseline_fit"],
)
def test_report_name_rejects_invalid_values(value: str) -> None:
    """Reject unsupported or noncanonical report names.

    Parameters
    ----------
    value : str
        Invalid report name supplied to the enum.
    """
    with pytest.raises(ValueError):
        ReportName(value)


def test_unit_key_mapping_identity_uses_parent_and_model() -> None:
    """Keep different parent/model pairs distinct as mapping keys."""
    key = UnitKey(parent_id=ParentId("parent-a"), model_id=ModelId("model-a"))
    same = UnitKey(parent_id=ParentId("parent-a"), model_id=ModelId("model-a"))
    other_parent = UnitKey(parent_id=ParentId("parent-b"), model_id=ModelId("model-a"))
    other_model = UnitKey(parent_id=ParentId("parent-a"), model_id=ModelId("model-b"))

    outcomes = {key: "first", other_parent: "second", other_model: "third"}

    assert key == same
    assert len(outcomes) == 3
    assert outcomes[same] == "first"
    assert outcomes[other_parent] == "second"
    assert outcomes[other_model] == "third"


@pytest.mark.parametrize("field", ["parent_id", "model_id"])
def test_unit_key_rejects_identity_mutation(field: str) -> None:
    """Prevent changes to either identity after construction.

    Parameters
    ----------
    field : str
        Identity field whose reassignment must fail.
    """
    key = UnitKey(parent_id=ParentId("parent-a"), model_id=ModelId("model-a"))

    with pytest.raises(FrozenInstanceError):
        setattr(key, field, "replacement")


def test_unit_key_requires_keyword_arguments() -> None:
    """Reject positional construction that could swap identity roles."""
    # Widen the callable signature to test invalid runtime usage without suppression.
    constructor: Callable[..., UnitKey] = UnitKey

    with pytest.raises(TypeError):
        constructor(ParentId("parent-a"), ModelId("model-a"))


def test_unit_key_has_no_instance_dictionary() -> None:
    """Keep the record's storage limited to its declared slots."""
    key = UnitKey(parent_id=ParentId("parent-a"), model_id=ModelId("model-a"))

    assert not hasattr(key, "__dict__")


@pytest.mark.parametrize(
    "value",
    [
        "model-a",
        " model-a ",
        " ",
        "\t",
        "part/one",
        "opaque:identifier",
        DatasetId("shared"),
        ModelId("shared"),
    ],
)
def test_validate_identifier_accepts_nonempty_strings(value: str) -> None:
    """Accept opaque strings without imposing formatting or identity scope.

    Parameters
    ----------
    value : str
        Nonempty identifier candidate.
    """
    validate_identifier(value, field="identifier")


@pytest.mark.parametrize("field", ["parent_id", "model_id"])
def test_validate_identifier_rejects_empty_strings(field: str) -> None:
    """Identify the affected field when rejecting an empty identifier.

    Parameters
    ----------
    field : str
        Field name expected in the error.
    """
    with pytest.raises(ValueError, match=rf"^{field} must be a nonempty string\.$"):
        validate_identifier("", field=field)


@pytest.mark.parametrize("value", [None, False, 0, 1.5, b"id", [], {}, object()])
def test_validate_identifier_rejects_non_strings(value: object) -> None:
    """Reject invalid runtime types without coercing them into strings.

    Parameters
    ----------
    value : object
        Non-string candidate supplied to the runtime validation boundary.
    """
    # The cast affects static typing only; the original object reaches validation.
    with pytest.raises(TypeError, match=r"^model_id must be a string; got "):
        validate_identifier(cast(str, value), field="model_id")


@pytest.mark.parametrize(
    "value",
    ["specimen-a", "", " ", 0, 1, -1, 10**400, 1.0, -1.5, 1e308, 5e-324, True, False],
)
def test_validate_label_accepts_supported_scalars(value: ScalarLabel) -> None:
    """Accept supported labels without imposing identifier formatting rules.

    Parameters
    ----------
    value : ScalarLabel
        Valid scalar label, including numerical boundary cases.
    """
    validate_label(value, field="sample_label")


@pytest.mark.parametrize("field", ["sample_label", "model_label", "group", "class_label"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_validate_label_rejects_nonfinite_floats(value: float, field: str) -> None:
    """Reject nonfinite labels and identify the affected field.

    Parameters
    ----------
    value : float
        Nonfinite label candidate.
    field : str
        Field name expected in the error.
    """
    with pytest.raises(ValueError, match=rf"^{field} must be finite; got "):
        validate_label(value, field=field)


@pytest.mark.parametrize("value", [None, b"label", 1j, [], {}, ("label",), object()])
def test_validate_label_rejects_unsupported_types(value: object) -> None:
    """Reject unsupported objects without converting them into labels.

    Parameters
    ----------
    value : object
        Unsupported label candidate.
    """
    # The cast affects static typing only; the original object reaches validation.
    with pytest.raises(
        TypeError,
        match=r"^sample_label must be a str, int, float, or bool; got ",
    ):
        validate_label(cast(ScalarLabel, value), field="sample_label")


@pytest.mark.parametrize(
    ("parent_id", "model_id"),
    [("parent-a", "model-a"), ("shared", "shared"), (" parent-a ", " model-a ")],
)
def test_validate_unit_key_accepts_valid_identifiers(parent_id: str, model_id: str) -> None:
    """Accept opaque identifiers without changing their values.

    Parameters
    ----------
    parent_id : str
        Parent identifier.
    model_id : str
        Model identifier.
    """
    key = UnitKey(parent_id=ParentId(parent_id), model_id=ModelId(model_id))

    validate_unit_key(key)

    assert key.parent_id == parent_id
    assert key.model_id == model_id


@pytest.mark.parametrize(
    "value",
    [None, "unit", ("parent-a", "model-a"), {"parent_id": "parent-a"}, object()],
)
def test_validate_unit_key_rejects_other_objects(value: object) -> None:
    """Reject objects that are not unit-key records.

    Parameters
    ----------
    value : object
        Invalid record candidate.
    """
    # The cast affects static typing only; validation receives the original object.
    with pytest.raises(TypeError, match=r"^value must be a UnitKey; got "):
        validate_unit_key(cast(UnitKey, value))


@pytest.mark.parametrize("field", ["parent_id", "model_id"])
@pytest.mark.parametrize(
    ("invalid_id", "error"),
    [("", ValueError), (None, TypeError), (1, TypeError), (b"id", TypeError)],
)
def test_validate_unit_key_rejects_invalid_identifiers(
    field: str,
    invalid_id: object,
    error: type[Exception],
) -> None:
    """Reject an invalid identity and identify its field.

    Parameters
    ----------
    field : str
        Identity field receiving the invalid value.
    invalid_id : object
        Empty or non-string identifier.
    error : type[Exception]
        Expected validation exception.
    """
    parent_id = invalid_id if field == "parent_id" else "parent-a"
    model_id = invalid_id if field == "model_id" else "model-a"
    # Construction permits candidates; casts preserve invalid values for validation.
    key = UnitKey(
        parent_id=cast(ParentId, parent_id),
        model_id=cast(ModelId, model_id),
    )

    with pytest.raises(error, match=rf"^{field} must be "):
        validate_unit_key(key)
