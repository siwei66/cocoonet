"""Tests for shared identity contracts."""

from typing import cast

import numpy as np
import pytest

from cocoonet._contracts.identity import (
    DatasetId,
    ModelId,
    ReportName,
    ScalarLabel,
    validate_identifier,
    validate_label,
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
    with pytest.raises(ValueError, match=rf"{field}.*nonempty.*empty string"):
        validate_identifier("", field=field)


@pytest.mark.parametrize("value", [None, False, 0, 1.5, b"id", [], {}, object()])
def test_validate_identifier_rejects_non_strings(value: object) -> None:
    """Reject invalid runtime types without coercing them into strings.

    Parameters
    ----------
    value : object
        Non-string candidate supplied to the runtime validation boundary.
    """
    with pytest.raises(TypeError) as error:
        validate_identifier(cast(str, value), field="model_id")

    message = str(error.value)
    assert "path: model_id" in message
    assert "expected: builtins.str" in message
    assert f"got type: {type(value).__module__}.{type(value).__qualname__}" in message


def test_validate_identifier_rejects_non_string_field() -> None:
    """Reject a diagnostic field name that is not a string."""
    with pytest.raises(TypeError) as error:
        validate_identifier("model-a", field=cast(str, 1))

    message = str(error.value)
    assert "path: field" in message
    assert "expected: builtins.str" in message
    assert "got type: builtins.int" in message
    assert "got value: 1" in message


@pytest.mark.parametrize(
    "value",
    ["specimen-a", "", " ", np.str_("specimen-a"), np.str_("")],
)
def test_validate_label_accepts_supported_scalars(value: ScalarLabel) -> None:
    """Accept supported labels without imposing identifier formatting rules.

    Parameters
    ----------
    value : ScalarLabel
        Standardized text label, including NumPy strings and empty labels.
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
    with pytest.raises(TypeError, match=rf"path: {field}"):
        validate_label(cast(ScalarLabel, value), field=field)


@pytest.mark.parametrize(
    "value",
    [None, b"label", 1j, [], {}, ("label",), object(), 0, 1, 10**400, 1.0, True, False, np.int64(1), np.bool_(True)],
)
def test_validate_label_rejects_unsupported_types(value: object) -> None:
    """Reject unsupported objects without converting them into labels.

    Parameters
    ----------
    value : object
        Unsupported label candidate.
    """
    with pytest.raises(TypeError) as error:
        validate_label(cast(ScalarLabel, value), field="sample_label")

    message = str(error.value)
    assert "path: sample_label" in message
    assert "expected: builtins.str | numpy.str_" in message
    assert f"got type: {type(value).__module__}.{type(value).__qualname__}" in message


def test_validate_label_rejects_non_string_field() -> None:
    """Reject a diagnostic field name that is not a string."""
    with pytest.raises(TypeError) as error:
        validate_label("sample-a", field=cast(str, 1))

    message = str(error.value)
    assert "path: field" in message
    assert "expected: builtins.str" in message
    assert "got type: builtins.int" in message
    assert "got value: 1" in message


@pytest.mark.parametrize("value", [np.str_("label"), np.str_(" label ")])
def test_string_scalars_are_preserved(value: np.str_) -> None:
    """Accept standardized NumPy identifiers and labels without conversion."""
    before = (type(value), value.tobytes())
    validate_identifier(value, field="sample_id")
    validate_label(value, field="sample_label")
    assert (type(value), value.tobytes()) == before


@pytest.mark.parametrize("validator", [validate_identifier, validate_label])
def test_diagnostic_field_requires_builtin_string(validator: object) -> None:
    """Keep field-path validation separate from accepted label subclasses."""
    from collections.abc import Callable

    check = cast(Callable[..., None], validator)
    with pytest.raises(TypeError) as error:
        check("sample-a", field=np.str_("sample_id"))
    assert "path: field" in str(error.value)
    assert "got type: numpy.str_" in str(error.value)
