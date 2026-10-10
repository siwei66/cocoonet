"""Tests for Cocoonet's direct runtime type validator."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import (
    Annotated,
    Any,
    Literal,
    NamedTuple,
    NewType,
    NotRequired,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

import numpy as np
import pandas as pd
import pytest
import torch

import cocoonet._type_validation as type_validation
from cocoonet._type_validation import (
    validate_args,
    validate_declared_types,
    validate_numpy_dtype,
    validate_pandas_dtype,
    validate_torch_dtype,
)

type TextValues = list[str]
type RecursiveValues = list[int | RecursiveValues]

UserIdentifier = NewType("UserIdentifier", str)
NumericType = TypeVar("NumericType", int, float)
IntegerBound = TypeVar("IntegerBound", bound=int)
UnrestrictedType = TypeVar("UnrestrictedType")


class Color(Enum):
    """Enumeration used by direct runtime type tests."""

    RED = "red"


@runtime_checkable
class NamedValue(Protocol):
    """Runtime-checkable protocol used by direct type tests."""

    @property
    def name(self) -> str:
        """Return the object's name."""


@dataclass(frozen=True)
class NamedObject:
    """Concrete protocol implementation used by direct type tests."""

    name: str


class SampleMapping(TypedDict):
    """Typed dictionary used by structural type tests."""

    name: str
    count: NotRequired[int]


class Coordinate(NamedTuple):
    """Named tuple used by structural type tests."""

    x: int
    y: int


@dataclass
class Source:
    """Nested source record used by structural type tests."""

    relative_path: str


@dataclass
class Sample:
    """Nested sample record used by structural type tests."""

    sources: tuple[Source, ...]


@dataclass
class Node:
    """Recursive dataclass used to verify cycle protection."""

    value: int
    next: Node | None = None


class Base:
    """Base class used by ``type[T]`` tests."""


class Derived(Base):
    """Derived class used by ``type[T]`` tests."""


class RepresentationError(RuntimeError):
    """Identify an intentional representation failure."""


class BrokenRepr:
    """Provide a hashable object whose representation fails."""

    def __repr__(self) -> str:
        """Raise instead of producing a representation.

        Returns
        -------
        str
            This method never returns.

        Raises
        ------
        RepresentationError
            Always raised.
        """
        raise RepresentationError("representation failed")


class PreviewTraversalError(RuntimeError):
    """Identify an intentional preview traversal failure."""


class FailingPreviewMapping(Mapping[object, object]):
    """Provide a mapping whose diagnostic traversal fails."""

    def __getitem__(self, key: object) -> object:
        """Reject every key lookup.

        Parameters
        ----------
        key
            Requested key.

        Returns
        -------
        object
            This method never returns.

        Raises
        ------
        KeyError
            Always raised.
        """
        raise KeyError(key)

    def __iter__(self) -> Iterator[object]:
        """Raise when mapping traversal begins.

        Returns
        -------
        Iterator[object]
            This method never returns.

        Raises
        ------
        PreviewTraversalError
            Always raised.
        """
        raise PreviewTraversalError("mapping traversal failed")

    def __len__(self) -> int:
        """Return the declared mapping size.

        Returns
        -------
        int
            Declared size.
        """
        return 1


class LiteralEqualityError(RuntimeError):
    """Identify an intentional literal equality failure."""


class ExplosiveLiteral(Enum):
    """Provide an enum literal whose equality operation fails."""

    VALUE = "value"

    def __eq__(self, other: object) -> bool:
        """Raise instead of comparing literal members.

        Parameters
        ----------
        other
            Other comparison value.

        Returns
        -------
        bool
            This method never returns.

        Raises
        ------
        LiteralEqualityError
            Always raised.
        """
        raise LiteralEqualityError("literal equality failed")

    __hash__ = Enum.__hash__


class TruthEvaluationError(RuntimeError):
    """Identify unintended truth conversion of an equality result."""


class NonBooleanEqualityResult:
    """Provide an equality result that must not be truth-tested."""

    def __bool__(self) -> bool:
        """Raise if validation attempts truth conversion.

        Returns
        -------
        bool
            This method never returns.

        Raises
        ------
        TruthEvaluationError
            Always raised.
        """
        raise TruthEvaluationError("equality result was truth-tested")


class NonBooleanLiteral(Enum):
    """Provide an enum whose equality returns a non-boolean object."""

    VALUE = "value"

    def __eq__(self, other: object) -> bool:
        """Return a non-boolean equality result at runtime.

        Parameters
        ----------
        other
            Other comparison value.

        Returns
        -------
        bool
            Runtime object that must not be truth-tested.
        """
        return cast(bool, NonBooleanEqualityResult())

    __hash__ = Enum.__hash__


class FieldAccessError(RuntimeError):
    """Identify an intentional structural field-access failure."""


@dataclass
class InaccessibleRecord:
    """Provide a dataclass whose declared field cannot be accessed."""

    value: int

    def __getattribute__(self, name: str) -> object:
        """Reject access to the declared field.

        Parameters
        ----------
        name
            Requested attribute name.

        Returns
        -------
        object
            Requested non-field attribute.

        Raises
        ------
        FieldAccessError
            If the declared field is requested.
        """
        if name == "value":
            raise FieldAccessError("field access failed")
        return object.__getattribute__(self, name)


class InstanceCheckError(TypeError):
    """Identify an intentional runtime instance-check failure."""


class FailingInstanceCheckMeta(type):
    """Provide a metaclass whose instance check fails."""

    def __instancecheck__(cls, instance: object) -> bool:
        """Raise instead of determining instance membership.

        Parameters
        ----------
        instance
            Candidate instance.

        Returns
        -------
        bool
            This method never returns.

        Raises
        ------
        InstanceCheckError
            Always raised.
        """
        raise InstanceCheckError("custom instance check failed")


class FailingInstanceTarget(metaclass=FailingInstanceCheckMeta):
    """Runtime class using the intentionally failing metaclass."""


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [
        (True, bool),
        (1, int),
        (1.5, float),
        ("value", str),
        (None, None),
        (object(), Any),
        (Color.RED, Color),
        (NamedObject("sample"), NamedValue),
    ],
)
def test_validate_declared_types_accepts_direct_types(
    value: object,
    expected_type: object,
) -> None:
    """Accept values satisfying direct runtime annotations."""
    validate_declared_types(value, expected_type=expected_type)


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [
        (True, int),
        (1, bool),
        (1, float),
        (1.0, int),
    ],
)
def test_validate_declared_types_keeps_scalars_strict(
    value: object,
    expected_type: object,
) -> None:
    """Reject implicit compatibility between distinct scalar types."""
    with pytest.raises(TypeError):
        validate_declared_types(value, expected_type=expected_type)


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [
        ("sample", str | int),
        (2, Union[str, int]),
        (None, str | None),
        (1, Literal[1]),
        ("sample", Annotated[str, "metadata"]),
        (["a", "b"], TextValues),
        ("sample", UserIdentifier),
        (1, NumericType),
        (1.5, NumericType),
        (1, IntegerBound),
        (object(), UnrestrictedType),
        (Derived, type[Base]),
        (lambda value: str(value), Callable[[int], str]),
    ],
)
def test_validate_declared_types_accepts_typing_constructs(
    value: object,
    expected_type: object,
) -> None:
    """Accept supported aliases and special typing constructs."""
    validate_declared_types(value, expected_type=expected_type)


def test_validate_declared_types_applies_strict_literal_matching() -> None:
    """Require literal value and runtime type to match."""
    with pytest.raises(TypeError):
        validate_declared_types(True, expected_type=Literal[1])


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [
        ([1, 2], list[int]),
        ({1, 2}, set[int]),
        (frozenset({1, 2}), frozenset[int]),
        ({"a": 1}, dict[str, int]),
        ({"a": 1}, Mapping[str, int]),
        ([1, 2], Sequence[int]),
        ((1, "a"), tuple[int, str]),
        ((1, 2), tuple[int, ...]),
        ((), tuple[()]),
    ],
)
def test_validate_declared_types_accepts_containers(
    value: object,
    expected_type: object,
) -> None:
    """Accept supported concrete and abstract containers."""
    validate_declared_types(value, expected_type=expected_type)


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [
        ([1, "two"], list[int]),
        ({1, "two"}, set[int]),
        ({"a": "one"}, dict[str, int]),
        ((1, 2), tuple[int, str]),
        ((1, 2, 3), tuple[int, int]),
        ((1,), tuple[()]),
    ],
)
def test_validate_declared_types_rejects_invalid_containers(
    value: object,
    expected_type: object,
) -> None:
    """Reject invalid members and incompatible tuple structures."""
    with pytest.raises(TypeError):
        validate_declared_types(value, expected_type=expected_type)


def test_validate_declared_types_validates_typed_dictionary() -> None:
    """Validate required, optional, and nested typed-dictionary entries."""
    validate_declared_types(
        {"name": "sample", "count": 2},
        expected_type=SampleMapping,
    )
    validate_declared_types(
        {"name": "sample"},
        expected_type=SampleMapping,
    )

    with pytest.raises(TypeError):
        validate_declared_types(
            {"name": "sample", "count": "two"},
            expected_type=SampleMapping,
        )


def test_validate_declared_types_reports_typed_dictionary_detail() -> None:
    """Report missing typed-dictionary keys as additional detail."""
    with pytest.raises(TypeError) as error:
        validate_declared_types({}, expected_type=SampleMapping)

    assert "detail: missing required keys: ['name']" in str(error.value)


def test_validate_declared_types_validates_named_tuple() -> None:
    """Inspect and validate declared named-tuple fields."""
    invalid = Coordinate(1, cast(int, "two"))

    with pytest.raises(TypeError, match=r"value\.y"):
        validate_declared_types(invalid, expected_type=Coordinate)


def test_validate_declared_types_reports_nested_dataclass_path() -> None:
    """Report the incompatible nested dataclass field."""
    sample = Sample(
        sources=(
            Source("images/a.png"),
            Source(cast(str, 42)),
        )
    )

    with pytest.raises(TypeError, match=r"value\.sources\[1\]\.relative_path"):
        validate_declared_types(sample, expected_type=Sample)


def test_validate_declared_types_protects_cycles() -> None:
    """Terminate validation of recursive dataclass and container graphs."""
    node = Node(1)
    node.next = node
    validate_declared_types(node, expected_type=Node)

    values: list[object] = []
    values.append(values)
    validate_declared_types(values, expected_type=RecursiveValues)


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [
        (np.array([1, 2]), np.ndarray),
        (np.float64(1.5), np.float64),
        (pd.DataFrame({"x": [1]}), pd.DataFrame),
        (pd.Series([1]), pd.Series),
        (pd.Index([1]), pd.Index),
        (torch.tensor([1.0]), torch.Tensor),
        (torch.float32, torch.dtype),
    ],
)
def test_validate_declared_types_accepts_framework_types(
    value: object,
    expected_type: object,
) -> None:
    """Accept concrete NumPy, pandas, and PyTorch runtime classes."""
    validate_declared_types(value, expected_type=expected_type)


def test_validate_declared_types_does_not_scan_dataframe_cells() -> None:
    """Validate a dataframe class without representing its cells."""
    dataframe = pd.DataFrame({"value": [BrokenRepr()]})

    validate_declared_types(dataframe, expected_type=pd.DataFrame)


def test_validate_declared_types_does_not_consume_iterator() -> None:
    """Reject an unsupported iterator annotation without consuming it."""
    iterator = iter([1, 2, 3])

    with pytest.raises(TypeError) as error:
        validate_declared_types(iterator, expected_type=Iterator[int])

    assert "detail: unsupported runtime annotation" in str(error.value)
    assert next(iterator) == 1


def test_ordinary_type_mismatch_uses_concise_error() -> None:
    """Report an ordinary mismatch without a redundant detail line."""
    with pytest.raises(TypeError) as error:
        validate_declared_types(
            {"sample": [1, "two"]},
            expected_type=dict[str, list[int]],
            field="records",
        )

    message = str(error.value)
    assert "Type validation failed" in message
    assert "path: records['sample'][1]" in message
    assert "expected: builtins.int" in message
    assert "got type: builtins.str" in message
    assert "got value: 'two'" in message
    assert "detail:" not in message


def test_structural_type_mismatch_adds_detail() -> None:
    """Add detail when structure cannot be inferred from the type names."""
    with pytest.raises(TypeError) as error:
        validate_declared_types(
            (1, "two", 3),
            expected_type=tuple[int, str],
        )

    message = str(error.value)
    assert "got type: builtins.tuple" in message
    assert "detail: fixed tuple length mismatch: expected 2, got 3" in message


@pytest.mark.parametrize(
    "value",
    [
        "x" * 2_000,
        list(range(1_000)),
        np.arange(1_000),
        pd.DataFrame(np.arange(1_000).reshape(100, 10)),
        torch.arange(1_000),
    ],
)
def test_validate_declared_types_bounds_previews(value: object) -> None:
    """Keep diagnostics bounded for large supported values."""
    with pytest.raises(TypeError) as error:
        validate_declared_types(value, expected_type=int)

    assert len(str(error.value)) < 1_000


def test_type_error_survives_broken_value_repr() -> None:
    """Report a mismatch when the actual value cannot be represented."""
    with pytest.raises(TypeError) as error:
        validate_declared_types(BrokenRepr(), expected_type=int)

    message = str(error.value)
    assert "got type:" in message
    assert "representation unavailable" in message


def test_type_error_survives_preview_traversal_failure() -> None:
    """Preserve a mismatch when diagnostic mapping traversal fails."""
    with pytest.raises(TypeError) as error:
        validate_declared_types(FailingPreviewMapping(), expected_type=int)

    message = str(error.value)
    assert "expected: builtins.int" in message
    assert "value preview unavailable" in message


def test_mapping_key_repr_failure_does_not_mask_nested_mismatch() -> None:
    """Reach an invalid mapping value when its key cannot be represented."""
    value = {BrokenRepr(): "wrong"}

    with pytest.raises(TypeError) as error:
        validate_declared_types(
            value,
            expected_type=dict[BrokenRepr, int],
        )

    message = str(error.value)
    assert "path: value.values()[0]" in message
    assert "expected: builtins.int" in message


def test_set_repr_failure_does_not_mask_member_mismatch() -> None:
    """Reject an invalid set member despite its broken representation."""
    with pytest.raises(TypeError) as error:
        validate_declared_types(
            {1, BrokenRepr()},
            expected_type=set[int],
        )

    message = str(error.value)
    assert "expected: builtins.int" in message
    assert "representation unavailable" in message


def test_literal_equality_exception_propagates() -> None:
    """Preserve an exception raised by literal equality."""
    with pytest.raises(LiteralEqualityError, match="literal equality failed"):
        validate_declared_types(
            ExplosiveLiteral.VALUE,
            expected_type=Literal[ExplosiveLiteral.VALUE],
        )


def test_literal_does_not_truth_test_equality_result() -> None:
    """Reject a non-boolean equality result without truth-testing it."""
    with pytest.raises(TypeError):
        validate_declared_types(
            NonBooleanLiteral.VALUE,
            expected_type=Literal[NonBooleanLiteral.VALUE],
        )


def test_structural_attribute_exception_propagates() -> None:
    """Preserve an exception raised while reading a structural field."""
    with pytest.raises(FieldAccessError, match="field access failed"):
        validate_declared_types(
            InaccessibleRecord(1),
            expected_type=InaccessibleRecord,
        )


def test_custom_instance_check_exception_propagates() -> None:
    """Preserve an exception raised by custom runtime type checking."""
    with pytest.raises(InstanceCheckError, match="custom instance check failed"):
        validate_declared_types(
            object(),
            expected_type=FailingInstanceTarget,
        )


@pytest.mark.parametrize(
    ("value", "expected_dtype"),
    [
        (np.array([1.0], dtype=np.float32), "float32"),
        (np.float64(1.5), np.float64),
        (np.array([1], dtype=np.int32), np.dtype("int32")),
    ],
)
def test_validate_numpy_dtype_accepts_normalized_specs(
    value: np.ndarray[Any, Any] | np.generic,
    expected_dtype: object,
) -> None:
    """Accept array and scalar dtypes through NumPy normalization."""
    validate_numpy_dtype(value, expected_dtype=expected_dtype)


def test_validate_numpy_dtype_reports_dtype_path() -> None:
    """Report a NumPy dtype mismatch at the value's dtype path."""
    value = np.array([1], dtype=np.int64)

    with pytest.raises(TypeError) as error:
        validate_numpy_dtype(value, expected_dtype=np.float32, field="features")

    message = str(error.value)
    assert "path: features.dtype" in message
    assert "expected: dtype('float32')" in message
    assert "got type: numpy.dtypes.Int64DType" in message
    assert "got value: dtype('int64')" in message
    assert "detail:" not in message


def test_validate_numpy_dtype_rejects_invalid_specification() -> None:
    """Report an invalid expected dtype through the shared diagnostic."""
    value = np.array([1], dtype=np.int64)

    with pytest.raises(TypeError) as error:
        validate_numpy_dtype(value, expected_dtype="not-a-dtype")

    message = str(error.value)
    assert "path: expected_dtype" in message
    assert "expected: numpy.dtype" in message
    assert "got type: builtins.str" in message
    assert "got value: 'not-a-dtype'" in message
    assert "detail: invalid NumPy dtype specification:" in message


def test_validate_numpy_dtype_rejects_non_numpy_value() -> None:
    """Reject values outside the supported NumPy array and scalar types."""
    with pytest.raises(TypeError, match=r"path: value"):
        validate_numpy_dtype(cast(Any, [1, 2]), expected_dtype=np.int64)


def test_validate_numpy_dtype_does_not_cast_or_replace_value() -> None:
    """Leave the validated NumPy value and its dtype unchanged."""
    value = np.array([1.5], dtype=np.float32)
    original_id = id(value)

    validate_numpy_dtype(value, expected_dtype=np.float32)

    assert id(value) == original_id
    assert value.dtype == np.dtype(np.float32)


def test_validate_declared_types_accepts_numpy_dtype_annotation() -> None:
    """Accept a parameterized NumPy array with its declared dtype."""
    validate_declared_types(
        np.array([1.0], dtype=np.float32),
        expected_type=np.ndarray[Any, np.dtype[np.float32]],
    )


def test_validate_declared_types_rejects_numpy_dtype_annotation_mismatch() -> None:
    """Reject a parameterized NumPy array with a different dtype."""
    with pytest.raises(TypeError, match=r"path: value\.dtype"):
        validate_declared_types(
            np.array([1], dtype=np.int64),
            expected_type=np.ndarray[Any, np.dtype[np.float32]],
        )


def test_numpy_dtype_annotation_preserves_union_fallback() -> None:
    """Reject one dtype branch internally before accepting a later union branch."""
    validate_declared_types(
        [1, 2],
        expected_type=np.ndarray[Any, np.dtype[np.float32]] | list[int],
    )


@pytest.mark.parametrize(
    ("value", "expected_dtype"),
    [
        (pd.Series([1.0], dtype="float32"), "float32"),
        (pd.Index([1, 2], dtype="int64"), int),
        (pd.Series([1, None], dtype="Int64"), pd.Int64Dtype()),
        (
            pd.DataFrame(
                {
                    "first": pd.Series([1.0], dtype="float64"),
                    "second": pd.Series([2.0], dtype="float64"),
                }
            ),
            "float64",
        ),
    ],
)
def test_validate_pandas_dtype_accepts_normalized_specs(
    value: pd.Series[Any] | pd.Index[Any] | pd.DataFrame,
    expected_dtype: object,
) -> None:
    """Accept NumPy and pandas extension dtypes after normalization."""
    validate_pandas_dtype(value, expected_dtype=expected_dtype)


def test_validate_pandas_dtype_reports_series_dtype_path() -> None:
    """Report a series dtype mismatch at its dtype path."""
    value = pd.Series([1], dtype="int64")

    with pytest.raises(TypeError) as error:
        validate_pandas_dtype(
            value,
            expected_dtype="float32",
            field="target",
        )

    message = str(error.value)
    assert "path: target.dtype" in message
    assert "expected: dtype('float32')" in message
    assert "got value: dtype('int64')" in message
    assert "detail:" not in message


def test_validate_pandas_dtype_reports_dataframe_column_position() -> None:
    """Report the first heterogeneous dataframe dtype by stable position."""
    value = pd.DataFrame(
        {
            "first": pd.Series([1.0], dtype="float64"),
            "second": pd.Series([2], dtype="int64"),
        }
    )

    with pytest.raises(TypeError) as error:
        validate_pandas_dtype(
            value,
            expected_dtype="float64",
            field="table",
        )

    message = str(error.value)
    assert "path: table.dtypes[1]" in message
    assert "expected: dtype('float64')" in message
    assert "got value: dtype('int64')" in message


def test_validate_pandas_dtype_rejects_invalid_specification() -> None:
    """Report an invalid expected pandas dtype through the shared diagnostic."""
    value = pd.Series([1], dtype="int64")

    with pytest.raises(TypeError) as error:
        validate_pandas_dtype(value, expected_dtype="not-a-dtype")

    message = str(error.value)
    assert "path: expected_dtype" in message
    assert "got type: builtins.str" in message
    assert "got value: 'not-a-dtype'" in message
    assert "detail: invalid pandas dtype specification:" in message


def test_validate_pandas_dtype_rejects_non_pandas_value() -> None:
    """Reject values outside the supported pandas object types."""
    with pytest.raises(TypeError, match=r"path: value"):
        validate_pandas_dtype(
            cast(Any, np.array([1])),
            expected_dtype="int64",
        )


def test_validate_pandas_dtype_does_not_cast_or_replace_value() -> None:
    """Leave the validated pandas object and its dtype unchanged."""
    value = pd.Series([1, None], dtype="Int64")
    original_id = id(value)
    original_dtype = value.dtype

    validate_pandas_dtype(value, expected_dtype="Int64")

    assert id(value) == original_id
    assert value.dtype == original_dtype


def test_validate_pandas_dtype_supports_column_specific_delegation() -> None:
    """Allow heterogeneous schemas through separate series validation."""
    value = pd.DataFrame(
        {
            "sample_label": pd.Series(["first"], dtype="string"),
            "target": pd.Series([1.5], dtype="float64"),
        }
    )

    validate_pandas_dtype(
        value["sample_label"],
        expected_dtype="string",
        field="table['sample_label']",
    )
    validate_pandas_dtype(
        value["target"],
        expected_dtype="float64",
        field="table['target']",
    )


@pytest.mark.parametrize(
    ("value", "expected_dtype"),
    [
        (torch.tensor([1.0], dtype=torch.float32), torch.float32),
        (torch.tensor([1], dtype=torch.int64), torch.int64),
        (torch.tensor([True], dtype=torch.bool), torch.bool),
    ],
)
def test_validate_torch_dtype_accepts_matching_dtype(
    value: torch.Tensor,
    expected_dtype: torch.dtype,
) -> None:
    """Accept tensors whose dtype exactly matches the required dtype."""
    validate_torch_dtype(value, expected_dtype=expected_dtype)


def test_validate_torch_dtype_reports_dtype_path() -> None:
    """Report a tensor dtype mismatch at the tensor's dtype path."""
    value = torch.tensor([1], dtype=torch.int64)

    with pytest.raises(TypeError) as error:
        validate_torch_dtype(
            value,
            expected_dtype=torch.float32,
            field="features",
        )

    message = str(error.value)
    assert "path: features.dtype" in message
    assert "expected: torch.float32" in message
    assert "got type: torch.dtype" in message
    assert "got value: torch.int64" in message
    assert "detail:" not in message


def test_validate_torch_dtype_rejects_invalid_expected_dtype() -> None:
    """Reject an expected dtype that is not a PyTorch dtype."""
    value = torch.tensor([1], dtype=torch.int64)

    with pytest.raises(TypeError) as error:
        validate_torch_dtype(
            value,
            expected_dtype=cast(Any, np.dtype("int64")),
        )

    message = str(error.value)
    assert "path: expected_dtype" in message
    assert "expected: torch.dtype" in message
    assert "got type:" in message
    assert "got value: dtype('int64')" in message


def test_validate_torch_dtype_rejects_non_tensor_value() -> None:
    """Reject values outside the PyTorch tensor type."""
    with pytest.raises(TypeError, match=r"path: value"):
        validate_torch_dtype(
            cast(Any, [1.0]),
            expected_dtype=torch.float32,
        )


def test_validate_torch_dtype_does_not_change_tensor() -> None:
    """Leave tensor identity, storage, values, dtype, and device unchanged."""
    value = torch.tensor(
        [1.0, 2.0],
        dtype=torch.float32,
        requires_grad=True,
    )
    original_id = id(value)
    original_data_pointer = value.data_ptr()
    original_values = value.detach().clone()
    original_device = value.device

    validate_torch_dtype(value, expected_dtype=torch.float32)

    assert id(value) == original_id
    assert value.data_ptr() == original_data_pointer
    assert torch.equal(value.detach(), original_values)
    assert value.dtype is torch.float32
    assert value.device == original_device
    assert value.requires_grad


def test_validate_args_validates_positional_keyword_and_default_arguments() -> None:
    """Validate ordinary arguments regardless of how Python supplies them."""

    @validate_args
    def describe(count: int, label: str = "sample") -> tuple[int, str]:
        """Return the supplied values after argument validation."""
        return count, label

    assert describe(2) == (2, "sample")
    assert describe(count=3, label="record") == (3, "record")

    with pytest.raises(TypeError, match=r"path: count"):
        describe(cast(int, "two"))

    with pytest.raises(TypeError, match=r"path: label"):
        describe(2, label=cast(str, 3))


def test_validate_args_validates_applied_default_values() -> None:
    """Validate an annotated default after applying the function signature."""
    body_calls: list[object] = []

    @validate_args
    def use_default(value: int = cast(int, "invalid")) -> None:
        """Record execution after default validation."""
        body_calls.append(value)

    with pytest.raises(TypeError, match=r"path: value"):
        use_default()

    assert body_calls == []


def test_validate_args_validates_methods() -> None:
    """Validate annotated method arguments while skipping unannotated self."""

    class Counter:
        """Provide a decorated instance method for validation."""

        @validate_args
        def add(self, value: int) -> int:
            """Return the supplied integer."""
            return value

    counter = Counter()
    assert counter.add(2) == 2

    with pytest.raises(TypeError, match=r"path: value"):
        counter.add(cast(int, "two"))


def test_validate_args_validates_variadic_arguments() -> None:
    """Validate every supplied variadic positional and keyword value."""

    @validate_args
    def collect(*values: int, **labels: str) -> None:
        """Accept validated variadic values."""
        return None

    collect(1, 2, first="sample")

    with pytest.raises(TypeError, match=r"path: values\[1\]"):
        collect(1, cast(int, "two"))

    with pytest.raises(TypeError, match=r"path: labels\['first'\]"):
        collect(first=cast(str, 1))


def test_validate_args_matches_direct_validation_diagnostics() -> None:
    """Produce the same diagnostic as a corresponding direct validation."""

    @validate_args
    def accept(records: dict[str, list[int]]) -> None:
        """Accept validated records."""
        return None

    invalid = {"sample": [1, "two"]}

    with pytest.raises(TypeError) as direct_error:
        validate_declared_types(
            invalid,
            expected_type=dict[str, list[int]],
            field="records",
        )

    with pytest.raises(TypeError) as decorated_error:
        accept(cast(dict[str, list[int]], invalid))

    assert str(decorated_error.value) == str(direct_error.value)


def test_validate_args_caches_signature_and_type_hints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve the signature and annotations once when decorating."""
    signature_calls = 0
    type_hint_calls = 0
    original_signature = inspect.signature
    original_get_type_hints = type_validation.get_type_hints

    def counted_signature(target: object) -> inspect.Signature:
        """Count and delegate one signature resolution."""
        nonlocal signature_calls
        signature_calls += 1
        return original_signature(cast(Callable[..., Any], target))

    def counted_get_type_hints(
        target: object,
        *,
        include_extras: bool = False,
    ) -> dict[str, object]:
        """Count and delegate one annotation resolution."""
        nonlocal type_hint_calls
        type_hint_calls += 1
        return original_get_type_hints(
            target,
            include_extras=include_extras,
        )

    monkeypatch.setattr(
        type_validation.inspect,
        "signature",
        counted_signature,
    )
    monkeypatch.setattr(
        type_validation,
        "get_type_hints",
        counted_get_type_hints,
    )

    @validate_args
    def identity(value: int) -> int:
        """Return one validated integer."""
        return value

    assert identity(1) == 1
    assert identity(2) == 2
    assert signature_calls == 1
    assert type_hint_calls == 1


def test_validate_args_preserves_callable_metadata_and_signature() -> None:
    """Preserve callable metadata and introspection through functools.wraps."""

    def original(value: int, label: str = "sample") -> tuple[int, str]:
        """Return the supplied record."""
        return value, label

    decorated = validate_args(original)

    assert decorated.__name__ == original.__name__
    assert decorated.__doc__ == original.__doc__
    assert cast(Any, decorated).__wrapped__ is original
    assert inspect.signature(decorated) == inspect.signature(original)


def test_validate_args_does_not_execute_body_after_type_failure() -> None:
    """Stop before executing the wrapped body after invalid input."""
    body_calls: list[object] = []

    @validate_args
    def record(value: int) -> None:
        """Record a validated value."""
        body_calls.append(value)

    with pytest.raises(TypeError, match=r"path: value"):
        record(cast(int, "invalid"))

    assert body_calls == []


def test_validate_args_propagates_body_exceptions_unchanged() -> None:
    """Preserve the exact exception raised by the wrapped callable."""
    body_error = RuntimeError("body failed")

    @validate_args
    def fail(value: int) -> None:
        """Raise the prepared body exception."""
        raise body_error

    with pytest.raises(RuntimeError) as error:
        fail(1)

    assert error.value is body_error


def test_validate_args_does_not_validate_return_value() -> None:
    """Leave return-value validation outside the Phase 1 decorator contract."""

    @validate_args
    def return_unchecked(value: int) -> int:
        """Return a deliberately mismatched value for contract verification."""
        return cast(int, "unchecked")

    assert return_unchecked(1) == "unchecked"
