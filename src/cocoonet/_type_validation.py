"""Shared strict runtime type validation for Cocoonet."""

from __future__ import annotations

import collections.abc as collections_abc
import inspect
from dataclasses import fields, is_dataclass
from functools import lru_cache, wraps
from itertools import islice
from types import NoneType, UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    NoReturn,
    NotRequired,
    ParamSpec,
    Required,
    TypeAliasType,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

import numpy as np
import pandas as pd
import torch

_VALUE_PREVIEW_LIMIT = 500
_ITEM_PREVIEW_LIMIT = 80
_PREVIEW_ITEM_LIMIT = 8
_DATAFRAME_PREVIEW_ROWS = 5
_DATAFRAME_PREVIEW_COLUMNS = 8

_Parameters = ParamSpec("_Parameters")
_ReturnType = TypeVar("_ReturnType")

_STRICT_SCALAR_TYPES: tuple[type[object], ...] = (
    bool,
    int,
    float,
    complex,
    str,
    bytes,
)


class _TypeMismatchError(Exception):
    """Carry one internal mismatch to the public validation boundary."""

    def __init__(
        self,
        path: str,
        expected_type: object,
        value: object,
        detail: str | None = None,
    ) -> None:
        """Initialize an internal type mismatch.

        Parameters
        ----------
        path
            Location of the mismatching value.
        expected_type
            Declared type the value failed to satisfy.
        value
            Actual mismatching value.
        detail
            Optional explanation that adds information beyond the expected and
            actual types.
        """
        super().__init__(detail)
        self.path = path
        self.expected_type = expected_type
        self.value = value
        self.detail = detail


def _type_name(type_: type[object]) -> str:
    """Return a qualified name for a runtime type.

    Parameters
    ----------
    type_
        Runtime type to describe.

    Returns
    -------
    str
        Module-qualified type name or a fixed fallback.
    """
    try:
        module = type_.__module__
        name = type_.__qualname__
    except Exception:
        return "<runtime type unavailable>"

    if module == "builtins":
        return f"builtins.{name}"
    return f"{module}.{name}"


def _truncate(text: str, limit: int) -> str:
    """Limit diagnostic text to a fixed character budget.

    Parameters
    ----------
    text
        Text to limit.
    limit
        Maximum number of original characters retained.

    Returns
    -------
    str
        Complete or abbreviated text.
    """
    if len(text) <= limit:
        return text

    omitted = len(text) - limit
    return f"{text[:limit]}... <{omitted} characters omitted>"


def _safe_repr(value: object, *, limit: int = _VALUE_PREVIEW_LIMIT) -> str:
    """Return a bounded representation without exposing ``repr`` failures.

    Parameters
    ----------
    value
        Value to represent.
    limit
        Maximum number of original characters retained.

    Returns
    -------
    str
        Bounded representation or a representation-failure marker.
    """
    try:
        representation = repr(value)
    except Exception as error:
        return f"<representation unavailable: {_type_name(type(error))}>"

    return _truncate(representation, limit)


def _is_new_type(expected_type: object) -> bool:
    """Return whether an annotation is a runtime ``NewType`` definition.

    Parameters
    ----------
    expected_type
        Annotation to inspect.

    Returns
    -------
    bool
        Whether the object exposes the runtime ``NewType`` contract.
    """
    return callable(expected_type) and hasattr(expected_type, "__supertype__")


def _generic_annotation_text(
    origin: object,
    arguments: tuple[object, ...],
) -> str:
    """Format a parameterized typing annotation.

    Parameters
    ----------
    origin
        Runtime annotation origin.
    arguments
        Annotation arguments.

    Returns
    -------
    str
        Readable annotation description.
    """
    if origin in (Union, UnionType):
        return " | ".join(_annotation_text(argument) for argument in arguments)

    if origin is Literal:
        values = ", ".join(_safe_repr(value, limit=_ITEM_PREVIEW_LIMIT) for value in arguments)
        return f"typing.Literal[{values}]"

    if origin is Annotated:
        return f"typing.Annotated[{_annotation_text(arguments[0])}, ...]"

    if origin is type:
        return f"builtins.type[{_annotation_text(arguments[0])}]"

    origin_text = _type_name(origin) if isinstance(origin, type) else _safe_repr(origin)
    arguments_text = ", ".join(_annotation_text(argument) for argument in arguments)
    return f"{origin_text}[{arguments_text}]"


def _annotation_text(expected_type: object) -> str:
    """Produce a readable description of an expected annotation.

    Parameters
    ----------
    expected_type
        Expected annotation.

    Returns
    -------
    str
        Human-readable annotation description.
    """
    if expected_type is Any:
        return "typing.Any"

    if expected_type is None or expected_type is NoneType:
        return "builtins.NoneType"

    if expected_type is Ellipsis:
        return "..."

    if isinstance(expected_type, TypeAliasType):
        module = expected_type.__module__
        name = expected_type.__name__
        return name if module == "builtins" else f"{module}.{name}"

    if isinstance(expected_type, TypeVar):
        return str(expected_type)

    if _is_new_type(expected_type):
        module = getattr(expected_type, "__module__", "")
        name = str(getattr(expected_type, "__name__", "<NewType>"))
        return name if module == "builtins" else f"{module}.{name}"

    if isinstance(expected_type, type):
        return _type_name(expected_type)

    origin = get_origin(expected_type)
    if origin is not None:
        return _generic_annotation_text(origin, get_args(expected_type))

    return _safe_repr(expected_type)


def _safe_annotation_text(expected_type: object) -> str:
    """Format an annotation without replacing an established mismatch.

    Parameters
    ----------
    expected_type
        Expected annotation.

    Returns
    -------
    str
        Annotation description or a formatting-failure marker.
    """
    try:
        return _annotation_text(expected_type)
    except Exception as error:
        return f"<expected type unavailable: {_type_name(type(error))}>"


def _preview_items(items: collections_abc.Iterable[object]) -> str:
    """Produce a bounded representation of iterable members.

    Parameters
    ----------
    items
        Items to preview.

    Returns
    -------
    str
        Comma-separated protected member representations.
    """
    collected = list(islice(items, _PREVIEW_ITEM_LIMIT + 1))
    has_more = len(collected) > _PREVIEW_ITEM_LIMIT
    displayed = collected[:_PREVIEW_ITEM_LIMIT]

    text = ", ".join(_safe_repr(item, limit=_ITEM_PREVIEW_LIMIT) for item in displayed)
    if has_more:
        text = f"{text}, ..."
    return text


def _scientific_value_preview(value: object) -> str | None:
    """Produce a bounded preview for a supported scientific object.

    Parameters
    ----------
    value
        Candidate NumPy, pandas, or PyTorch value.

    Returns
    -------
    str | None
        Bounded framework preview, or ``None`` for another value category.
    """
    if isinstance(value, np.ndarray):
        values = _preview_items(value.flat)
        return _truncate(
            f"ndarray(shape={value.shape}, dtype={value.dtype}, size={value.size}, values=[{values}])",
            _VALUE_PREVIEW_LIMIT,
        )

    if isinstance(value, np.generic):
        return _truncate(
            f"{type(value).__name__}(dtype={value.dtype}, value={_safe_repr(value)})",
            _VALUE_PREVIEW_LIMIT,
        )

    if isinstance(value, pd.DataFrame):
        preview = value.iloc[:_DATAFRAME_PREVIEW_ROWS, :_DATAFRAME_PREVIEW_COLUMNS]
        return _truncate(
            (
                f"DataFrame(shape={value.shape}, columns={_safe_repr(list(preview.columns))}, "
                f"dtypes={_safe_repr(list(preview.dtypes))}, values=\n{preview.to_string()})"
            ),
            _VALUE_PREVIEW_LIMIT,
        )

    if isinstance(value, (pd.Series, pd.Index)):
        values = _preview_items(value)
        return _truncate(
            f"{type(value).__name__}(length={len(value)}, dtype={value.dtype}, values=[{values}])",
            _VALUE_PREVIEW_LIMIT,
        )

    if isinstance(value, torch.Tensor):
        metadata = (
            f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
            f"device={value.device}, requires_grad={value.requires_grad}"
        )
        if value.device.type != "cpu":
            return f"{metadata}, values=<not read from non-CPU device>)"

        try:
            # ``view`` avoids copying a complete tensor solely for diagnostics.
            flat = value.detach().view(-1)
            values = _preview_items(flat[:_PREVIEW_ITEM_LIMIT].tolist())
        except Exception as error:
            values = f"<unavailable: {_type_name(type(error))}>"

        return _truncate(
            f"{metadata}, values=[{values}])",
            _VALUE_PREVIEW_LIMIT,
        )

    return None


def _value_preview(value: object) -> str:
    """Produce a framework-aware bounded value preview.

    Parameters
    ----------
    value
        Value included in a type-error diagnostic.

    Returns
    -------
    str
        Bounded value preview.

    Notes
    -----
    This function may encounter user-defined traversal or formatting behavior.
    Its caller contains those failures after the mismatch has been established.
    """
    scientific_preview = _scientific_value_preview(value)
    if scientific_preview is not None:
        return scientific_preview

    if isinstance(value, collections_abc.Mapping):
        entries = list(islice(value.items(), _PREVIEW_ITEM_LIMIT + 1))
        has_more = len(entries) > _PREVIEW_ITEM_LIMIT
        displayed = entries[:_PREVIEW_ITEM_LIMIT]

        values = ", ".join(
            (f"{_safe_repr(key, limit=_ITEM_PREVIEW_LIMIT)}: " f"{_safe_repr(item, limit=_ITEM_PREVIEW_LIMIT)}")
            for key, item in displayed
        )
        if has_more:
            values = f"{values}, ..."

        return _truncate(
            f"{type(value).__name__}(length={len(value)}, values={{{values}}})",
            _VALUE_PREVIEW_LIMIT,
        )

    if isinstance(value, (list, tuple, set, frozenset)):
        values = _preview_items(value)
        return _truncate(
            f"{type(value).__name__}(length={len(value)}, values=[{values}])",
            _VALUE_PREVIEW_LIMIT,
        )

    return _safe_repr(value)


def _safe_value_preview(value: object) -> str:
    """Generate a preview without replacing an established mismatch.

    Parameters
    ----------
    value
        Actual mismatching value.

    Returns
    -------
    str
        Bounded preview or a preview-failure marker.
    """
    try:
        return _value_preview(value)
    except Exception as error:
        return f"<value preview unavailable: {_type_name(type(error))}>"


def _format_mismatch(mismatch: _TypeMismatchError) -> str:
    """Format a complete public type-validation diagnostic.

    Parameters
    ----------
    mismatch
        Internal mismatch information.

    Returns
    -------
    str
        Public error message.
    """
    lines = [
        "Type validation failed",
        f"  path: {mismatch.path}",
        f"  expected: {_safe_annotation_text(mismatch.expected_type)}",
        f"  got type: {_type_name(type(mismatch.value))}",
        f"  got value: {_safe_value_preview(mismatch.value)}",
    ]
    if mismatch.detail is not None:
        lines.append(f"  detail: {mismatch.detail}")
    return "\n".join(lines)


def _raise_public_type_error(mismatch: _TypeMismatchError) -> NoReturn:
    """Raise the public exception for an internal type mismatch.

    Parameters
    ----------
    mismatch
        Internal mismatch information.

    Raises
    ------
    TypeError
        Always raised with the shared validation diagnostic.
    """
    raise TypeError(_format_mismatch(mismatch)) from None


def _fail(
    path: str,
    expected_type: object,
    value: object,
    detail: str | None = None,
) -> NoReturn:
    """Raise one internal type mismatch.

    Parameters
    ----------
    path
        Location of the mismatch.
    expected_type
        Failed annotation.
    value
        Mismatching value.
    detail
        Optional explanation that adds information beyond the expected and
        actual types.

    Raises
    ------
    _TypeMismatchError
        Always raised.
    """
    raise _TypeMismatchError(path, expected_type, value, detail)


def _normalize_numpy_dtype(expected_dtype: object, *, path: str) -> np.dtype[Any]:
    """Normalize one NumPy dtype specification.

    Parameters
    ----------
    expected_dtype
        Object accepted by NumPy's dtype constructor.
    path
        Diagnostic path for an invalid specification.

    Returns
    -------
    numpy.dtype
        Normalized NumPy dtype.

    Raises
    ------
    _TypeMismatchError
        If NumPy cannot interpret the dtype specification.
    """
    try:
        return cast(np.dtype[Any], np.dtype(cast(Any, expected_dtype)))
    except (TypeError, ValueError) as error:
        _fail(
            path,
            np.dtype,
            expected_dtype,
            f"invalid NumPy dtype specification: {error}",
        )


def _validate_numpy_dtype_value(
    value: np.ndarray[Any, Any] | np.generic,
    *,
    expected_dtype: object,
    field: str,
) -> None:
    """Validate one NumPy value against a dtype specification.

    Parameters
    ----------
    value
        NumPy array or scalar whose dtype is inspected.
    expected_dtype
        Object accepted by NumPy's dtype constructor.
    field
        Diagnostic path for the value.

    Raises
    ------
    _TypeMismatchError
        If the dtype specification is invalid or the dtype differs.
    """
    normalized_dtype = _normalize_numpy_dtype(expected_dtype, path="expected_dtype")
    if value.dtype != normalized_dtype:
        _fail(f"{field}.dtype", normalized_dtype, value.dtype)


def _validate_numpy_array_annotation(
    value: object,
    expected_type: object,
    path: str,
) -> None:
    """Validate a parameterized NumPy array annotation.

    Parameters
    ----------
    value
        Candidate NumPy array.
    expected_type
        Complete ``numpy.ndarray`` annotation.
    path
        Current diagnostic path.

    Raises
    ------
    _TypeMismatchError
        If the value is not an array or its declared dtype is invalid or does
        not match.
    """
    if not isinstance(value, np.ndarray):
        _fail(path, expected_type, value)

    arguments = get_args(expected_type)
    if len(arguments) != 2:
        _fail(
            path,
            expected_type,
            value,
            "NumPy array annotation must declare shape and dtype arguments",
        )

    dtype_annotation = arguments[1]
    if dtype_annotation is Any:
        return

    if get_origin(dtype_annotation) is not np.dtype:
        _fail(
            path,
            expected_type,
            value,
            "NumPy array dtype annotation must use numpy.dtype[T]",
        )

    dtype_arguments = get_args(dtype_annotation)
    if len(dtype_arguments) != 1:
        _fail(
            path,
            expected_type,
            value,
            "NumPy dtype annotation must declare one scalar type",
        )

    _validate_numpy_dtype_value(
        value,
        expected_dtype=dtype_arguments[0],
        field=path,
    )


def _mapping_path(path: str, key: object, index: int) -> str:
    """Construct a safe path for a mapping value.

    Parameters
    ----------
    path
        Parent mapping path.
    key
        Mapping key.
    index
        Key position in mapping iteration order.

    Returns
    -------
    str
        Nested mapping path.

    Notes
    -----
    A failed key representation is replaced by an index-based path and cannot
    prevent validation of the corresponding value.
    """
    key_text = _safe_repr(key, limit=_ITEM_PREVIEW_LIMIT)
    if key_text.startswith("<representation unavailable:"):
        return f"{path}.values()[{index}]"
    return f"{path}[{key_text}]"


def _literal_matches(value: object, allowed: object) -> bool:
    """Return whether a value strictly matches one literal member.

    Parameters
    ----------
    value
        Actual value.
    allowed
        Declared literal value.

    Returns
    -------
    bool
        Whether runtime type and value match strictly.

    Notes
    -----
    Equality exceptions propagate unchanged because no reliable comparison
    result exists. A non-boolean equality result is not truth-tested.
    """
    if type(value) is not type(allowed):
        return False

    equality_result = value == allowed
    return equality_result is True


def _validate_union(
    value: object,
    expected_type: object,
    path: str,
    visited: set[tuple[int, int]],
) -> None:
    """Validate a value against any member of a union.

    Parameters
    ----------
    value
        Value to validate.
    expected_type
        Union annotation.
    path
        Current diagnostic path.
    visited
        Previously visited value/annotation pairs.

    Raises
    ------
    _TypeMismatchError
        If no union member accepts the value.
    """
    for member_type in get_args(expected_type):
        # A rejected branch must not affect cycle tracking in later branches.
        try:
            _validate(value, member_type, path, visited.copy())
        except _TypeMismatchError:
            continue
        return

    _fail(path, expected_type, value)


def _validate_type_variable(
    value: object,
    expected_type: TypeVar,
    path: str,
    visited: set[tuple[int, int]],
) -> None:
    """Validate a constrained, bounded, or unrestricted TypeVar.

    Parameters
    ----------
    value
        Value to validate.
    expected_type
        TypeVar annotation.
    path
        Current diagnostic path.
    visited
        Previously visited value/annotation pairs.

    Raises
    ------
    _TypeMismatchError
        If no constraint accepts the value.
    """
    if expected_type.__constraints__:
        for constraint in expected_type.__constraints__:
            try:
                _validate(value, constraint, path, visited.copy())
            except _TypeMismatchError:
                continue
            return

        _fail(
            path,
            expected_type,
            value,
            "value does not satisfy any declared TypeVar constraint",
        )

    if expected_type.__bound__ is not None:
        _validate(value, expected_type.__bound__, path, visited)


def _class_matches(value: type[object], expected_type: object) -> bool:
    """Return whether a runtime class satisfies a ``type[T]`` constraint.

    Parameters
    ----------
    value
        Runtime class supplied as the value.
    expected_type
        Declared superclass constraint.

    Returns
    -------
    bool
        Whether the class satisfies the supported constraint.
    """
    if expected_type is Any:
        return True

    if isinstance(expected_type, TypeAliasType):
        return _class_matches(value, expected_type.__value__)

    if _is_new_type(expected_type):
        return _class_matches(value, cast(Any, expected_type).__supertype__)

    if isinstance(expected_type, TypeVar):
        if expected_type.__constraints__:
            return any(_class_matches(value, constraint) for constraint in expected_type.__constraints__)

        if expected_type.__bound__ is None:
            return True
        return _class_matches(value, expected_type.__bound__)

    origin = get_origin(expected_type)
    if origin in (Union, UnionType):
        return any(_class_matches(value, member) for member in get_args(expected_type))

    return isinstance(expected_type, type) and issubclass(value, expected_type)


def _validate_class_value(
    value: object,
    expected_type: object,
    path: str,
) -> None:
    """Validate a runtime class against a ``type[T]`` annotation.

    Parameters
    ----------
    value
        Candidate runtime class.
    expected_type
        Complete ``type[T]`` annotation.
    path
        Current diagnostic path.

    Raises
    ------
    _TypeMismatchError
        If the value is not a class or violates its superclass constraint.
    """
    if not isinstance(value, type):
        _fail(path, expected_type, value)

    if not _class_matches(value, get_args(expected_type)[0]):
        _fail(
            path,
            expected_type,
            value,
            "class does not satisfy the declared superclass constraint",
        )


def _validate_alias_or_variable(
    value: object,
    expected_type: object,
    path: str,
    visited: set[tuple[int, int]],
) -> bool:
    """Validate an alias, NewType, or TypeVar when applicable.

    Parameters
    ----------
    value
        Value to validate.
    expected_type
        Candidate alias or variable annotation.
    path
        Current diagnostic path.
    visited
        Previously visited value/annotation pairs.

    Returns
    -------
    bool
        Whether this helper recognized and validated the annotation.
    """
    if isinstance(expected_type, TypeAliasType):
        _validate(value, expected_type.__value__, path, visited)
        return True

    if _is_new_type(expected_type):
        _validate(
            value,
            cast(Any, expected_type).__supertype__,
            path,
            visited,
        )
        return True

    if isinstance(expected_type, TypeVar):
        _validate_type_variable(value, expected_type, path, visited)
        return True

    return False


def _validate_special(
    value: object,
    expected_type: object,
    path: str,
    visited: set[tuple[int, int]],
) -> bool:
    """Validate aliases and special typing constructs when applicable.

    Parameters
    ----------
    value
        Value to validate.
    expected_type
        Candidate special annotation.
    path
        Current diagnostic path.
    visited
        Previously visited value/annotation pairs.

    Returns
    -------
    bool
        Whether this helper recognized and validated the annotation.
    """
    if _validate_alias_or_variable(value, expected_type, path, visited):
        return True

    origin = get_origin(expected_type)
    arguments = get_args(expected_type)

    if origin in (Required, NotRequired, Annotated):
        _validate(value, arguments[0], path, visited)
        return True

    if origin in (Union, UnionType):
        _validate_union(value, expected_type, path, visited)
        return True

    if origin is Literal:
        if not any(_literal_matches(value, allowed) for allowed in arguments):
            _fail(path, expected_type, value)
        return True

    if origin is type:
        _validate_class_value(value, expected_type, path)
        return True

    if origin is collections_abc.Callable:
        if not callable(value):
            _fail(path, expected_type, value)
        return True

    return False


def _validate_typed_dictionary(
    value: object,
    expected_type: object,
    path: str,
    visited: set[tuple[int, int]],
) -> None:
    """Validate a dictionary against a typed-dictionary definition.

    Parameters
    ----------
    value
        Candidate dictionary.
    expected_type
        Typed-dictionary class.
    path
        Current diagnostic path.
    visited
        Previously visited value/annotation pairs.

    Raises
    ------
    _TypeMismatchError
        If keys or nested values do not satisfy the definition.
    """
    if not isinstance(value, dict):
        _fail(path, expected_type, value)

    annotations = _get_structured_type_hints(cast(collections_abc.Hashable, expected_type))
    # Derive required keys from resolved hints because postponed annotations
    # can make TypedDict.__required_keys__ misclassify NotRequired fields.
    total = bool(getattr(expected_type, "__total__", True))
    required_keys: list[str] = []
    for key, field_type in annotations.items():
        origin = get_origin(field_type)
        if origin is Required or (origin is not NotRequired and total):
            required_keys.append(key)
    required_keys.sort()

    missing_keys = [key for key in required_keys if key not in value]
    if missing_keys:
        _fail(
            path,
            expected_type,
            value,
            f"missing required keys: {missing_keys!r}",
        )

    unexpected_keys = [key for key in value if type(key) is not str or key not in annotations]
    if unexpected_keys:
        keys_text = ", ".join(_safe_repr(key, limit=_ITEM_PREVIEW_LIMIT) for key in unexpected_keys)
        _fail(
            path,
            expected_type,
            value,
            f"unexpected keys: [{keys_text}]",
        )

    for key, field_type in annotations.items():
        if key in value:
            _validate(value[key], field_type, f"{path}[{key!r}]", visited)


def _validate_structured_class(
    value: object,
    expected_type: object,
    path: str,
    visited: set[tuple[int, int]],
) -> bool:
    """Validate typed dictionaries, named tuples, and dataclasses.

    Parameters
    ----------
    value
        Value to validate.
    expected_type
        Candidate structural class.
    path
        Current diagnostic path.
    visited
        Previously visited value/annotation pairs.

    Returns
    -------
    bool
        Whether this helper recognized and validated the annotation.

    Notes
    -----
    Attribute-access exceptions propagate unchanged. Converting them into type
    mismatches would imply that the field value was read successfully.
    """
    if is_typeddict(expected_type):
        _validate_typed_dictionary(value, expected_type, path, visited)
        return True

    is_named_tuple = (
        isinstance(expected_type, type)
        and tuple in getattr(expected_type, "__mro__", ())
        and hasattr(expected_type, "_fields")
        and hasattr(expected_type, "__annotations__")
    )
    is_dataclass_type = isinstance(expected_type, type) and is_dataclass(expected_type)

    if not is_named_tuple and not is_dataclass_type:
        return False

    runtime_type = cast(type[object], expected_type)
    if not isinstance(value, runtime_type):
        _fail(path, expected_type, value)

    annotations = _get_structured_type_hints(cast(collections_abc.Hashable, expected_type))
    field_types = (
        annotations
        if is_named_tuple
        else {
            field.name: annotations[field.name]
            for field in fields(cast(Any, runtime_type))
            if field.name in annotations
        }
    )

    for field_name, field_type in field_types.items():
        _validate(getattr(value, field_name), field_type, f"{path}.{field_name}", visited)

    return True


@lru_cache(maxsize=256)
def _get_structured_type_hints(
    expected_type: collections_abc.Hashable,
) -> dict[str, object]:
    """Resolve and cache annotations for one structured runtime type.

    Parameters
    ----------
    expected_type
        Typed dictionary, named tuple, or dataclass type.

    Returns
    -------
    dict[str, object]
        Resolved field annotations, including ``Annotated`` metadata.

    Notes
    -----
    The bounded cache avoids repeated annotation resolution while limiting
    retention of dynamically created structured classes.
    """
    return get_type_hints(expected_type, include_extras=True)


def _validate_tuple(
    value: object,
    expected_type: object,
    path: str,
    visited: set[tuple[int, int]],
) -> None:
    """Validate fixed, variadic, or empty tuple annotations.

    Parameters
    ----------
    value
        Candidate tuple.
    expected_type
        Tuple annotation.
    path
        Current diagnostic path.
    visited
        Previously visited value/annotation pairs.

    Raises
    ------
    _TypeMismatchError
        If tuple structure or members are incompatible.
    """
    if not isinstance(value, tuple):
        _fail(path, expected_type, value)

    arguments = get_args(expected_type)
    if not arguments:
        if value:
            _fail(path, expected_type, value, "expected an empty tuple")
        return

    if len(arguments) == 2 and arguments[1] is Ellipsis:
        for index, item in enumerate(value):
            _validate(item, arguments[0], f"{path}[{index}]", visited)
        return

    if len(value) != len(arguments):
        _fail(
            path,
            expected_type,
            value,
            f"fixed tuple length mismatch: expected {len(arguments)}, got {len(value)}",
        )

    for index, (item, item_type) in enumerate(zip(value, arguments, strict=True)):
        _validate(item, item_type, f"{path}[{index}]", visited)


def _validate_mapping(
    value: object,
    expected_type: object,
    path: str,
    visited: set[tuple[int, int]],
) -> None:
    """Validate mapping key and value types.

    Parameters
    ----------
    value
        Candidate mapping.
    expected_type
        Mapping annotation.
    path
        Current diagnostic path.
    visited
        Previously visited value/annotation pairs.

    Raises
    ------
    _TypeMismatchError
        If mapping structure, keys, or values are incompatible.
    """
    if not isinstance(value, collections_abc.Mapping):
        _fail(path, expected_type, value)

    arguments = get_args(expected_type)
    if len(arguments) != 2:
        _fail(
            path,
            expected_type,
            value,
            "mapping annotation must declare key and value types",
        )

    key_type, item_type = arguments
    mapping = cast(collections_abc.Mapping[object, object], value)

    for index, (key, item) in enumerate(mapping.items()):
        _validate(key, key_type, f"{path}.keys()[{index}]", visited)
        _validate(item, item_type, _mapping_path(path, key, index), visited)


def _validate_collection(
    value: object,
    expected_type: object,
    origin: object,
    path: str,
    visited: set[tuple[int, int]],
) -> None:
    """Validate a homogeneous collection annotation.

    Parameters
    ----------
    value
        Candidate collection.
    expected_type
        Complete collection annotation.
    origin
        Concrete or abstract collection origin.
    path
        Current diagnostic path.
    visited
        Previously visited value/annotation pairs.

    Raises
    ------
    _TypeMismatchError
        If collection structure or members are incompatible.
    """
    runtime_types: dict[object, type[object]] = {
        list: list,
        set: set,
        frozenset: frozenset,
        collections_abc.Sequence: collections_abc.Sequence,
        collections_abc.Set: collections_abc.Set,
    }
    runtime_type = runtime_types[origin]

    if not isinstance(value, runtime_type):
        _fail(path, expected_type, value)

    arguments = get_args(expected_type)
    if len(arguments) != 1:
        _fail(
            path,
            expected_type,
            value,
            "collection annotation must declare one member type",
        )

    items = cast(collections_abc.Iterable[object], value)
    if origin in (set, frozenset, collections_abc.Set):
        # Protected representations provide stable ordinary traversal without
        # allowing a broken repr to abort validation.
        items = sorted(
            items,
            key=lambda item: (
                _type_name(type(item)),
                _safe_repr(item, limit=_ITEM_PREVIEW_LIMIT),
            ),
        )

    for index, item in enumerate(items):
        _validate(item, arguments[0], f"{path}[{index}]", visited)


def _validate_container(
    value: object,
    expected_type: object,
    path: str,
    visited: set[tuple[int, int]],
) -> bool:
    """Validate supported parameterized containers when applicable.

    Parameters
    ----------
    value
        Value to validate.
    expected_type
        Candidate container annotation.
    path
        Current diagnostic path.
    visited
        Previously visited value/annotation pairs.

    Returns
    -------
    bool
        Whether this helper recognized and validated the annotation.
    """
    origin = get_origin(expected_type)

    if origin is np.ndarray:
        _validate_numpy_array_annotation(value, expected_type, path)
        return True

    if origin is tuple:
        _validate_tuple(value, expected_type, path, visited)
        return True

    if origin in (
        list,
        set,
        frozenset,
        collections_abc.Sequence,
        collections_abc.Set,
    ):
        _validate_collection(value, expected_type, origin, path, visited)
        return True

    if origin in (dict, collections_abc.Mapping):
        _validate_mapping(value, expected_type, path, visited)
        return True

    return False


def _validate_runtime_class(
    value: object,
    expected_type: object,
    path: str,
) -> None:
    """Validate an ordinary runtime class annotation.

    Parameters
    ----------
    value
        Value to validate.
    expected_type
        Expected runtime class.
    path
        Current diagnostic path.

    Raises
    ------
    _TypeMismatchError
        If the annotation is unsupported or the value has the wrong type.

    Notes
    -----
    Exceptions from custom ``__instancecheck__`` behavior propagate unchanged.
    """
    if not isinstance(expected_type, type):
        _fail(
            path,
            expected_type,
            value,
            "unsupported runtime annotation",
        )

    if any(expected_type is scalar_type for scalar_type in _STRICT_SCALAR_TYPES):
        valid = type(value) is expected_type
    else:
        is_protocol = bool(getattr(expected_type, "_is_protocol", False))
        is_runtime_protocol = bool(getattr(expected_type, "_is_runtime_protocol", False))
        if is_protocol and not is_runtime_protocol:
            _fail(
                path,
                expected_type,
                value,
                "protocol is not runtime-checkable",
            )

        valid = isinstance(value, expected_type)

    if not valid:
        _fail(path, expected_type, value)


def _validate(
    value: object,
    expected_type: object,
    path: str,
    visited: set[tuple[int, int]],
) -> None:
    """Recursively validate one value against one supported annotation.

    Parameters
    ----------
    value
        Value to validate.
    expected_type
        Expected runtime annotation.
    path
        Current diagnostic path.
    visited
        Previously visited value/annotation pairs used for cycle protection.

    Raises
    ------
    _TypeMismatchError
        If the value does not satisfy the annotation.
    """
    if expected_type is Any:
        return

    if expected_type is None:
        expected_type = NoneType

    pair = (id(value), id(expected_type))
    if pair in visited:
        return
    visited.add(pair)

    if _validate_special(value, expected_type, path, visited):
        return

    if _validate_structured_class(value, expected_type, path, visited):
        return

    if _validate_container(value, expected_type, path, visited):
        return

    _validate_runtime_class(value, expected_type, path)


def validate_declared_types(
    value: object,
    *,
    expected_type: object,
    field: str = "value",
) -> None:
    """Validate a value against a supported runtime type annotation.

    Validation is strict and does not coerce, reconstruct, normalize, repair,
    consume, or mutate the supplied value. Nested stable structures are
    inspected recursively. Type failures identify the nested path, expected
    annotation, actual qualified type, and a bounded preview of the actual
    value.

    Operations required to inspect user-defined values, including equality,
    attribute access, container traversal, and custom runtime type checks, may
    raise their original exceptions. Those exceptions are not converted into
    ordinary type mismatches because the corresponding check could not be
    completed reliably.

    Parameters
    ----------
    value
        Value to validate.
    expected_type
        Runtime annotation the value must satisfy.
    field
        Initial path used in validation diagnostics.

    Raises
    ------
    TypeError
        If the value does not satisfy the annotation, the annotation is
        unsupported, or ``field`` is not a string.

    Notes
    -----
    Recursive container and structured annotations validate every reachable
    declared member, and set-like collections are sorted to produce a stable
    failure path. Repeated validation of very large Python containers or use
    inside per-sample, per-row, or per-batch loops can therefore add material
    overhead. Dataset-scale content rules should use the applicable vectorized
    NumPy, pandas, or PyTorch operations.
    Node usage scope: Shared - Used by both root and worker nodes.
    """
    try:
        if type(field) is not str:
            _fail("field", str, field)

        _validate(value, expected_type, field, set())
    except _TypeMismatchError as mismatch:
        _raise_public_type_error(mismatch)


def validate_numpy_dtype(
    value: np.ndarray[Any, Any] | np.generic,
    *,
    expected_dtype: object,
    field: str = "value",
) -> None:
    """Require a NumPy array or scalar to have the expected dtype.

    The expected dtype is normalized with ``numpy.dtype``. Validation does
    not cast, copy, or mutate the supplied value.

    Parameters
    ----------
    value
        NumPy array or scalar whose dtype is validated.
    expected_dtype
        Object accepted by NumPy's dtype constructor.
    field
        Initial path used in validation diagnostics.

    Raises
    ------
    TypeError
        If the value is not a NumPy array or scalar, the dtype specification
        is invalid, or the value has a different dtype.

    Notes
    -----
    Node usage scope: Shared - Used by both root and worker nodes.
    """
    try:
        validate_declared_types(
            value,
            expected_type=np.ndarray | np.generic,
            field=field,
        )
        _validate_numpy_dtype_value(
            value,
            expected_dtype=expected_dtype,
            field=field,
        )
    except _TypeMismatchError as mismatch:
        _raise_public_type_error(mismatch)


def _normalize_pandas_dtype(expected_dtype: object, *, path: str) -> object:
    """Normalize one pandas dtype specification.

    Parameters
    ----------
    expected_dtype
        Object accepted by pandas' dtype normalization API.
    path
        Diagnostic path for an invalid specification.

    Returns
    -------
    object
        Normalized NumPy or pandas extension dtype.

    Raises
    ------
    _TypeMismatchError
        If pandas cannot interpret the dtype specification.
    """
    try:
        return pd.api.types.pandas_dtype(expected_dtype)
    except TypeError as error:
        _fail(
            path,
            "pandas-compatible dtype",
            expected_dtype,
            f"invalid pandas dtype specification: {error}",
        )


def _validate_pandas_dtype_value(
    value: pd.Series[Any] | pd.Index[Any] | pd.DataFrame,
    *,
    expected_dtype: object,
    field: str,
) -> None:
    """Validate one pandas object against a dtype specification.

    Parameters
    ----------
    value
        Series, index, or dataframe whose dtype is inspected.
    expected_dtype
        Object accepted by pandas' dtype normalization API.
    field
        Diagnostic path for the pandas object.

    Raises
    ------
    _TypeMismatchError
        If the dtype specification is invalid or an inspected dtype differs.
    """
    normalized_dtype = _normalize_pandas_dtype(
        expected_dtype,
        path="expected_dtype",
    )

    if isinstance(value, pd.DataFrame):
        # Dataframes may contain multiple columns, so report the stable
        # positional dtype path of the first mismatching column.
        for index, actual_dtype in enumerate(value.dtypes):
            if actual_dtype != normalized_dtype:
                _fail(
                    f"{field}.dtypes[{index}]",
                    normalized_dtype,
                    actual_dtype,
                )
        return

    if value.dtype != normalized_dtype:
        _fail(f"{field}.dtype", normalized_dtype, value.dtype)


def validate_pandas_dtype(
    value: pd.Series[Any] | pd.Index[Any] | pd.DataFrame,
    *,
    expected_dtype: object,
    field: str = "value",
) -> None:
    """Require a pandas series, index, or dataframe to have the expected dtype.

    The expected dtype is normalized with pandas' dtype API. For a dataframe,
    every column must have the normalized dtype. Validation does not cast,
    copy, or mutate the supplied object. Callers requiring heterogeneous
    dataframe schemas validate each selected series separately with a
    column-specific field path.

    Parameters
    ----------
    value
        Series, index, or dataframe whose dtype is validated.
    expected_dtype
        Object accepted by pandas' dtype normalization API.
    field
        Initial path used in validation diagnostics.

    Raises
    ------
    TypeError
        If the value is not a supported pandas object, the dtype specification
        is invalid, or an inspected dtype differs.

    Notes
    -----
    Dataframe validation checks every column until the first mismatch. Calling
    this helper repeatedly for very wide dataframes can add overhead
    proportional to the number of inspected columns. A heterogeneous schema
    should validate only its selected series with column-specific calls.
    Node usage scope: Shared - Used by both root and worker nodes.
    """
    try:
        validate_declared_types(
            value,
            expected_type=pd.Series | pd.Index | pd.DataFrame,
            field=field,
        )
        _validate_pandas_dtype_value(
            value,
            expected_dtype=expected_dtype,
            field=field,
        )
    except _TypeMismatchError as mismatch:
        _raise_public_type_error(mismatch)


def validate_torch_dtype(
    value: torch.Tensor,
    *,
    expected_dtype: torch.dtype,
    field: str = "value",
) -> None:
    """Require a PyTorch tensor to have the expected dtype.

    Validation compares the tensor's dtype directly. It does not cast, copy,
    detach, mutate, or move the tensor between devices.

    Parameters
    ----------
    value
        PyTorch tensor whose dtype is validated.
    expected_dtype
        Required PyTorch dtype.
    field
        Initial path used in validation diagnostics.

    Raises
    ------
    TypeError
        If ``value`` is not a tensor, ``expected_dtype`` is not a PyTorch
        dtype, or the tensor has a different dtype.

    Notes
    -----
    Node usage scope: Shared - Used by both root and worker nodes.
    """
    try:
        validate_declared_types(
            value,
            expected_type=torch.Tensor,
            field=field,
        )
        validate_declared_types(
            expected_dtype,
            expected_type=torch.dtype,
            field="expected_dtype",
        )

        if value.dtype != expected_dtype:
            _fail(f"{field}.dtype", expected_dtype, value.dtype)
    except _TypeMismatchError as mismatch:
        _raise_public_type_error(mismatch)


def validate_args(
    func: collections_abc.Callable[_Parameters, _ReturnType],
) -> collections_abc.Callable[_Parameters, _ReturnType]:
    """Validate a callable's annotated input arguments at runtime.

    The callable signature, resolved type hints, and annotated parameter plan
    are cached when the decorator is applied. Each annotated input is
    delegated to ``validate_declared_types`` before the wrapped body executes.
    The decorator performs no independent type matching and does not validate
    the return value.

    Parameters
    ----------
    func
        Callable whose annotated input arguments are validated.

    Returns
    -------
    collections.abc.Callable
        Wrapped callable with its original parameter and return typing.

    Raises
    ------
    TypeError
        If a supplied or defaulted annotated argument does not satisfy its
        resolved annotation, or if normal Python signature binding fails.

    Notes
    -----
    Every invocation performs signature binding and validates all annotated
    inputs. Decorated functions with large recursive container arguments, or
    functions called once per sample, row, batch, prediction, or metric
    element, may incur material overhead. This decorator is intended for API,
    configuration, and contract-validation boundaries rather than hot
    computational loops.
    Node usage scope: Shared - Used by both root and worker nodes.
    """
    signature = inspect.signature(func)
    type_hints = get_type_hints(func, include_extras=True)
    validation_plan = tuple(
        (
            name,
            parameter.kind,
            type_hints[name],
            parameter.default,
        )
        for name, parameter in signature.parameters.items()
        if name in type_hints
    )

    @wraps(func)
    def wrapper(
        *args: _Parameters.args,
        **kwargs: _Parameters.kwargs,
    ) -> _ReturnType:
        """Bind and validate one invocation before calling the wrapped body."""
        bound_arguments = signature.bind(*args, **kwargs)

        for name, parameter_kind, expected_type, default in validation_plan:
            if name in bound_arguments.arguments:
                argument = bound_arguments.arguments[name]
            elif default is not inspect.Parameter.empty:
                argument = default
            else:
                # An absent variadic parameter contains no values to validate.
                continue

            if parameter_kind is inspect.Parameter.VAR_POSITIONAL:
                positional_values = cast(tuple[object, ...], argument)
                for index, item in enumerate(positional_values):
                    validate_declared_types(
                        item,
                        expected_type=expected_type,
                        field=f"{name}[{index}]",
                    )
                continue

            if parameter_kind is inspect.Parameter.VAR_KEYWORD:
                keyword_values = cast(dict[str, object], argument)
                for key, item in keyword_values.items():
                    validate_declared_types(
                        item,
                        expected_type=expected_type,
                        field=f"{name}[{key!r}]",
                    )
                continue

            validate_declared_types(
                argument,
                expected_type=expected_type,
                field=name,
            )

        return func(*args, **kwargs)

    return wrapper
