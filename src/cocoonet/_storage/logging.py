"""Structured JSON Lines records shared by Cocoonet storage."""

import asyncio
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NoReturn
from weakref import WeakKeyDictionary

import aiofiles

from cocoonet._contracts.failure import (
    FailureRecord,
    FailureScope,
    TraceContext,
    validate_failure_record,
    validate_trace_context,
)
from cocoonet._contracts.identity import (
    AttemptId,
    FailureId,
    FoldId,
    ModelId,
    OperationId,
    ParentId,
    ReportName,
    SampleId,
    WorkerId,
)
from cocoonet._storage.metadata import NodeRole
from cocoonet._storage.paths import RootStorageLayout
from cocoonet._storage.publication import StorageError
from cocoonet._type_validation import validate_declared_types


class LogChannel(StrEnum):
    """Identify the destination category of one structured log record.

    Node usage scope: Shared - Used by both root and worker logging.
    """

    RUNNING = "running"
    ERROR = "error"


type LogScalar = None | bool | int | float | str


@dataclass(frozen=True, slots=True)
class StructuredLogRecord:
    """Record one structured running or error event.

    Parameters
    ----------
    timestamp_utc_ns
        UTC timestamp represented as integer nanoseconds.
    node_role
        Role of the node producing the record.
    channel
        Running or error log channel.
    event
        Event name owned by the phase producing the record.
    message
        Human-readable event description.
    context
        Applicable scientific or operational trace coordinates.
    failure
        Structured failure information when the event records a failure.
    attributes
        Ordered unique scalar event attributes.

    Notes
    -----
    Phase 2 defines the storage representation but does not define a catalogue
    of event names or required event-specific attributes. Construction does not
    validate, encode, route, or append the record.

    Node usage scope: Shared - Used by both root and worker logging.
    """

    timestamp_utc_ns: int
    node_role: NodeRole
    channel: LogChannel
    event: str
    message: str
    context: TraceContext | None
    failure: FailureRecord | None
    attributes: tuple[tuple[str, LogScalar], ...]


_LOG_RECORD_KEYS = (
    "timestamp_utc_ns",
    "node_role",
    "channel",
    "event",
    "message",
    "context",
    "failure",
    "attributes",
)

_TRACE_CONTEXT_KEYS = (
    "stage",
    "parent_id",
    "model_id",
    "attempt_id",
    "worker_id",
    "fold_id",
    "sample_id",
    "class_label",
    "report_name",
    "operation_id",
)

_FAILURE_KEYS = (
    "failure_id",
    "scope",
    "context",
    "exception_type",
    "message",
    "traceback",
    "origin_failure_id",
)

_ROOT_LOG_LOCKS: WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[Path, asyncio.Lock],
] = WeakKeyDictionary()


def validate_structured_log_record(  # noqa: C901
    value: StructuredLogRecord,
    /,
) -> None:
    """Validate one structured log record.

    Parameters
    ----------
    value
        Candidate structured log record.

    Raises
    ------
    TypeError
        If the record or a field has an incorrect runtime type.
    ValueError
        If an attribute key is empty or duplicated, an attribute float is
        nonfinite, nested context or failure data is invalid, or supplied
        context and failure coordinates conflict.

    Notes
    -----
    This validator does not define event names, require event-specific
    attributes, capture exceptions, route records, or perform file I/O.

    Node usage scope: Shared - Used by both root and worker logging.
    """
    if not isinstance(value, StructuredLogRecord):
        validate_declared_types(
            value,
            expected_type=StructuredLogRecord,
        )

    validate_declared_types(
        value.timestamp_utc_ns,
        expected_type=int,
        field="timestamp_utc_ns",
    )
    validate_declared_types(
        value.node_role,
        expected_type=NodeRole,
        field="node_role",
    )
    validate_declared_types(
        value.channel,
        expected_type=LogChannel,
        field="channel",
    )
    validate_declared_types(
        value.event,
        expected_type=str,
        field="event",
    )
    validate_declared_types(
        value.message,
        expected_type=str,
        field="message",
    )
    validate_declared_types(
        value.attributes,
        expected_type=tuple[tuple[str, LogScalar], ...],
        field="attributes",
    )

    if value.context is not None:
        validate_trace_context(value.context)

    if value.failure is not None:
        validate_failure_record(value.failure)

    seen_attribute_keys: set[str] = set()
    for index, (key, attribute_value) in enumerate(value.attributes):
        if not key:
            raise ValueError(f"attributes[{index}][0] must be a nonempty string; " "got an empty string.")
        if key in seen_attribute_keys:
            raise ValueError(f"attributes[{index}][0] must be unique; " f"got duplicate {key!r}.")
        seen_attribute_keys.add(key)

        if isinstance(attribute_value, float) and not math.isfinite(attribute_value):
            raise ValueError(f"attributes[{index}][1] must be finite; " f"got {attribute_value!r}.")

    if value.context is None or value.failure is None:
        return

    context_pairs = (
        (
            "stage",
            value.context.stage,
            value.failure.context.stage,
        ),
        (
            "parent_id",
            value.context.parent_id,
            value.failure.context.parent_id,
        ),
        (
            "model_id",
            value.context.model_id,
            value.failure.context.model_id,
        ),
        (
            "attempt_id",
            value.context.attempt_id,
            value.failure.context.attempt_id,
        ),
        (
            "worker_id",
            value.context.worker_id,
            value.failure.context.worker_id,
        ),
        (
            "fold_id",
            value.context.fold_id,
            value.failure.context.fold_id,
        ),
        (
            "sample_id",
            value.context.sample_id,
            value.failure.context.sample_id,
        ),
        (
            "class_label",
            value.context.class_label,
            value.failure.context.class_label,
        ),
        (
            "report_name",
            value.context.report_name,
            value.failure.context.report_name,
        ),
        (
            "operation_id",
            value.context.operation_id,
            value.failure.context.operation_id,
        ),
    )

    for field, context_value, failure_value in context_pairs:
        if context_value is not None and failure_value is not None and context_value != failure_value:
            raise ValueError(
                f"context.{field} must agree with failure.context.{field}; "
                f"got {context_value!r} and {failure_value!r}."
            )


def _trace_context_payload(
    value: TraceContext,
    /,
) -> dict[str, object]:
    """Return one deterministic JSON-compatible trace-context object."""
    return {
        "stage": value.stage,
        "parent_id": value.parent_id,
        "model_id": value.model_id,
        "attempt_id": value.attempt_id,
        "worker_id": value.worker_id,
        "fold_id": value.fold_id,
        "sample_id": value.sample_id,
        "class_label": value.class_label,
        "report_name": (None if value.report_name is None else value.report_name.value),
        "operation_id": value.operation_id,
    }


def _failure_payload(
    value: FailureRecord,
    /,
) -> dict[str, object]:
    """Return one deterministic JSON-compatible failure object."""
    return {
        "failure_id": value.failure_id,
        "scope": value.scope.value,
        "context": _trace_context_payload(value.context),
        "exception_type": value.exception_type,
        "message": value.message,
        "traceback": value.traceback,
        "origin_failure_id": value.origin_failure_id,
    }


def encode_structured_log_record(
    value: StructuredLogRecord,
    /,
) -> bytes:
    """Encode one validated record as one compact UTF-8 JSON line.

    Parameters
    ----------
    value
        Structured log record to validate and encode.

    Returns
    -------
    bytes
        One compact UTF-8 JSON object followed by exactly one LF character.

    Raises
    ------
    TypeError
        If the record or a field has an incorrect runtime type.
    ValueError
        If the record is invalid or contains a nonfinite number.

    Notes
    -----
    JSON object order follows the fixed schema and supplied attribute order.
    Non-ASCII text is preserved rather than escaped.

    This function performs no file I/O, routing, staging, or append operation.

    Node usage scope: Shared - Used by both root and worker logging.
    """
    validate_structured_log_record(value)

    payload: dict[str, object] = {
        "timestamp_utc_ns": value.timestamp_utc_ns,
        "node_role": value.node_role.value,
        "channel": value.channel.value,
        "event": value.event,
        "message": value.message,
        "context": (None if value.context is None else _trace_context_payload(value.context)),
        "failure": (None if value.failure is None else _failure_payload(value.failure)),
        "attributes": dict(value.attributes),
    }

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return f"{encoded}\n".encode("utf-8")


def _json_object_from_pairs(
    pairs: list[tuple[str, object]],
    /,
) -> dict[str, object]:
    """Build one JSON object while rejecting duplicate keys."""
    result: dict[str, object] = {}

    for key, value in pairs:
        if key in result:
            raise ValueError(f"Structured log JSON contains duplicate key {key!r}.")
        result[key] = value

    return result


def _reject_json_constant(value: str, /) -> NoReturn:
    """Reject nonstandard nonfinite JSON numeric constants."""
    raise ValueError(f"Structured log JSON must not contain nonfinite constant {value!r}.")


def _require_json_object(
    value: object,
    *,
    field: str,
) -> dict[str, object]:
    """Return a JSON object or reject another JSON value."""
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object; " f"got {type(value).__module__}.{type(value).__qualname__}.")

    for key in value:
        if not isinstance(key, str):
            raise ValueError(f"{field} must contain only string keys.")

    return value


def _require_exact_keys(
    value: dict[str, object],
    expected: tuple[str, ...],
    *,
    field: str,
) -> None:
    """Require exact deterministic JSON object keys and order."""
    observed = tuple(value)
    if observed != expected:
        raise ValueError(f"{field} must contain keys {expected!r} in order; " f"got {observed!r}.")


def _required_string(
    value: object,
    *,
    field: str,
) -> str:
    """Return one required JSON string."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a JSON string; " f"got {type(value).__module__}.{type(value).__qualname__}.")
    return value


def _optional_string(
    value: object,
    *,
    field: str,
) -> str | None:
    """Return one optional JSON string."""
    if value is None:
        return None
    return _required_string(value, field=field)


def _decode_log_scalar(
    value: object,
    *,
    field: str,
) -> LogScalar:
    """Decode one JSON scalar accepted by the structured-log schema."""
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{field} must be finite; got {value!r}.")
        return value

    raise ValueError(
        f"{field} must be null, bool, int, float, or string; "
        f"got {type(value).__module__}.{type(value).__qualname__}."
    )


def _decode_trace_context(
    value: object,
    /,
) -> TraceContext | None:
    """Decode one optional trace-context JSON object."""
    if value is None:
        return None

    payload = _require_json_object(value, field="context")
    _require_exact_keys(
        payload,
        _TRACE_CONTEXT_KEYS,
        field="context",
    )

    report_name_text = _optional_string(
        payload["report_name"],
        field="context.report_name",
    )
    try:
        report_name = None if report_name_text is None else ReportName(report_name_text)
    except ValueError as error:
        raise ValueError(f"context.report_name has unknown value {report_name_text!r}.") from error

    return TraceContext(
        stage=_required_string(
            payload["stage"],
            field="context.stage",
        ),
        parent_id=(
            None
            if (
                text := _optional_string(
                    payload["parent_id"],
                    field="context.parent_id",
                )
            )
            is None
            else ParentId(text)
        ),
        model_id=(
            None
            if (
                text := _optional_string(
                    payload["model_id"],
                    field="context.model_id",
                )
            )
            is None
            else ModelId(text)
        ),
        attempt_id=(
            None
            if (
                text := _optional_string(
                    payload["attempt_id"],
                    field="context.attempt_id",
                )
            )
            is None
            else AttemptId(text)
        ),
        worker_id=(
            None
            if (
                text := _optional_string(
                    payload["worker_id"],
                    field="context.worker_id",
                )
            )
            is None
            else WorkerId(text)
        ),
        fold_id=(
            None
            if (
                text := _optional_string(
                    payload["fold_id"],
                    field="context.fold_id",
                )
            )
            is None
            else FoldId(text)
        ),
        sample_id=(
            None
            if (
                text := _optional_string(
                    payload["sample_id"],
                    field="context.sample_id",
                )
            )
            is None
            else SampleId(text)
        ),
        class_label=_optional_string(
            payload["class_label"],
            field="context.class_label",
        ),
        report_name=report_name,
        operation_id=(
            None
            if (
                text := _optional_string(
                    payload["operation_id"],
                    field="context.operation_id",
                )
            )
            is None
            else OperationId(text)
        ),
    )


def _decode_failure(
    value: object,
    /,
) -> FailureRecord | None:
    """Decode one optional failure JSON object."""
    if value is None:
        return None

    payload = _require_json_object(value, field="failure")
    _require_exact_keys(
        payload,
        _FAILURE_KEYS,
        field="failure",
    )

    context = _decode_trace_context(payload["context"])
    if context is None:
        raise ValueError("failure.context must be a JSON object; got null.")

    scope_text = _required_string(
        payload["scope"],
        field="failure.scope",
    )
    try:
        scope = FailureScope(scope_text)
    except ValueError as error:
        raise ValueError(f"failure.scope has unknown value {scope_text!r}.") from error

    origin_text = _optional_string(
        payload["origin_failure_id"],
        field="failure.origin_failure_id",
    )

    return FailureRecord(
        failure_id=FailureId(
            _required_string(
                payload["failure_id"],
                field="failure.failure_id",
            )
        ),
        scope=scope,
        context=context,
        exception_type=_required_string(
            payload["exception_type"],
            field="failure.exception_type",
        ),
        message=_required_string(
            payload["message"],
            field="failure.message",
        ),
        traceback=_required_string(
            payload["traceback"],
            field="failure.traceback",
        ),
        origin_failure_id=(None if origin_text is None else FailureId(origin_text)),
    )


def decode_structured_log_record(
    data: bytes,
    /,
) -> StructuredLogRecord:
    """Decode one complete structured UTF-8 JSON log line.

    Parameters
    ----------
    data
        One complete LF-terminated structured log record.

    Returns
    -------
    StructuredLogRecord
        Reconstructed and validated structured record.

    Raises
    ------
    TypeError
        If ``data`` is not bytes.
    ValueError
        If encoding, line boundaries, JSON syntax, object schema, nested
        records, attributes, or context agreement is invalid.

    Notes
    -----
    This diagnostic boundary accepts exactly one complete line. It does not
    read a log file, repair or ignore an incomplete line, route a record, or
    perform file I/O.

    Node usage scope: Shared - Used by both root and worker logging.
    """
    validate_declared_types(
        data,
        expected_type=bytes,
        field="data",
    )

    if data.startswith(b"\xef\xbb\xbf"):
        raise ValueError("Structured log data must not contain a UTF-8 BOM.")

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Structured log data must contain valid UTF-8.") from error

    if not text.endswith("\n") or text.count("\n") != 1:
        raise ValueError("Structured log data must contain exactly one complete LF-terminated line.")

    body = text[:-1]
    if not body or "\r" in body:
        raise ValueError("Structured log data must contain one nonempty JSON object with LF line ending.")

    try:
        decoded = json.loads(
            body,
            object_pairs_hook=_json_object_from_pairs,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("Structured log data contains malformed JSON.") from error

    payload = _require_json_object(decoded, field="record")
    _require_exact_keys(
        payload,
        _LOG_RECORD_KEYS,
        field="record",
    )

    timestamp = payload["timestamp_utc_ns"]
    if type(timestamp) is not int:
        raise ValueError(
            "record.timestamp_utc_ns must be a JSON integer; "
            f"got {type(timestamp).__module__}.{type(timestamp).__qualname__}."
        )

    node_role_text = _required_string(
        payload["node_role"],
        field="record.node_role",
    )
    try:
        node_role = NodeRole(node_role_text)
    except ValueError as error:
        raise ValueError(f"record.node_role has unknown value {node_role_text!r}.") from error

    channel_text = _required_string(
        payload["channel"],
        field="record.channel",
    )
    try:
        channel = LogChannel(channel_text)
    except ValueError as error:
        raise ValueError(f"record.channel has unknown value {channel_text!r}.") from error

    attributes_payload = _require_json_object(
        payload["attributes"],
        field="record.attributes",
    )
    attributes = tuple(
        (
            key,
            _decode_log_scalar(
                attribute_value,
                field=f"record.attributes[{key!r}]",
            ),
        )
        for key, attribute_value in attributes_payload.items()
    )

    record = StructuredLogRecord(
        timestamp_utc_ns=timestamp,
        node_role=node_role,
        channel=channel,
        event=_required_string(
            payload["event"],
            field="record.event",
        ),
        message=_required_string(
            payload["message"],
            field="record.message",
        ),
        context=_decode_trace_context(payload["context"]),
        failure=_decode_failure(payload["failure"]),
        attributes=attributes,
    )
    validate_structured_log_record(record)
    return record


def _root_log_lock(path: Path, /) -> asyncio.Lock:
    """Return the event-loop-local append lock for one root log file.

    Parameters
    ----------
    path
        Destination root log path.

    Returns
    -------
    asyncio.Lock
        Lock shared by appends to this path in the current event loop.

    Notes
    -----
    Locks are separated by destination so unrelated root log files do not
    block one another. Event-loop keys are weakly held so completed loops do
    not remain retained by the registry.

    Node usage scope: Root - Used only for root log append serialization.
    """
    loop = asyncio.get_running_loop()
    locks = _ROOT_LOG_LOCKS.get(loop)

    if locks is None:
        locks = {}
        _ROOT_LOG_LOCKS[loop] = locks

    lock = locks.get(path)
    if lock is None:
        lock = asyncio.Lock()
        locks[path] = lock

    return lock


async def append_root_log(
    layout: RootStorageLayout,
    record: StructuredLogRecord,
    /,
) -> None:
    """Append one complete structured record to its root log channel.

    Parameters
    ----------
    layout
        Root storage paths containing the running and error log destinations.
    record
        Valid root-owned structured log record.

    Raises
    ------
    TypeError
        If ``layout``, ``record``, or a nested field has an incorrect type.
    ValueError
        If the record is invalid or is not owned by the root node.
    StorageError
        If opening, writing, or closing the destination log fails.

    Notes
    -----
    The record is validated and encoded before the destination is opened.
    ``running`` records append to ``root_running.log`` and ``error`` records
    append to ``root_error.log`` through one binary append operation.

    This function does not create directories, stage, truncate, rotate,
    rewrite, repair, synchronize, or decode log content. Existing incomplete
    final content remains untouched.

    Node usage scope: Root - Used only for root running and error logs.
    """
    validate_declared_types(
        layout,
        expected_type=RootStorageLayout,
        field="layout",
    )
    encoded = encode_structured_log_record(record)

    if record.node_role is not NodeRole.ROOT:
        raise ValueError("append_root_log requires record.node_role='root'; " f"got {record.node_role.value!r}.")

    if record.channel is LogChannel.RUNNING:
        destination = layout.root_running_log
    elif record.channel is LogChannel.ERROR:
        destination = layout.root_error_log
    else:
        raise ValueError("append_root_log requires channel 'running' or 'error'; " f"got {record.channel.value!r}.")

    async with _root_log_lock(destination):
        try:
            async with aiofiles.open(destination, "ab") as log_handle:
                await log_handle.write(encoded)
        except OSError as error:
            raise StorageError(f"Failed to append root log {str(destination)!r}.") from error
