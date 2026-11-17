"""Tests for structured storage log records and their one-line codec."""

import asyncio
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import cocoonet._storage.logging as storage_logging
from cocoonet._contracts.failure import (
    FailureRecord,
    FailureScope,
    TraceContext,
)
from cocoonet._contracts.identity import (
    AttemptId,
    FailureId,
    ModelId,
    ParentId,
    WorkerId,
)
from cocoonet._storage.logging import (
    LogChannel,
    StructuredLogRecord,
    append_root_log,
    decode_structured_log_record,
    encode_structured_log_record,
    validate_structured_log_record,
)
from cocoonet._storage.metadata import NodeRole
from cocoonet._storage.paths import build_root_storage_layout
from cocoonet._storage.publication import StorageError, staged_file_path

_CONTEXT = TraceContext(
    stage="validation",
    parent_id=ParentId("parent-1"),
    model_id=ModelId("1"),
    attempt_id=AttemptId("attempt-1"),
    worker_id=WorkerId("worker-1"),
)

_FAILURE = FailureRecord(
    failure_id=FailureId("failure-1"),
    scope=FailureScope.FIT_PREDICTION,
    context=_CONTEXT,
    exception_type="builtins.RuntimeError",
    message="Model evaluation failed.",
    traceback="Traceback content",
)


def _record(
    *,
    timestamp_utc_ns: int = 1,
    node_role: NodeRole = NodeRole.ROOT,
    channel: LogChannel = LogChannel.RUNNING,
    event: str = "attempt_started",
    message: str = "Attempt started.",
    context: TraceContext | None = None,
    failure: FailureRecord | None = None,
    attributes: tuple[
        tuple[str, None | bool | int | float | str],
        ...,
    ] = (),
) -> StructuredLogRecord:
    """Build one structured log record."""
    return StructuredLogRecord(
        timestamp_utc_ns=timestamp_utc_ns,
        node_role=node_role,
        channel=channel,
        event=event,
        message=message,
        context=context,
        failure=failure,
        attributes=attributes,
    )


def test_structured_log_record_preserves_fields_and_is_immutable() -> None:
    """Preserve supplied values and prevent record reassignment."""
    record = _record(
        attributes=(("attempt", 1),),
    )

    assert record.timestamp_utc_ns == 1
    assert record.node_role is NodeRole.ROOT
    assert record.channel is LogChannel.RUNNING
    assert record.event == "attempt_started"
    assert record.message == "Attempt started."
    assert record.context is None
    assert record.failure is None
    assert record.attributes == (("attempt", 1),)

    with pytest.raises(FrozenInstanceError):
        record.__setattr__("event", "changed")


def test_validate_structured_log_record_accepts_context_and_failure() -> None:
    """Accept one traceable structured error record."""
    validate_structured_log_record(
        _record(
            node_role=NodeRole.WORKER,
            channel=LogChannel.ERROR,
            event="validation_failed",
            message="Evaluation failed — details retained.",
            context=_CONTEXT,
            failure=_FAILURE,
            attributes=(
                ("retryable", False),
                ("attempt_number", 2),
                ("duration", 1.25),
                ("detail", None),
            ),
        )
    )


@pytest.mark.parametrize(
    "attributes",
    (
        (("", 1),),
        (("duplicate", 1), ("duplicate", 2)),
        (("value", float("nan")),),
        (("value", float("inf")),),
        (("value", float("-inf")),),
    ),
)
def test_validate_structured_log_record_rejects_invalid_attributes(
    attributes: tuple[
        tuple[str, None | bool | int | float | str],
        ...,
    ],
) -> None:
    """Reject empty, duplicate, or nonfinite attributes."""
    with pytest.raises(ValueError):
        validate_structured_log_record(_record(attributes=attributes))


def test_validate_structured_log_record_rejects_context_disagreement() -> None:
    """Reject contradictory top-level and failure trace coordinates."""
    conflicting = replace(
        _CONTEXT,
        model_id=ModelId("2"),
    )

    with pytest.raises(ValueError, match=r"context\.model_id"):
        validate_structured_log_record(
            _record(
                channel=LogChannel.ERROR,
                context=conflicting,
                failure=_FAILURE,
            )
        )


def test_encode_structured_log_record_is_deterministic() -> None:
    """Encode one compact object with fixed keys and attribute order."""
    record = _record(
        attributes=(
            ("attempt", 1),
            ("ready", True),
        ),
    )

    expected = (
        '{"timestamp_utc_ns":1,"node_role":"root","channel":"running",'
        '"event":"attempt_started","message":"Attempt started.",'
        '"context":null,"failure":null,'
        '"attributes":{"attempt":1,"ready":true}}\n'
    ).encode("utf-8")

    assert encode_structured_log_record(record) == expected


def test_structured_log_codec_round_trips_non_ascii_failure() -> None:
    """Round-trip structured context, failure, and non-ASCII text."""
    record = _record(
        timestamp_utc_ns=123456789,
        node_role=NodeRole.WORKER,
        channel=LogChannel.ERROR,
        event="validation_failed",
        message="Evaluation failed — result retained.",
        context=_CONTEXT,
        failure=_FAILURE,
        attributes=(
            ("folds_completed", 2),
            ("retryable", False),
            ("detail", None),
        ),
    )

    encoded = encode_structured_log_record(record)

    assert b"\\u2014" not in encoded
    assert decode_structured_log_record(encoded) == record


def test_structured_log_codec_preserves_attribute_order() -> None:
    """Preserve supplied attribute order through JSON encoding and decoding."""
    record = _record(
        attributes=(
            ("third", 3),
            ("first", 1),
            ("second", 2),
        ),
    )

    decoded = decode_structured_log_record(encode_structured_log_record(record))

    assert decoded.attributes == record.attributes


@pytest.mark.parametrize(
    "data",
    (
        b"",
        b"{}\n{}\n",
        b"{}",
        b"{}\r\n",
        b"\xef\xbb\xbf{}\n",
        b"\xff\n",
        b'{"event":"first","event":"second"}\n',
        b'{"value":NaN}\n',
    ),
)
def test_decode_structured_log_record_rejects_invalid_line(
    data: bytes,
) -> None:
    """Reject incomplete, multiple, malformed, or nonstandard JSON lines."""
    with pytest.raises(ValueError):
        decode_structured_log_record(data)


def test_decode_structured_log_record_rejects_wrong_key_order() -> None:
    """Reject a record whose physical object order differs from the schema."""
    encoded = (
        '{"node_role":"root","timestamp_utc_ns":1,"channel":"running",'
        '"event":"attempt_started","message":"Attempt started.",'
        '"context":null,"failure":null,"attributes":{}}\n'
    ).encode("utf-8")

    with pytest.raises(ValueError, match="keys"):
        decode_structured_log_record(encoded)


def test_decode_structured_log_record_rejects_nonfinite_attribute() -> None:
    """Reject nonfinite attribute values during decoding."""
    encoded = (
        '{"timestamp_utc_ns":1,"node_role":"root","channel":"running",'
        '"event":"attempt_started","message":"Attempt started.",'
        '"context":null,"failure":null,'
        '"attributes":{"duration":Infinity}}\n'
    ).encode("utf-8")

    with pytest.raises(ValueError, match="nonfinite"):
        decode_structured_log_record(encoded)


def test_append_root_log_routes_running_and_error_records(
    tmp_path: Path,
) -> None:
    """Route complete records to their exact root channel files."""

    async def exercise() -> None:
        layout = build_root_storage_layout(tmp_path)
        layout.logs.mkdir(parents=True)

        running = _record(
            event="root_started",
            message="Root started.",
        )
        error = _record(
            timestamp_utc_ns=2,
            channel=LogChannel.ERROR,
            event="root_error",
            message="Root operation failed.",
        )

        await append_root_log(layout, running)
        await append_root_log(layout, error)

        assert decode_structured_log_record(layout.root_running_log.read_bytes()) == running
        assert decode_structured_log_record(layout.root_error_log.read_bytes()) == error
        assert not staged_file_path(layout.root_running_log).exists()
        assert not staged_file_path(layout.root_error_log).exists()

    asyncio.run(exercise())


def test_append_root_log_rejects_worker_record(
    tmp_path: Path,
) -> None:
    """Reject worker-owned records without creating a root log."""

    async def exercise() -> None:
        layout = build_root_storage_layout(tmp_path)
        layout.logs.mkdir(parents=True)
        record = _record(node_role=NodeRole.WORKER)

        with pytest.raises(ValueError, match="node_role='root'"):
            await append_root_log(layout, record)

        assert not layout.root_running_log.exists()
        assert not layout.root_error_log.exists()

    asyncio.run(exercise())


def test_append_root_log_preserves_concurrent_complete_lines(
    tmp_path: Path,
) -> None:
    """Append concurrent records without interleaving their encoded lines."""

    async def exercise() -> None:
        layout = build_root_storage_layout(tmp_path)
        layout.logs.mkdir(parents=True)
        records = tuple(
            _record(
                timestamp_utc_ns=index,
                event=f"event-{index}",
                message=f"Message {index}.",
            )
            for index in range(20)
        )

        await asyncio.gather(*(append_root_log(layout, record) for record in records))

        lines = layout.root_running_log.read_bytes().splitlines(keepends=True)
        decoded = tuple(decode_structured_log_record(line) for line in lines)

        assert len(lines) == len(records)
        assert set(decoded) == set(records)
        assert all(line.endswith(b"\n") for line in lines)

    asyncio.run(exercise())


def test_append_root_log_serializes_writes_independent_of_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent overlapping writes without relying on OS append behavior."""
    active_writes = 0
    maximum_active_writes = 0

    class FakeLogHandle:
        """Expose an instrumented asynchronous append handle."""

        async def __aenter__(self) -> "FakeLogHandle":
            return self

        async def __aexit__(
            self,
            exception_type: object,
            exception: object,
            traceback: object,
        ) -> None:
            return None

        async def write(self, data: bytes) -> int:
            nonlocal active_writes, maximum_active_writes
            active_writes += 1
            maximum_active_writes = max(
                maximum_active_writes,
                active_writes,
            )
            await asyncio.sleep(0)
            active_writes -= 1
            return len(data)

    def fake_open(path: object, mode: str) -> FakeLogHandle:
        return FakeLogHandle()

    async def exercise() -> None:
        layout = build_root_storage_layout(tmp_path)
        records = tuple(
            _record(
                timestamp_utc_ns=index,
                event=f"event-{index}",
            )
            for index in range(10)
        )

        await asyncio.gather(*(append_root_log(layout, record) for record in records))

    monkeypatch.setattr(storage_logging.aiofiles, "open", fake_open)
    asyncio.run(exercise())

    assert maximum_active_writes == 1


def test_append_root_log_leaves_incomplete_final_content_untouched(
    tmp_path: Path,
) -> None:
    """Leave a crash-like incomplete final line present without repair."""

    async def exercise() -> None:
        layout = build_root_storage_layout(tmp_path)
        layout.logs.mkdir(parents=True)
        record = _record()
        encoded = encode_structured_log_record(record)

        await append_root_log(layout, record)
        incomplete = b'{"timestamp_utc_ns":'
        with layout.root_running_log.open("ab") as log_handle:
            log_handle.write(incomplete)

        content = layout.root_running_log.read_bytes()

        assert content == encoded + incomplete
        assert decode_structured_log_record(content[: len(encoded)]) == record
        with pytest.raises(ValueError):
            decode_structured_log_record(content[len(encoded) :])

    asyncio.run(exercise())


def test_append_root_log_wraps_operational_failure(
    tmp_path: Path,
) -> None:
    """Wrap a failed append with its exact destination path."""

    async def exercise() -> None:
        layout = build_root_storage_layout(tmp_path)
        record = _record()

        with pytest.raises(
            StorageError,
            match="Failed to append root log",
        ) as captured:
            await append_root_log(layout, record)

        assert repr(str(layout.root_running_log)) in str(captured.value)
        assert isinstance(captured.value.__cause__, OSError)

    asyncio.run(exercise())
