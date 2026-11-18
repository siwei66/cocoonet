"""Tests for minimal durable execution-state record validation."""

import asyncio
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import cast

import pytest

from cocoonet._contracts.identity import (
    AttemptId,
    ModelId,
    ParentId,
    ScalarLabel,
    WorkerId,
)
from cocoonet._contracts.state import AttemptStatus
from cocoonet._storage.execution_state import (
    ExecutionStateRecord,
    decode_execution_state_csv,
    encode_execution_state_csv,
    load_execution_state,
    replace_execution_state,
    validate_execution_state_records,
)
from cocoonet._storage.publication import staged_file_path

_PARENT_ID = ParentId("a" * 16)
_MODEL_ID = ModelId("1")
_MODEL_LABEL = "model-a"
_ATTEMPT_ID = AttemptId("attempt-1")
_WORKER_ID = WorkerId("worker-1")


def _record(
    *,
    parent_id: ParentId = _PARENT_ID,
    worker_id: WorkerId | None = None,
    model_label: ScalarLabel = _MODEL_LABEL,
    model_id: ModelId = _MODEL_ID,
    attempt_id: AttemptId = _ATTEMPT_ID,
    attempt_status: AttemptStatus = AttemptStatus.NOT_ALLOCATED,
) -> ExecutionStateRecord:
    """Build one valid minimal execution-state record."""
    return ExecutionStateRecord(
        parent_id=parent_id,
        worker_id=worker_id,
        model_label=model_label,
        model_id=model_id,
        attempt_id=attempt_id,
        attempt_status=attempt_status,
    )


def test_execution_state_record_preserves_fields_and_is_immutable() -> None:
    """Preserve the six approved fields and prevent reassignment."""
    record = _record()

    assert record.parent_id == _PARENT_ID
    assert record.worker_id is None
    assert record.model_label == _MODEL_LABEL
    assert record.model_id == _MODEL_ID
    assert record.attempt_id == _ATTEMPT_ID
    assert record.attempt_status is AttemptStatus.NOT_ALLOCATED

    with pytest.raises(FrozenInstanceError):
        record.__setattr__("attempt_status", AttemptStatus.RUNNING)


def test_validate_execution_state_records_accepts_empty_sequence() -> None:
    """Accept an empty worker-side state sequence."""
    validate_execution_state_records(())


def test_validate_execution_state_records_accepts_every_status() -> None:
    """Accept every established Phase 1 attempt status."""
    records = tuple(
        _record(
            worker_id=(
                None
                if status
                in (
                    AttemptStatus.NOT_ALLOCATED,
                    AttemptStatus.ALLOCATION_FAILED,
                )
                else _WORKER_ID
            ),
            attempt_id=AttemptId(f"attempt-{index}"),
            attempt_status=status,
        )
        for index, status in enumerate(AttemptStatus)
    )

    validate_execution_state_records(records)


def test_validate_execution_state_records_accepts_repeated_model_attempts() -> None:
    """Accept distinct attempts belonging to the same model unit."""
    records = (
        _record(),
        _record(
            worker_id=_WORKER_ID,
            attempt_id=AttemptId("attempt-2"),
            attempt_status=AttemptStatus.MODEL_SYNCHRONIZING,
        ),
    )

    validate_execution_state_records(records)


def test_validate_execution_state_records_rejects_wrong_outer_type() -> None:
    """Reject a non-tuple record collection."""
    with pytest.raises(TypeError):
        validate_execution_state_records(cast(tuple[ExecutionStateRecord, ...], []))


def test_validate_execution_state_records_rejects_wrong_record_type() -> None:
    """Reject a tuple member that is not an execution-state record."""
    with pytest.raises(TypeError, match=r"records\[0\]"):
        validate_execution_state_records((cast(ExecutionStateRecord, None),))


@pytest.mark.parametrize(
    "record",
    (
        _record(parent_id=ParentId("invalid")),
        _record(model_id=ModelId("01")),
        _record(attempt_id=AttemptId("")),
        _record(worker_id=WorkerId("")),
    ),
)
def test_validate_execution_state_records_rejects_invalid_identifier(
    record: ExecutionStateRecord,
) -> None:
    """Reject invalid managed or opaque identities."""
    with pytest.raises(ValueError):
        validate_execution_state_records((record,))


def test_validate_execution_state_records_rejects_invalid_label_type() -> None:
    """Reject a model label outside the standardized string family."""
    record = _record(model_label=cast(ScalarLabel, 1))

    with pytest.raises(TypeError, match="model_label"):
        validate_execution_state_records((record,))


def test_validate_execution_state_records_rejects_invalid_status_type() -> None:
    """Reject a raw string in place of an AttemptStatus member."""
    record = _record(attempt_status=cast(AttemptStatus, "not_allocated"))

    with pytest.raises(TypeError, match="attempt_status"):
        validate_execution_state_records((record,))


def test_validate_execution_state_records_rejects_parent_mismatch() -> None:
    """Require every row in one file to identify the same parent."""
    records = (
        _record(),
        _record(
            parent_id=ParentId("b" * 16),
            attempt_id=AttemptId("attempt-2"),
        ),
    )

    with pytest.raises(ValueError, match="parent_id"):
        validate_execution_state_records(records)


def test_validate_execution_state_records_rejects_duplicate_attempt_id() -> None:
    """Require attempt identities to be unique within the parent."""
    with pytest.raises(ValueError, match="attempt_id"):
        validate_execution_state_records((_record(), _record()))


@pytest.mark.parametrize(
    "second",
    (
        _record(
            model_label="different-label",
            attempt_id=AttemptId("attempt-2"),
        ),
        _record(
            model_id=ModelId("2"),
            attempt_id=AttemptId("attempt-2"),
        ),
    ),
)
def test_validate_execution_state_records_rejects_model_association_change(
    second: ExecutionStateRecord,
) -> None:
    """Preserve the one-to-one model identity and label association."""
    with pytest.raises(ValueError):
        validate_execution_state_records((_record(), second))


def test_encode_execution_state_csv_encodes_empty_sequence() -> None:
    """Encode an empty state sequence as the exact header."""
    assert encode_execution_state_csv(()) == (
        b"parent_id,worker_id,model_label,model_id,attempt_id," b"attempt_status\n"
    )


def test_encode_execution_state_csv_encodes_unassigned_attempt() -> None:
    """Encode an undefined worker identity as an empty CSV field."""
    assert encode_execution_state_csv((_record(),)) == (
        b"parent_id,worker_id,model_label,model_id,attempt_id,"
        b"attempt_status\n"
        b"aaaaaaaaaaaaaaaa,,model-a,1,attempt-1,not_allocated\n"
    )


def test_encode_execution_state_csv_preserves_order_and_status_values() -> None:
    """Preserve supplied row order and encode exact enum values."""
    records = tuple(
        _record(
            worker_id=(
                None
                if status
                in (
                    AttemptStatus.NOT_ALLOCATED,
                    AttemptStatus.ALLOCATION_FAILED,
                )
                else _WORKER_ID
            ),
            attempt_id=AttemptId(f"attempt-{index}"),
            attempt_status=status,
        )
        for index, status in enumerate(AttemptStatus)
    )
    expected_rows = "".join(
        (
            f"aaaaaaaaaaaaaaaa,"
            f"{'' if record.worker_id is None else record.worker_id},"
            f"model-a,1,{record.attempt_id},{record.attempt_status.value}\n"
        )
        for record in records
    )
    expected = ("parent_id,worker_id,model_label,model_id,attempt_id," f"attempt_status\n{expected_rows}").encode(
        "utf-8"
    )

    assert encode_execution_state_csv(records) == expected


def test_encode_execution_state_csv_quotes_non_ascii_text() -> None:
    """Encode non-ASCII labels and RFC-compatible special characters."""
    record = _record(
        worker_id=_WORKER_ID,
        model_label='model—,"alpha"\nnext',
        attempt_status=AttemptStatus.RUNNING,
    )

    expected = (
        "parent_id,worker_id,model_label,model_id,attempt_id,"
        "attempt_status\n"
        'aaaaaaaaaaaaaaaa,worker-1,"model—,""alpha""\nnext",'
        "1,attempt-1,running\n"
    ).encode("utf-8")

    assert encode_execution_state_csv((record,)) == expected


def test_encode_execution_state_csv_validates_before_encoding() -> None:
    """Reject inconsistent records before producing CSV bytes."""
    duplicate = _record()

    with pytest.raises(ValueError, match="attempt_id"):
        encode_execution_state_csv((_record(), duplicate))


def test_decode_execution_state_csv_round_trips_ordered_records() -> None:
    """Reconstruct typed records while preserving supplied row order."""
    records = (
        _record(),
        _record(
            worker_id=_WORKER_ID,
            model_label='model—,"alpha"\nnext',
            model_id=ModelId("2"),
            attempt_id=AttemptId("attempt-2"),
            attempt_status=AttemptStatus.RUNNING,
        ),
    )

    assert decode_execution_state_csv(encode_execution_state_csv(records)) == records


def test_decode_execution_state_csv_decodes_header_only() -> None:
    """Decode the exact header as an empty record sequence."""
    data = b"parent_id,worker_id,model_label,model_id,attempt_id," b"attempt_status\n"

    assert decode_execution_state_csv(data) == ()


def test_decode_execution_state_csv_rejects_wrong_input_type() -> None:
    """Reject text supplied instead of encoded bytes."""
    with pytest.raises(TypeError, match="data"):
        decode_execution_state_csv(cast(bytes, "not-bytes"))


@pytest.mark.parametrize(
    "data",
    (
        b"",
        (b"\xef\xbb\xbf" b"parent_id,worker_id,model_label,model_id,attempt_id," b"attempt_status\n"),
        b"\xff",
        (b"parent_id,worker_id,model_label,model_id,attempt_id," b"attempt_status"),
        (b"parent_id,worker_id,model_label,model_id,attempt_id," b"attempt_status\r\n"),
        (b"worker_id,parent_id,model_label,model_id,attempt_id," b"attempt_status\n"),
    ),
)
def test_decode_execution_state_csv_rejects_invalid_document(
    data: bytes,
) -> None:
    """Reject invalid encoding, BOM, newline, or header boundaries."""
    with pytest.raises(ValueError):
        decode_execution_state_csv(data)


@pytest.mark.parametrize(
    "row",
    (
        b"aaaaaaaaaaaaaaaa,,model-a,1,attempt-1\n",
        (b"aaaaaaaaaaaaaaaa,,model-a,1,attempt-1," b"not_allocated,extra\n"),
        b'aaaaaaaaaaaaaaaa,,"unclosed,1,attempt-1,not_allocated\n',
    ),
)
def test_decode_execution_state_csv_rejects_malformed_row(
    row: bytes,
) -> None:
    """Reject rows with invalid field counts or quoting."""
    header = b"parent_id,worker_id,model_label,model_id,attempt_id," b"attempt_status\n"

    with pytest.raises(ValueError):
        decode_execution_state_csv(header + row)


def test_decode_execution_state_csv_rejects_unknown_status() -> None:
    """Reject a status outside the established Phase 1 vocabulary."""
    data = (
        b"parent_id,worker_id,model_label,model_id,attempt_id,"
        b"attempt_status\n"
        b"aaaaaaaaaaaaaaaa,,model-a,1,attempt-1,unknown\n"
    )

    with pytest.raises(ValueError, match="unknown attempt_status"):
        decode_execution_state_csv(data)


@pytest.mark.parametrize(
    "rows",
    (
        b"invalid,,model-a,1,attempt-1,not_allocated\n",
        (
            b"aaaaaaaaaaaaaaaa,,model-a,1,attempt-1,not_allocated\n"
            b"aaaaaaaaaaaaaaaa,worker-1,model-a,1,attempt-1,running\n"
        ),
        (
            b"aaaaaaaaaaaaaaaa,,model-a,1,attempt-1,not_allocated\n"
            b"aaaaaaaaaaaaaaaa,worker-1,different,1,attempt-2,running\n"
        ),
        (
            b"aaaaaaaaaaaaaaaa,,model-a,1,attempt-1,not_allocated\n"
            b"aaaaaaaaaaaaaaaa,worker-1,model-a,2,attempt-2,running\n"
        ),
    ),
)
def test_decode_execution_state_csv_rejects_invalid_records(
    rows: bytes,
) -> None:
    """Apply Unit 21 identity, uniqueness, and association validation."""
    header = b"parent_id,worker_id,model_label,model_id,attempt_id," b"attempt_status\n"

    with pytest.raises(ValueError):
        decode_execution_state_csv(header + rows)


def test_replace_execution_state_writes_and_replaces_atomically(
    tmp_path: Path,
) -> None:
    """Write initial state and replace mutable fields without reordering."""

    async def exercise() -> None:
        path = tmp_path / "execution_state.csv"
        initial = _record()

        await replace_execution_state(path, (initial,))
        assert await load_execution_state(path) == (initial,)

        assigned = replace(
            initial,
            worker_id=_WORKER_ID,
            attempt_status=AttemptStatus.MODEL_SYNCHRONIZING,
        )
        appended = _record(
            model_id=ModelId("2"),
            model_label="model-b",
            attempt_id=AttemptId("attempt-2"),
        )

        await replace_execution_state(
            path,
            (assigned, appended),
            expected_records=(initial,),
        )

        assert await load_execution_state(path) == (assigned, appended)
        assert not staged_file_path(path).exists()

    asyncio.run(exercise())


def test_replace_execution_state_requires_current_record_for_change(
    tmp_path: Path,
) -> None:
    """Require an exact current-record precondition for an existing-row change."""

    async def exercise() -> None:
        path = tmp_path / "execution_state.csv"
        initial = _record(
            worker_id=_WORKER_ID,
            attempt_status=AttemptStatus.MODEL_SYNCHRONIZING,
        )
        updated = replace(
            initial,
            attempt_status=AttemptStatus.MODEL_SYNCHRONIZED,
        )

        await replace_execution_state(path, (initial,))

        with pytest.raises(ValueError, match="exact current record"):
            await replace_execution_state(path, (updated,))

        assert await load_execution_state(path) == (initial,)

    asyncio.run(exercise())


def test_replace_execution_state_rejects_stale_competing_update(
    tmp_path: Path,
) -> None:
    """Permit one competing revision and reject the stale competitor."""

    async def exercise() -> None:
        path = tmp_path / "execution_state.csv"
        initial = _record(
            worker_id=_WORKER_ID,
            attempt_status=AttemptStatus.MODEL_SYNCHRONIZED,
        )
        running = replace(
            initial,
            attempt_status=AttemptStatus.RUNNING,
        )
        failed = replace(
            initial,
            attempt_status=AttemptStatus.ATTEMPT_FAILED,
        )

        await replace_execution_state(path, (initial,))

        outcomes = await asyncio.gather(
            replace_execution_state(
                path,
                (running,),
                expected_records=(initial,),
            ),
            replace_execution_state(
                path,
                (failed,),
                expected_records=(initial,),
            ),
            return_exceptions=True,
        )

        assert sum(outcome is None for outcome in outcomes) == 1
        assert sum(isinstance(outcome, ValueError) for outcome in outcomes) == 1
        assert await load_execution_state(path) in (
            (running,),
            (failed,),
        )
        assert not staged_file_path(path).exists()

    asyncio.run(exercise())


def test_replace_execution_state_rejects_stale_status_regression(
    tmp_path: Path,
) -> None:
    """Prevent an old observation from replacing a newer published status."""

    async def exercise() -> None:
        path = tmp_path / "execution_state.csv"
        initial = _record(
            worker_id=_WORKER_ID,
            attempt_status=AttemptStatus.MODEL_SYNCHRONIZING,
        )
        synchronized = replace(
            initial,
            attempt_status=AttemptStatus.MODEL_SYNCHRONIZED,
        )

        await replace_execution_state(path, (initial,))
        await replace_execution_state(
            path,
            (synchronized,),
            expected_records=(initial,),
        )

        with pytest.raises(ValueError, match="stale"):
            await replace_execution_state(
                path,
                (initial,),
                expected_records=(initial,),
            )

        assert await load_execution_state(path) == (synchronized,)

    asyncio.run(exercise())


def test_replace_execution_state_preserves_concurrently_appended_rows(
    tmp_path: Path,
) -> None:
    """Serialize competing append proposals without losing either attempt."""

    async def exercise() -> None:
        path = tmp_path / "execution_state.csv"
        initial = _record()
        second = _record(
            model_id=ModelId("2"),
            model_label="model-b",
            attempt_id=AttemptId("attempt-2"),
        )
        third = _record(
            model_id=ModelId("3"),
            model_label="model-c",
            attempt_id=AttemptId("attempt-3"),
        )

        await replace_execution_state(path, (initial,))
        await asyncio.gather(
            replace_execution_state(path, (initial, second)),
            replace_execution_state(path, (initial, third)),
        )

        assert await load_execution_state(path) == (
            initial,
            second,
            third,
        )

    asyncio.run(exercise())


def test_replace_execution_state_rejects_worker_reassignment(
    tmp_path: Path,
) -> None:
    """Prevent an established worker identity from changing or becoming empty."""

    async def exercise() -> None:
        path = tmp_path / "execution_state.csv"
        assigned = _record(
            worker_id=_WORKER_ID,
            attempt_status=AttemptStatus.RUNNING,
        )
        await replace_execution_state(path, (assigned,))

        changed = replace(
            assigned,
            worker_id=WorkerId("worker-2"),
        )
        with pytest.raises(ValueError, match="worker_id"):
            await replace_execution_state(path, (changed,))

        cleared = replace(assigned, worker_id=None)
        with pytest.raises(ValueError, match="worker_id"):
            await replace_execution_state(path, (cleared,))

        assert await load_execution_state(path) == (assigned,)

    asyncio.run(exercise())


def test_replace_execution_state_rejects_existing_row_reordering(
    tmp_path: Path,
) -> None:
    """Reject a proposal that changes the published attempt identity order."""

    async def exercise() -> None:
        path = tmp_path / "execution_state.csv"
        first = _record()
        second = _record(
            model_id=ModelId("2"),
            model_label="model-b",
            attempt_id=AttemptId("attempt-2"),
        )
        await replace_execution_state(path, (first, second))

        with pytest.raises(ValueError, match="identity order"):
            await replace_execution_state(path, (second, first))

        assert await load_execution_state(path) == (first, second)

    asyncio.run(exercise())


def test_load_execution_state_ignores_stale_partial_file(
    tmp_path: Path,
) -> None:
    """Read only final state and leave an unrelated stale staging file intact."""

    async def exercise() -> None:
        path = tmp_path / "execution_state.csv"
        record = _record(
            worker_id=_WORKER_ID,
            attempt_status=AttemptStatus.RESULT_SYNCHRONIZED,
        )
        await replace_execution_state(path, (record,))

        staging = staged_file_path(path)
        staging.write_bytes(b"stale incomplete state")

        assert await load_execution_state(path) == (record,)
        assert staging.read_bytes() == b"stale incomplete state"

    asyncio.run(exercise())


def test_load_execution_state_does_not_require_result_bundle(
    tmp_path: Path,
) -> None:
    """Return informational status without treating it as completion evidence."""

    async def exercise() -> None:
        path = tmp_path / "execution_state.csv"
        record = _record(
            worker_id=_WORKER_ID,
            attempt_status=AttemptStatus.RESULT_SYNCHRONIZED,
        )
        await replace_execution_state(path, (record,))

        assert not (tmp_path / "model_1").exists()
        assert await load_execution_state(path) == (record,)

    asyncio.run(exercise())
