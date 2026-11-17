"""Tests for minimal durable execution-state record validation."""

from dataclasses import FrozenInstanceError
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
    encode_execution_state_csv,
    validate_execution_state_records,
)

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
        model_label='模型,"alpha"\nnext',
        attempt_status=AttemptStatus.RUNNING,
    )

    expected = (
        "parent_id,worker_id,model_label,model_id,attempt_id,"
        "attempt_status\n"
        'aaaaaaaaaaaaaaaa,worker-1,"模型,""alpha""\nnext",'
        "1,attempt-1,running\n"
    ).encode("utf-8")

    assert encode_execution_state_csv((record,)) == expected


def test_encode_execution_state_csv_validates_before_encoding() -> None:
    """Reject inconsistent records before producing CSV bytes."""
    duplicate = _record()

    with pytest.raises(ValueError, match="attempt_id"):
        encode_execution_state_csv((_record(), duplicate))
