"""Minimal durable execution-state record contracts."""

import csv
import io
from dataclasses import dataclass

from cocoonet._contracts.identity import (
    AttemptId,
    ModelId,
    ParentId,
    ScalarLabel,
    WorkerId,
    validate_identifier,
    validate_label,
)
from cocoonet._contracts.state import AttemptStatus
from cocoonet._storage.paths import (
    _validate_parent_id,
    _validate_positive_decimal_identifier,
)
from cocoonet._type_validation import validate_declared_types

EXECUTION_STATE_COLUMNS: tuple[str, ...] = (
    "parent_id",
    "worker_id",
    "model_label",
    "model_id",
    "attempt_id",
    "attempt_status",
)


@dataclass(frozen=True, slots=True)
class ExecutionStateRecord:
    """Record one minimal durable execution-attempt observation.

    Parameters
    ----------
    parent_id
        Identity of the fixed parent task.
    worker_id
        Assigned worker identity, or ``None`` before assignment.
    model_label
        Fixed user-provided label of the model unit.
    model_id
        Identity of the fixed model unit.
    attempt_id
        Identity of this execution attempt.
    attempt_status
        Current approved attempt workflow state.

    Notes
    -----
    This record intentionally excludes parent progress, detailed attempt
    observations, stop disposition, timestamps, result artifacts, worker
    health, reception state, operations, and failure details.

    It stores informational attempt state only. It does not perform lifecycle
    transitions or establish model completion.

    Node usage scope: Shared - Used by both root and worker storage.
    """

    parent_id: ParentId
    worker_id: WorkerId | None
    model_label: ScalarLabel
    model_id: ModelId
    attempt_id: AttemptId
    attempt_status: AttemptStatus


def validate_execution_state_records(
    value: tuple[ExecutionStateRecord, ...],
    /,
) -> None:
    """Validate one ordered sequence of minimal attempt-state records.

    Parameters
    ----------
    value
        Candidate execution-state records for one parent.

    Raises
    ------
    TypeError
        If the sequence, a record, or one of its fields has an incorrect type.
    ValueError
        If an identity is invalid, parent identities differ, attempt identities
        repeat, or model identity and label associations are inconsistent.

    Notes
    -----
    Validation preserves input order and permits an empty sequence. It does not
    infer node role, perform transitions, inspect result bundles, compare a
    previous file revision, or determine model completion.

    Node usage scope: Shared - Used by both root and worker storage.
    """
    validate_declared_types(
        value,
        expected_type=tuple,
    )

    parent_id: ParentId | None = None
    seen_attempt_ids: set[AttemptId] = set()
    labels_by_model_id: dict[ModelId, ScalarLabel] = {}
    model_ids_by_label: dict[ScalarLabel, ModelId] = {}

    for index, record in enumerate(value):
        if not isinstance(record, ExecutionStateRecord):
            validate_declared_types(
                record,
                expected_type=ExecutionStateRecord,
                field=f"records[{index}]",
            )

        _validate_parent_id(record.parent_id)
        _validate_positive_decimal_identifier(
            record.model_id,
            field=f"records[{index}].model_id",
        )
        validate_identifier(
            record.attempt_id,
            field=f"records[{index}].attempt_id",
        )
        validate_label(
            record.model_label,
            field=f"records[{index}].model_label",
        )
        validate_declared_types(
            record.attempt_status,
            expected_type=AttemptStatus,
            field=f"records[{index}].attempt_status",
        )

        if record.worker_id is not None:
            validate_identifier(
                record.worker_id,
                field=f"records[{index}].worker_id",
            )

        if parent_id is None:
            parent_id = record.parent_id
        elif record.parent_id != parent_id:
            raise ValueError(f"records[{index}].parent_id must equal {parent_id!r}; " f"got {record.parent_id!r}.")

        if record.attempt_id in seen_attempt_ids:
            raise ValueError(f"records[{index}].attempt_id must be unique; " f"got duplicate {record.attempt_id!r}.")
        seen_attempt_ids.add(record.attempt_id)

        known_label = labels_by_model_id.get(record.model_id)
        if known_label is not None and record.model_label != known_label:
            raise ValueError(
                f"records[{index}].model_label must preserve the label "
                f"associated with model_id {record.model_id!r}; "
                f"expected {known_label!r}, got {record.model_label!r}."
            )

        known_model_id = model_ids_by_label.get(record.model_label)
        if known_model_id is not None and record.model_id != known_model_id:
            raise ValueError(
                f"records[{index}].model_id must preserve the identity "
                f"associated with model_label {record.model_label!r}; "
                f"expected {known_model_id!r}, got {record.model_id!r}."
            )

        labels_by_model_id[record.model_id] = record.model_label
        model_ids_by_label[record.model_label] = record.model_id


def encode_execution_state_csv(
    value: tuple[ExecutionStateRecord, ...],
    /,
) -> bytes:
    """Encode ordered execution-state records as deterministic UTF-8 CSV.

    Parameters
    ----------
    value
        Execution-state records to validate and encode in their supplied order.

    Returns
    -------
    bytes
        UTF-8 CSV containing the exact six-column header and one row per
        supplied record.

    Raises
    ------
    TypeError
        If the record sequence or one of its fields has an incorrect type.
    ValueError
        If record identities or associations are invalid.

    Notes
    -----
    ``worker_id=None`` is encoded as an empty field. Attempt status is encoded
    by its exact enum value. The standard-library CSV writer provides
    RFC-compatible quoting, and every physical record uses ``\n`` line
    endings.

    This function performs no decoding, file I/O, state transition, sorting,
    persistence, or result inference.

    Node usage scope: Shared - Used by root and worker storage.
    """
    validate_execution_state_records(value)

    output = io.StringIO(newline="")
    writer = csv.writer(
        output,
        delimiter=",",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writerow(EXECUTION_STATE_COLUMNS)

    for record in value:
        writer.writerow(
            (
                record.parent_id,
                "" if record.worker_id is None else record.worker_id,
                record.model_label,
                record.model_id,
                record.attempt_id,
                record.attempt_status.value,
            )
        )

    return output.getvalue().encode("utf-8")


def decode_execution_state_csv(  # noqa: C901
    data: bytes,
    /,
) -> tuple[ExecutionStateRecord, ...]:
    """Decode strict UTF-8 CSV into ordered execution-state records.

    Parameters
    ----------
    data
        Complete execution-state CSV bytes.

    Returns
    -------
    tuple[ExecutionStateRecord, ...]
        Validated records in their persisted order.

    Raises
    ------
    TypeError
        If ``data`` is not bytes.
    ValueError
        If UTF-8, header, newline, row, field, identity, status, uniqueness, or
        model-association validation fails.

    Notes
    -----
    The decoder accepts only the exact six-column schema. Empty ``worker_id``
    fields reconstruct as ``None``; every other field remains required by its
    owning validator.

    This function performs no file I/O, transition, sorting, recovery, result
    inspection, or model-completion inference.

    Node usage scope: Shared - Used by root and worker storage.
    """
    validate_declared_types(
        data,
        expected_type=bytes,
        field="data",
    )

    if data.startswith(b"\xef\xbb\xbf"):
        raise ValueError("Execution-state CSV must not contain a UTF-8 BOM.")

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Execution-state CSV must contain valid UTF-8.") from error

    if not text:
        raise ValueError("Execution-state CSV must contain the exact header.")
    if not text.endswith("\n"):
        raise ValueError("Execution-state CSV must end with a newline.")

    # Reject carriage returns used as physical record endings while permitting
    # them inside correctly quoted field content.
    in_quotes = False
    at_field_start = True
    index = 0
    while index < len(text):
        character = text[index]

        if in_quotes:
            if character == '"':
                if index + 1 < len(text) and text[index + 1] == '"':
                    index += 2
                    continue
                in_quotes = False
        elif character == '"' and at_field_start:
            in_quotes = True
            at_field_start = False
        elif character == ",":
            at_field_start = True
        elif character == "\n":
            at_field_start = True
        elif character == "\r":
            raise ValueError("Execution-state CSV physical records must use LF line endings.")
        else:
            at_field_start = False

        index += 1

    try:
        rows = list(
            csv.reader(
                io.StringIO(text, newline=""),
                delimiter=",",
                strict=True,
            )
        )
    except csv.Error as error:
        raise ValueError("Execution-state CSV contains malformed CSV.") from error

    if not rows or tuple(rows[0]) != EXECUTION_STATE_COLUMNS:
        raise ValueError("Execution-state CSV must contain the exact six-column header.")

    records: list[ExecutionStateRecord] = []
    for row_index, row in enumerate(rows[1:], start=2):
        if len(row) != len(EXECUTION_STATE_COLUMNS):
            raise ValueError(
                f"Execution-state CSV row {row_index} must contain exactly "
                f"{len(EXECUTION_STATE_COLUMNS)} fields; got {len(row)}."
            )

        (
            parent_id,
            worker_id,
            model_label,
            model_id,
            attempt_id,
            attempt_status,
        ) = row

        try:
            status = AttemptStatus(attempt_status)
        except ValueError as error:
            raise ValueError(
                f"Execution-state CSV row {row_index} has unknown " f"attempt_status {attempt_status!r}."
            ) from error

        records.append(
            ExecutionStateRecord(
                parent_id=ParentId(parent_id),
                worker_id=None if worker_id == "" else WorkerId(worker_id),
                model_label=model_label,
                model_id=ModelId(model_id),
                attempt_id=AttemptId(attempt_id),
                attempt_status=status,
            )
        )

    result = tuple(records)
    validate_execution_state_records(result)
    return result
