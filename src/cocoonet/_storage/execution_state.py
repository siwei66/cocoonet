"""Minimal durable execution-state record contracts."""

import asyncio
import csv
import io
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

import aiofiles

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
    PathInput,
    _validate_parent_id,
    _validate_positive_decimal_identifier,
    resolve_physical_path,
)
from cocoonet._storage.publication import (
    publish_staged_file,
    staged_file_path,
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

_EXECUTION_STATE_LOCKS: WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[Path, asyncio.Lock],
] = WeakKeyDictionary()


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


def _execution_state_lock(path: Path, /) -> asyncio.Lock:
    """Return the event-loop-local writer lock for one physical state file.

    Parameters
    ----------
    path
        Canonical physical path of the final execution-state file.

    Returns
    -------
    asyncio.Lock
        Lock shared by state replacements for this path in the current event
        loop.

    Notes
    -----
    Each root or worker owns one asynchronous writer event loop. Separating
    locks by path prevents unrelated root and worker state files from blocking
    one another.

    Node usage scope: Shared - Used by root and worker state writers.
    """
    loop = asyncio.get_running_loop()
    locks = _EXECUTION_STATE_LOCKS.get(loop)

    if locks is None:
        locks = {}
        _EXECUTION_STATE_LOCKS[loop] = locks

    lock = locks.get(path)
    if lock is None:
        lock = asyncio.Lock()
        locks[path] = lock

    return lock


def _merge_execution_state_records(
    current: tuple[ExecutionStateRecord, ...],
    proposed: tuple[ExecutionStateRecord, ...],
    expected_records: tuple[ExecutionStateRecord, ...],
    /,
) -> tuple[ExecutionStateRecord, ...]:
    """Merge one proposed state revision without losing newer observations.

    Parameters
    ----------
    current
        Currently published records in their preserved identity order.
    proposed
        Validated records supplied for the next state revision.
    expected_records
        Exact previously observed records required for changed existing rows.

    Returns
    -------
    tuple[ExecutionStateRecord, ...]
        Updated existing rows followed by any newly proposed attempts.

    Raises
    ------
    ValueError
        If an expected record is stale or unrelated to the proposal, a changed
        existing row lacks its exact current precondition, an existing identity
        is reordered or changed, an established worker assignment is removed
        or changed, or the merged records violate the state-record contract.

    Notes
    -----
    Existing rows omitted from a concurrent proposal remain unchanged. New
    attempt identities are appended after every already published row. An
    existing row may change only when its exact current record was supplied as
    an optimistic concurrency precondition.

    This function does not apply attempt-status transition policy.

    Node usage scope: Shared - Used by root and worker state writers.
    """
    validate_execution_state_records(expected_records)

    merged = list(current)
    current_positions = {record.attempt_id: position for position, record in enumerate(current)}
    proposed_ids = {record.attempt_id for record in proposed}
    expected_by_id = {record.attempt_id: record for record in expected_records}

    for expected_record in expected_records:
        if expected_record.attempt_id not in proposed_ids:
            raise ValueError("Each expected execution-state record must correspond to " "a proposed existing attempt.")

        current_position = current_positions.get(expected_record.attempt_id)
        if current_position is None or current[current_position] != expected_record:
            raise ValueError("Execution-state revision is stale for attempt " f"{expected_record.attempt_id!r}.")

    next_current_position = 0
    saw_new_attempt = False

    for proposed_record in proposed:
        current_position = current_positions.get(proposed_record.attempt_id)

        if current_position is None:
            saw_new_attempt = True
            merged.append(proposed_record)
            continue

        if saw_new_attempt or current_position != next_current_position:
            raise ValueError("Existing execution-state attempts must preserve their " "published identity order.")

        current_record = current[current_position]
        if (
            proposed_record.parent_id != current_record.parent_id
            or proposed_record.model_label != current_record.model_label
            or proposed_record.model_id != current_record.model_id
        ):
            raise ValueError(
                f"Existing attempt {proposed_record.attempt_id!r} must " "preserve its parent and model identity."
            )

        if current_record.worker_id is not None and proposed_record.worker_id != current_record.worker_id:
            raise ValueError(
                f"Existing attempt {proposed_record.attempt_id!r} must "
                f"preserve assigned worker_id {current_record.worker_id!r}."
            )

        if proposed_record != current_record and expected_by_id.get(proposed_record.attempt_id) != current_record:
            raise ValueError(
                "Changing existing attempt "
                f"{proposed_record.attempt_id!r} requires its exact current "
                "record in expected_records."
            )

        merged[current_position] = proposed_record
        next_current_position += 1

    result = tuple(merged)
    validate_execution_state_records(result)
    return result


async def replace_execution_state(
    path: PathInput,
    records: tuple[ExecutionStateRecord, ...],
    /,
    *,
    expected_records: tuple[ExecutionStateRecord, ...] = (),
) -> None:
    """Atomically replace one serialized execution-state CSV file.

    Parameters
    ----------
    path
        Final ``execution_state.csv`` path.
    records
        Proposed records containing existing rows in published order and any
        new attempts after them.
    expected_records
        Exact previously observed records for existing rows changed by this
        revision. New and unchanged rows require no precondition.

    Raises
    ------
    TypeError
        If the path, sequence, record, or field has an incorrect type.
    ValueError
        If the path does not identify ``execution_state.csv``, the revision is
        stale or lacks a required current-record precondition, or the proposal
        violates schema, identity, ordering, or worker-assignment requirements.
    OSError
        If asynchronous staging-file I/O fails.

    Notes
    -----
    Validation and encoding finish before the staging file is opened. Updates
    to the same physical path are serialized in the owning event loop. Within
    that lock, changed existing rows are compared with their exact expected
    records before publication. Publication uses the focused mutable staged-file
    helper.

    This function does not validate status transitions, inspect result bundles,
    or infer model completion.

    Node usage scope: Shared - Used by root and worker state writers.
    """
    staging = staged_file_path(path)
    final = staging.with_name(staging.name.removesuffix(".partial"))

    if final.name != "execution_state.csv":
        raise ValueError(
            "Execution-state replacement requires a final file named " f"'execution_state.csv'; got {str(final)!r}."
        )

    encoded = encode_execution_state_csv(records)
    validate_execution_state_records(expected_records)
    lock = _execution_state_lock(resolve_physical_path(final))

    async with lock:
        current = await load_execution_state(final) if final.exists() else ()
        records = _merge_execution_state_records(
            current,
            records,
            expected_records,
        )
        encoded = encode_execution_state_csv(records)

        async with aiofiles.open(staging, "wb") as staging_handle:
            await staging_handle.write(encoded)

        publish_staged_file(final, replace=True)


async def load_execution_state(
    path: PathInput,
    /,
) -> tuple[ExecutionStateRecord, ...]:
    """Load and decode one final execution-state CSV file.

    Parameters
    ----------
    path
        Final ``execution_state.csv`` path.

    Returns
    -------
    tuple[ExecutionStateRecord, ...]
        Validated records in persisted order.

    Raises
    ------
    TypeError
        If the path is not a supported string-producing path value.
    ValueError
        If the path does not identify ``execution_state.csv`` or the final
        file does not contain a valid execution-state document.
    OSError
        If reading the final file fails.

    Notes
    -----
    Only the final file is read. A sibling ``.partial`` file is ignored and
    remains untouched. Loaded status claims do not establish result-bundle
    existence or model completion.

    Node usage scope: Shared - Used by root and worker state readers.
    """
    staging = staged_file_path(path)
    final = staging.with_name(staging.name.removesuffix(".partial"))

    if final.name != "execution_state.csv":
        raise ValueError(
            "Execution-state loading requires a final file named " f"'execution_state.csv'; got {str(final)!r}."
        )

    async with aiofiles.open(final, "rb") as final_handle:
        data = await final_handle.read()

    return decode_execution_state_csv(data)
