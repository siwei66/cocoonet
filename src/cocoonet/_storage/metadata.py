"""Storage metadata contracts shared by Cocoonet nodes."""

import csv
import io
import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from cocoonet import get_version
from cocoonet._contracts.compatibility import (
    CompatibilityMetadata,
    validate_compatibility_metadata,
)
from cocoonet._type_validation import validate_declared_types

_METADATA_FIELDS = (
    "storage_schema_version",
    "node_role",
    "cocoonet_version",
    "python_major",
    "python_minor",
    "python_micro",
    "python_release_level",
)


class NodeRole(StrEnum):
    """Identify the node role owning one storage metadata record.

    Node usage scope: Shared - Used by both root and worker storage.
    """

    ROOT = "root"
    WORKER = "worker"


@dataclass(frozen=True, slots=True)
class CocoonetMetadata:
    """Record the storage schema and runtime compatibility metadata.

    Parameters
    ----------
    storage_schema_version
        Exact Cocoonet-managed storage schema version.
    node_role
        Role of the node owning the associated storage.
    compatibility
        Complete Cocoonet and Python runtime compatibility metadata.

    Notes
    -----
    This immutable record preserves candidate values without performing
    validation. It does not capture the current runtime, perform compatibility
    comparison, encode storage, or modify the filesystem.

    Node usage scope: Shared - Used by both root and worker storage.
    """

    storage_schema_version: int
    node_role: NodeRole
    compatibility: CompatibilityMetadata


def validate_cocoonet_metadata(
    value: CocoonetMetadata,
    /,
) -> None:
    """Validate one Cocoonet storage metadata record.

    Parameters
    ----------
    value
        Candidate storage metadata.

    Raises
    ------
    TypeError
        If the record or one of its fields has an incorrect runtime type.
    ValueError
        If the storage schema version is not exactly ``1`` or the nested
        compatibility metadata is invalid.

    Notes
    -----
    Phase 2 supports exactly storage schema version ``1``. This function
    validates metadata only; it does not compare root and worker runtimes,
    encode CSV data, inspect the environment, or modify storage.

    Node usage scope: Shared - Used by both root and worker storage.
    """
    if not isinstance(value, CocoonetMetadata):
        validate_declared_types(
            value,
            expected_type=CocoonetMetadata,
        )

    validate_declared_types(
        value.storage_schema_version,
        expected_type=int,
        field="storage_schema_version",
    )
    if value.storage_schema_version != 1:
        raise ValueError(
            "Invalid Cocoonet metadata:\n"
            "  field: storage_schema_version\n"
            "  expected: exactly 1\n"
            f"  got: {value.storage_schema_version!r}"
        )

    validate_declared_types(
        value.node_role,
        expected_type=NodeRole,
        field="node_role",
    )
    validate_compatibility_metadata(value.compatibility)


def current_compatibility_metadata() -> CompatibilityMetadata:
    """Capture compatibility metadata for the current runtime.

    Returns
    -------
    CompatibilityMetadata
        Complete installed Cocoonet version and current Python runtime data.

    Notes
    -----
    This function captures metadata only. It does not compare runtimes,
    negotiate dependencies, encode storage, or modify the filesystem.

    Node usage scope: Shared - Used by both root and worker storage.
    """
    metadata = CompatibilityMetadata(
        cocoonet_version=get_version(),
        python_version=(
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
        ),
        python_release_level=sys.version_info.releaselevel,
    )
    validate_compatibility_metadata(metadata)
    return metadata


def encode_cocoonet_metadata_csv(
    value: CocoonetMetadata,
    /,
) -> bytes:
    """Encode validated Cocoonet metadata as canonical UTF-8 CSV bytes.

    Parameters
    ----------
    value
        Storage metadata to encode.

    Returns
    -------
    bytes
        Exact schema-controlled CSV using UTF-8 and ``\n`` line endings.

    Raises
    ------
    TypeError
        If the record or one of its fields has an incorrect runtime type.
    ValueError
        If the metadata violates the storage or compatibility contract.

    Notes
    -----
    Encoding performs no compatibility comparison or filesystem access.

    Node usage scope: Shared - Used by both root and worker storage.
    """
    validate_cocoonet_metadata(value)

    python_major, python_minor, python_micro = value.compatibility.python_version
    values = (
        str(value.storage_schema_version),
        value.node_role.value,
        value.compatibility.cocoonet_version,
        str(python_major),
        str(python_minor),
        str(python_micro),
        value.compatibility.python_release_level,
    )
    rows = [
        ("field", "value"),
        *zip(_METADATA_FIELDS, values, strict=True),
    ]

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def decode_cocoonet_metadata_csv(  # noqa: C901
    data: bytes,
    /,
) -> CocoonetMetadata:
    """Decode strict schema-controlled Cocoonet metadata CSV bytes.

    Parameters
    ----------
    data
        Candidate UTF-8 CSV bytes.

    Returns
    -------
    CocoonetMetadata
        Validated storage metadata.

    Raises
    ------
    TypeError
        If ``data`` is not exactly bytes.
    ValueError
        If UTF-8, CSV structure, field order, field values, or canonical byte
        representation is invalid.

    Notes
    -----
    Decoding reconstructs and validates metadata only. It does not compare
    root and worker compatibility, deserialize executable payloads, or access
    the filesystem.

    Node usage scope: Shared - Used by both root and worker storage.
    """
    validate_declared_types(
        data,
        expected_type=bytes,
        field="data",
    )
    if data.startswith(b"\xef\xbb\xbf"):
        raise ValueError("Cocoonet metadata CSV must not contain a UTF-8 BOM.")

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Cocoonet metadata CSV must be valid UTF-8.") from error

    try:
        rows = list(
            csv.reader(
                io.StringIO(text, newline=""),
                strict=True,
            )
        )
    except csv.Error as error:
        raise ValueError("Cocoonet metadata CSV is malformed.") from error

    expected_row_count = len(_METADATA_FIELDS) + 1
    if len(rows) != expected_row_count:
        raise ValueError("Cocoonet metadata CSV must contain exactly " f"{expected_row_count} rows; got {len(rows)}.")
    if any(len(row) != 2 for row in rows):
        raise ValueError("Each Cocoonet metadata CSV row must contain exactly two fields.")
    if rows[0] != ["field", "value"]:
        raise ValueError("Cocoonet metadata CSV header must be exactly 'field,value'.")

    observed_fields = tuple(row[0] for row in rows[1:])
    if observed_fields != _METADATA_FIELDS:
        raise ValueError(
            "Cocoonet metadata CSV fields must occur exactly in this order: "
            f"{_METADATA_FIELDS!r}; got {observed_fields!r}."
        )

    integer_rows = (
        ("storage_schema_version", 1),
        ("python_major", 4),
        ("python_minor", 5),
        ("python_micro", 6),
    )
    parsed_integers: dict[str, int] = {}
    for field, row_index in integer_rows:
        text_value = rows[row_index][1]
        try:
            integer_value = int(text_value)
        except ValueError as error:
            raise ValueError(
                f"Cocoonet metadata field {field!r} must be a canonical " f"decimal integer; got {text_value!r}."
            ) from error

        if str(integer_value) != text_value:
            raise ValueError(
                f"Cocoonet metadata field {field!r} must be a canonical " f"decimal integer; got {text_value!r}."
            )
        parsed_integers[field] = integer_value

    node_role_text = rows[2][1]
    try:
        node_role = NodeRole(node_role_text)
    except ValueError as error:
        raise ValueError(
            "Cocoonet metadata field 'node_role' must be 'root' or " f"'worker'; got {node_role_text!r}."
        ) from error

    release_level_text = rows[7][1]
    release_level: Literal["alpha", "beta", "candidate", "final"]
    match release_level_text:
        case "alpha":
            release_level = "alpha"
        case "beta":
            release_level = "beta"
        case "candidate":
            release_level = "candidate"
        case "final":
            release_level = "final"
        case _:
            raise ValueError(
                "Cocoonet metadata field 'python_release_level' must be "
                "'alpha', 'beta', 'candidate', or 'final'; "
                f"got {release_level_text!r}."
            )

    metadata = CocoonetMetadata(
        storage_schema_version=parsed_integers["storage_schema_version"],
        node_role=node_role,
        compatibility=CompatibilityMetadata(
            cocoonet_version=rows[3][1],
            python_version=(
                parsed_integers["python_major"],
                parsed_integers["python_minor"],
                parsed_integers["python_micro"],
            ),
            python_release_level=release_level,
        ),
    )
    validate_cocoonet_metadata(metadata)

    if encode_cocoonet_metadata_csv(metadata) != data:
        raise ValueError("Cocoonet metadata CSV bytes are not in canonical form.")

    return metadata
