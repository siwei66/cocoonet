"""Tests for Cocoonet storage metadata contracts."""

import sys
from dataclasses import FrozenInstanceError, fields
from typing import Any

import pytest

from cocoonet import get_version
from cocoonet._contracts.compatibility import CompatibilityMetadata
from cocoonet._storage.metadata import (
    CocoonetMetadata,
    NodeRole,
    current_compatibility_metadata,
    decode_cocoonet_metadata_csv,
    encode_cocoonet_metadata_csv,
    validate_cocoonet_metadata,
)

VALID_COMPATIBILITY = CompatibilityMetadata(
    cocoonet_version="1.2.3",
    python_version=(3, 13, 2),
    python_release_level="final",
)

CANONICAL_METADATA = CocoonetMetadata(
    storage_schema_version=1,
    node_role=NodeRole.ROOT,
    compatibility=VALID_COMPATIBILITY,
)

CANONICAL_METADATA_BYTES = (
    b"field,value\n"
    b"storage_schema_version,1\n"
    b"node_role,root\n"
    b"cocoonet_version,1.2.3\n"
    b"python_major,3\n"
    b"python_minor,13\n"
    b"python_micro,2\n"
    b"python_release_level,final\n"
)


def test_node_role_has_exact_storage_values() -> None:
    """Expose only the root and worker storage roles."""
    assert tuple(NodeRole) == (NodeRole.ROOT, NodeRole.WORKER)
    assert NodeRole.ROOT.value == "root"
    assert NodeRole.WORKER.value == "worker"


def test_cocoonet_metadata_preserves_immutable_fields() -> None:
    """Preserve the exact immutable storage metadata fields."""
    metadata = CocoonetMetadata(
        storage_schema_version=1,
        node_role=NodeRole.ROOT,
        compatibility=VALID_COMPATIBILITY,
    )

    assert tuple(field.name for field in fields(metadata)) == (
        "storage_schema_version",
        "node_role",
        "compatibility",
    )
    assert metadata.compatibility is VALID_COMPATIBILITY

    with pytest.raises(FrozenInstanceError):
        metadata.__setattr__("node_role", NodeRole.WORKER)


@pytest.mark.parametrize("node_role", tuple(NodeRole))
def test_validate_cocoonet_metadata_accepts_each_node_role(
    node_role: NodeRole,
) -> None:
    """Accept schema version one for either storage-owning node."""
    metadata = CocoonetMetadata(
        storage_schema_version=1,
        node_role=node_role,
        compatibility=VALID_COMPATIBILITY,
    )

    validate_cocoonet_metadata(metadata)


@pytest.mark.parametrize("storage_schema_version", (0, 2))
def test_validate_cocoonet_metadata_rejects_other_schema_versions(
    storage_schema_version: int,
) -> None:
    """Reject storage schema versions outside the Phase 2 contract."""
    metadata = CocoonetMetadata(
        storage_schema_version=storage_schema_version,
        node_role=NodeRole.ROOT,
        compatibility=VALID_COMPATIBILITY,
    )

    with pytest.raises(ValueError) as error:
        validate_cocoonet_metadata(metadata)

    assert str(error.value) == (
        "Invalid Cocoonet metadata:\n"
        "  field: storage_schema_version\n"
        "  expected: exactly 1\n"
        f"  got: {storage_schema_version!r}"
    )


def test_validate_cocoonet_metadata_rejects_boolean_schema_version() -> None:
    """Reject a boolean despite its relationship to the integer type."""
    metadata = CocoonetMetadata(
        storage_schema_version=True,
        node_role=NodeRole.ROOT,
        compatibility=VALID_COMPATIBILITY,
    )

    with pytest.raises(TypeError, match=r"storage_schema_version"):
        validate_cocoonet_metadata(metadata)


def test_validate_cocoonet_metadata_rejects_plain_text_node_role() -> None:
    """Require the declared node-role enum rather than equivalent text."""
    invalid_role: Any = "root"
    metadata = CocoonetMetadata(
        storage_schema_version=1,
        node_role=invalid_role,
        compatibility=VALID_COMPATIBILITY,
    )

    with pytest.raises(TypeError, match=r"node_role"):
        validate_cocoonet_metadata(metadata)


def test_validate_cocoonet_metadata_rejects_invalid_compatibility() -> None:
    """Apply the shared compatibility validator to nested metadata."""
    metadata = CocoonetMetadata(
        storage_schema_version=1,
        node_role=NodeRole.WORKER,
        compatibility=CompatibilityMetadata(
            cocoonet_version="",
            python_version=(3, 13, 2),
            python_release_level="final",
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"nonempty complete Cocoonet version string",
    ):
        validate_cocoonet_metadata(metadata)


def test_validate_cocoonet_metadata_rejects_wrong_record_type() -> None:
    """Reject values outside the storage metadata record boundary."""
    invalid_metadata: Any = object()

    with pytest.raises(TypeError, match=r"CocoonetMetadata"):
        validate_cocoonet_metadata(invalid_metadata)


def test_current_compatibility_metadata_captures_runtime() -> None:
    """Capture the installed package and complete current Python version."""
    metadata = current_compatibility_metadata()

    assert metadata.cocoonet_version == get_version()
    assert metadata.python_version == (
        sys.version_info.major,
        sys.version_info.minor,
        sys.version_info.micro,
    )
    assert metadata.python_release_level == sys.version_info.releaselevel


def test_encode_cocoonet_metadata_csv_emits_exact_utf8_bytes() -> None:
    """Emit exact rows, order, newline, and UTF-8 content."""
    metadata = CocoonetMetadata(
        storage_schema_version=1,
        node_role=NodeRole.WORKER,
        compatibility=CompatibilityMetadata(
            cocoonet_version="1.2.3+δοκιμή",
            python_version=(3, 14, 0),
            python_release_level="candidate",
        ),
    )

    encoded = encode_cocoonet_metadata_csv(metadata)

    assert encoded == (
        "field,value\n"
        "storage_schema_version,1\n"
        "node_role,worker\n"
        "cocoonet_version,1.2.3+δοκιμή\n"
        "python_major,3\n"
        "python_minor,14\n"
        "python_micro,0\n"
        "python_release_level,candidate\n"
    ).encode("utf-8")


def test_cocoonet_metadata_csv_round_trip_preserves_record() -> None:
    """Round-trip validated metadata through canonical CSV bytes."""
    encoded = encode_cocoonet_metadata_csv(CANONICAL_METADATA)

    assert encoded == CANONICAL_METADATA_BYTES
    assert decode_cocoonet_metadata_csv(encoded) == CANONICAL_METADATA


@pytest.mark.parametrize(
    ("data", "expected"),
    (
        (
            CANONICAL_METADATA_BYTES.replace(
                b"storage_schema_version,1",
                b"storage_schema_version,2",
            ),
            "storage_schema_version",
        ),
        (
            CANONICAL_METADATA_BYTES.replace(
                b"node_role,root",
                b"node_role,coordinator",
            ),
            "node_role",
        ),
    ),
)
def test_decode_cocoonet_metadata_csv_rejects_schema_or_role(
    data: bytes,
    expected: str,
) -> None:
    """Reject unsupported storage schema versions and node roles."""
    with pytest.raises(ValueError, match=expected):
        decode_cocoonet_metadata_csv(data)


@pytest.mark.parametrize(
    "data",
    (
        CANONICAL_METADATA_BYTES.replace(b"node_role,root\n", b""),
        CANONICAL_METADATA_BYTES.replace(
            b"node_role,root\n",
            b"node_role,root\nnode_role,root\n",
        ),
        CANONICAL_METADATA_BYTES.replace(
            b"storage_schema_version,1\nnode_role,root\n",
            b"node_role,root\nstorage_schema_version,1\n",
        ),
        CANONICAL_METADATA_BYTES.replace(
            b"node_role,root",
            b"unexpected_field,root",
        ),
    ),
)
def test_decode_cocoonet_metadata_csv_rejects_invalid_field_structure(
    data: bytes,
) -> None:
    """Reject missing, duplicate, reordered, and unknown fields."""
    with pytest.raises(ValueError):
        decode_cocoonet_metadata_csv(data)


@pytest.mark.parametrize(
    ("data", "expected"),
    (
        (
            CANONICAL_METADATA_BYTES.replace(
                b"python_major,3",
                b"python_major,-1",
            ),
            "python_version",
        ),
        (
            CANONICAL_METADATA_BYTES.replace(
                b"python_minor,13",
                b"python_minor,01",
            ),
            "python_minor",
        ),
        (
            CANONICAL_METADATA_BYTES.replace(
                b"python_release_level,final",
                b"python_release_level,preview",
            ),
            "python_release_level",
        ),
    ),
)
def test_decode_cocoonet_metadata_csv_rejects_invalid_runtime_values(
    data: bytes,
    expected: str,
) -> None:
    """Reject invalid version components and Python release levels."""
    with pytest.raises(ValueError, match=expected):
        decode_cocoonet_metadata_csv(data)


@pytest.mark.parametrize(
    "data",
    (
        b"\xef\xbb\xbf" + CANONICAL_METADATA_BYTES,
        b"\xff" + CANONICAL_METADATA_BYTES,
        CANONICAL_METADATA_BYTES.replace(b"\n", b"\r\n"),
        CANONICAL_METADATA_BYTES + b"unexpected,data\n",
    ),
)
def test_decode_cocoonet_metadata_csv_rejects_noncanonical_bytes(
    data: bytes,
) -> None:
    """Reject BOM, invalid UTF-8, alternate newlines, and trailing data."""
    with pytest.raises(ValueError):
        decode_cocoonet_metadata_csv(data)


def test_decode_cocoonet_metadata_csv_rejects_serialized_payload() -> None:
    """Reject serialized Python payload bytes without deserializing them."""
    serialized_payload = b"\x80\x04csubprocess\nPopen\n."

    with pytest.raises(ValueError, match=r"valid UTF-8"):
        decode_cocoonet_metadata_csv(serialized_payload)
