"""Tests for plain Cocoonet compatibility declarations."""

from dataclasses import FrozenInstanceError, fields
from typing import Literal, cast

import pytest

from cocoonet._contracts.compatibility import (
    SUPPORTED_PYTHON_SERIES,
    CompatibilityMetadata,
    require_compatible,
    validate_compatibility_metadata,
)


def test_compatibility_metadata_preserves_plain_runtime_metadata() -> None:
    """Preserve only the runtime values required for compatibility decisions."""
    metadata = CompatibilityMetadata(
        cocoonet_version="1.2.3",
        python_version=(3, 13, 2),
        python_release_level="final",
    )

    assert metadata.cocoonet_version == "1.2.3"
    assert metadata.python_version == (3, 13, 2)
    assert metadata.python_release_level == "final"
    assert tuple(field.name for field in fields(metadata)) == (
        "cocoonet_version",
        "python_version",
        "python_release_level",
    )


def test_compatibility_metadata_rejects_field_mutation() -> None:
    """Prevent reassignment of recorded compatibility metadata."""
    metadata = CompatibilityMetadata(
        cocoonet_version="1.2.3",
        python_version=(3, 12, 10),
        python_release_level="final",
    )

    with pytest.raises(FrozenInstanceError):
        metadata.__setattr__("cocoonet_version", "1.2.4")


def test_supported_python_series_matches_declared_project_range() -> None:
    """Expose exactly the Python series declared by the current release."""
    assert SUPPORTED_PYTHON_SERIES == (
        (3, 12),
        (3, 13),
        (3, 14),
    )


def test_validate_compatibility_metadata_accepts_plain_metadata() -> None:
    """Accept well-formed complete Cocoonet and Python runtime metadata."""
    metadata = CompatibilityMetadata(
        cocoonet_version="1.2.3+worker",
        python_version=(3, 14, 0),
        python_release_level="final",
    )

    validate_compatibility_metadata(metadata)

    assert metadata.cocoonet_version == "1.2.3+worker"
    assert metadata.python_version == (3, 14, 0)


def test_validate_compatibility_metadata_rejects_empty_cocoonet_version() -> None:
    """Reject an empty complete Cocoonet version string."""
    metadata = CompatibilityMetadata(
        cocoonet_version="",
        python_version=(3, 12, 0),
        python_release_level="final",
    )

    with pytest.raises(ValueError) as error:
        validate_compatibility_metadata(metadata)

    assert str(error.value) == (
        "Invalid compatibility metadata:\n"
        "  field: cocoonet_version\n"
        "  expected: nonempty complete Cocoonet version string\n"
        "  got: ''"
    )


@pytest.mark.parametrize(
    ("python_version", "expected"),
    (
        (cast(tuple[int, int, int], (3, 12)), "fixed tuple length mismatch"),
        (cast(tuple[int, int, int], (3, "12", 0)), "python_version[1]"),
        (cast(tuple[int, int, int], (3, True, 0)), "python_version[1]"),
    ),
)
def test_validate_compatibility_metadata_rejects_invalid_python_version_type(
    python_version: tuple[int, int, int],
    expected: str,
) -> None:
    """Reject malformed, noninteger, and boolean Python version components."""
    metadata = CompatibilityMetadata(
        cocoonet_version="1.2.3",
        python_version=python_version,
        python_release_level="final",
    )

    with pytest.raises(TypeError) as error:
        validate_compatibility_metadata(metadata)

    message = str(error.value)
    assert expected in message
    assert "python_version" in message


def test_validate_compatibility_metadata_rejects_negative_python_component() -> None:
    """Reject a negative Python version component."""
    metadata = CompatibilityMetadata(
        cocoonet_version="1.2.3",
        python_version=(3, -1, 0),
        python_release_level="final",
    )

    with pytest.raises(ValueError) as error:
        validate_compatibility_metadata(metadata)

    assert str(error.value) == (
        "Invalid compatibility metadata:\n"
        "  field: python_version[1]\n"
        "  expected: nonnegative integer version component\n"
        "  got: -1"
    )


def test_validate_compatibility_metadata_rejects_unknown_release_level() -> None:
    """Reject a Python release level outside the declared vocabulary."""
    metadata = CompatibilityMetadata(
        cocoonet_version="1.2.3",
        python_version=(3, 12, 0),
        python_release_level=cast(
            Literal["alpha", "beta", "candidate", "final"],
            "preview",
        ),
    )

    with pytest.raises(TypeError) as error:
        validate_compatibility_metadata(metadata)

    message = str(error.value)
    assert "python_release_level" in message
    assert "alpha" in message
    assert "preview" in message


def test_validate_compatibility_metadata_rejects_wrong_outer_record() -> None:
    """Reject a value that is not a compatibility-metadata record."""
    with pytest.raises(TypeError) as error:
        validate_compatibility_metadata(
            cast(CompatibilityMetadata, None),
        )

    message = str(error.value)
    assert "CompatibilityMetadata" in message
    assert "NoneType" in message
    assert "None" in message


def test_require_compatible_accepts_every_supported_python_series_pair() -> None:
    """Allow differing final Python versions throughout the supported range."""
    # Exercise every ordered root/worker series pairing.
    for root_series in SUPPORTED_PYTHON_SERIES:
        for worker_series in SUPPORTED_PYTHON_SERIES:
            root = CompatibilityMetadata(
                cocoonet_version="1.2.3",
                python_version=(*root_series, 1),
                python_release_level="final",
            )
            worker = CompatibilityMetadata(
                cocoonet_version="1.2.3",
                python_version=(*worker_series, 9),
                python_release_level="final",
            )

            require_compatible(root, worker)


@pytest.mark.parametrize(
    "worker_version",
    (
        "1.2.4",
        "1.2.3rc1",
        "1.2.3+worker",
    ),
)
def test_require_compatible_rejects_complete_cocoonet_version_difference(
    worker_version: str,
) -> None:
    """Reject patch, prerelease, and local-suffix differences without normalization."""
    root = CompatibilityMetadata(
        cocoonet_version="1.2.3",
        python_version=(3, 12, 0),
        python_release_level="final",
    )
    worker = CompatibilityMetadata(
        cocoonet_version=worker_version,
        python_version=(3, 12, 0),
        python_release_level="final",
    )

    with pytest.raises(ValueError) as error:
        require_compatible(root, worker)

    message = str(error.value)
    assert message.startswith("Incompatible Cocoonet versions:")
    assert "root='1.2.3'" in message
    assert f"worker={worker_version!r}" in message
    assert "install the same complete Cocoonet version" in message


@pytest.mark.parametrize(
    ("role", "release_level"),
    (
        ("root", "candidate"),
        ("worker", "alpha"),
    ),
)
def test_require_compatible_rejects_nonfinal_python_release(
    role: str,
    release_level: Literal["alpha", "beta", "candidate", "final"],
) -> None:
    """Reject a nonfinal Python runtime on either endpoint."""
    root = CompatibilityMetadata(
        cocoonet_version="1.2.3",
        python_version=(3, 12, 0),
        python_release_level=release_level if role == "root" else "final",
    )
    worker = CompatibilityMetadata(
        cocoonet_version="1.2.3",
        python_version=(3, 13, 0),
        python_release_level=release_level if role == "worker" else "final",
    )

    with pytest.raises(ValueError) as error:
        require_compatible(root, worker)

    message = str(error.value)
    assert message.startswith("Incompatible Python runtime:")
    assert f"runtime: {role}" in message
    assert "expected: final Python release" in message
    assert f"release_level={release_level!r}" in message


@pytest.mark.parametrize(
    ("role", "python_version"),
    (
        ("root", (3, 11, 9)),
        ("worker", (3, 15, 0)),
    ),
)
def test_require_compatible_rejects_unsupported_python_series(
    role: str,
    python_version: tuple[int, int, int],
) -> None:
    """Reject an unsupported Python series on either endpoint."""
    root = CompatibilityMetadata(
        cocoonet_version="1.2.3",
        python_version=python_version if role == "root" else (3, 12, 0),
        python_release_level="final",
    )
    worker = CompatibilityMetadata(
        cocoonet_version="1.2.3",
        python_version=python_version if role == "worker" else (3, 13, 0),
        python_release_level="final",
    )

    with pytest.raises(ValueError) as error:
        require_compatible(root, worker)

    message = str(error.value)
    assert message.startswith("Incompatible Python runtime:")
    assert f"runtime: {role}" in message
    assert "supported Python series (3.12, 3.13, 3.14)" in message
    assert ".".join(str(component) for component in python_version) in message


def test_require_compatible_identifies_invalid_worker_metadata() -> None:
    """Identify the endpoint containing invalid compatibility metadata."""
    root = CompatibilityMetadata(
        cocoonet_version="1.2.3",
        python_version=(3, 12, 0),
        python_release_level="final",
    )
    worker = CompatibilityMetadata(
        cocoonet_version="",
        python_version=(3, 13, 0),
        python_release_level="final",
    )

    with pytest.raises(ValueError) as error:
        require_compatible(root, worker)

    message = str(error.value)
    assert message.startswith("worker metadata is invalid:")
    assert "cocoonet_version" in message
    assert "nonempty complete Cocoonet version string" in message
