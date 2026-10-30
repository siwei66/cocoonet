"""Plain compatibility contracts shared by Cocoonet components."""

from dataclasses import dataclass
from typing import Literal

from cocoonet._type_validation import validate_declared_types

SUPPORTED_PYTHON_SERIES: tuple[tuple[int, int], ...] = (
    (3, 12),
    (3, 13),
    (3, 14),
)


@dataclass(frozen=True, kw_only=True, slots=True)
class CompatibilityMetadata:
    """Record plain runtime metadata used for compatibility decisions.

    Parameters
    ----------
    cocoonet_version : str
        Complete installed Cocoonet version string.
    python_version : tuple[int, int, int]
        Python major, minor, and micro version numbers.
    python_release_level : Literal["alpha", "beta", "candidate", "final"]
        Python interpreter release level.

    Notes
    -----
    This immutable record contains metadata only. Construction preserves
    candidate values without validating them or deciding compatibility. It
    contains no model, report, executable payload, dependency inventory, or
    environment-repair behavior.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    cocoonet_version: str
    python_version: tuple[int, int, int]
    python_release_level: Literal["alpha", "beta", "candidate", "final"]


def validate_compatibility_metadata(value: CompatibilityMetadata) -> None:
    """Validate plain runtime compatibility metadata.

    Parameters
    ----------
    value : CompatibilityMetadata
        Candidate Cocoonet and Python runtime metadata.

    Raises
    ------
    TypeError
        If the record or one of its fields has an incorrect runtime type.
    ValueError
        If the Cocoonet version is empty or a Python version component is
        negative.

    Notes
    -----
    The Cocoonet version remains an opaque complete string. This function does
    not normalize versions, compare environments, inspect executable payloads,
    negotiate dependencies, query a network, or repair an environment.

    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Validate the outer record before accessing its metadata.
    if not isinstance(value, CompatibilityMetadata):
        validate_declared_types(
            value,
            expected_type=CompatibilityMetadata,
        )

    # Validate the complete Cocoonet version as a nonempty opaque string.
    validate_declared_types(
        value.cocoonet_version,
        expected_type=str,
        field="cocoonet_version",
    )
    if value.cocoonet_version == "":
        raise ValueError(
            "Invalid compatibility metadata:\n"
            "  field: cocoonet_version\n"
            "  expected: nonempty complete Cocoonet version string\n"
            "  got: ''"
        )

    # Validate the exact Python version shape and strict integer components.
    validate_declared_types(
        value.python_version,
        expected_type=tuple[int, int, int],
        field="python_version",
    )
    for index, component in enumerate(value.python_version):
        if component < 0:
            raise ValueError(
                "Invalid compatibility metadata:\n"
                f"  field: python_version[{index}]\n"
                "  expected: nonnegative integer version component\n"
                f"  got: {component!r}"
            )

    # Validate the interpreter release-level vocabulary.
    validate_declared_types(
        value.python_release_level,
        expected_type=Literal["alpha", "beta", "candidate", "final"],
        field="python_release_level",
    )


def require_compatible(
    root: CompatibilityMetadata,
    worker: CompatibilityMetadata,
) -> None:
    """Require compatible Cocoonet and Python runtime metadata.

    Parameters
    ----------
    root : CompatibilityMetadata
        Runtime metadata for the coordinating root.
    worker : CompatibilityMetadata
        Runtime metadata for the candidate worker.

    Raises
    ------
    TypeError
        If either metadata record or one of its fields has an incorrect runtime
        type.
    ValueError
        If metadata is invalid, complete Cocoonet versions differ, a Python
        runtime is not a final release, or a Python series is unsupported.

    Notes
    -----
    Complete Cocoonet version strings must match exactly. Root and worker Python
    versions may differ when both are final releases in the declared supported
    series.

    This pure guard does not normalize versions, load model or report payloads,
    negotiate dependencies, query a network, repair an environment, or change
    worker state.

    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Validate both metadata records and identify which endpoint is invalid.
    for role, metadata in (("root", root), ("worker", worker)):
        try:
            validate_compatibility_metadata(metadata)
        except (TypeError, ValueError) as error:
            raise type(error)(f"{role} metadata is invalid: {error}") from error

    # Reject framework-version differences before Python compatibility checks.
    if root.cocoonet_version != worker.cocoonet_version:
        raise ValueError(
            "Incompatible Cocoonet versions:\n"
            f"  expected: worker version to exactly equal root version "
            f"{root.cocoonet_version!r}\n"
            f"  got: root={root.cocoonet_version!r}, "
            f"worker={worker.cocoonet_version!r}\n"
            "  detail: install the same complete Cocoonet version on root and worker"
        )

    # Require final Python releases within the declared supported series.
    supported = ", ".join(f"{major}.{minor}" for major, minor in SUPPORTED_PYTHON_SERIES)
    for role, metadata in (("root", root), ("worker", worker)):
        if metadata.python_release_level != "final":
            raise ValueError(
                "Incompatible Python runtime:\n"
                f"  runtime: {role}\n"
                "  expected: final Python release\n"
                f"  got: release_level={metadata.python_release_level!r}, "
                f"version={metadata.python_version!r}"
            )

        series = metadata.python_version[:2]
        if series not in SUPPORTED_PYTHON_SERIES:
            version = ".".join(str(component) for component in metadata.python_version)
            raise ValueError(
                "Incompatible Python runtime:\n"
                f"  runtime: {role}\n"
                f"  expected: supported Python series ({supported})\n"
                f"  got: Python {version}"
            )
