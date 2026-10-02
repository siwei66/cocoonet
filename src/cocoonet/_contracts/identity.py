"""Shared scalar types and opaque identities for internal component contracts."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import NewType

# Preserve scalar label values without coercing them to strings.
# Finite-float and uniqueness requirements are enforced by later validators.
type ScalarLabel = str | int | float | bool

# Target admissibility is checked separately from label validity.
type TargetValue = str | int | float | bool

# Distinct static types prevent accidental interchange of identity roles.
# Runtime value and association checks belong to the planned validators.
DatasetId = NewType("DatasetId", str)
SampleId = NewType("SampleId", str)
ModelId = NewType("ModelId", str)
ParentId = NewType("ParentId", str)
AttemptId = NewType("AttemptId", str)
FoldId = NewType("FoldId", str)
WorkerId = NewType("WorkerId", str)
PartitionId = NewType("PartitionId", str)
EvaluationConfigId = NewType("EvaluationConfigId", str)
FailureId = NewType("FailureId", str)
OperationId = NewType("OperationId", str)
ArtifactId = NewType("ArtifactId", str)


class ReportName(StrEnum):
    """Identify report categories shared by outcomes and failure context."""

    VALIDATION = "validation"
    PERFORMANCE = "performance"
    RESIDUAL = "residual"
    VISUALIZATION = "visualization"
    INFLUENCE = "influence"


@dataclass(frozen=True, kw_only=True, slots=True)
class UnitKey:
    """Identify one model-evaluation unit within its parent.

    Parameters
    ----------
    parent_id : ParentId
        Parent evaluation containing the unit.
    model_id : ModelId
        Fixed model input evaluated by the unit.

    Notes
    -----
    Construction stores candidate identities without validating their values.
    Dataset and evaluation settings resolve through the parent.
    """

    parent_id: ParentId
    model_id: ModelId


def validate_identifier(value: str, *, field: str) -> None:
    """Validate an opaque identifier without changing its representation.

    Parameters
    ----------
    value : str
        Identifier to validate.
    field : str
        Field name included in validation errors.

    Raises
    ------
    TypeError
        If the identifier is not a string.
    ValueError
        If the identifier is empty.
    """
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string; got {type(value).__name__}.")
    if not value:
        raise ValueError(f"{field} must be a nonempty string.")


def validate_label(value: ScalarLabel, *, field: str) -> None:
    """Validate a scalar label without coercing its value.

    Parameters
    ----------
    value : ScalarLabel
        Sample, model, group, or class label to validate.
    field : str
        Field name included in validation errors.

    Raises
    ------
    TypeError
        If the label is not a string, integer, float, or boolean.
    ValueError
        If a float label is not finite.
    """
    if not isinstance(value, (str, int, float, bool)):
        raise TypeError(f"{field} must be a str, int, float, or bool; got {type(value).__name__}.")
    # Check floats only: arbitrary-size integer labels need no float conversion.
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{field} must be finite; got {value!r}.")


def validate_unit_key(value: UnitKey) -> None:
    """Validate the two identifiers forming a unit key.

    Parameters
    ----------
    value : UnitKey
        Candidate parent/model identity pair.

    Raises
    ------
    TypeError
        If the value is not a UnitKey or either identifier is not a string.
    ValueError
        If either identifier is empty.
    """
    if not isinstance(value, UnitKey):
        raise TypeError(f"value must be a UnitKey; got {type(value).__name__}.")
    validate_identifier(value.parent_id, field="parent_id")
    validate_identifier(value.model_id, field="model_id")
