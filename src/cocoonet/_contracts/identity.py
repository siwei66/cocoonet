"""Shared scalar types and opaque identities for internal component contracts."""

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType

import numpy as np

from cocoonet._type_validation import validate_declared_types

# Standardized labels retain their string representation without conversion.
# Consuming validators enforce scoped uniqueness and associations.
type ScalarLabel = str | np.str_

# Classification targets are strings; regression targets are floating scalars.
# Leave NumPy precision unrestricted and preserve each value's representation.
# Consuming validators enforce finiteness and the task-specific target type.
type TargetValue = ScalarLabel | float | np.floating

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
    """Identify report categories shared by outcomes and failure context.

    Notes
    -----
    Node usage scope: Shared - Used by both root and worker nodes.
    """

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
    Node usage scope: Shared - Used by both root and worker nodes.
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
        If the identifier is not a string or ``field`` is not a string.
    ValueError
        If the identifier is empty.

    Notes
    -----
    Node usage scope: Shared - Used by both root and worker nodes.
    """
    validate_declared_types(
        value,
        expected_type=str,
        field=field,
    )

    if not value:
        raise ValueError(f"{field} must be a nonempty string.")


def validate_label(value: ScalarLabel, *, field: str) -> None:
    """Validate a standardized string label without conversion.

    Parameters
    ----------
    value : ScalarLabel
        Standardized sample, model, group, or class label.
    field : str
        Field path included in validation errors.

    Raises
    ------
    TypeError
        If the label is not a string or the diagnostic field path
        is not a built-in string.

    Notes
    -----
    String subclasses, including NumPy string scalars, are accepted
    without changing their type or value. Input conversion belongs to
    standardization. This check does not enforce scoped uniqueness
    or nonempty-label requirements.
    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Validate diagnostic metadata even when the label is accepted directly.
    validate_declared_types(field, expected_type=str, field="field")

    # Accept string families without exact-type restrictions or conversion.
    if isinstance(value, (str, np.str_)):
        return

    # Reuse the existing bounded diagnostics for rejected label values.
    validate_declared_types(value, expected_type=ScalarLabel, field=field)


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

    Notes
    -----
    Node usage scope: Shared - Used by both root and worker nodes.
    """
    validate_declared_types(
        value,
        expected_type=UnitKey,
    )

    validate_identifier(value.parent_id, field="parent_id")
    validate_identifier(value.model_id, field="model_id")
