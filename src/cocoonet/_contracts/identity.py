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
# Local validators check values; consuming boundaries check identity associations.
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
    """Identify report categories for callers and failure context.

    Notes
    -----
    A complete result's fixed report slot supplies its report category.
    Independently consumed reports receive that category from their caller;
    ReportOutcome does not store a duplicate name.
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
    Allocation and operational records use this independently consumed key.
    The consuming boundary verifies that the model belongs to the parent;
    dataset and evaluation settings resolve through that parent.
    Node usage scope: Shared - Used by both root and worker nodes.
    """

    parent_id: ParentId
    model_id: ModelId


def validate_identifier(value: str, *, field: str) -> None:
    """Validate a nonempty opaque string identifier without conversion.

    Parameters
    ----------
    value : str
        Identifier to validate, including supported string subclasses.
    field : str
        Field path included in validation errors.

    Raises
    ------
    TypeError
        If the identifier is not a string or the diagnostic field path
        is not a built-in string.
    ValueError
        If the identifier is empty.

    Notes
    -----
    String subclasses, including NumPy string scalars, retain their type
    and value. Whitespace and spelling are preserved. This check does not
    infer an identifier format, enforce uniqueness, or resolve associations.
    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Share the string-family check without imposing exact built-in types.
    validate_label(value, field=field)

    # Identifiers require content, but whitespace has no special meaning.
    if not value:
        raise ValueError(f"{field} must be a nonempty string; got an empty string.")


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
    """Validate a unit key's record type and two opaque identifiers.

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
    Each identifier is validated without conversion. This function does
    not resolve a parent or establish that its model collection contains
    the referenced model; those checks belong to the consuming boundary.
    Validation does not mutate the key.
    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Check the outer record without recursively validating its fields.
    # Incorrect outer types use the established bounded diagnostic path.
    if not isinstance(value, UnitKey):
        validate_declared_types(value, expected_type=UnitKey)

    # Delegate each field's string and nonempty requirements to its owner.
    validate_identifier(value.parent_id, field="parent_id")
    validate_identifier(value.model_id, field="model_id")
