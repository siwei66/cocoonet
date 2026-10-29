"""Standardized prediction records for internal component contracts."""

from dataclasses import dataclass

import numpy as np

from cocoonet._contracts.data import FoldRecord
from cocoonet._contracts.identity import (
    FailureId,
    FoldId,
    SampleId,
    ScalarLabel,
    validate_identifier,
    validate_label,
)
from cocoonet._type_validation import validate_declared_types


@dataclass(frozen=True, kw_only=True, slots=True)
class RegressionPredictions:
    """Record standardized regression predictions for one testing fold.

    Parameters
    ----------
    fold_id : FoldId
        Identity of the realized fold that produced these predictions.
    sample_ids : tuple[SampleId, ...]
        Testing-sample identities in the exact saved fold order.
    values : tuple[float | np.floating | None, ...]
        Predicted target values aligned one-to-one with ``sample_ids``.
    failure_ids : tuple[tuple[FailureId, ...], ...]
        Value-level failure references aligned one-to-one with ``values``.

    Notes
    -----
    A present value is a finite Python or NumPy floating-point scalar, has an
    empty aligned failure tuple, and is rendered as its numeric value in
    human-readable output. Its scalar type and precision are preserved without
    conversion to a Python float. NumPy floating-point precision is unrestricted.
    ``None`` means that prediction is computationally unavailable, requires a
    nonempty aligned failure tuple, and is rendered as ``NA``.

    Integers, booleans, complex numbers, arrays, and tensors are not valid
    prediction scalars. Tensor extraction and device handling belong to model
    adapters.

    Non-applicable ``None`` is not permitted because every saved testing sample
    requires a prediction result. ``NaN`` and positive or negative infinity
    are not valid standardized regression predictions and therefore are not
    rendered as ``NaN``, ``Inf``, or ``-Inf`` from this record.

    Construction preserves candidate values without validating alignment,
    identities, finiteness, or failure references. Those rules belong to
    ``validate_regression_predictions``.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    fold_id: FoldId
    sample_ids: tuple[SampleId, ...]
    values: tuple[float | np.floating | None, ...]
    failure_ids: tuple[tuple[FailureId, ...], ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class ClassificationPredictions:
    """Record standardized classification predictions for one testing fold.

    Parameters
    ----------
    fold_id : FoldId
        Identity of the realized fold that produced these predictions.
    sample_ids : tuple[SampleId, ...]
        Testing-sample identities in the exact saved fold order.
    class_labels : tuple[ScalarLabel, ...]
        Class identities in the exact parent-defined column order.
    predicted_labels : tuple[ScalarLabel | None, ...]
        Predicted class labels aligned one-to-one with ``sample_ids``.
    label_failure_ids : tuple[tuple[FailureId, ...], ...]
        Value-level failure references aligned with ``predicted_labels``.
    probabilities : tuple[tuple[float | np.floating | None, ...], ...]
        Per-sample class probabilities whose columns align with
        ``class_labels``.
    probability_failure_ids : tuple[tuple[tuple[FailureId, ...], ...], ...]
        Value-level failure references aligned with every probability cell.

    Notes
    -----
    A present predicted label belongs to ``class_labels``, has an empty aligned
    failure tuple, and is rendered as that label in human-readable output.
    ``None`` means that the predicted label is computationally unavailable,
    requires a nonempty aligned failure tuple, and is rendered as ``NA``.

    A present probability is a finite Python or NumPy floating-point scalar in
    the inclusive interval ``[0, 1]``, has an empty aligned failure tuple, and
    is rendered as its numeric value. Its scalar type and precision are
    preserved without conversion to a Python float. NumPy floating-point
    precision is unrestricted. A probability of ``None`` is computationally
    unavailable, requires a nonempty aligned failure tuple, and is rendered
    as ``NA``.

    Integers, booleans, complex numbers, arrays, and tensors are not valid
    probability scalars. Tensor extraction and device handling belong to model
    adapters.

    Non-applicable ``None`` is not permitted for predicted labels or
    probability cells because every saved testing sample requires those
    results. ``NaN`` and positive or negative infinity are not valid
    standardized probabilities and therefore are not rendered as ``NaN``,
    ``Inf``, or ``-Inf`` from this record.

    Construction preserves candidate values without validating alignment,
    class membership, matrix shape, numerical admissibility, or failure
    references. Those rules belong to
    ``validate_classification_predictions``.

    Node usage scope: Shared - Used by both root and worker nodes.
    """

    fold_id: FoldId
    sample_ids: tuple[SampleId, ...]
    class_labels: tuple[ScalarLabel, ...]
    predicted_labels: tuple[ScalarLabel | None, ...]
    label_failure_ids: tuple[tuple[FailureId, ...], ...]
    probabilities: tuple[tuple[float | np.floating | None, ...], ...]
    probability_failure_ids: tuple[
        tuple[tuple[FailureId, ...], ...],
        ...,
    ]


type PredictionBatch = RegressionPredictions | ClassificationPredictions


def validate_regression_predictions(  # noqa: C901
    value: RegressionPredictions,
    *,
    fold: FoldRecord,
) -> None:
    """Validate regression values, saved sample order, and failure references.

    Parameters
    ----------
    value : RegressionPredictions
        Candidate fold predictions and position-aligned failure references.
    fold : FoldRecord
        Realized fold supplying the authoritative testing-sample order.

    Raises
    ------
    TypeError
        If a record, tuple, identifier, or present prediction has an
        incorrect runtime type.
    ValueError
        If an identity is empty, fold/sample alignment or vector length is
        invalid, a prediction is nonfinite, or failure associations are invalid.

    Notes
    -----
    Present predictions are Python or NumPy floating-point scalars, preserved
    without conversion or precision restriction. Finiteness is checked without
    narrowing the supplied scalar. Integers, booleans, complex numbers, arrays,
    and tensors are rejected. Present predictions require empty failure tuples;
    None requires at least one failure ID. NaN, infinity, and non-applicable
    None are invalid.
    Fold identifiers are validated here; dataset membership, masks, group
    separation, and train/test disjointness belong to schedule validation.
    Failure resolution belongs to the complete result boundary. This function
    preserves inputs and performs no conversion, reordering, or data access.
    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Validate record types before accessing prediction and fold fields.
    if not isinstance(value, RegressionPredictions):
        validate_declared_types(value, expected_type=RegressionPredictions)
    if not isinstance(fold, FoldRecord):
        validate_declared_types(fold, expected_type=FoldRecord, field="fold")

    # Validate fold identities and each supplied sample-identity sequence.
    validate_identifier(value.fold_id, field="fold_id")
    validate_identifier(fold.fold_id, field="fold.fold_id")
    for field, sample_ids in (
        ("sample_ids", value.sample_ids),
        ("fold.training_sample_ids", fold.training_sample_ids),
        ("fold.testing_sample_ids", fold.testing_sample_ids),
    ):
        validate_declared_types(sample_ids, expected_type=tuple, field=field)
        for index, sample_id in enumerate(sample_ids):
            validate_identifier(sample_id, field=f"{field}[{index}]")

    # Require the saved fold identity and exact testing-sample order.
    if value.fold_id != fold.fold_id:
        raise ValueError(f"fold_id must equal fold.fold_id {fold.fold_id!r}; got {value.fold_id!r}.")
    if len(value.sample_ids) != len(fold.testing_sample_ids):
        raise ValueError(
            f"sample_ids must contain {len(fold.testing_sample_ids)} saved testing entries; "
            f"got {len(value.sample_ids)}."
        )
    for index, (sample_id, expected_id) in enumerate(zip(value.sample_ids, fold.testing_sample_ids, strict=True)):
        if sample_id != expected_id:
            raise ValueError(
                f"sample_ids[{index}] must equal fold.testing_sample_ids[{index}] "
                f"{expected_id!r}; got {sample_id!r}."
            )

    # Require one prediction and one failure-reference tuple per testing sample.
    for field, entries in (("values", value.values), ("failure_ids", value.failure_ids)):
        validate_declared_types(entries, expected_type=tuple, field=field)
        if len(entries) != len(value.sample_ids):
            raise ValueError(f"{field} must contain {len(value.sample_ids)} entries; got {len(entries)}.")

    # Validate each prediction and its applicable failure identifiers.
    for index, (prediction, failure_ids) in enumerate(zip(value.values, value.failure_ids, strict=True)):
        validate_declared_types(failure_ids, expected_type=tuple, field=f"failure_ids[{index}]")
        for failure_index, failure_id in enumerate(failure_ids):
            validate_identifier(failure_id, field=f"failure_ids[{index}][{failure_index}]")
        if prediction is None:
            if not failure_ids:
                raise ValueError(f"values[{index}] is None and requires at least one failure ID; got zero.")
        else:
            # Require a floating scalar and check finiteness without narrowing precision.
            if not isinstance(prediction, (float, np.floating)):
                validate_declared_types(prediction, expected_type=float | np.floating, field=f"values[{index}]")
            if not np.isfinite(prediction):
                raise ValueError(f"values[{index}] must be finite; got {prediction!r}.")
            if failure_ids:
                raise ValueError(
                    f"values[{index}] is present and requires an empty failure_ids entry; "
                    f"got {len(failure_ids)} failure references."
                )


def _validate_classification_context(
    value: ClassificationPredictions,
    *,
    fold: FoldRecord,
    class_labels: tuple[ScalarLabel, ...],
) -> None:
    """Validate classification coordinates and authoritative sequence alignment.

    Parameters
    ----------
    value : ClassificationPredictions
        Candidate record whose outer type has been checked by the caller.
    fold : FoldRecord
        Realized fold whose outer type has been checked by the caller.
    class_labels : tuple[ScalarLabel, ...]
        Parent-provided class sequence; its tuple type is checked here.

    Raises
    ------
    TypeError
        If a coordinate sequence, identifier, or class label has an
        incorrect runtime type.
    ValueError
        If an identity is empty or fold, sample, or class alignment is invalid.

    Notes
    -----
    This helper validates identifier and label representations before comparing
    ordered coordinates. Class-vocabulary uniqueness and nonemptiness belong
    to _validate_predicted_labels. Dataset membership and partition rules
    require schedule context. No input is mutated.
    """
    # Validate fold identities and the supplied sample-identity sequences.
    validate_identifier(value.fold_id, field="fold_id")
    validate_identifier(fold.fold_id, field="fold.fold_id")
    for field, sample_ids in (
        ("sample_ids", value.sample_ids),
        ("fold.training_sample_ids", fold.training_sample_ids),
        ("fold.testing_sample_ids", fold.testing_sample_ids),
    ):
        validate_declared_types(sample_ids, expected_type=tuple, field=field)
        for index, sample_id in enumerate(sample_ids):
            validate_identifier(sample_id, field=f"{field}[{index}]")

    # Validate the standardized class labels in both ordered vocabularies.
    for field, labels in (("value.class_labels", value.class_labels), ("class_labels", class_labels)):
        validate_declared_types(labels, expected_type=tuple, field=field)
        for index, label in enumerate(labels):
            validate_label(label, field=f"{field}[{index}]")

    # Require the exact fold identity supplied by the saved partition.
    if value.fold_id != fold.fold_id:
        raise ValueError(f"fold_id must equal fold.fold_id {fold.fold_id!r}; got {value.fold_id!r}.")

    # Require equal sequence lengths and identify the first differing coordinate.
    for field, entries, expected_field, expected in (
        ("sample_ids", value.sample_ids, "fold.testing_sample_ids", fold.testing_sample_ids),
        ("value.class_labels", value.class_labels, "class_labels", class_labels),
    ):
        if len(entries) != len(expected):
            raise ValueError(f"{field} must contain {len(expected)} entries from {expected_field}; got {len(entries)}.")
        for index, (entry, expected_entry) in enumerate(zip(entries, expected, strict=True)):
            if entry != expected_entry:
                raise ValueError(
                    f"{field}[{index}] must equal {expected_field}[{index}] " f"{expected_entry!r}; got {entry!r}."
                )


def _validate_classification_shapes(value: ClassificationPredictions) -> None:
    """Validate tuple containers and aligned classification dimensions.

    Parameters
    ----------
    value : ClassificationPredictions
        Record with sample and class coordinates validated by
        _validate_classification_context.

    Raises
    ------
    TypeError
        If a vector, matrix, or matrix row is not a tuple.
    ValueError
        If a vector length or matrix row width is inconsistent with the
        validated sample and class counts.

    Notes
    -----
    This helper checks the containers needed for positional traversal.
    Predicted-label values, probability values, and per-value failure tuples
    belong to their respective value validators. No input is mutated.
    """
    # Require one entry or matrix row per saved testing sample.
    sample_count = len(value.sample_ids)
    class_count = len(value.class_labels)
    for field, entries in (
        ("predicted_labels", value.predicted_labels),
        ("label_failure_ids", value.label_failure_ids),
        ("probabilities", value.probabilities),
        ("probability_failure_ids", value.probability_failure_ids),
    ):
        validate_declared_types(entries, expected_type=tuple, field=field)
        if len(entries) != sample_count:
            raise ValueError(f"{field} must contain {sample_count} sample entries; got {len(entries)}.")

    # Require one probability position and failure position per declared class.
    for field, rows in (
        ("probabilities", value.probabilities),
        ("probability_failure_ids", value.probability_failure_ids),
    ):
        for sample_index, row in enumerate(rows):
            validate_declared_types(row, expected_type=tuple, field=f"{field}[{sample_index}]")
            if len(row) != class_count:
                raise ValueError(f"{field}[{sample_index}] must contain {class_count} class entries; got {len(row)}.")


def _validate_predicted_labels(
    value: ClassificationPredictions,
    *,
    class_labels: tuple[ScalarLabel, ...],
) -> None:
    """Validate the class vocabulary, predicted labels, and failure provenance.

    Parameters
    ----------
    value : ClassificationPredictions
        Record whose coordinate sequences and aligned outer containers
        have passed context and shape validation.
    class_labels : tuple[ScalarLabel, ...]
        Ordered string labels validated by _validate_classification_context.

    Raises
    ------
    TypeError
        If a present prediction, failure tuple, or failure ID has an
        incorrect runtime type.
    ValueError
        If the vocabulary is empty or duplicated, a label is unknown,
        a failure ID is empty, or a value/failure relationship is invalid.

    Notes
    -----
    A single class is valid. The local membership set preserves the stored
    vocabulary and avoids repeated linear searches for predicted labels.
    Each present label is validated without conversion. None requires
    failure provenance. No input is mutated or failure reference resolved.
    """
    # Require a nonempty unique class vocabulary and build a local membership set.
    if not class_labels:
        raise ValueError("class_labels must contain at least one class label; got zero.")
    known_labels: set[ScalarLabel] = set()
    for index, class_label in enumerate(class_labels):
        if class_label in known_labels:
            raise ValueError(f"class_labels[{index}] must be unique; got duplicate {class_label!r}.")
        known_labels.add(class_label)

    # Validate each predicted label and its applicable failure identifiers.
    for sample_index, (label, failure_ids) in enumerate(
        zip(value.predicted_labels, value.label_failure_ids, strict=True)
    ):
        validate_declared_types(failure_ids, expected_type=tuple, field=f"label_failure_ids[{sample_index}]")
        for failure_index, failure_id in enumerate(failure_ids):
            validate_identifier(failure_id, field=f"label_failure_ids[{sample_index}][{failure_index}]")
        if label is None:
            if not failure_ids:
                raise ValueError(
                    f"predicted_labels[{sample_index}] is None and requires at least one failure ID; got zero."
                )
        else:
            validate_label(label, field=f"predicted_labels[{sample_index}]")
            if label not in known_labels:
                raise ValueError(f"predicted_labels[{sample_index}] must belong to class_labels; got {label!r}.")
            if failure_ids:
                raise ValueError(
                    f"predicted_labels[{sample_index}] is present and requires an empty label_failure_ids entry; "
                    f"got {len(failure_ids)} failure references."
                )


def _validate_probability_values(value: ClassificationPredictions) -> None:
    """Validate probability values and their aligned failure provenance.

    Parameters
    ----------
    value : ClassificationPredictions
        Record whose sample/class coordinates and aligned matrix rows
        have passed context and shape validation.

    Raises
    ------
    TypeError
        If a present probability, failure tuple, or failure ID has an
        incorrect runtime type.
    ValueError
        If a probability is nonfinite or outside [0, 1], a failure ID
        is empty, or a value/failure relationship is invalid.

    Notes
    -----
    Probabilities are Python or NumPy floating-point scalars, preserved without
    conversion or precision restriction. Finiteness and bounds are checked
    without narrowing the supplied scalar. Integers, booleans, complex numbers,
    arrays, and tensors are rejected. Each None cell requires failure provenance
    independently of its predicted label and other probability cells. This
    helper does not normalize rows, resolve failure IDs, or mutate input.
    """
    # Validate each probability cell and its applicable failure identifiers.
    for sample_index, (probability_row, failure_row) in enumerate(
        zip(value.probabilities, value.probability_failure_ids, strict=True)
    ):
        for class_index, (probability, failure_ids) in enumerate(zip(probability_row, failure_row, strict=True)):
            value_field = f"probabilities[{sample_index}][{class_index}]"
            failure_field = f"probability_failure_ids[{sample_index}][{class_index}]"
            validate_declared_types(failure_ids, expected_type=tuple, field=failure_field)
            for failure_index, failure_id in enumerate(failure_ids):
                validate_identifier(failure_id, field=f"{failure_field}[{failure_index}]")
            if probability is None:
                if not failure_ids:
                    raise ValueError(f"{value_field} is None and requires at least one failure ID; got zero.")
            else:
                # Require a floating scalar and check finiteness without narrowing precision.
                if not isinstance(probability, (float, np.floating)):
                    validate_declared_types(probability, expected_type=float | np.floating, field=value_field)
                if not np.isfinite(probability):
                    raise ValueError(f"{value_field} must be finite; got {probability!r}.")
                if not 0.0 <= probability <= 1.0:
                    raise ValueError(f"{value_field} must be within [0, 1]; got {probability!r}.")
                if failure_ids:
                    raise ValueError(
                        f"{value_field} is present and requires an empty {failure_field} entry; "
                        f"got {len(failure_ids)} failure references."
                    )


def validate_classification_predictions(
    value: ClassificationPredictions,
    *,
    fold: FoldRecord,
    class_labels: tuple[ScalarLabel, ...],
) -> None:
    """Validate classification predictions against saved fold and class order.

    Parameters
    ----------
    value : ClassificationPredictions
        Candidate predicted labels, probabilities, and aligned failure IDs.
    fold : FoldRecord
        Realized fold supplying the authoritative testing-sample order.
    class_labels : tuple[ScalarLabel, ...]
        Nonempty unique parent class vocabulary in probability-column order.

    Raises
    ------
    TypeError
        If a record, container, identifier, label, or probability has an
        incorrect runtime type.
    ValueError
        If identities, vocabulary, alignment, dimensions, values, or their
        failure associations violate the prediction contract.

    Notes
    -----
    Present labels belong to the authoritative vocabulary. Present
    probabilities are finite Python or NumPy floating-point scalars in [0, 1],
    preserving their type and precision without conversion. Integers, booleans,
    complex numbers, arrays, and tensors are rejected as probabilities.
    A present result has no failure references; None requires at least one
    aligned failure ID. NaN, infinity, and non-applicable None are invalid.
    Each helper owns its stated structural and semantic checks. Dataset
    membership and partition eligibility require schedule context; failure
    resolution requires the complete result. Validation does not alter input,
    reorder coordinates, normalize probabilities, or access source data.
    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Validate record types before accessing prediction and fold fields.
    if not isinstance(value, ClassificationPredictions):
        validate_declared_types(value, expected_type=ClassificationPredictions)
    if not isinstance(fold, FoldRecord):
        validate_declared_types(fold, expected_type=FoldRecord, field="fold")

    # Validate exact fold, sample, and class associations.
    _validate_classification_context(value, fold=fold, class_labels=class_labels)

    # Validate tuple dimensions before traversing aligned result positions.
    _validate_classification_shapes(value)

    # Validate class membership and independent label/probability failure associations.
    _validate_predicted_labels(value, class_labels=class_labels)
    _validate_probability_values(value)
