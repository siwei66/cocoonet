"""Standardized prediction records for internal component contracts."""

from dataclasses import dataclass
from math import isfinite

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
    values : tuple[float | None, ...]
        Predicted target values aligned one-to-one with ``sample_ids``.
    failure_ids : tuple[tuple[FailureId, ...], ...]
        Value-level failure references aligned one-to-one with ``values``.

    Notes
    -----
    A present value is a finite ``float``, has an empty aligned failure tuple,
    and is rendered as its numeric value in human-readable output. ``None``
    means that prediction is computationally unavailable, requires a nonempty
    aligned failure tuple, and is rendered as ``NA``.

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
    values: tuple[float | None, ...]
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
    probabilities : tuple[tuple[float | None, ...], ...]
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

    A present probability is a finite ``float`` in the inclusive interval
    ``[0, 1]``, has an empty aligned failure tuple, and is rendered as its
    numeric value. A probability of ``None`` is computationally unavailable,
    requires a nonempty aligned failure tuple, and is rendered as ``NA``.

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
    probabilities: tuple[tuple[float | None, ...], ...]
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
    """Validate standardized regression predictions against one fold.

    Parameters
    ----------
    value : RegressionPredictions
        Ordered regression predictions and value-level failure references to
        validate.
    fold : FoldRecord
        Realized fold supplying the authoritative testing-sample sequence.

    Raises
    ------
    TypeError
        If the prediction record, fold, or a nested field has an incorrect
        runtime type.
    ValueError
        If an identity is empty, the fold or testing-sample sequence does not
        match, vectors are misaligned, a present value is nonfinite or has
        failure references, or an unavailable value lacks failure references.

    Notes
    -----
    Present predictions must be finite floats with empty failure-reference
    tuples. ``None`` means computationally unavailable and requires at least
    one failure ID. Non-applicable ``None``, ``NaN``, and infinity are invalid.

    This validator preserves input order and values. It does not reorder
    samples, coerce numbers, resolve failure IDs, or access dataset contents.

    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Validate both declared structures before applying alignment rules.
    validate_declared_types(
        value,
        expected_type=RegressionPredictions,
    )
    validate_declared_types(
        fold,
        expected_type=FoldRecord,
    )

    # Validate the identities consumed directly by this prediction boundary.
    validate_identifier(value.fold_id, field="fold_id")
    validate_identifier(fold.fold_id, field="fold.fold_id")
    for index, sample_id in enumerate(value.sample_ids):
        validate_identifier(sample_id, field=f"sample_ids[{index}]")
    for index, sample_id in enumerate(fold.testing_sample_ids):
        validate_identifier(
            sample_id,
            field=f"fold.testing_sample_ids[{index}]",
        )

    # Require the exact fold identity and saved testing-sample order.
    if value.fold_id != fold.fold_id:
        raise ValueError(f"fold_id must equal fold.fold_id {fold.fold_id!r}; " f"got {value.fold_id!r}.")
    if value.sample_ids != fold.testing_sample_ids:
        raise ValueError(
            "sample_ids must exactly match fold.testing_sample_ids; "
            f"expected {fold.testing_sample_ids!r}, got {value.sample_ids!r}."
        )

    # Require one prediction and one failure-reference tuple per testing sample.
    expected_length = len(value.sample_ids)
    if len(value.values) != expected_length:
        raise ValueError(f"values must contain {expected_length} entries; " f"got {len(value.values)}.")
    if len(value.failure_ids) != expected_length:
        raise ValueError(f"failure_ids must contain {expected_length} entries; " f"got {len(value.failure_ids)}.")

    # Validate each value together with its aligned failure provenance.
    for index, (prediction, failure_ids) in enumerate(zip(value.values, value.failure_ids, strict=True)):
        if prediction is None:
            if not failure_ids:
                raise ValueError(f"values[{index}] is None and requires at least one " "failure ID.")
        else:
            if failure_ids:
                raise ValueError(f"values[{index}] is present and requires an empty " "failure_ids entry.")
            if not isfinite(prediction):
                raise ValueError(f"values[{index}] must be finite; got {prediction!r}.")

        for failure_index, failure_id in enumerate(failure_ids):
            validate_identifier(
                failure_id,
                field=f"failure_ids[{index}][{failure_index}]",
            )


def _validate_classification_context(
    value: ClassificationPredictions,
    *,
    fold: FoldRecord,
    class_labels: tuple[ScalarLabel, ...],
) -> None:
    """Validate classification identities and authoritative alignment.

    Parameters
    ----------
    value : ClassificationPredictions
        Classification prediction record to validate.
    fold : FoldRecord
        Realized fold supplying the authoritative testing-sample sequence.
    class_labels : tuple[ScalarLabel, ...]
        Authoritative class identities in probability-column order.

    Raises
    ------
    TypeError
        If a label or identity has an incorrect runtime type.
    ValueError
        If an identity or label is invalid, or the fold, samples, or classes
        do not exactly match their authoritative sequences.
    """
    validate_identifier(value.fold_id, field="fold_id")
    validate_identifier(fold.fold_id, field="fold.fold_id")
    for index, sample_id in enumerate(value.sample_ids):
        validate_identifier(sample_id, field=f"sample_ids[{index}]")
    for index, sample_id in enumerate(fold.testing_sample_ids):
        validate_identifier(
            sample_id,
            field=f"fold.testing_sample_ids[{index}]",
        )
    for index, label in enumerate(value.class_labels):
        validate_label(label, field=f"value.class_labels[{index}]")
    for index, label in enumerate(class_labels):
        validate_label(label, field=f"class_labels[{index}]")

    if value.fold_id != fold.fold_id:
        raise ValueError(f"fold_id must equal fold.fold_id {fold.fold_id!r}; " f"got {value.fold_id!r}.")
    if value.sample_ids != fold.testing_sample_ids:
        raise ValueError(
            "sample_ids must exactly match fold.testing_sample_ids; "
            f"expected {fold.testing_sample_ids!r}, got {value.sample_ids!r}."
        )
    if value.class_labels != class_labels:
        raise ValueError(
            "value.class_labels must exactly match class_labels; "
            f"expected {class_labels!r}, got {value.class_labels!r}."
        )


def _validate_classification_shapes(value: ClassificationPredictions) -> None:
    """Validate sample-vector and probability-matrix dimensions.

    Parameters
    ----------
    value : ClassificationPredictions
        Classification prediction record whose aligned dimensions are
        validated.

    Raises
    ------
    ValueError
        If a sample-aligned vector or probability matrix has an incompatible
        length or row width.
    """
    sample_count = len(value.sample_ids)
    class_count = len(value.class_labels)

    if len(value.predicted_labels) != sample_count:
        raise ValueError(
            f"predicted_labels must contain {sample_count} sample entries; " f"got {len(value.predicted_labels)}."
        )
    if len(value.label_failure_ids) != sample_count:
        raise ValueError(
            f"label_failure_ids must contain {sample_count} sample entries; " f"got {len(value.label_failure_ids)}."
        )
    if len(value.probabilities) != sample_count:
        raise ValueError(f"probabilities must contain {sample_count} sample rows; " f"got {len(value.probabilities)}.")
    if len(value.probability_failure_ids) != sample_count:
        raise ValueError(
            "probability_failure_ids must contain "
            f"{sample_count} sample rows; "
            f"got {len(value.probability_failure_ids)}."
        )

    for sample_index, probabilities in enumerate(value.probabilities):
        if len(probabilities) != class_count:
            raise ValueError(
                f"probabilities[{sample_index}] must contain {class_count} " f"class entries; got {len(probabilities)}."
            )
    for sample_index, failure_row in enumerate(value.probability_failure_ids):
        if len(failure_row) != class_count:
            raise ValueError(
                f"probability_failure_ids[{sample_index}] must contain "
                f"{class_count} class entries; got {len(failure_row)}."
            )


def _validate_predicted_labels(
    value: ClassificationPredictions,
    *,
    class_labels: tuple[ScalarLabel, ...],
) -> None:
    """Validate predicted labels and their aligned failure provenance.

    Parameters
    ----------
    value : ClassificationPredictions
        Classification prediction record to validate.
    class_labels : tuple[ScalarLabel, ...]
        Authoritative class identities accepted for present predictions.

    Raises
    ------
    ValueError
        If a present prediction is outside the authoritative class set or has
        failure references, an unavailable prediction lacks a failure
        reference, or a failure identity is empty.
    """
    for sample_index, (label, failure_ids) in enumerate(
        zip(
            value.predicted_labels,
            value.label_failure_ids,
            strict=True,
        )
    ):
        if label is None:
            if not failure_ids:
                raise ValueError(f"predicted_labels[{sample_index}] is None and requires " "at least one failure ID.")
        else:
            if label not in class_labels:
                raise ValueError(f"predicted_labels[{sample_index}] must belong to " f"class_labels; got {label!r}.")
            if failure_ids:
                raise ValueError(
                    f"predicted_labels[{sample_index}] is present and " "requires an empty label_failure_ids entry."
                )

        for failure_index, failure_id in enumerate(failure_ids):
            validate_identifier(
                failure_id,
                field=(f"label_failure_ids[{sample_index}]" f"[{failure_index}]"),
            )


def _validate_probability_values(value: ClassificationPredictions) -> None:
    """Validate probability cells and their aligned failure provenance.

    Parameters
    ----------
    value : ClassificationPredictions
        Classification prediction record to validate.

    Raises
    ------
    ValueError
        If a present probability is nonfinite, outside ``[0, 1]``, or has
        failure references; an unavailable probability lacks a failure
        reference; or a failure identity is empty.
    """
    for sample_index, (probability_row, failure_row) in enumerate(
        zip(
            value.probabilities,
            value.probability_failure_ids,
            strict=True,
        )
    ):
        # Each probability cell is validated with its aligned failure tuple.
        for class_index, (probability, failure_ids) in enumerate(zip(probability_row, failure_row, strict=True)):
            if probability is None:
                if not failure_ids:
                    raise ValueError(
                        f"probabilities[{sample_index}][{class_index}] is " "None and requires at least one failure ID."
                    )
            else:
                if failure_ids:
                    raise ValueError(
                        f"probabilities[{sample_index}][{class_index}] is "
                        "present and requires an empty "
                        "probability_failure_ids entry."
                    )
                if not isfinite(probability):
                    raise ValueError(
                        f"probabilities[{sample_index}][{class_index}] must " f"be finite; got {probability!r}."
                    )
                if not 0.0 <= probability <= 1.0:
                    raise ValueError(
                        f"probabilities[{sample_index}][{class_index}] must " f"be within [0, 1]; got {probability!r}."
                    )

            for failure_index, failure_id in enumerate(failure_ids):
                validate_identifier(
                    failure_id,
                    field=(f"probability_failure_ids[{sample_index}]" f"[{class_index}][{failure_index}]"),
                )


def validate_classification_predictions(
    value: ClassificationPredictions,
    *,
    fold: FoldRecord,
    class_labels: tuple[ScalarLabel, ...],
) -> None:
    """Validate standardized classification predictions against one fold.

    Parameters
    ----------
    value : ClassificationPredictions
        Ordered classification predictions and value-level failure references
        to validate.
    fold : FoldRecord
        Realized fold supplying the authoritative testing-sample sequence.
    class_labels : tuple[ScalarLabel, ...]
        Authoritative class identities in probability-column order.

    Raises
    ------
    TypeError
        If the prediction record, fold, class-label sequence, or a nested field
        has an incorrect runtime type.
    ValueError
        If identities or labels are invalid, authoritative sequences do not
        match, vectors or matrices are misaligned, a present result is invalid
        or has failure references, or an unavailable result lacks failure
        references.

    Notes
    -----
    Present predicted labels must belong to ``class_labels`` and have empty
    failure-reference tuples. Present probabilities must be finite floats in
    ``[0, 1]`` and have empty failure-reference tuples. ``None`` means
    computationally unavailable and requires at least one aligned failure ID.

    This validator preserves input order and values. It does not reorder
    samples or classes, coerce labels or probabilities, normalize probability
    rows, resolve failure IDs, or access dataset contents.

    Node usage scope: Shared - Used by both root and worker nodes.
    """
    validate_declared_types(
        value,
        expected_type=ClassificationPredictions,
    )
    validate_declared_types(
        fold,
        expected_type=FoldRecord,
    )
    validate_declared_types(
        class_labels,
        expected_type=tuple[ScalarLabel, ...],
        field="class_labels",
    )

    _validate_classification_context(
        value,
        fold=fold,
        class_labels=class_labels,
    )
    _validate_classification_shapes(value)
    _validate_predicted_labels(value, class_labels=class_labels)
    _validate_probability_values(value)
