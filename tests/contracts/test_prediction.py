"""Tests for standardized prediction contracts."""

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from math import isfinite
from typing import cast

import numpy as np
import pytest
import torch

from cocoonet._contracts.data import FoldRecord
from cocoonet._contracts.identity import (
    FailureId,
    FoldId,
    SampleId,
    ScalarLabel,
)
from cocoonet._contracts.prediction import (
    ClassificationPredictions,
    PredictionBatch,
    RegressionPredictions,
    validate_classification_predictions,
    validate_regression_predictions,
)
from cocoonet._type_validation import validate_declared_types


def test_regression_predictions_preserve_aligned_values_and_failures() -> None:
    """Preserve ordered regression values and their value-level failures."""
    predictions = RegressionPredictions(
        fold_id=FoldId("fold-a"),
        sample_ids=(
            SampleId("sample-a"),
            SampleId("sample-b"),
            SampleId("sample-c"),
        ),
        values=(1.25, None, -3.5),
        failure_ids=(
            (),
            (
                FailureId("failure-a"),
                FailureId("failure-b"),
            ),
            (),
        ),
    )
    same = RegressionPredictions(
        fold_id=FoldId("fold-a"),
        sample_ids=(
            SampleId("sample-a"),
            SampleId("sample-b"),
            SampleId("sample-c"),
        ),
        values=(1.25, None, -3.5),
        failure_ids=(
            (),
            (
                FailureId("failure-a"),
                FailureId("failure-b"),
            ),
            (),
        ),
    )

    assert predictions.fold_id == FoldId("fold-a")
    assert predictions.sample_ids == (
        SampleId("sample-a"),
        SampleId("sample-b"),
        SampleId("sample-c"),
    )
    assert predictions.values == (1.25, None, -3.5)
    assert predictions.failure_ids == (
        (),
        (
            FailureId("failure-a"),
            FailureId("failure-b"),
        ),
        (),
    )
    assert predictions == same


def test_regression_predictions_accept_empty_aligned_vectors() -> None:
    """Represent a fold with no permitted testing samples."""
    predictions = RegressionPredictions(
        fold_id=FoldId("fold-a"),
        sample_ids=(),
        values=(),
        failure_ids=(),
    )

    assert predictions.sample_ids == ()
    assert predictions.values == ()
    assert predictions.failure_ids == ()


def test_regression_predictions_reject_field_mutation() -> None:
    """Prevent reassignment of standardized prediction data."""
    predictions = RegressionPredictions(
        fold_id=FoldId("fold-a"),
        sample_ids=(SampleId("sample-a"),),
        values=(1.25,),
        failure_ids=((),),
    )

    with pytest.raises(FrozenInstanceError):
        predictions.__setattr__("values", (2.5,))


def test_classification_predictions_preserve_aligned_results() -> None:
    """Preserve class order and independently unavailable prediction values."""
    predictions = ClassificationPredictions(
        fold_id=FoldId("fold-a"),
        sample_ids=(
            SampleId("sample-a"),
            SampleId("sample-b"),
        ),
        class_labels=("negative", "positive"),
        predicted_labels=("positive", None),
        label_failure_ids=(
            (),
            (FailureId("label-failure"),),
        ),
        probabilities=(
            (0.25, 0.75),
            (None, 0.6),
        ),
        probability_failure_ids=(
            (
                (),
                (),
            ),
            (
                (FailureId("probability-failure"),),
                (),
            ),
        ),
    )
    same = ClassificationPredictions(
        fold_id=FoldId("fold-a"),
        sample_ids=(
            SampleId("sample-a"),
            SampleId("sample-b"),
        ),
        class_labels=("negative", "positive"),
        predicted_labels=("positive", None),
        label_failure_ids=(
            (),
            (FailureId("label-failure"),),
        ),
        probabilities=(
            (0.25, 0.75),
            (None, 0.6),
        ),
        probability_failure_ids=(
            (
                (),
                (),
            ),
            (
                (FailureId("probability-failure"),),
                (),
            ),
        ),
    )

    assert predictions.fold_id == FoldId("fold-a")
    assert predictions.sample_ids == (
        SampleId("sample-a"),
        SampleId("sample-b"),
    )
    assert predictions.class_labels == ("negative", "positive")
    assert predictions.predicted_labels == ("positive", None)
    assert predictions.label_failure_ids == (
        (),
        (FailureId("label-failure"),),
    )
    assert predictions.probabilities == (
        (0.25, 0.75),
        (None, 0.6),
    )
    assert predictions.probability_failure_ids == (
        (
            (),
            (),
        ),
        (
            (FailureId("probability-failure"),),
            (),
        ),
    )
    assert predictions == same


def test_classification_predictions_accept_empty_sample_rows() -> None:
    """Retain class order for a fold with no permitted testing samples."""
    predictions = ClassificationPredictions(
        fold_id=FoldId("fold-a"),
        sample_ids=(),
        class_labels=("negative", "positive"),
        predicted_labels=(),
        label_failure_ids=(),
        probabilities=(),
        probability_failure_ids=(),
    )

    assert predictions.sample_ids == ()
    assert predictions.class_labels == ("negative", "positive")
    assert predictions.predicted_labels == ()
    assert predictions.label_failure_ids == ()
    assert predictions.probabilities == ()
    assert predictions.probability_failure_ids == ()


def test_classification_predictions_reject_field_mutation() -> None:
    """Prevent reassignment of standardized classification predictions."""
    predictions = ClassificationPredictions(
        fold_id=FoldId("fold-a"),
        sample_ids=(SampleId("sample-a"),),
        class_labels=("negative", "positive"),
        predicted_labels=("positive",),
        label_failure_ids=((),),
        probabilities=((0.25, 0.75),),
        probability_failure_ids=(((), ()),),
    )

    with pytest.raises(FrozenInstanceError):
        predictions.__setattr__("predicted_labels", ("negative",))


@pytest.mark.parametrize(
    "predictions",
    [
        RegressionPredictions(
            fold_id=FoldId("fold-a"),
            sample_ids=(SampleId("sample-a"),),
            values=(1.25,),
            failure_ids=((),),
        ),
        ClassificationPredictions(
            fold_id=FoldId("fold-a"),
            sample_ids=(SampleId("sample-a"),),
            class_labels=("negative", "positive"),
            predicted_labels=("positive",),
            label_failure_ids=((),),
            probabilities=((0.25, 0.75),),
            probability_failure_ids=(((), ()),),
        ),
    ],
)
def test_prediction_batch_accepts_both_prediction_records(
    predictions: PredictionBatch,
) -> None:
    """Accept either standardized prediction record through the union alias.

    Parameters
    ----------
    predictions : PredictionBatch
        Regression or classification prediction record to validate.
    """
    validate_declared_types(
        predictions,
        expected_type=PredictionBatch,
    )


@pytest.fixture
def regression_fold() -> FoldRecord:
    """Provide one realized fold with an ordered regression testing sequence."""
    return FoldRecord(
        fold_id=FoldId("fold-a"),
        training_sample_ids=(SampleId("sample-train"),),
        testing_sample_ids=(
            SampleId("sample-a"),
            SampleId("sample-b"),
        ),
    )


def test_validate_regression_predictions_accepts_aligned_values_without_mutation(
    regression_fold: FoldRecord,
) -> None:
    """Accept finite and unavailable values while preserving the record.

    Parameters
    ----------
    regression_fold : FoldRecord
        Authoritative fold used for prediction alignment.
    """
    predictions = RegressionPredictions(
        fold_id=FoldId("fold-a"),
        sample_ids=(
            SampleId("sample-a"),
            SampleId("sample-b"),
        ),
        values=(1.25, None),
        failure_ids=(
            (),
            (FailureId("failure-a"),),
        ),
    )
    original = RegressionPredictions(
        fold_id=FoldId("fold-a"),
        sample_ids=(
            SampleId("sample-a"),
            SampleId("sample-b"),
        ),
        values=(1.25, None),
        failure_ids=(
            (),
            (FailureId("failure-a"),),
        ),
    )

    validate_regression_predictions(
        predictions,
        fold=regression_fold,
    )

    assert predictions == original


def test_validate_regression_predictions_accepts_empty_testing_sequence() -> None:
    """Accept empty aligned vectors for a fold without testing samples."""
    fold = FoldRecord(
        fold_id=FoldId("fold-a"),
        training_sample_ids=(SampleId("sample-train"),),
        testing_sample_ids=(),
    )
    predictions = RegressionPredictions(
        fold_id=FoldId("fold-a"),
        sample_ids=(),
        values=(),
        failure_ids=(),
    )

    validate_regression_predictions(predictions, fold=fold)


def test_validate_regression_predictions_rejects_wrong_fold(
    regression_fold: FoldRecord,
) -> None:
    """Reject predictions associated with another fold identity.

    Parameters
    ----------
    regression_fold : FoldRecord
        Authoritative fold used for prediction alignment.
    """
    predictions = RegressionPredictions(
        fold_id=FoldId("fold-b"),
        sample_ids=regression_fold.testing_sample_ids,
        values=(1.25, 2.5),
        failure_ids=((), ()),
    )

    with pytest.raises(ValueError) as error:
        validate_regression_predictions(predictions, fold=regression_fold)

    expected_details = "fold_id must equal fold.fold_id 'fold-a'; got 'fold-b'."
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


@pytest.mark.parametrize(
    "sample_ids",
    [
        (
            SampleId("sample-b"),
            SampleId("sample-a"),
        ),
        (
            SampleId("sample-a"),
            SampleId("sample-unknown"),
        ),
        (SampleId("sample-a"),),
    ],
)
def test_validate_regression_predictions_rejects_misaligned_samples(
    regression_fold: FoldRecord,
    sample_ids: tuple[SampleId, ...],
) -> None:
    """Reject reordered, unknown, or omitted testing-sample identities.

    Parameters
    ----------
    regression_fold : FoldRecord
        Authoritative fold used for prediction alignment.
    sample_ids : tuple[SampleId, ...]
        Invalid prediction sample sequence.
    """
    predictions = RegressionPredictions(
        fold_id=FoldId("fold-a"),
        sample_ids=sample_ids,
        values=tuple(1.0 for _ in sample_ids),
        failure_ids=tuple(() for _ in sample_ids),
    )

    with pytest.raises(ValueError) as error:
        validate_regression_predictions(predictions, fold=regression_fold)

    message = str(error.value)
    assert "sample_ids" in message
    if len(sample_ids) != len(regression_fold.testing_sample_ids):
        assert "2" in message and f"got {len(sample_ids)}" in message
    else:
        index = next(
            i
            for i, pair in enumerate(zip(sample_ids, regression_fold.testing_sample_ids, strict=True))
            if pair[0] != pair[1]
        )
        assert f"sample_ids[{index}]" in message
        assert repr(regression_fold.testing_sample_ids[index]) in message
        assert repr(sample_ids[index]) in message


@pytest.mark.parametrize(
    ("values", "failure_ids", "message"),
    [
        (
            (1.25,),
            ((), ()),
            "values must contain 2 entries; got 1.",
        ),
        (
            (1.25, 2.5),
            ((),),
            "failure_ids must contain 2 entries; got 1.",
        ),
    ],
)
def test_validate_regression_predictions_rejects_vector_length_mismatch(
    regression_fold: FoldRecord,
    values: tuple[float | None, ...],
    failure_ids: tuple[tuple[FailureId, ...], ...],
    message: str,
) -> None:
    """Reject prediction and failure vectors not aligned to testing samples.

    Parameters
    ----------
    regression_fold : FoldRecord
        Authoritative fold used for prediction alignment.
    values : tuple[float | None, ...]
        Candidate prediction vector.
    failure_ids : tuple[tuple[FailureId, ...], ...]
        Candidate failure-reference vector.
    message : str
        Expected validation error.
    """
    predictions = RegressionPredictions(
        fold_id=FoldId("fold-a"),
        sample_ids=regression_fold.testing_sample_ids,
        values=values,
        failure_ids=failure_ids,
    )

    with pytest.raises(ValueError) as error:
        validate_regression_predictions(predictions, fold=regression_fold)

    expected_details = message
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


@pytest.mark.parametrize(
    ("values", "failure_ids", "message"),
    [
        (
            (None, 2.5),
            ((), ()),
            "values[0] is None and requires at least one failure ID; got zero.",
        ),
        (
            (1.25, 2.5),
            ((FailureId("failure-a"),), ()),
            "values[0] is present and requires an empty failure_ids entry; got 1 failure references.",
        ),
    ],
)
def test_validate_regression_predictions_rejects_invalid_failure_relationship(
    regression_fold: FoldRecord,
    values: tuple[float | None, ...],
    failure_ids: tuple[tuple[FailureId, ...], ...],
    message: str,
) -> None:
    """Reject missing or contradictory value-level failure provenance.

    Parameters
    ----------
    regression_fold : FoldRecord
        Authoritative fold used for prediction alignment.
    values : tuple[float | None, ...]
        Candidate prediction vector.
    failure_ids : tuple[tuple[FailureId, ...], ...]
        Candidate failure-reference vector.
    message : str
        Expected validation error.
    """
    predictions = RegressionPredictions(
        fold_id=FoldId("fold-a"),
        sample_ids=regression_fold.testing_sample_ids,
        values=values,
        failure_ids=failure_ids,
    )

    with pytest.raises(ValueError) as error:
        validate_regression_predictions(predictions, fold=regression_fold)

    expected_details = message
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_validate_regression_predictions_rejects_nonfinite_value(
    regression_fold: FoldRecord,
    invalid_value: float,
) -> None:
    """Reject nonfinite standardized regression predictions.

    Parameters
    ----------
    regression_fold : FoldRecord
        Authoritative fold used for prediction alignment.
    invalid_value : float
        Nonfinite prediction to reject.
    """
    predictions = RegressionPredictions(
        fold_id=FoldId("fold-a"),
        sample_ids=regression_fold.testing_sample_ids,
        values=(invalid_value, 2.5),
        failure_ids=((), ()),
    )

    with pytest.raises(ValueError) as error:
        validate_regression_predictions(predictions, fold=regression_fold)

    expected_details = f"values[0] must be finite; got {invalid_value!r}."
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


@pytest.mark.parametrize(
    ("invalid_value", "expected_message"),
    [
        (
            True,
            "Type validation failed\n"
            "  path: values[0]\n"
            "  expected: builtins.float | numpy.floating\n"
            "  got type: builtins.bool\n"
            "  got value: True",
        ),
        (
            1,
            "Type validation failed\n"
            "  path: values[0]\n"
            "  expected: builtins.float | numpy.floating\n"
            "  got type: builtins.int\n"
            "  got value: 1",
        ),
        (
            "1.25",
            "Type validation failed\n"
            "  path: values[0]\n"
            "  expected: builtins.float | numpy.floating\n"
            "  got type: builtins.str\n"
            "  got value: '1.25'",
        ),
    ],
)
def test_validate_regression_predictions_rejects_wrong_value_type(
    regression_fold: FoldRecord,
    invalid_value: object,
    expected_message: str,
) -> None:
    """Reject booleans, integers, and text as standardized predictions.

    Parameters
    ----------
    regression_fold : FoldRecord
        Authoritative fold used for prediction alignment.
    invalid_value : object
        Incorrectly typed prediction value.
    expected_message : str
        Required field, expected type, and observed type/value diagnostic details.
    """
    predictions = RegressionPredictions(
        fold_id=FoldId("fold-a"),
        sample_ids=regression_fold.testing_sample_ids,
        values=(
            cast(float | None, invalid_value),
            2.5,
        ),
        failure_ids=((), ()),
    )

    with pytest.raises(TypeError) as error:
        validate_regression_predictions(predictions, fold=regression_fold)

    expected_details = expected_message
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


def test_validate_regression_predictions_rejects_empty_failure_id(
    regression_fold: FoldRecord,
) -> None:
    """Reject an empty value-level failure identity.

    Parameters
    ----------
    regression_fold : FoldRecord
        Authoritative fold used for prediction alignment.
    """
    predictions = RegressionPredictions(
        fold_id=FoldId("fold-a"),
        sample_ids=regression_fold.testing_sample_ids,
        values=(None, 2.5),
        failure_ids=((FailureId(""),), ()),
    )

    with pytest.raises(ValueError) as error:
        validate_regression_predictions(predictions, fold=regression_fold)

    expected_details = "failure_ids[0][0] must be a nonempty string; got an empty string."
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


@pytest.fixture
def classification_fold() -> FoldRecord:
    """Provide one fold with an ordered classification testing sequence."""
    return FoldRecord(
        fold_id=FoldId("fold-a"),
        training_sample_ids=(SampleId("sample-train"),),
        testing_sample_ids=(SampleId("sample-a"), SampleId("sample-b")),
    )


_DEFAULT_CLASSIFICATION_FOLD_ID = FoldId("fold-a")


def _classification_predictions(
    *,
    fold_id: FoldId = _DEFAULT_CLASSIFICATION_FOLD_ID,
    sample_ids: tuple[SampleId, ...] = (
        SampleId("sample-a"),
        SampleId("sample-b"),
    ),
    class_labels: tuple[object, ...] = ("negative", "positive"),
    predicted_labels: tuple[object | None, ...] = ("positive", "negative"),
    label_failure_ids: tuple[tuple[FailureId, ...], ...] = ((), ()),
    probabilities: tuple[tuple[object | None, ...], ...] = (
        (0.2, 0.7),
        (0.6, 0.3),
    ),
    probability_failure_ids: tuple[
        tuple[tuple[FailureId, ...], ...],
        ...,
    ] = (((), ()), ((), ())),
) -> ClassificationPredictions:
    """Build classification predictions for focused validation tests."""
    return ClassificationPredictions(
        fold_id=fold_id,
        sample_ids=sample_ids,
        class_labels=cast(tuple[ScalarLabel, ...], class_labels),
        predicted_labels=cast(
            tuple[ScalarLabel | None, ...],
            predicted_labels,
        ),
        label_failure_ids=label_failure_ids,
        probabilities=cast(
            tuple[tuple[float | None, ...], ...],
            probabilities,
        ),
        probability_failure_ids=probability_failure_ids,
    )


def test_validate_classification_predictions_accepts_independent_unavailability(
    classification_fold: FoldRecord,
) -> None:
    """Accept independently unavailable labels and probability cells."""
    predictions = _classification_predictions(
        predicted_labels=("positive", None),
        label_failure_ids=((), (FailureId("label-failure"),)),
        probabilities=((0.2, 0.7), (None, 0.6)),
        probability_failure_ids=(
            ((), ()),
            ((FailureId("probability-failure"),), ()),
        ),
    )
    original = predictions

    validate_classification_predictions(
        predictions,
        fold=classification_fold,
        class_labels=("negative", "positive"),
    )

    assert predictions == original


def test_validate_classification_predictions_accepts_empty_testing_sequence() -> None:
    """Accept empty aligned sample rows while retaining class order."""
    fold = FoldRecord(
        fold_id=FoldId("fold-a"),
        training_sample_ids=(SampleId("sample-train"),),
        testing_sample_ids=(),
    )
    predictions = _classification_predictions(
        sample_ids=(),
        predicted_labels=(),
        label_failure_ids=(),
        probabilities=(),
        probability_failure_ids=(),
    )

    validate_classification_predictions(
        predictions,
        fold=fold,
        class_labels=("negative", "positive"),
    )


def test_validate_classification_predictions_rejects_wrong_fold(
    classification_fold: FoldRecord,
) -> None:
    """Reject classification predictions associated with another fold."""
    predictions = _classification_predictions(fold_id=FoldId("fold-b"))

    with pytest.raises(ValueError) as error:
        validate_classification_predictions(
            predictions,
            fold=classification_fold,
            class_labels=("negative", "positive"),
        )

    expected_details = "fold_id must equal fold.fold_id 'fold-a'; got 'fold-b'."
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


def test_validate_classification_predictions_rejects_sample_misalignment(
    classification_fold: FoldRecord,
) -> None:
    """Reject a reordered classification testing-sample sequence."""
    predictions = _classification_predictions(
        sample_ids=(SampleId("sample-b"), SampleId("sample-a")),
    )

    with pytest.raises(ValueError) as error:
        validate_classification_predictions(
            predictions,
            fold=classification_fold,
            class_labels=("negative", "positive"),
        )

    message = str(error.value)
    assert "sample_ids[0]" in message and "fold.testing_sample_ids[0]" in message
    assert "'sample-a'" in message and "'sample-b'" in message


@pytest.mark.parametrize(
    "candidate_labels",
    [
        ("positive", "negative"),
        ("negative", "unknown"),
        ("negative",),
    ],
)
def test_validate_classification_predictions_rejects_class_misalignment(
    classification_fold: FoldRecord,
    candidate_labels: tuple[object, ...],
) -> None:
    """Reject reordered, unknown, or omitted class identities."""
    predictions = _classification_predictions(class_labels=candidate_labels)

    with pytest.raises(ValueError) as error:
        validate_classification_predictions(
            predictions,
            fold=classification_fold,
            class_labels=("negative", "positive"),
        )

    message = str(error.value)
    assert "class_labels" in message
    if len(candidate_labels) != 2:
        assert "2" in message and f"got {len(candidate_labels)}" in message
    else:
        index = 0 if candidate_labels[0] != "negative" else 1
        assert f"class_labels[{index}]" in message
        assert repr(("negative", "positive")[index]) in message
        assert repr(candidate_labels[index]) in message


@pytest.mark.parametrize(
    ("predictions", "message"),
    [
        (
            _classification_predictions(predicted_labels=("positive",)),
            "predicted_labels must contain 2 sample entries; got 1.",
        ),
        (
            _classification_predictions(label_failure_ids=((),)),
            "label_failure_ids must contain 2 sample entries; got 1.",
        ),
        (
            _classification_predictions(probabilities=((0.2, 0.7),)),
            "probabilities must contain 2 sample entries; got 1.",
        ),
        (
            _classification_predictions(probability_failure_ids=(((), ()),)),
            "probability_failure_ids must contain 2 sample entries; got 1.",
        ),
    ],
)
def test_validate_classification_predictions_rejects_sample_vector_lengths(
    classification_fold: FoldRecord,
    predictions: ClassificationPredictions,
    message: str,
) -> None:
    """Reject sample-aligned vectors and matrices with wrong lengths."""
    with pytest.raises(ValueError) as error:
        validate_classification_predictions(
            predictions,
            fold=classification_fold,
            class_labels=("negative", "positive"),
        )

    expected_details = message
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


@pytest.mark.parametrize(
    ("predictions", "message"),
    [
        (
            _classification_predictions(probabilities=((0.2,), (0.6, 0.3))),
            "probabilities[0] must contain 2 class entries; got 1.",
        ),
        (
            _classification_predictions(
                probability_failure_ids=(((),), ((), ())),
            ),
            "probability_failure_ids[0] must contain 2 class entries; got 1.",
        ),
    ],
)
def test_validate_classification_predictions_rejects_ragged_rows(
    classification_fold: FoldRecord,
    predictions: ClassificationPredictions,
    message: str,
) -> None:
    """Reject probability and failure rows with wrong class widths."""
    with pytest.raises(ValueError) as error:
        validate_classification_predictions(
            predictions,
            fold=classification_fold,
            class_labels=("negative", "positive"),
        )

    expected_details = message
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


def test_validate_classification_predictions_rejects_unknown_predicted_label(
    classification_fold: FoldRecord,
) -> None:
    """Reject a present predicted label outside the authoritative classes."""
    predictions = _classification_predictions(
        predicted_labels=("unknown", "negative"),
    )

    with pytest.raises(ValueError) as error:
        validate_classification_predictions(
            predictions,
            fold=classification_fold,
            class_labels=("negative", "positive"),
        )

    expected_details = "predicted_labels[0] must belong to class_labels; got 'unknown'."
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


@pytest.mark.parametrize(
    ("predictions", "message"),
    [
        (
            _classification_predictions(predicted_labels=(None, "negative")),
            "predicted_labels[0] is None and requires at least one failure ID; got zero.",
        ),
        (
            _classification_predictions(
                label_failure_ids=((FailureId("label-failure"),), ()),
            ),
            "predicted_labels[0] is present and requires an empty "
            "label_failure_ids entry; got 1 failure references.",
        ),
        (
            _classification_predictions(probabilities=((None, 0.7), (0.6, 0.3))),
            "probabilities[0][0] is None and requires at least one failure ID; got zero.",
        ),
        (
            _classification_predictions(
                probability_failure_ids=(
                    ((FailureId("probability-failure"),), ()),
                    ((), ()),
                ),
            ),
            "probabilities[0][0] is present and requires an empty "
            "probability_failure_ids[0][0] entry; got 1 failure references.",
        ),
    ],
)
def test_validate_classification_predictions_rejects_invalid_failure_relationship(
    classification_fold: FoldRecord,
    predictions: ClassificationPredictions,
    message: str,
) -> None:
    """Reject missing or contradictory classification failure provenance."""
    with pytest.raises(ValueError) as error:
        validate_classification_predictions(
            predictions,
            fold=classification_fold,
            class_labels=("negative", "positive"),
        )

    expected_details = message
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


@pytest.mark.parametrize(
    "invalid_probability",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        -0.1,
        1.1,
    ],
)
def test_validate_classification_predictions_rejects_invalid_probability(
    classification_fold: FoldRecord,
    invalid_probability: float,
) -> None:
    """Reject nonfinite and out-of-range classification probabilities."""
    predictions = _classification_predictions(
        probabilities=((invalid_probability, 0.7), (0.6, 0.3)),
    )

    with pytest.raises(ValueError) as error:
        validate_classification_predictions(
            predictions,
            fold=classification_fold,
            class_labels=("negative", "positive"),
        )

    if isfinite(invalid_probability):
        expected = "probabilities[0][0] must be within [0, 1]; " f"got {invalid_probability!r}."
    else:
        expected = "probabilities[0][0] must be finite; " f"got {invalid_probability!r}."
    expected_details = expected
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


def test_validate_classification_predictions_rejects_boolean_probability(
    classification_fold: FoldRecord,
) -> None:
    """Reject a boolean as a standardized probability value."""
    predictions = _classification_predictions(
        probabilities=((True, 0.7), (0.6, 0.3)),
    )

    with pytest.raises(TypeError) as error:
        validate_classification_predictions(
            predictions,
            fold=classification_fold,
            class_labels=("negative", "positive"),
        )

    expected_details = (
        "Type validation failed\n"
        "  path: probabilities[0][0]\n"
        "  expected: builtins.float | numpy.floating\n"
        "  got type: builtins.bool\n"
        "  got value: True"
    )
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


@pytest.mark.parametrize(
    ("predictions", "message"),
    [
        (
            _classification_predictions(
                predicted_labels=(None, "negative"),
                label_failure_ids=((FailureId(""),), ()),
            ),
            "label_failure_ids[0][0] must be a nonempty string; got an empty string.",
        ),
        (
            _classification_predictions(
                probabilities=((None, 0.7), (0.6, 0.3)),
                probability_failure_ids=(
                    ((FailureId(""),), ()),
                    ((), ()),
                ),
            ),
            "probability_failure_ids[0][0][0] must be a nonempty string; got an empty string.",
        ),
    ],
)
def test_validate_classification_predictions_rejects_empty_failure_id(
    classification_fold: FoldRecord,
    predictions: ClassificationPredictions,
    message: str,
) -> None:
    """Reject empty classification value-level failure identities."""
    with pytest.raises(ValueError) as error:
        validate_classification_predictions(
            predictions,
            fold=classification_fold,
            class_labels=("negative", "positive"),
        )

    expected_details = message
    for detail in expected_details.splitlines():
        assert detail.strip() in str(error.value)


@pytest.mark.parametrize(
    "number",
    [
        np.float16(0.5),
        np.float32(0.5),
        np.float64(0.5),
        np.longdouble("1.00000000000000001"),
        np.finfo(np.longdouble).max,
        np.nextafter(np.longdouble(0), np.longdouble(1)),
        -0.0,
    ],
)
def test_regression_preserves_floating_scalar(number: float | np.floating, regression_fold: FoldRecord) -> None:
    """Accept finite floating representations without narrowing their range or precision."""
    values = (number, 1.0)
    predictions = RegressionPredictions(
        fold_id=regression_fold.fold_id,
        sample_ids=regression_fold.testing_sample_ids,
        values=values,
        failure_ids=((), ()),
    )
    validate_regression_predictions(predictions, fold=regression_fold)
    assert predictions.values is values and predictions.values[0] is number


@pytest.mark.parametrize(
    "number",
    [
        0.0,
        1.0,
        np.float16(0.5),
        np.float32(0.5),
        np.float64(0.5),
        np.nextafter(np.longdouble(1), np.longdouble(0)),
        np.nextafter(np.longdouble(0), np.longdouble(1)),
    ],
)
def test_probabilities_preserve_floating_scalars(number: float | np.floating, classification_fold: FoldRecord) -> None:
    """Preserve probability precision and accept rows without imposing a sum tolerance."""
    predictions = _classification_predictions(
        probabilities=((number, 1.0), (0.0, 0.0)), predicted_labels=(np.str_("positive"), np.str_("negative"))
    )
    validate_classification_predictions(predictions, fold=classification_fold, class_labels=("negative", "positive"))
    assert predictions.probabilities[0][0] is number
    assert predictions.probabilities[1] == (0.0, 0.0)
    assert type(predictions.predicted_labels[0]) is np.str_


@pytest.mark.parametrize(
    "invalid", [np.int64(1), np.bool_(False), 1j, np.array(0.5), np.array([0.5]), torch.tensor(0.5)]
)
def test_prediction_values_reject_nonfloating_scalars(
    invalid: object,
    regression_fold: FoldRecord,
    classification_fold: FoldRecord,
) -> None:
    """Keep raw arrays and tensors at the adapter boundary instead of coercing them."""
    regression = RegressionPredictions(
        fold_id=regression_fold.fold_id,
        sample_ids=regression_fold.testing_sample_ids,
        values=(cast(float, invalid), 1.0),
        failure_ids=((), ()),
    )
    with pytest.raises(TypeError) as error:
        validate_regression_predictions(regression, fold=regression_fold)
    assert "values[0]" in str(error.value) and "expected:" in str(error.value)
    classification = _classification_predictions(probabilities=((invalid, 0.5), (0.2, 0.8)))
    with pytest.raises(TypeError) as error:
        validate_classification_predictions(
            classification, fold=classification_fold, class_labels=("negative", "positive")
        )
    assert "probabilities[0][0]" in str(error.value) and "got type:" in str(error.value)


@pytest.mark.parametrize("number", [np.float16("nan"), np.float32("inf"), np.float64("-inf"), np.longdouble("nan")])
def test_numpy_nonfinite_predictions_are_rejected(
    number: np.floating,
    regression_fold: FoldRecord,
    classification_fold: FoldRecord,
) -> None:
    """Apply the finite-model-output rule to every supported scalar family."""
    regression = RegressionPredictions(
        fold_id=regression_fold.fold_id,
        sample_ids=regression_fold.testing_sample_ids,
        values=(number, 1.0),
        failure_ids=((), ()),
    )
    with pytest.raises(ValueError, match=r"values\[0\].*finite"):
        validate_regression_predictions(regression, fold=regression_fold)
    with pytest.raises(ValueError, match=r"probabilities\[0\]\[0\].*finite"):
        validate_classification_predictions(
            _classification_predictions(probabilities=((number, 0.5), (0.2, 0.8))),
            fold=classification_fold,
            class_labels=("negative", "positive"),
        )


@pytest.mark.parametrize(
    "number", [np.nextafter(np.longdouble(1), np.longdouble(2)), np.nextafter(np.longdouble(0), np.longdouble(-1))]
)
def test_probability_bounds_do_not_round_or_clip(number: np.floating, classification_fold: FoldRecord) -> None:
    """Reject values just outside the closed interval at their supplied precision."""
    with pytest.raises(ValueError, match=r"probabilities\[0\]\[0\].*within \[0, 1\]"):
        validate_classification_predictions(
            _classification_predictions(probabilities=((number, 0.5), (0.2, 0.8))),
            fold=classification_fold,
            class_labels=("negative", "positive"),
        )


@pytest.mark.parametrize("labels", [(), ("same", np.str_("same"))])
def test_class_vocabulary_requires_nonempty_unique_text(labels: tuple[ScalarLabel, ...]) -> None:
    """Check vocabulary semantics even when the fold has no testing samples."""
    fold = FoldRecord(fold_id=FoldId("fold-a"), training_sample_ids=(), testing_sample_ids=())
    predictions = ClassificationPredictions(
        fold_id=fold.fold_id,
        sample_ids=(),
        class_labels=labels,
        predicted_labels=(),
        label_failure_ids=(),
        probabilities=(),
        probability_failure_ids=(),
    )
    with pytest.raises(ValueError) as error:
        validate_classification_predictions(predictions, fold=fold, class_labels=labels)
    assert "class_labels" in str(error.value)
    assert ("unique" if labels else "at least one") in str(error.value)


def test_single_class_predictions_are_valid(classification_fold: FoldRecord) -> None:
    """Accept the explicitly permitted one-class vocabulary."""
    predictions = _classification_predictions(
        class_labels=("only",),
        predicted_labels=("only", "only"),
        probabilities=((1.0,), (0.5,)),
        probability_failure_ids=(((),), ((),)),
    )
    validate_classification_predictions(predictions, fold=classification_fold, class_labels=("only",))


def test_prediction_boundaries_reject_wrong_records(regression_fold: FoldRecord) -> None:
    """Check prediction and fold records before accessing their coordinates."""
    regression = RegressionPredictions(
        fold_id=regression_fold.fold_id,
        sample_ids=regression_fold.testing_sample_ids,
        values=(1.0, 2.0),
        failure_ids=((), ()),
    )
    with pytest.raises(TypeError, match="RegressionPredictions"):
        validate_regression_predictions(cast(RegressionPredictions, None), fold=regression_fold)
    with pytest.raises(TypeError, match="FoldRecord"):
        validate_regression_predictions(regression, fold=cast(FoldRecord, None))
    with pytest.raises(TypeError, match="ClassificationPredictions"):
        validate_classification_predictions(
            cast(ClassificationPredictions, None), fold=regression_fold, class_labels=("negative", "positive")
        )
    with pytest.raises(TypeError, match="FoldRecord"):
        validate_classification_predictions(
            _classification_predictions(), fold=cast(FoldRecord, None), class_labels=("negative", "positive")
        )


@pytest.mark.parametrize(
    "field,invalid",
    [("sample_ids", []), ("values", []), ("failure_ids", []), ("failure_ids", ([], ())), ("failure_ids", ((1,), ()))],
)
def test_regression_rejects_malformed_containers(field: str, invalid: object, regression_fold: FoldRecord) -> None:
    """Validate immutable vectors and failure entries at every nesting level."""
    predictions = RegressionPredictions(
        fold_id=regression_fold.fold_id,
        sample_ids=regression_fold.testing_sample_ids,
        values=(1.0, 2.0),
        failure_ids=((), ()),
    )
    with pytest.raises(TypeError) as error:
        validate_regression_predictions(
            # Widen only test construction so runtime validators receive the invalid candidate.
            cast(Callable[..., RegressionPredictions], replace)(predictions, **{field: invalid}),
            fold=regression_fold,
        )
    assert field in str(error.value)


@pytest.mark.parametrize(
    "field,invalid",
    [
        ("sample_ids", []),
        ("class_labels", []),
        ("class_labels", (1, "positive")),
        ("predicted_labels", []),
        ("predicted_labels", (1, "negative")),
        ("label_failure_ids", []),
        ("label_failure_ids", ([], ())),
        ("label_failure_ids", ((1,), ())),
        ("probabilities", []),
        ("probabilities", ([0.2, 0.8], (0.6, 0.4))),
        ("probability_failure_ids", []),
        ("probability_failure_ids", ([(), ()], ((), ()))),
        ("probability_failure_ids", (([], ()), ((), ()))),
        ("probability_failure_ids", (((1,), ()), ((), ()))),
    ],
)
def test_classification_rejects_malformed_containers(
    field: str,
    invalid: object,
    classification_fold: FoldRecord,
) -> None:
    """Validate rows, scalar labels, and failure entries without recursive payload validation."""
    with pytest.raises(TypeError) as error:
        validate_classification_predictions(
            cast(Callable[..., ClassificationPredictions], replace)(_classification_predictions(), **{field: invalid}),
            fold=classification_fold,
            class_labels=("negative", "positive"),
        )
    assert field in str(error.value)


def test_alignment_diagnostic_preserves_full_identifiers(regression_fold: FoldRecord) -> None:
    """Identify the first differing coordinate without dumping the complete sequence."""
    expected = SampleId("sample-" + "a" * 120 + "-expected")
    actual = SampleId("sample-" + "a" * 120 + "-actual")
    fold = replace(regression_fold, testing_sample_ids=(expected, SampleId("unrelated-tail")))
    predictions = RegressionPredictions(
        fold_id=fold.fold_id, sample_ids=(actual, SampleId("unrelated-tail")), values=(1.0, 2.0), failure_ids=((), ())
    )
    with pytest.raises(ValueError) as error:
        validate_regression_predictions(predictions, fold=fold)
    assert expected in str(error.value) and actual in str(error.value)
    assert "sample_ids[0]" in str(error.value)
    assert "unrelated-tail" not in str(error.value)
