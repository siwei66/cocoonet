"""Tests for dataset and partition contracts."""

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from typing import cast

import numpy as np
import pytest

from cocoonet._contracts.data import (
    FoldRecord,
    PartitionReference,
    PartitionSchedule,
    SampleRecord,
    SourceReference,
    validate_partition_schedule,
    validate_sample_record,
    validate_source_reference,
)
from cocoonet._contracts.identity import DatasetId, FoldId, PartitionId, SampleId, TargetValue


def test_source_reference_preserves_exact_path_identity() -> None:
    """Preserve path spelling and distinguish different logical references."""
    reference = "Samples/source.csv"
    same = "Samples/source.csv"
    different_case = "samples/source.csv"
    different_file = "Samples/other.csv"

    for candidate in (reference, same, different_case, different_file):
        validate_source_reference(candidate)

    assert reference == "Samples/source.csv"
    assert reference == same
    assert reference != different_case
    assert reference != different_file


@pytest.mark.parametrize(
    "relative_path",
    [
        "source.csv",
        "Samples/source.csv",
        "tables/source file.csv",
        "dataset/feature:name.csv",
        "measurements/样本.csv",
    ],
)
def test_validate_source_reference_accepts_logical_relative_paths(
    relative_path: str,
) -> None:
    """Accept nonempty slash-separated logical references without normalization."""
    reference = relative_path

    validate_source_reference(reference)

    assert reference == relative_path


@pytest.mark.parametrize(
    ("relative_path", "expected_message"),
    [
        ("", "nonempty string"),
        (
            "/dataset/source.csv",
            "leading '/'",
        ),
        (
            "C:/dataset/source.csv",
            "no drive prefix",
        ),
        (
            "c:dataset/source.csv",
            "no drive prefix",
        ),
        (
            r"dataset\source.csv",
            "backslash",
        ),
        (
            ".",
            "must not be empty, '.', or '..'",
        ),
        (
            "..",
            "must not be empty, '.', or '..'",
        ),
        (
            "dataset/./source.csv",
            "must not be empty, '.', or '..'",
        ),
        (
            "dataset/../source.csv",
            "must not be empty, '.', or '..'",
        ),
        (
            "dataset//source.csv",
            "must not be empty, '.', or '..'",
        ),
        (
            "dataset/",
            "must not be empty, '.', or '..'",
        ),
    ],
)
def test_validate_source_reference_rejects_invalid_logical_paths(
    relative_path: str,
    expected_message: str,
) -> None:
    """Reject logical paths that are empty, platform-specific, or ambiguous."""
    reference = relative_path

    with pytest.raises(ValueError) as error:
        validate_source_reference(reference)

    assert expected_message in str(error.value)


@pytest.mark.parametrize(
    "value",
    [None, 1, True, b"source.csv", {"relative_path": "source.csv"}, object()],
)
def test_validate_source_reference_rejects_other_objects(value: object) -> None:
    """Reject values that are not logical path strings."""
    with pytest.raises(TypeError) as error:
        validate_source_reference(cast(SourceReference, value))

    message = str(error.value)
    assert "path: source" in message
    assert "expected: builtins.str" in message
    assert f"got type: {type(value).__module__}.{type(value).__qualname__}" in message


def test_validate_source_reference_rejects_non_string_path() -> None:
    """Reject a source reference whose logical path is not text."""
    reference = cast(str, 1)

    with pytest.raises(TypeError) as error:
        validate_source_reference(reference)

    message = str(error.value)
    assert "path: source" in message
    assert "expected: builtins.str" in message
    assert "got type: builtins.int" in message
    assert "got value: 1" in message


def test_validate_source_reference_does_not_require_physical_file() -> None:
    """Validate a logical reference without checking host filesystem state."""
    reference = "not-created/feature:name.csv"

    validate_source_reference(reference)

    assert reference == "not-created/feature:name.csv"


@pytest.mark.parametrize("target", ["positive", 2.5])
def test_sample_record_preserves_metadata_association(target: TargetValue) -> None:
    """Preserve distinct sample identifiers, eligibility, target, and sources.

    Parameters
    ----------
    target : TargetValue
        Classification or regression target associated with the sample.
    """
    sources = (
        "tables/samples.csv",
        "measurements/specimen-a.csv",
    )
    sample = SampleRecord(
        sample_id=SampleId("sample-7"),
        sample_label="specimen-a",
        group="batch-2",
        train_mask=1,
        test_mask=0,
        target=target,
        sources=sources,
    )

    assert (
        sample.sample_id,
        sample.sample_label,
        sample.group,
        sample.train_mask,
        sample.test_mask,
        sample.target,
        sample.sources,
    ) == (SampleId("sample-7"), "specimen-a", "batch-2", 1, 0, target, sources)


@pytest.mark.parametrize(
    "field",
    ["sample_id", "sample_label", "group", "train_mask", "test_mask", "target", "sources"],
)
def test_sample_record_rejects_metadata_mutation(field: str) -> None:
    """Prevent reassignment of the sample's recorded metadata.

    Parameters
    ----------
    field : str
        Metadata field whose reassignment must fail.
    """
    sample = SampleRecord(
        sample_id=SampleId("sample-7"),
        sample_label="specimen-a",
        group="batch-2",
        train_mask=1,
        test_mask=0,
        target="positive",
        sources=("tables/samples.csv",),
    )

    # Exercise runtime mutation rejection without a statically invalid assignment.
    with pytest.raises(FrozenInstanceError):
        sample.__setattr__(field, object())


@pytest.fixture
def valid_sample_record() -> SampleRecord:
    """Provide a valid normalized sample record for focused validator tests."""
    return SampleRecord(
        sample_id=SampleId("sample-7"),
        sample_label="specimen-a",
        group="batch-2",
        train_mask=1,
        test_mask=1,
        target="positive",
        sources=("tables/samples.csv",),
    )


@pytest.mark.parametrize(
    ("train_mask", "test_mask"),
    [(0, 0), (0, 1), (1, 0), (1, 1)],
)
def test_validate_sample_record_accepts_normalized_masks_and_unique_sources(
    valid_sample_record: SampleRecord,
    train_mask: int,
    test_mask: int,
) -> None:
    """Accept every normalized mask combination and multiple unique sources."""
    sources = (
        "tables/samples.csv",
        "measurements/specimen-a.csv",
    )
    sample = replace(
        valid_sample_record,
        train_mask=train_mask,
        test_mask=test_mask,
        target=2.5,
        sources=sources,
    )

    validate_sample_record(sample)

    assert sample.train_mask == train_mask
    assert sample.test_mask == test_mask
    assert sample.target == 2.5
    assert sample.sources == sources


def test_validate_sample_record_rejects_other_objects() -> None:
    """Reject values that are not sample records."""
    value = object()

    with pytest.raises(TypeError) as error:
        validate_sample_record(cast(SampleRecord, value))

    message = str(error.value)
    assert "path: value" in message
    assert "expected: cocoonet._contracts.data.SampleRecord" in message
    assert "got type: builtins.object" in message


@pytest.mark.parametrize(
    ("field", "train_mask", "test_mask"),
    [
        ("train_mask", -1, 1),
        ("train_mask", 2, 1),
        ("test_mask", 1, -1),
        ("test_mask", 1, 2),
    ],
)
def test_validate_sample_record_rejects_masks_outside_binary_domain(
    valid_sample_record: SampleRecord,
    field: str,
    train_mask: int,
    test_mask: int,
) -> None:
    """Reject integer eligibility masks other than exactly zero or one."""
    sample = replace(
        valid_sample_record,
        train_mask=train_mask,
        test_mask=test_mask,
    )

    with pytest.raises(ValueError, match=rf"{field} must be exactly 0 or 1"):
        validate_sample_record(sample)


@pytest.mark.parametrize(
    ("field", "train_mask", "test_mask"),
    [
        ("train_mask", True, 1),
        ("test_mask", 1, False),
    ],
)
def test_validate_sample_record_rejects_boolean_masks(
    valid_sample_record: SampleRecord,
    field: str,
    train_mask: object,
    test_mask: object,
) -> None:
    """Reject booleans even though bool is a subclass of int."""
    sample = replace(
        valid_sample_record,
        train_mask=cast(int, train_mask),
        test_mask=cast(int, test_mask),
    )

    with pytest.raises(TypeError) as error:
        validate_sample_record(sample)

    message = str(error.value)
    assert f"path: {field}" in message
    assert "expected: builtins.int" in message
    assert "got type: builtins.bool" in message


def test_validate_sample_record_validates_identity_and_labels(
    valid_sample_record: SampleRecord,
) -> None:
    """Apply the established identifier and scalar-label value contracts."""
    invalid_records = (
        (replace(valid_sample_record, sample_id=SampleId("")), "sample_id"),
        (replace(valid_sample_record, sample_label=cast(str, float("nan"))), "sample_label"),
        (replace(valid_sample_record, group=cast(str, float("inf"))), "group"),
    )

    for sample, expected_field in invalid_records:
        error_type = ValueError if expected_field == "sample_id" else TypeError
        with pytest.raises(error_type) as error:
            validate_sample_record(sample)

        assert expected_field in str(error.value)


def test_validate_sample_record_rejects_non_scalar_target(
    valid_sample_record: SampleRecord,
) -> None:
    """Reject a target outside the declared scalar target vocabulary."""
    sample = replace(
        valid_sample_record,
        target=cast(TargetValue, ["positive"]),
    )

    with pytest.raises(TypeError) as error:
        validate_sample_record(sample)

    message = str(error.value)
    assert "path: target" in message
    assert "got type: builtins.list" in message


def test_validate_sample_record_requires_a_source(
    valid_sample_record: SampleRecord,
) -> None:
    """Reject a sample with no associated source reference."""
    sample = replace(valid_sample_record, sources=())

    with pytest.raises(
        ValueError,
        match="sources must contain at least one path",
    ):
        validate_sample_record(sample)


def test_validate_sample_record_rejects_duplicate_sources(
    valid_sample_record: SampleRecord,
) -> None:
    """Reject repeated logical source references within one sample."""
    source = "tables/samples.csv"
    sample = replace(valid_sample_record, sources=(source, source))

    with pytest.raises(ValueError) as error:
        validate_sample_record(sample)

    message = str(error.value)
    assert "must be unique" in message
    assert "sources[1]" in message
    assert "'tables/samples.csv'" in message


def test_validate_sample_record_validates_each_source(
    valid_sample_record: SampleRecord,
) -> None:
    """Report the position of an invalid logical source reference."""
    sample = replace(
        valid_sample_record,
        sources=("../samples.csv",),
    )

    with pytest.raises(ValueError) as error:
        validate_sample_record(sample)

    message = str(error.value)
    assert "sources[0] is invalid" in message
    assert "must not be empty, '.', or '..'" in message


def test_validate_sample_record_allows_sources_shared_by_samples(
    valid_sample_record: SampleRecord,
) -> None:
    """Allow distinct samples to reference the same underlying source file."""
    shared_source = "tables/samples.csv"
    first = replace(valid_sample_record, sources=(shared_source,))
    second = replace(
        valid_sample_record,
        sample_id=SampleId("sample-8"),
        sample_label="specimen-b",
        sources=(shared_source,),
    )

    validate_sample_record(first)
    validate_sample_record(second)


def test_partition_reference_equality_uses_both_identities() -> None:
    """Distinguish references when either partition or dataset identity differs."""
    reference = PartitionReference(
        partition_id=PartitionId("partition-a"),
        dataset_id=DatasetId("dataset-a"),
    )
    same = PartitionReference(
        partition_id=PartitionId("partition-a"),
        dataset_id=DatasetId("dataset-a"),
    )
    other_partition = PartitionReference(
        partition_id=PartitionId("partition-b"),
        dataset_id=DatasetId("dataset-a"),
    )
    other_dataset = PartitionReference(
        partition_id=PartitionId("partition-a"),
        dataset_id=DatasetId("dataset-b"),
    )

    assert reference == same
    assert reference != other_partition
    assert reference != other_dataset


@pytest.mark.parametrize("field", ["partition_id", "dataset_id"])
def test_partition_reference_rejects_identity_mutation(field: str) -> None:
    """Prevent reassignment of either recorded identity.

    Parameters
    ----------
    field : str
        Identity field whose reassignment must fail.
    """
    reference = PartitionReference(
        partition_id=PartitionId("partition-a"),
        dataset_id=DatasetId("dataset-a"),
    )

    # Exercise runtime mutation rejection without a statically invalid assignment.
    with pytest.raises(FrozenInstanceError):
        reference.__setattr__(field, "replacement")


def test_fold_record_preserves_identity_and_membership_order() -> None:
    """Preserve the fold identity and distinct ordered population memberships."""
    fold = FoldRecord(
        fold_id=FoldId("fold-2"),
        training_sample_ids=(SampleId("sample-7"), SampleId("sample-3")),
        testing_sample_ids=(SampleId("sample-9"), SampleId("sample-1")),
    )

    assert fold.fold_id == FoldId("fold-2")
    assert fold.training_sample_ids == (SampleId("sample-7"), SampleId("sample-3"))
    assert fold.testing_sample_ids == (SampleId("sample-9"), SampleId("sample-1"))


@pytest.mark.parametrize(
    "field",
    ["fold_id", "training_sample_ids", "testing_sample_ids"],
)
def test_fold_record_rejects_identity_or_membership_mutation(field: str) -> None:
    """Prevent reassignment of the fold identity or recorded memberships.

    Parameters
    ----------
    field : str
        Identity or membership field whose reassignment must fail.
    """
    fold = FoldRecord(
        fold_id=FoldId("fold-2"),
        training_sample_ids=(SampleId("sample-7"),),
        testing_sample_ids=(SampleId("sample-9"),),
    )

    # Exercise runtime mutation rejection without a statically invalid assignment.
    with pytest.raises(FrozenInstanceError):
        fold.__setattr__(field, object())


def test_partition_schedule_preserves_association_and_order() -> None:
    """Preserve the partition reference and ordered sample and fold records."""
    reference = PartitionReference(
        partition_id=PartitionId("partition-a"),
        dataset_id=DatasetId("dataset-a"),
    )
    samples = tuple(
        SampleRecord(
            sample_id=SampleId(sample_id),
            sample_label=label,
            group=label,
            train_mask=1,
            test_mask=1,
            target=2.5,
            sources=("tables/samples.csv",),
        )
        for sample_id, label in [
            ("sample-7", "specimen-b"),
            ("sample-3", "specimen-a"),
        ]
    )
    folds = (
        FoldRecord(
            fold_id=FoldId("fold-2"),
            training_sample_ids=(SampleId("sample-7"),),
            testing_sample_ids=(SampleId("sample-3"),),
        ),
        FoldRecord(
            fold_id=FoldId("fold-1"),
            training_sample_ids=(SampleId("sample-3"),),
            testing_sample_ids=(SampleId("sample-7"),),
        ),
    )

    schedule = PartitionSchedule(
        reference=reference,
        samples=samples,
        folds=folds,
    )

    assert schedule.reference == reference
    assert schedule.samples == samples
    assert schedule.folds == folds


@pytest.mark.parametrize("field", ["reference", "samples", "folds"])
def test_partition_schedule_rejects_field_mutation(field: str) -> None:
    """Prevent reassignment of the candidate schedule's recorded fields.

    Parameters
    ----------
    field : str
        Schedule field whose reassignment must fail.
    """
    schedule = PartitionSchedule(
        reference=PartitionReference(
            partition_id=PartitionId("partition-a"),
            dataset_id=DatasetId("dataset-a"),
        ),
        samples=(),
        folds=(),
    )

    # Exercise runtime mutation rejection without a statically invalid assignment.
    with pytest.raises(FrozenInstanceError):
        schedule.__setattr__(field, object())


@pytest.fixture
def valid_partition_schedule(
    valid_sample_record: SampleRecord,
) -> PartitionSchedule:
    """Provide a valid schedule with order-independent sample associations."""
    shared_source = "tables/samples.csv"
    first = replace(
        valid_sample_record,
        sample_id=SampleId("sample-1"),
        sample_label="specimen-a",
        group="group-a",
        sources=(shared_source,),
    )
    second = replace(
        valid_sample_record,
        sample_id=SampleId("sample-2"),
        sample_label="specimen-b",
        group="group-b",
        sources=(shared_source,),
    )
    return PartitionSchedule(
        reference=PartitionReference(
            partition_id=PartitionId("partition-a"),
            dataset_id=DatasetId("dataset-a"),
        ),
        samples=(second, first),
        folds=(
            FoldRecord(
                fold_id=FoldId("fold-1"),
                training_sample_ids=(first.sample_id,),
                testing_sample_ids=(second.sample_id,),
            ),
        ),
    )


def test_validate_partition_schedule_accepts_valid_associations(
    valid_partition_schedule: PartitionSchedule,
) -> None:
    """Accept ID-based memberships independent of sample tuple order."""
    validate_partition_schedule(valid_partition_schedule)

    assert valid_partition_schedule.samples[0].sample_id == SampleId("sample-2")
    assert valid_partition_schedule.folds[0].training_sample_ids == (SampleId("sample-1"),)


def test_validate_partition_schedule_allows_empty_populations_and_cross_fold_reuse(
    valid_partition_schedule: PartitionSchedule,
) -> None:
    """Allow empty populations and repeated sample use in different folds."""
    schedule = replace(
        valid_partition_schedule,
        folds=(
            valid_partition_schedule.folds[0],
            FoldRecord(
                fold_id=FoldId("fold-2"),
                training_sample_ids=(SampleId("sample-2"),),
                testing_sample_ids=(SampleId("sample-1"),),
            ),
            FoldRecord(
                fold_id=FoldId("fold-3"),
                training_sample_ids=(),
                testing_sample_ids=(),
            ),
        ),
    )

    validate_partition_schedule(schedule)


def test_validate_partition_schedule_allows_repeated_groups_within_population(
    valid_partition_schedule: PartitionSchedule,
) -> None:
    """Allow group repetition when the group remains on one side of a fold."""
    shared_source = "tables/samples.csv"
    samples = (
        SampleRecord(
            sample_id=SampleId("sample-1"),
            sample_label="specimen-a",
            group="train-group",
            train_mask=1,
            test_mask=1,
            target="positive",
            sources=(shared_source,),
        ),
        SampleRecord(
            sample_id=SampleId("sample-2"),
            sample_label="specimen-b",
            group="train-group",
            train_mask=1,
            test_mask=1,
            target="positive",
            sources=(shared_source,),
        ),
        SampleRecord(
            sample_id=SampleId("sample-3"),
            sample_label="specimen-c",
            group="test-group",
            train_mask=1,
            test_mask=1,
            target="negative",
            sources=(shared_source,),
        ),
    )
    schedule = replace(
        valid_partition_schedule,
        samples=samples,
        folds=(
            FoldRecord(
                fold_id=FoldId("fold-1"),
                training_sample_ids=(
                    SampleId("sample-1"),
                    SampleId("sample-2"),
                ),
                testing_sample_ids=(SampleId("sample-3"),),
            ),
        ),
    )

    validate_partition_schedule(schedule)


def test_validate_partition_schedule_rejects_other_objects() -> None:
    """Reject values that are not partition schedules."""
    value = object()

    with pytest.raises(TypeError) as error:
        validate_partition_schedule(cast(PartitionSchedule, value))

    message = str(error.value)
    assert "path: value" in message
    assert "expected: cocoonet._contracts.data.PartitionSchedule" in message
    assert "got type: builtins.object" in message


@pytest.mark.parametrize("field", ["samples", "folds"])
def test_validate_partition_schedule_requires_records(
    valid_partition_schedule: PartitionSchedule,
    field: str,
) -> None:
    """Require nonempty sample and fold collections."""
    schedule = (
        replace(valid_partition_schedule, samples=())
        if field == "samples"
        else replace(valid_partition_schedule, folds=())
    )

    with pytest.raises(ValueError, match=rf"{field} must contain at least one"):
        validate_partition_schedule(schedule)


def test_validate_partition_schedule_validates_nested_records(
    valid_partition_schedule: PartitionSchedule,
) -> None:
    """Validate partition, sample, and fold identities at their schedule paths."""
    invalid_sample = replace(
        valid_partition_schedule.samples[0],
        sources=("../samples.csv",),
    )
    invalid_schedules = (
        (
            replace(
                valid_partition_schedule,
                reference=replace(
                    valid_partition_schedule.reference,
                    partition_id=PartitionId(""),
                ),
            ),
            "reference.partition_id",
        ),
        (
            replace(
                valid_partition_schedule,
                samples=(
                    invalid_sample,
                    valid_partition_schedule.samples[1],
                ),
            ),
            "samples[0]",
        ),
        (
            replace(
                valid_partition_schedule,
                folds=(
                    replace(
                        valid_partition_schedule.folds[0],
                        fold_id=FoldId(""),
                    ),
                ),
            ),
            "folds[0].fold_id",
        ),
    )

    for schedule, expected_field in invalid_schedules:
        with pytest.raises(ValueError) as error:
            validate_partition_schedule(schedule)

        assert expected_field in str(error.value)


def test_validate_partition_schedule_rejects_duplicate_scoped_values(
    valid_partition_schedule: PartitionSchedule,
) -> None:
    """Reject duplicate sample IDs, sample labels, and fold IDs."""
    first, second = valid_partition_schedule.samples
    duplicate_schedules = (
        (
            replace(
                valid_partition_schedule,
                samples=(
                    first,
                    replace(second, sample_id=first.sample_id),
                ),
            ),
            "sample_id must be unique",
        ),
        (
            replace(
                valid_partition_schedule,
                samples=(
                    first,
                    replace(second, sample_label=first.sample_label),
                ),
            ),
            "sample_label must be unique",
        ),
        (
            replace(
                valid_partition_schedule,
                folds=(
                    valid_partition_schedule.folds[0],
                    valid_partition_schedule.folds[0],
                ),
            ),
            "fold_id must be unique",
        ),
    )

    for schedule, expected_message in duplicate_schedules:
        with pytest.raises(ValueError) as error:
            validate_partition_schedule(schedule)

        assert expected_message in str(error.value)


def test_validate_partition_schedule_rejects_invalid_memberships(
    valid_partition_schedule: PartitionSchedule,
) -> None:
    """Reject unknown, repeated, and train-test-overlapping memberships."""
    invalid_folds = (
        (
            FoldRecord(
                fold_id=FoldId("fold-unknown"),
                training_sample_ids=(SampleId("unknown"),),
                testing_sample_ids=(),
            ),
            "reference a known sample_id",
        ),
        (
            FoldRecord(
                fold_id=FoldId("fold-repeated"),
                training_sample_ids=(
                    SampleId("sample-1"),
                    SampleId("sample-1"),
                ),
                testing_sample_ids=(),
            ),
            "must be unique within its population",
        ),
        (
            FoldRecord(
                fold_id=FoldId("fold-overlap"),
                training_sample_ids=(SampleId("sample-1"),),
                testing_sample_ids=(SampleId("sample-1"),),
            ),
            "must be absent from testing",
        ),
    )

    for fold, expected_message in invalid_folds:
        schedule = replace(valid_partition_schedule, folds=(fold,))

        with pytest.raises(ValueError) as error:
            validate_partition_schedule(schedule)

        assert expected_message in str(error.value)


@pytest.mark.parametrize(
    ("population", "train_mask", "test_mask", "expected_mask"),
    [
        ("training", 0, 1, "train_mask"),
        ("testing", 1, 0, "test_mask"),
    ],
)
def test_validate_partition_schedule_enforces_membership_masks(
    valid_partition_schedule: PartitionSchedule,
    population: str,
    train_mask: int,
    test_mask: int,
    expected_mask: str,
) -> None:
    """Reject memberships forbidden by their corresponding eligibility mask."""
    restricted = replace(
        valid_partition_schedule.samples[1],
        train_mask=train_mask,
        test_mask=test_mask,
    )
    schedule = replace(
        valid_partition_schedule,
        samples=(valid_partition_schedule.samples[0], restricted),
        folds=(
            FoldRecord(
                fold_id=FoldId("fold-restricted"),
                training_sample_ids=((restricted.sample_id,) if population == "training" else ()),
                testing_sample_ids=((restricted.sample_id,) if population == "testing" else ()),
            ),
        ),
    )

    with pytest.raises(ValueError) as error:
        validate_partition_schedule(schedule)

    assert f"requires {expected_mask} == 1; got 0" in str(error.value)


def test_validate_partition_schedule_rejects_group_leakage(
    valid_partition_schedule: PartitionSchedule,
) -> None:
    """Reject a group shared by distinct training and testing samples."""
    first, second = valid_partition_schedule.samples
    schedule = replace(
        valid_partition_schedule,
        samples=(
            first,
            replace(second, group=first.group),
        ),
    )

    with pytest.raises(ValueError) as error:
        validate_partition_schedule(schedule)

    message = str(error.value)
    assert "folds[0]" in message
    assert "group absent from testing" in message
    assert "duplicate group" in message


@pytest.mark.parametrize(
    "target",
    [
        np.str_("positive"),
        0.0,
        -0.0,
        np.float16(0.5),
        np.float32(0.5),
        np.float64(0.5),
        np.longdouble("1.00000000000000001"),
        np.finfo(np.longdouble).max,
        np.nextafter(np.longdouble(0), np.longdouble(1)),
    ],
)
def test_sample_target_preserves_scalar_precision(valid_sample_record: SampleRecord, target: TargetValue) -> None:
    """Preserve standardized text and finite floats, including dtype limits."""
    sample = replace(valid_sample_record, target=target)
    validate_sample_record(sample)
    assert sample.target is target


@pytest.mark.parametrize(
    "target", [float("nan"), float("inf"), float("-inf"), np.float32("nan"), np.float64("inf"), np.longdouble("-inf")]
)
def test_sample_rejects_nonfinite_target(valid_sample_record: SampleRecord, target: TargetValue) -> None:
    """Reject nonfinite input targets without treating them as report NaN."""
    with pytest.raises(ValueError) as error:
        validate_sample_record(replace(valid_sample_record, target=target))
    assert "target" in str(error.value) and "finite" in str(error.value)
    assert repr(target) in str(error.value)


@pytest.mark.parametrize("target", [0, True, np.int64(1), np.bool_(False), 1j, np.array(1.0)])
def test_sample_rejects_unstandardized_target(valid_sample_record: SampleRecord, target: object) -> None:
    """Reject integers, booleans, complex values, and arrays without coercion."""
    with pytest.raises(TypeError) as error:
        validate_sample_record(replace(valid_sample_record, target=cast(TargetValue, target)))
    assert "path: target" in str(error.value)
    assert "expected:" in str(error.value) and "got type:" in str(error.value)


@pytest.mark.parametrize(
    "field,invalid",
    [
        ("sources", ["samples.csv"]),
        ("sources", (1,)),
        ("sample_label", 1),
        ("group", True),
        ("train_mask", 1.0),
        ("test_mask", np.int64(1)),
    ],
)
def test_sample_rejects_invalid_field_types(valid_sample_record: SampleRecord, field: str, invalid: object) -> None:
    """Validate standardized field types before indexing or hashing values."""
    # Widen only test construction so runtime validators receive the invalid candidate.
    sample = cast(Callable[..., SampleRecord], replace)(valid_sample_record, **{field: invalid})
    with pytest.raises(TypeError) as error:
        validate_sample_record(sample)
    assert field in str(error.value)
    assert "expected:" in str(error.value) and "got type:" in str(error.value)


@pytest.mark.parametrize(
    "field,invalid", [("reference", None), ("samples", []), ("folds", []), ("samples", (None,)), ("folds", (None,))]
)
def test_schedule_rejects_invalid_record_and_container_types(
    valid_partition_schedule: PartitionSchedule,
    field: str,
    invalid: object,
) -> None:
    """Validate each independently supplied schedule component."""
    schedule = cast(Callable[..., PartitionSchedule], replace)(valid_partition_schedule, **{field: invalid})
    with pytest.raises(TypeError) as error:
        validate_partition_schedule(schedule)
    assert field in str(error.value)
    assert "expected:" in str(error.value) and "got type:" in str(error.value)


@pytest.mark.parametrize("field", ["training_sample_ids", "testing_sample_ids"])
@pytest.mark.parametrize("invalid,error_type", [([], TypeError), ((1,), TypeError), (("",), ValueError)])
def test_schedule_rejects_invalid_membership_types(
    valid_partition_schedule: PartitionSchedule,
    field: str,
    invalid: object,
    error_type: type[Exception],
) -> None:
    """Validate tuple membership and each identity before resolving samples."""
    fold = cast(Callable[..., FoldRecord], replace)(valid_partition_schedule.folds[0], **{field: invalid})
    with pytest.raises(error_type) as error:
        validate_partition_schedule(replace(valid_partition_schedule, folds=(fold,)))
    assert f"folds[0].{field}" in str(error.value)


def test_duplicate_source_diagnostic_preserves_full_reference(valid_sample_record: SampleRecord) -> None:
    """Identify an entire duplicate path even when its useful suffix is long."""
    source = "sources/" + "a" * 120 + "/distinguishing-file.csv"
    with pytest.raises(ValueError) as error:
        validate_sample_record(replace(valid_sample_record, sources=(source, source)))
    assert source in str(error.value)
    assert "sources[1]" in str(error.value)


def test_sample_paths_keep_case_distinctions(valid_sample_record: SampleRecord) -> None:
    """Treat differently spelled logical paths as distinct without filesystem normalization."""
    sources = ("Samples/source.csv", "samples/source.csv", "Samples/other.csv")
    sample = replace(valid_sample_record, sources=sources)
    validate_sample_record(sample)
    assert sample.sources is sources
