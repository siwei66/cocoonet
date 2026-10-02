"""Tests for dataset and partition contracts."""

from dataclasses import FrozenInstanceError

import pytest

from cocoonet._contracts.data import (
    DatasetReference,
    FoldRecord,
    PartitionReference,
    PartitionSchedule,
    SampleRecord,
    SourceReference,
)
from cocoonet._contracts.identity import DatasetId, FoldId, PartitionId, SampleId, TargetValue


def test_dataset_reference_equality_uses_dataset_id() -> None:
    """Treat matching dataset identities as equal and different identities as distinct."""
    reference = DatasetReference(dataset_id=DatasetId("dataset-a"))
    same = DatasetReference(dataset_id=DatasetId("dataset-a"))
    other = DatasetReference(dataset_id=DatasetId("dataset-b"))

    assert reference == same
    assert reference != other


def test_dataset_reference_rejects_identity_mutation() -> None:
    """Prevent reassignment of the referenced dataset identity."""
    reference = DatasetReference(dataset_id=DatasetId("dataset-a"))

    # Exercise runtime mutation rejection without a statically invalid assignment.
    with pytest.raises(FrozenInstanceError):
        reference.__setattr__("dataset_id", DatasetId("dataset-b"))


def test_source_reference_preserves_exact_path_identity() -> None:
    """Preserve path spelling and distinguish different logical references."""
    reference = SourceReference(relative_path="Samples/source.csv")
    same = SourceReference(relative_path="Samples/source.csv")
    different_case = SourceReference(relative_path="samples/source.csv")
    different_file = SourceReference(relative_path="Samples/other.csv")

    assert reference.relative_path == "Samples/source.csv"
    assert reference == same
    assert reference != different_case
    assert reference != different_file


def test_source_reference_rejects_path_mutation() -> None:
    """Prevent redirection of an existing source reference."""
    reference = SourceReference(relative_path="Samples/source.csv")

    # Exercise runtime mutation rejection without a statically invalid assignment.
    with pytest.raises(FrozenInstanceError):
        reference.__setattr__("relative_path", "Samples/other.csv")


@pytest.mark.parametrize("target", ["positive", 2.5])
def test_sample_record_preserves_metadata_association(target: TargetValue) -> None:
    """Preserve distinct sample identifiers, eligibility, target, and sources.

    Parameters
    ----------
    target : TargetValue
        Classification or regression target associated with the sample.
    """
    sources = (
        SourceReference(relative_path="tables/samples.csv"),
        SourceReference(relative_path="measurements/specimen-a.csv"),
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
        sources=(SourceReference(relative_path="tables/samples.csv"),),
    )

    # Exercise runtime mutation rejection without a statically invalid assignment.
    with pytest.raises(FrozenInstanceError):
        sample.__setattr__(field, object())


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
            sources=(SourceReference(relative_path="tables/samples.csv"),),
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
