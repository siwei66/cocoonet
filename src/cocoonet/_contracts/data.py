"""Dataset and partition records for internal component contracts."""

from dataclasses import dataclass

from cocoonet._contracts.identity import DatasetId, FoldId, PartitionId, SampleId, ScalarLabel, TargetValue


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetReference:
    """Reference one exact dataset representation by its assigned identity.

    Parameters
    ----------
    dataset_id : DatasetId
        Identity assigned to the dataset representation.

    Notes
    -----
    Construction stores the candidate identity without validating it.
    This record does not load data or verify dataset contents.
    """

    dataset_id: DatasetId


@dataclass(frozen=True, kw_only=True, slots=True)
class SourceReference:
    """Reference a source file relative to its dataset root.

    Parameters
    ----------
    relative_path : str
        Logical dataset-relative path, interpreted within the dataset context.

    Notes
    -----
    Construction preserves the candidate path without validating or normalizing it.
    This record does not access the filesystem.
    """

    relative_path: str


@dataclass(frozen=True, kw_only=True, slots=True)
class SampleRecord:
    """Associate a sample's identity, metadata, target, and source references.

    Parameters
    ----------
    sample_id : SampleId
        Internal identity scoped to the dataset.
    sample_label : ScalarLabel
        User-facing sample identifier, distinct from the target.
    group : ScalarLabel
        Group identity used by applicable splitting strategies.
    train_mask : int
        Training eligibility: 1 permits selection and 0 excludes it.
    test_mask : int
        Testing eligibility, independently specified from training eligibility.
    target : TargetValue
        True class label or single regression target.
    sources : tuple[SourceReference, ...]
        Dataset-relative source references associated with the sample.

    Notes
    -----
    Training and testing populations use this same record structure.
    Construction does not normalize or validate candidate field values.
    Feature data and computed predictions are represented separately.
    """

    sample_id: SampleId
    sample_label: ScalarLabel
    group: ScalarLabel
    train_mask: int
    test_mask: int
    target: TargetValue
    sources: tuple[SourceReference, ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class PartitionReference:
    """Reference a realized partition schedule and its associated dataset.

    Parameters
    ----------
    partition_id : PartitionId
        Identity assigned to the complete realized partition schedule.
    dataset_id : DatasetId
        Identity of the dataset representation associated with the schedule.

    Notes
    -----
    Construction stores candidate identities without validating their association.
    This reference neither stores partition memberships nor accesses an artifact.
    """

    partition_id: PartitionId
    dataset_id: DatasetId


@dataclass(frozen=True, kw_only=True, slots=True)
class FoldRecord:
    """Record a fold's identity and realized sample memberships.

    Parameters
    ----------
    fold_id : FoldId
        Identity scoped to the partition schedule.
    training_sample_ids : tuple[SampleId, ...]
        Ordered internal sample identities in the training population.
    testing_sample_ids : tuple[SampleId, ...]
        Ordered internal sample identities in the testing population.

    Notes
    -----
    Construction preserves candidate memberships without validating them.
    Empty populations are structurally representable.
    This record does not generate or alter a split.
    """

    fold_id: FoldId
    training_sample_ids: tuple[SampleId, ...]
    testing_sample_ids: tuple[SampleId, ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class PartitionSchedule:
    """Associate a complete realized partition with its dataset sample metadata.

    Parameters
    ----------
    reference : PartitionReference
        Identities of the partition schedule and its associated dataset.
    samples : tuple[SampleRecord, ...]
        Ordered sample records retaining identities, labels, and groups.
    folds : tuple[FoldRecord, ...]
        Ordered folds containing the realized training and testing memberships.

    Notes
    -----
    Valid schedules require disjoint training and testing groups in every fold.
    Construction preserves candidate values without validating the schedule.
    This record does not generate groups, alter memberships, or persist an artifact.
    """

    reference: PartitionReference
    samples: tuple[SampleRecord, ...]
    folds: tuple[FoldRecord, ...]
