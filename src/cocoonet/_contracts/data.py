"""Dataset and partition records for internal component contracts."""

from dataclasses import dataclass

from cocoonet._contracts.identity import (
    DatasetId,
    FoldId,
    PartitionId,
    SampleId,
    ScalarLabel,
    TargetValue,
    validate_identifier,
    validate_label,
)
from cocoonet._type_validation import validate_declared_types


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
    Node usage scope: Shared - Used by both root and worker nodes.
    """

    relative_path: str


def validate_source_reference(value: SourceReference) -> None:
    """Validate a platform-independent logical source-file reference.

    Parameters
    ----------
    value : SourceReference
        Logical dataset-relative source reference to validate.

    Raises
    ------
    TypeError
        If ``value`` is not a SourceReference or ``relative_path`` is not a
        string.
    ValueError
        If the logical path is empty, absolute, drive-prefixed, contains a
        backslash, or contains an empty, current-directory, or parent-directory
        segment.

    Notes
    -----
    Validation is lexical and does not access the filesystem, resolve
    symlinks, establish physical containment, or normalize the supplied path.
    Physical-path resolution belongs to the later dataset I/O boundary.
    Node usage scope: Shared - Used by both root and worker nodes.
    """
    validate_declared_types(
        value,
        expected_type=SourceReference,
    )

    path = value.relative_path
    if not path:
        raise ValueError("relative_path must be nonempty.")

    if path.startswith("/"):
        raise ValueError("relative_path must be relative and must not start with '/'.")

    if len(path) >= 2 and path[0].isalpha() and path[1] == ":":
        raise ValueError("relative_path must not contain a drive prefix.")

    if "\\" in path:
        raise ValueError("relative_path must use '/' separators; backslashes are not allowed.")

    segments = path.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise ValueError("relative_path must not contain empty, '.', or '..' segments; " f"got {path!r}.")


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
    Node usage scope: Shared - Used by both root and worker nodes.
    """

    sample_id: SampleId
    sample_label: ScalarLabel
    group: ScalarLabel
    train_mask: int
    test_mask: int
    target: TargetValue
    sources: tuple[SourceReference, ...]


def validate_sample_record(value: SampleRecord) -> None:
    """Validate one normalized sample metadata record.

    Parameters
    ----------
    value : SampleRecord
        Sample identity, metadata, target, eligibility masks, and source
        references to validate.

    Raises
    ------
    TypeError
        If the record or any field has an incorrect runtime type.
    ValueError
        If the sample identity is empty, a label is invalid, an eligibility
        mask is not 0 or 1, no source is present, a source is invalid, or
        source references are duplicated.

    Notes
    -----
    This validator does not normalize input, modify targets, access source
    files, or enforce dataset-wide sample-ID or sample-label uniqueness.
    Target admissibility for a particular task belongs to the later dataset
    input design.
    Node usage scope: Shared - Used by both root and worker nodes.
    """
    validate_declared_types(
        value,
        expected_type=SampleRecord,
    )

    validate_identifier(value.sample_id, field="sample_id")
    validate_label(value.sample_label, field="sample_label")
    validate_label(value.group, field="group")

    for field, mask in (
        ("train_mask", value.train_mask),
        ("test_mask", value.test_mask),
    ):
        if mask not in (0, 1):
            raise ValueError(f"{field} must be exactly 0 or 1; got {mask!r}.")

    if not value.sources:
        raise ValueError("sources must contain at least one source reference.")

    source_paths: set[str] = set()
    for index, source in enumerate(value.sources):
        try:
            validate_source_reference(source)
        except (TypeError, ValueError) as error:
            raise type(error)(f"sources[{index}] is invalid: {error}") from error

        if source.relative_path in source_paths:
            raise ValueError(
                "sources must contain unique source references; "
                f"duplicate path at sources[{index}]: {source.relative_path!r}."
            )
        source_paths.add(source.relative_path)


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
    Node usage scope: Shared - Used by both root and worker nodes.
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
    Node usage scope: Shared - Used by both root and worker nodes.
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
    Node usage scope: Shared - Used by both root and worker nodes.
    """

    reference: PartitionReference
    samples: tuple[SampleRecord, ...]
    folds: tuple[FoldRecord, ...]


def validate_partition_schedule(value: PartitionSchedule) -> None:  # noqa: C901
    """Validate a complete realized partition schedule.

    Parameters
    ----------
    value : PartitionSchedule
        Dataset association, sample metadata, and realized fold memberships
        to validate.

    Raises
    ------
    TypeError
        If the schedule or any nested field has an incorrect runtime type.
    ValueError
        If an identity is invalid, samples or folds are absent, scoped
        identities or labels are duplicated, a membership is unknown or
        repeated, training and testing memberships overlap, a mask forbids
        a recorded membership, or a group crosses a fold boundary.

    Notes
    -----
    Validation preserves tuple order and the supplied memberships. It does
    not infer a splitting strategy, regenerate a split, require nonempty
    fold populations, require every sample to appear, or access persistent
    partition artifacts.
    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Validate the complete nested record structure before accessing fields.
    validate_declared_types(
        value,
        expected_type=PartitionSchedule,
    )

    # Validate the partition and associated dataset identities.
    validate_identifier(
        value.reference.partition_id,
        field="reference.partition_id",
    )
    validate_identifier(
        value.reference.dataset_id,
        field="reference.dataset_id",
    )

    # Require sample metadata because memberships cannot resolve without it.
    if not value.samples:
        raise ValueError("samples must contain at least one sample record.")

    # Validate every sample and build the ID lookup used by fold validation.
    # Sample IDs and user-facing labels are independently unique.
    sample_by_id: dict[SampleId, SampleRecord] = {}
    sample_labels: set[ScalarLabel] = set()
    for sample_index, sample in enumerate(value.samples):
        try:
            validate_sample_record(sample)
        except (TypeError, ValueError) as error:
            raise type(error)(f"samples[{sample_index}] is invalid: {error}") from error

        if sample.sample_id in sample_by_id:
            raise ValueError(
                "samples must contain unique sample_id values; "
                f"duplicate at samples[{sample_index}]: {sample.sample_id!r}."
            )
        if sample.sample_label in sample_labels:
            raise ValueError(
                "samples must contain unique sample_label values; "
                f"duplicate at samples[{sample_index}]: {sample.sample_label!r}."
            )

        sample_by_id[sample.sample_id] = sample
        sample_labels.add(sample.sample_label)

    # Require at least one realized fold while permitting empty populations
    # inside an individual fold.
    if not value.folds:
        raise ValueError("folds must contain at least one fold record.")

    # Validate each fold in its authoritative tuple order.
    fold_ids: set[FoldId] = set()
    for fold_index, fold in enumerate(value.folds):
        fold_field = f"folds[{fold_index}]"

        # Validate fold identity and uniqueness within this schedule.
        validate_identifier(fold.fold_id, field=f"{fold_field}.fold_id")
        if fold.fold_id in fold_ids:
            raise ValueError(
                "folds must contain unique fold_id values; " f"duplicate at {fold_field}: {fold.fold_id!r}."
            )
        fold_ids.add(fold.fold_id)

        # Validate both populations using the same membership rules while
        # applying the corresponding training or testing eligibility mask.
        for membership_field, memberships, mask_field in (
            (
                "training_sample_ids",
                fold.training_sample_ids,
                "train_mask",
            ),
            (
                "testing_sample_ids",
                fold.testing_sample_ids,
                "test_mask",
            ),
        ):
            seen_memberships: set[SampleId] = set()
            for membership_index, sample_id in enumerate(memberships):
                membership_path = f"{fold_field}.{membership_field}[{membership_index}]"

                # Validate each membership identity before lookup.
                validate_identifier(sample_id, field=membership_path)

                # Reject repetition within one fold population.
                if sample_id in seen_memberships:
                    raise ValueError(
                        f"{fold_field}.{membership_field} must not repeat "
                        f"sample IDs; duplicate at index {membership_index}: "
                        f"{sample_id!r}."
                    )
                seen_memberships.add(sample_id)

                # Require every membership to resolve to this schedule's
                # exact sample collection.
                member = sample_by_id.get(sample_id)
                if member is None:
                    raise ValueError(f"{membership_path} references unknown sample_id " f"{sample_id!r}.")

                # Enforce the eligibility mask corresponding to the current
                # training or testing population.
                if getattr(member, mask_field) != 1:
                    raise ValueError(
                        f"{membership_path} references sample_id " f"{sample_id!r}, whose {mask_field} is not 1."
                    )

        # Require training and testing sample memberships to be disjoint.
        overlapping_ids = set(fold.training_sample_ids) & set(fold.testing_sample_ids)
        if overlapping_ids:
            duplicate_id = next(sample_id for sample_id in fold.training_sample_ids if sample_id in overlapping_ids)
            raise ValueError(
                f"{fold_field} contains sample_id {duplicate_id!r} in both " "training and testing memberships."
            )

        # Enforce group separation unconditionally. Repeated groups remain
        # valid within either population when they do not cross the boundary.
        testing_groups = {sample_by_id[sample_id].group for sample_id in fold.testing_sample_ids}
        for sample_id in fold.training_sample_ids:
            group = sample_by_id[sample_id].group
            if group in testing_groups:
                raise ValueError(f"{fold_field} places group {group!r} in both training " "and testing populations.")
