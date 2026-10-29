"""Dataset and partition records for internal component contracts."""

from dataclasses import dataclass

import numpy as np

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

# Logical paths carry no row locator or filesystem behavior.
type SourceReference = str


def validate_source_reference(value: SourceReference) -> None:
    """Validate a logical dataset-relative source path without conversion.

    Parameters
    ----------
    value : SourceReference
        Nonempty slash-separated standardized dataset-relative logical file path string.

    Raises
    ------
    TypeError
        If the source reference is not a string.
    ValueError
        If the path is empty, absolute, drive-prefixed, contains a backslash,
        or contains an empty, current-directory, or parent-directory segment.

    Notes
    -----
    Validation preserves spelling and case. It does not access files,
    resolve symlinks, establish physical containment, or normalize paths.
    Physical-path resolution belongs to the dataset I/O boundary.
    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Validate the string once without narrowing supported string subclasses.
    validate_identifier(value, field="source")

    # Reject absolute and drive-qualified syntax without interpreting OS paths.
    if value.startswith("/"):
        raise ValueError("source must be relative; got a leading '/'.")
    if len(value) >= 2 and value[0].isalpha() and value[1] == ":":
        raise ValueError(f"source must have no drive prefix; got {value[:2]!r}.")

    # Logical references have one separator convention on every platform.
    if "\\" in value:
        raise ValueError("source must use '/' separators; got a backslash.")

    # Report only the offending segment, never the complete potentially long path.
    for index, segment in enumerate(value.split("/")):
        if segment in ("", ".", ".."):
            raise ValueError(f"source segment {index} must not be empty, '.', or '..'; got {segment!r}.")


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
        Standardized string classification target or finite floating-point
        regression target, preserving its scalar type and value.
    sources : tuple[SourceReference, ...]
        Logical dataset-relative path strings for the sample's source files.

    Notes
    -----
    Training and testing populations use this same record structure.
    Construction does not normalize or validate candidate field values.
    Inline samples reference their input table; file-backed samples reference
    their X-bearing files. Managed reference/metadata tables are not sources.
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
    """Validate standardized sample metadata without converting its values.

    Parameters
    ----------
    value : SampleRecord
        Candidate identity, labels, masks, target, and logical source paths.

    Raises
    ------
    TypeError
        If the record or a field has an incorrect runtime type.
    ValueError
        If the identity is empty, a mask is not 0 or 1, a floating target
        is nonfinite, or sources are empty, invalid, or duplicated.

    Notes
    -----
    This validator does not normalize input, modify targets, access source
    files, or enforce dataset-wide sample-ID or sample-label uniqueness.
    It accepts standardized string or finite floating-point targets without
    conversion. Consumers with evaluation context enforce task compatibility
    and classification class membership. Source references must be unique
    within each sample, but are not compared across samples.
    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Check the outer record; field validity belongs to the checks below.
    if not isinstance(value, SampleRecord):
        validate_declared_types(value, expected_type=SampleRecord)

    # Validate identity and string families without exact-built-in restrictions.
    validate_identifier(value.sample_id, field="sample_id")
    validate_label(value.sample_label, field="sample_label")
    validate_label(value.group, field="group")

    # Validate target families before checking finiteness at their own precision.
    if not isinstance(value.target, (str, np.str_, float, np.floating)):
        validate_declared_types(value.target, expected_type=TargetValue, field="target")
    if isinstance(value.target, (float, np.floating)) and not np.isfinite(value.target):
        raise ValueError(f"target must be finite; got {value.target!r}.")

    # Masks use exact integer semantics, including explicit rejection of bool.
    for field, mask in (
        ("train_mask", value.train_mask),
        ("test_mask", value.test_mask),
    ):
        validate_declared_types(mask, expected_type=int, field=field)
        if mask not in (0, 1):
            raise ValueError(f"{field} must be exactly 0 or 1; got {mask!r}.")

    # Validate the container without recursively checking its source strings.
    validate_declared_types(value.sources, expected_type=tuple, field="sources")
    if not value.sources:
        raise ValueError("sources must contain at least one path; got an empty tuple.")

    # Validate each path once; duplicate diagnostics use a bounded string prefix.
    source_paths: set[str] = set()
    for index, source in enumerate(value.sources):
        try:
            validate_source_reference(source)
        except (TypeError, ValueError) as error:
            raise type(error)(f"sources[{index}] is invalid: {error}") from error

        if source in source_paths:
            raise ValueError(f"sources[{index}] must be unique; duplicate path prefix: {source!r}.")
        source_paths.add(source)


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
    """Validate a complete realized schedule and its sample associations.

    Parameters
    ----------
    value : PartitionSchedule
        Candidate partition reference, sample metadata, and ordered folds.

    Raises
    ------
    TypeError
        If a record, container, or field has an incorrect runtime type.
    ValueError
        If sample metadata is invalid, samples or folds are empty, identities
        or labels are duplicated, memberships are unknown or repeated,
        a mask forbids membership, or training/testing samples or groups duplicate.

    Notes
    -----
    Sample validators own sample validity; this function owns partition
    identities, fold structure, and cross-record associations. Local
    dictionaries and sets index validated records without mutating input.
    Validation preserves order and memberships. It does not infer task kind
    or splitting strategy, regenerate splits, require nonempty fold
    populations or complete sample coverage, or access persistent artifacts.
    Node usage scope: Shared - Used by both root and worker nodes.
    """
    # Validate outer records without recursively traversing their fields.
    if not isinstance(value, PartitionSchedule):
        validate_declared_types(value, expected_type=PartitionSchedule)
    if not isinstance(value.reference, PartitionReference):
        validate_declared_types(value.reference, expected_type=PartitionReference, field="reference")

    # Validate the fixed dataset/partition identity pair without artifact access.
    validate_identifier(value.reference.partition_id, field="reference.partition_id")
    validate_identifier(value.reference.dataset_id, field="reference.dataset_id")

    # Require ordered, nonempty containers; their elements are checked below.
    validate_declared_types(value.samples, expected_type=tuple, field="samples")
    validate_declared_types(value.folds, expected_type=tuple, field="folds")
    if not value.samples:
        raise ValueError("samples must contain at least one record; got an empty tuple.")
    if not value.folds:
        raise ValueError("folds must contain at least one record; got an empty tuple.")

    # Validate each sample once and index its independently unique ID and label.
    sample_by_id: dict[SampleId, SampleRecord] = {}
    sample_labels: set[ScalarLabel] = set()
    for sample_index, sample in enumerate(value.samples):
        try:
            validate_sample_record(sample)
        except (TypeError, ValueError) as error:
            raise type(error)(f"samples[{sample_index}] is invalid: {error}") from error

        if sample.sample_id in sample_by_id:
            raise ValueError(
                f"samples[{sample_index}].sample_id must be unique; " f"duplicate ID prefix: {sample.sample_id!r}."
            )
        if sample.sample_label in sample_labels:
            raise ValueError(
                f"samples[{sample_index}].sample_label must be unique; "
                f"duplicate label prefix: {sample.sample_label!r}."
            )
        sample_by_id[sample.sample_id] = sample
        sample_labels.add(sample.sample_label)

    # Visit folds in saved order and validate each fold's own fields once.
    fold_ids: set[FoldId] = set()
    for fold_index, fold in enumerate(value.folds):
        fold_field = f"folds[{fold_index}]"
        if not isinstance(fold, FoldRecord):
            validate_declared_types(fold, expected_type=FoldRecord, field=fold_field)

        # Fold identities are unique within this schedule.
        validate_identifier(fold.fold_id, field=f"{fold_field}.fold_id")
        if fold.fold_id in fold_ids:
            raise ValueError(f"{fold_field}.fold_id must be unique; duplicate ID prefix: {fold.fold_id!r}.")
        fold_ids.add(fold.fold_id)

        # Both populations share membership rules and have independent masks.
        for membership_field, memberships, mask_field in (
            ("training_sample_ids", fold.training_sample_ids, "train_mask"),
            ("testing_sample_ids", fold.testing_sample_ids, "test_mask"),
        ):
            population_path = f"{fold_field}.{membership_field}"
            validate_declared_types(memberships, expected_type=tuple, field=population_path)
            seen_memberships: set[SampleId] = set()
            for membership_index, sample_id in enumerate(memberships):
                membership_path = f"{population_path}[{membership_index}]"

                # Validate each identity before hashing it or resolving its sample.
                validate_identifier(sample_id, field=membership_path)
                if sample_id in seen_memberships:
                    raise ValueError(
                        f"{membership_path} must be unique within its population; "
                        f"duplicate ID prefix: {sample_id!r}."
                    )
                seen_memberships.add(sample_id)

                # Every membership must resolve within this exact sample collection.
                member = sample_by_id.get(sample_id)
                if member is None:
                    raise ValueError(
                        f"{membership_path} must reference a known sample_id; " f"unknown ID prefix: {sample_id!r}."
                    )

                # Eligibility is permission for the corresponding population only.
                mask = getattr(member, mask_field)
                if mask != 1:
                    raise ValueError(f"{membership_path} requires {mask_field} == 1; got {mask!r}.")

        # Require disjoint sample memberships without comparing source-file sets.
        testing_ids = set(fold.testing_sample_ids)
        for index, sample_id in enumerate(fold.training_sample_ids):
            if sample_id in testing_ids:
                raise ValueError(
                    f"{fold_field}.training_sample_ids[{index}] must be absent from testing; "
                    f"duplicate ID prefix: {sample_id!r}."
                )

        # Groups may repeat within a population but never cross the fold boundary.
        testing_groups = {sample_by_id[sample_id].group for sample_id in fold.testing_sample_ids}
        for index, sample_id in enumerate(fold.training_sample_ids):
            group = sample_by_id[sample_id].group
            if group in testing_groups:
                raise ValueError(
                    f"{fold_field}.training_sample_ids[{index}] must have a group absent from testing; "
                    f"duplicate group prefix: {group!r}."
                )
