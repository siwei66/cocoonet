# Phase 1: Shared identities, records, and component contracts

Status: proposed for review; implementation is not authorized.

## 1. Authority, scope, and approval gates

This plan implements only Phase 1, **Shared identities, records, and component contracts**, in
[the project roadmap](../roadmap.md). It follows the complete root [AGENTS.md](../../AGENTS.md) and
[architecture](../architecture.md), particularly Sections 2–3, 4.3–4.4, 5–7, 9, and 10.1.
The three governing documents were read in full before drafting this plan.

The goal is a small, typed, testable vocabulary that later components can share without redefining identity,
scientific inputs, prediction alignment, failure scope, or result acceptance. Phase 1 supplies logical records and
pure structural/cross-record validation. It supplies no working computation, worker, storage service, or scheduler.

The architectural impact is concretization of already required shared contracts, not a new architecture or
performance improvement. Proposed names, containers, and validation interfaces below are Phase 1 implementation
choices submitted for approval. They are not claims that these mechanisms already exist. Detailed scientific,
filesystem, model-adapter, lifecycle, and protocol designs stay with their assigned phases.

Approval of this document and authorization to implement a particular unit are separate gates. For each later
implementation response:

1. Inspect the applicable code and tests, apply both development reviews required by `AGENTS.md`, and resolve applicable
   Section 9 representation decisions before presenting the next unit for approval.
2. Implement only that unit and its directly corresponding tests and minimal supporting edits.
3. Run applicable checks, explain the result, suggest a manual commit message, and stop for approval of the next unit.

A class declaration does not authorize implementing its methods. An exact helper bundle needs separate, explicit
approval identifying every included function/method. If implementation requires changing a contract in this plan,
stop, propose a roadmap amendment, and obtain approval before continuing. Do not commit automatically.

This planning task changes only this file. It does not authorize source code, tests, dependencies, or changes to the
governing documents.

### 1.1 In scope

- Dataset-, parent-, unit-, attempt-, model-, sample-, fold-, worker-, and operation-reference scopes.
- A reusable computation's configuration snapshot, distinct from a fixed parent evaluation snapshot.
- Logical normalized sample metadata, realized fold memberships, and references to dataset/model/configuration inputs.
- Separate consolidation, execution, worker availability/readiness, stop, report, reception/acceptance, and operation
  records. These describe facts; they do not implement transitions or scheduling decisions.
- Standardized regression/classification prediction alignment and availability information.
- Generic report outcomes, traceable failures, and one complete attempt-result envelope, including failure-only results.
- Exact Cocoonet version compatibility and supported-Python metadata checks without executable payload loading.
- Component ownership and input/output obligations, with pure validators and directly corresponding tests.

### 1.2 Out of scope

- Filesystem inspection or mutation, directory ownership enforcement, serialization/deserialization, durable writes,
  artifact filenames, log writers, recovery journals, and publication mechanisms: Phase 2 and later consumers.
- The public computation class, initialization, configuration mutation, copying, bulk operations, and configuration
  transition algorithms: Phase 3. No `start()`, `add_dataset()`, or worker-management implementation belongs here.
- Dataframe adapters, feature/tensor storage, dataset identity generation/content verification, partition generation,
  split reuse/restoration, and dataset registration: Phase 4.
- Executable model payloads, API probing, reconstruction, fitting, prediction production, subprocesses, device handling,
  or isolation enforcement: Phases 5–11.
- Metric formulas/calculations, concrete metric/report payload schemas, visualization arrays, influence aggregation,
  and model-specific output conversion: their scientific phases.
- Result packaging/persistence, runnable local coordination, delivery/retry loops, interruption/recovery, online
  connections, synchronization, duplicate scheduling, cleanup, or external service integration: Phases 12–20.
- Wire formats, credentials, timers, IPC, identifier encodings/generation algorithms, public service policies, and
  new dependencies. No generic plugin, schema registry, event bus, or persistence framework is introduced.

## 2. Entry and completion criteria

### 2.1 Entry criteria

At the original Phase 1 inspection, the repository contained release scaffolding without domain implementation or tests.
Shared contracts, validators, and corresponding tests have since been implemented; their presence does not establish
completion of every Phase 1 requirement.
Before the first implementation unit:

- The user approves this plan and separately authorizes that exact unit.
- No unresolved Phase 1 contract conflict remains; the nonblocking documentation discrepancy in Section 11 is recorded.
- Reinspect the repository for intervening changes and applicable instructions.
- Preserve `get_version()` and existing dependencies/tool settings. Use standard-library records, enums, and typing;
  Phase 1 needs no new third-party dependency.

### 2.2 Completion criteria

Phase 1 is complete only when all approved declarations and validators in Sections 3–8 exist, their directly
corresponding tests pass, and:

- Invalid associations are rejected descriptively, including cross-parent/unit/attempt, sample/label, class-column,
  selected-result, and operation-progress mismatches.
- Inputs cannot be silently rewritten through mutable nested containers in the Phase 1 records.
- Incomplete configuration is representable without a parent, dataset, model, or worker; it cannot be confused with a
  validated parent execution snapshot.
- Model references and fixed configuration references can be shared by duplicate attempts without copying or running
  executable model objects.
- Scientific undefined values, computational failures, stopped work, failed delivery, acceptance, and worker-process
  availability remain independently observable.
- A complete failure-only envelope and a complete envelope containing failed reports that preserve successful results
  can both be represented; neither implies delivery, acceptance, or worker termination.
- Compatibility is decided using plain metadata; supported differing Python versions do not cause rejection.
- Component obligations are covered by record-level tests without fake workers, sockets, splitters, model execution,
  files, or simulated lifecycle engines.
- Applicable pytest, mypy, Black, and Ruff checks pass under the existing configuration. Unavailable checks are
  explicitly reported, not treated as passing.

Completion does not establish a usable evaluation workflow or permit beginning Phase 2 without its own planning and
implementation approvals.

Phase 1 completion requires declarations, validators, component boundaries, and tests to agree
with the schemas and scientific and lifecycle invariants specified in this roadmap.

## 3. Representation and validation conventions

### 3.1 Internal surface and file boundaries

Plan a private `src/cocoonet/_contracts/` package with these cohesive modules:

| Module | Responsibility |
| --- | --- |
| `identity.py` | Typed identifiers, shared scalar aliases, identity validation. |
| `data.py` | Sample metadata, source references, dataset/partition references and realized memberships. |
| `execution.py` | Model/evaluation references, computation snapshot, fixed parent and attempt inputs. |
| `state.py` | Orthogonal runtime observations, stop, bounded operation, and result reception records. |
| `failure.py` | Trace context and failure records. |
| `prediction.py` | Sample-aligned predictions with class identities and value availability. |
| `result.py` | Generic report outcome and complete attempt-result envelope. |
| `compatibility.py` | Plain version metadata and the compatibility guard. |

Avoid one module per dataclass or one class per architecture component. Keep imports in the dependency order in
Section 9; data records do not import runtime state, failures do not embed reports, and reports reference failures by
identifier. Use qualified imports where necessary to keep names clear. The package initializer remains minimal.

There is **no new user-facing public API in Phase 1**. Keep `cocoonet.get_version() -> str` unchanged and do not export
private contracts from the top-level package. Signatures below are internal component interfaces, stable for their
later consumers. Future public computation/worker APIs are explicitly deferred.

### 3.2 Containers and validation behavior

- Apply the representation rules in `AGENTS.md`. Small frozen dataclasses are suitable for fixed metadata, but are not
  required for every payload. Use closed status vocabularies where needed.
- Choose tuples, scientific arrays/tables, or explicitly defined mappings according to their use. Document ownership and
  prevent later mutation of committed inputs; a frozen outer record or read-only array flag alone does not establish
  this guarantee.
- Records must not perform hidden filesystem actions or lifecycle transitions.
- Use `NewType` over `str` for opaque identifiers. Their spelling is not a UUID, digest, filename, path, or ordering
  contract. Generation and storage/wire encoding remain deferred. Runtime validation requires nonempty strings
  without silently stripping, normalizing, or converting them. Separate types are a static safeguard; association
  validators enforce identity scope at runtime.
- Standardized sample identifiers, sample labels, groups, and class identities are text, following `AGENTS.md`. Define
  any separately permitted model-label domain explicitly.
- Standardized classification targets are text; regression targets use the selected finite floating-point
  representation. Do not use a broader shared alias that admits invalid standardized values.
- Input standardization performs permitted conversions. Contract validators check standardized values without coercion.

ScalarLabel accepts standardized Python and NumPy string scalars. TargetValue accepts those strings,
Python floating-point scalars, and NumPy floating-point scalars without restricting precision.
Validation preserves scalar types and values; it does not convert them to Python built-ins.
Integer and boolean values are not standardized labels or targets. A separately permitted model-label
domain requires an explicit definition.

Standalone sample validation rejects nonfinite floating-point targets without narrowing their precision.
Task-specific validation uses explicit evaluation context, not inference from target values. The dataset
API owns input standardization and floating-point representation selection.

- Counts are nonnegative integers, excluding `bool`; eligibility flags are integers exactly `0` or `1`, excluding
  `bool` in the normalized record. Adapters may later accept other input forms but must output this representation.
- `validate_*` functions return `None` and never mutate their arguments. They raise `TypeError` for incorrect runtime
  types and `ValueError` for invalid values, associations, or shapes. Use these built-in exceptions rather than
  introducing a general exception hierarchy.
- Cocoonet-generated runtime type-validation `TypeError` diagnostics identify the field path, expected type, observed
  type, and observed value. They show short scalar values directly and use a bounded, type-aware preview for
  potentially large values without allowing preview failure to replace the established mismatch.
- Contract-validator `ValueError` diagnostics identify the affected context, expected value or condition, and observed
  value or condition. Use a concise sentence for simple violations and a structured multiline form when several
  context fields are needed. Keep observed information bounded; do not render complete large payloads or collections.
  Tests verify the required diagnostic information under AGENTS.md.
  Exact message wording is required only where explicitly contractual.
- Construction does not establish semantic validity. Validate before publication and at independent consuming boundaries
  under `AGENTS.md`. Internal helpers may rely on documented preconditions when ownership prevents intervening
  invalidation. Keep validation explicit and directly testable.
- Validators perform only the checks defined here. They do not execute or inspect user model behavior, determine
  scientific equivalence, inspect source contents, or claim to establish durable persistence or isolation.

These are internal representation decisions, not public registration signatures. Resolve affected schemas before
dependent implementation.

### 3.3 Identifier inventory

Declare `DatasetId`, `SampleId`, `ModelId`, `ParentId`, `AttemptId`, `FoldId`, `WorkerId`, `PartitionId`,
`EvaluationConfigId`, `FailureId`, `OperationId`, and `ArtifactId` as distinct opaque string aliases.

| Identity | Scope and invariant |
| --- | --- |
| `DatasetId` | One exact representation, including sample associations, features, targets, and metadata. Changed representation needs a new identity. |
| `SampleId` | Unique only within its dataset; generated by Cocoonet, paired one-to-one with the supplied sample label. |
| `ModelId` | Cocoonet-assigned reference to the fixed supplied executable input. It is not a model-description hash. |
| `ParentId` | One fixed evaluation configuration and coherent result set. Copy/resume does not generate a replacement. |
| Logical unit: `(parent_id, model_id)` | Exactly one model in a parent; a composite association, not a separate identity record. No independent dataset key in the unit. |
| `AttemptId` | One execution of a unit; duplicate attempts share parent/model identities and fixed inputs. |
| `FoldId` | Unique within one realized partition schedule; tuple order is the authoritative schedule order. |
| `PartitionId` | Opaque handle for a complete realized schedule, not its filename or an additional authoritative copy. |
| `EvaluationConfigId` | Opaque handle for a fixed evaluation-settings snapshot; not a claim of content equivalence. |
| `WorkerId` | Execution target reference, independent of a parent and external public-pool identity. |
| `FailureId`, `OperationId` | Unique within the owning recorded execution/operation context; prevent ambiguous references. |
| `ArtifactId` | Logical handle resolved by the owning storage component; contains no Phase 1 path/publication mechanism. |

Independently consumed unit references carry `parent_id: ParentId` and `model_id: ModelId` directly.
Validate both with `validate_identifier`; consuming boundaries validate their association.
Use temporary `(parent_id, model_id)` tuples for indexing or uniqueness when needed; do not declare
another unit-key type, store a duplicate composite key, or add a compatibility property.

Do not introduce a computation ID simply to duplicate parent identity. A configuration snapshot may exist before any
parent. Root/user authentication identity and pool allocation IDs are not designed here.

Planned leaf functions:

| Signature | Responsibility | Direct tests |
| --- | --- | --- |
| `validate_identifier(value: str, *, field: str) -> None` | Enforce opaque nonempty string representation; no encoding inference. | Empty/non-string rejection; preserve spelling; unrelated ID types may have equal text without sharing scope. |
| `validate_label(value: str, *, field: str) -> None` | Enforce the standardized text label domain from Section 3.2; any separate model-label domain requires its own explicit definition. | Text acceptance; non-text/custom object rejection; no conversion. |

## 4. Data, configuration, and fixed execution records

### 4.1 Dataset and partition records

All fields in the following tables are required unless an explicit default is shown. `tuple[T, ...]` preserves order.

| Record | Fields |
| --- | --- |
| `SourceReference` | String type alias: `type SourceReference = str`; no wrapper record or `relative_path` attribute. |
| `SampleRecord` | `sample_id: SampleId`, `sample_label: str`, `group: str`, `train_mask: int`, `test_mask: int`, `target: TargetValue` (classification: text; regression: selected finite floating-point representation, validated according to task under Section 3.2), `sources: tuple[SourceReference, ...]` |
| `PartitionReference` | `partition_id: PartitionId`, `dataset_id: DatasetId` |
| `FoldRecord` | `fold_id: FoldId`, `training_sample_ids: tuple[SampleId, ...]`, `testing_sample_ids: tuple[SampleId, ...]` |
| `PartitionSchedule` | `reference: PartitionReference`, `samples: tuple[SampleRecord, ...]`, `folds: tuple[FoldRecord, ...]` |

`PartitionReference` is the single identity handle for a prepared dataset/partition association.
Its `dataset_id` identifies the exact dataset representation, and its `partition_id` identifies
the complete realized partition schedule. No separate dataset-reference or dataset-binding record
is used. `PartitionSchedule`, configuration snapshots, and committed parents carry this reference
directly.

`PartitionReference` contains identities only. It does not contain sample memberships, dataset
content, physical paths, or the authoritative partition artifact. Wherever a parent and schedule
are consumed together, the parent's partition reference must equal `PartitionSchedule.reference`.

`SampleRecord` is common population metadata; it is neither an input dataframe nor a feature-data container. Features,
tensor layouts, and file readers remain in Phase 4. Both training and testing refer to these same records; predictions
are separate associated results. At least one source reference is required per sample, including inline samples that
originate in source dataset files. Several samples may share a file and one sample may reference several files.

Choose a platform-independent logical source syntax here: nonempty `/`-separated relative paths, with no empty, `.`
or `..` segment, leading slash, backslash, or drive prefix. This is an internal logical reference, not an OS path;
Phase 4 maps actual source paths to it. Reject invalid syntax rather than repairing it. This lexical check does not
resolve symlinks, establish containment, access a file, or replace Phase 2 ownership validation. Preserve spelling and
case; physical encoding and local storage layouts remain deferred.

SampleRecord.sources contains logical path strings directly. Validate their syntax and uniqueness
without normalization or filesystem access. Inline samples reference their input table; file-backed
samples reference their X-bearing files. Managed reference/metadata tables are not sample sources.

Schedule invariants:

- Samples and folds are nonempty collections, with unique sample IDs, sample labels, and fold IDs in their scopes.
  Groups may repeat. IDs and labels stay paired; reordering a sample tuple does not change their association.
- Each fold references known sample IDs without repetition within either population; train/test IDs are disjoint.
- Training members have `train_mask == 1`; testing members have `test_mask == 1`.
- No group appears in both training and testing populations of one fold.
- Source-file overlap is allowed even when samples/groups must be disjoint. Source references within one sample are
  unique; this does not forbid reuse across samples or folds.
- Empty fold populations are structurally representable. Phase 1 does not invent an empty-population split policy or
  reject masked populations merely to manufacture a computable metric. Phase 4 defines strategy-level admissibility;
  later execution/report components apply their actual input requirements and defined failure/NaN rules.
- The structural validator does not infer a splitting method, regenerate a split, require all samples to be tested
  for every strategy, or reject repeat appearances across folds. Strategy-specific coverage belongs to Phase 4.
- This is the complete logical schedule. Parent records hold a reference to it, not an embedded second authoritative
  storage copy. In-memory/worker inputs may contain it; durable single-file storage belongs to Phase 2/4.

Group-separation clarification: every fold must have disjoint training and testing groups, regardless of
splitting method. Every sample has a group value; Phase 4 normalization assigns distinct per-sample values
when groups are absent and preserves supplied groups. Phase 1 validates existing values and memberships;
it does not generate groups or alter partitions.

`PartitionSchedule` contains its partition reference, sample metadata, and ordered folds. No flag enables or disables
group separation.

`validate_partition_schedule(value: PartitionSchedule) -> None` checks group separation unconditionally.
Its tests must reject a shared group across distinct training/testing samples and accept both separated
repeated groups and distinct per-sample groups, subject to the other schedule invariants. Strategy-specific
compatibility remains Phase 4 work and must preserve the universal separation rule.

### 4.2 Configuration and execution records

| Record / enum | Fields / values |
| --- | --- |
| `TaskKind` | `regression`, `classification` |
| `ModelFamily` | `sklearn`, `sklearn-like`, `pytorch`, `cocoonet` |
| `ModelReference` | `model_id: ModelId`, `model_label: ScalarLabel`, `family: ModelFamily`, `task_kind: TaskKind` |
| `EvaluationReference` | `configuration_id: EvaluationConfigId`, `task_kind: TaskKind`, `class_labels: tuple[ScalarLabel, ...]`, `influence_requested: bool` |
| `ComputationSnapshot` | `root_dir: str`, `consolidation: ConsolidationState`, `partition: PartitionReference | None`, `evaluation: EvaluationReference | None`, `models: tuple[ModelReference, ...]`, `worker_ids: tuple[WorkerId, ...]`, `parent_id: ParentId | None` |
| `ParentEvaluation` | `parent_id: ParentId`, `partition: PartitionReference`, `evaluation: EvaluationReference`, `models: tuple[ModelReference, ...]` |
| `AttemptInputs` | `attempt_id: AttemptId`, `parent_id: ParentId`, `model_id: ModelId`, `worker_id: WorkerId` |

ModelFamily members are SKLEARN="sklearn", SKLEARN_LIKE="sklearn-like", PYTORCH="pytorch",
and COCOONET="cocoonet". COCOONET identifies models constructed through Cocoonet-native composition
tools, including native ensembles and nested native compositions. Each constituent retains its own
family identity. Standard scikit-learn ensembles and compositions remain SKLEARN.
Phase 1 validates these tags without inspecting model behavior or defining composition APIs.

`ConsolidationState` has exactly `initialized`, `consolidated`, and `unconsolidated`.

`root_dir` is the user-selected location recorded as a nonempty string; no normalization or filesystem validation
happens in Phase 1. The configuration component later enforces ownership before initializing/reassigning it.

`ComputationSnapshot` is a read-only description, not the mutable public computation. It intentionally
permits no active dataset/partition association, no settings, no models, and no workers in initialized
state. `partition=None` means that the prospective configuration has no active dataset and realized
partition. `parent_id` is absent for initialized state and required for consolidated/unconsolidated
state; in unconsolidated state it identifies the previous execution, not a new parent for the
prospective configuration. Models and worker references are unique, including model labels within
the prospective configuration. An incomplete prospective configuration after a permitted removal is
representable. The validator does not compare changed configuration with an old parent or perform a
state transition.

Each `ModelReference` identifies the fixed supplied model input; the owning configuration component retains that
input and later creates the payload. No executable object is stored here. Each `EvaluationReference` identifies all
fixed task-wide evaluation settings owned by that component. The explicit task kind, class order, and influence flag
are the parts required by shared result validation, not an exhaustive settings schema. Later settings producers must
bind an immutable snapshot to the handle and must not reuse it for changed settings. No model may override it.

Regression has an empty `class_labels` tuple. Classification has a nonempty unique tuple in the authoritative class
order. Do not impose a minimum of two classes at this structural layer: missing/insufficient classes can be actual
scientific operation failures. Classifier adapters later map outputs into the declared order without fabricating
missing probabilities. All models in a parent match its task kind; their families may differ.

Parent and attempt invariants:

- A parent has one partition reference, one evaluation reference, and at least one model reference.
  Model IDs and model labels are unique within the parent; every model agrees with its task-wide task kind.
- Each model defines one logical unit identified by the parent's parent_id and that model's model_id.
  ParentEvaluation stores model references directly, without embedded unit records or a second unit collection.
- Evaluation configuration, dataset, partition, and model inputs remain fixed throughout each attempt's
  lifetime. Supplied model/framework parallelism settings remain part of those fixed inputs.
- AttemptInputs carries its routing identities directly. Require parent_id to match the supplied parent
  and resolve model_id through that parent's models; reject unknown models and mismatched parents.
  Duplicate attempts share parent/model identities and fixed scientific inputs; attempt identities differ,
  and workers may differ.
- Worker n_attempt and effective attempt capacity belong to worker configuration and execution control,
  not the parent's scientific inputs. Worker-only configuration edits do not create another parent.
- Attempts remain separate from the parent; a parent does not own worker lifetime.
- Copying a configuration does not create identities, runtime resources, or artifacts. Resumption and prospective
  edits follow architecture Section 2.3; transition/copy algorithms are Phase 3/15, not methods on these records.

### 4.3 Planned data/execution validators and tests

| Signature | Responsibility | Direct tests |
| --- | --- | --- |
| `validate_source_reference(value: SourceReference) -> None` | Validate logical relative syntax only. | Valid nested reference; empty/absolute/drive/backslash/dot/traversal forms rejected; no filesystem access. |
| `validate_sample_record(value: SampleRecord) -> None` | Validate identity/label/group, standardized target type and floating-point finiteness, normalized masks, and nonempty unique sources; do not infer task kind. | Python/NumPy scalar acceptance without conversion; reject integer/boolean targets and nonfinite floating targets; preserve extended precision; valid/invalid masks and sources. |
| `validate_partition_schedule(value: PartitionSchedule) -> None` | Validate complete logical associations and per-fold eligibility/disjointness/group rules. | Duplicate ID/label/fold; unknown ID; forbidden membership; group leakage; legal file overlap; reordered metadata; empty populations; repeated cross-fold use. |
| `validate_model_reference(value: ModelReference) -> None` | Check identity/label and declared family/task-kind tags. | Exact four-family enum vocabulary, including sklearn-like and cocoonet; valid/invalid family and task-kind tags; no model object/API inspection or derived descriptions. |
| `validate_evaluation_reference(value: EvaluationReference) -> None` | Check settings handle, task kind, class order, influence flag. | Regression/classification class rules; equality collisions; one class allowed; no sorting/coercion. |
| `validate_computation_snapshot(value: ComputationSnapshot) -> None` | Check structural consolidation/parent relationships, the optional partition and evaluation references, and model/worker uniqueness while allowing incomplete configuration. | All three states; zero models/workers/partition; present partition with valid or invalid partition/dataset identity; missing required parent; unchanged parent after worker-list edit; snapshot unchanged by validation. |
| `validate_parent_evaluation(value: ParentEvaluation) -> None` | Check partition-reference identities, fixed parent structure, nonempty models, unique model IDs/labels, and agreement with the task-wide task kind. | Invalid partition/dataset identity; empty models; mixed task kinds; duplicate IDs/labels; same label in different parents allowed. |
| `validate_attempt_inputs(value: AttemptInputs, *, parent: ParentEvaluation) -> None` | Resolve the exact unit/model context and validate attempt/worker identities. | Unknown/cross-parent unit; duplicates with different attempts/workers; no independent scientific override. |

When an execution record contains a `PartitionReference`, its validator validates both
`partition_id` and `dataset_id` with `validate_identifier`. Execution-record validation does not
load the partition artifact or verify dataset contents. Validators receiving both a parent and a
schedule require `parent.partition == schedule.reference`.

Collection association checks use ID-indexed lookup and sets where possible; do not introduce quadratic pairwise
searches or caches. Runtime label comparisons follow Section 3.2. No identity generator is implemented in Phase 1.

## 5. Traceable failures and value semantics

### 5.1 Trace and failure records

| Record / enum | Fields / values |
| --- | --- |
| `FailureScope` | `fit_prediction`, `report`, `measure`, `sample_influence`, `data_preparation`, `data_sync`, `worker_management`, `result_transfer`, `result_persistence`, `worker_process` |
| `TraceContext` | `stage: str`, `parent_id: ParentId | None = None`, `model_id: ModelId | None = None`, `attempt_id: AttemptId | None = None`, `worker_id: WorkerId | None = None`, `fold_id: FoldId | None = None`, `sample_id: SampleId | None = None`, `class_label: ScalarLabel | None = None`, `report_name: ReportName | None = None`, `operation_id: OperationId | None = None` |
| `FailureRecord` | `failure_id: FailureId`, `scope: FailureScope`, `context: TraceContext`, `exception_type: str`, `message: str`, `traceback: str`, `origin_failure_id: FailureId | None = None` |

`ReportName` is the shared enum `validation`, `performance`, `residual`, `visualization`, `influence`; place it in
`identity.py` to avoid a failure/result import cycle. Stage names are descriptive nonempty strings rather than a
premature exhaustive enum of future algorithms. Initial reconstruction and primary-fold model initialization, fitting,
and prediction use `fit_prediction` scope with distinct stages such as `reconstruction`, `fold_initialization`, `fit`,
and `prediction`. `reconstruction_failed` identifies failure before primary validation begins. After successful
reconstruction, any primary-fold initialization, fit, or prediction failure produces `validation_failed`. Influence
removal-model failures use `sample_influence` scope and do not produce another attempt-level outcome.
Configuration and startup errors are reported directly through the root-side API and are not represented by
`FailureScope` or `FailureRecord`.

- Scope describes the affected object or operation, not the whole process. A worker-process failure identifies an
  unavailable worker process and is not inferred from any scientific, transfer, persistence, synchronization, or
  management error.
- Model context requires parent context; attempt context requires parent/model/worker context; fold/sample context
  requires parent context. Class-specific context requires a class label. `None` means not applicable/known, not an
  anonymous replacement ID. A known class or sample must never be dropped from an error record.
- Report/measure failure requires `report_name`; sample-influence failure requires model, attempt, fold, sample, and
  influence report context. Data-preparation, data-sync, worker-management, result-transfer, and result-persistence
  failures require an operation ID. Fit/prediction failures included in attempt results carry that attempt's
  parent/model/worker context.
- Exception type and message are nonempty. Preserve the full supplied traceback as a string. It may be empty only
  for a failure detected without a caught exception; never replace a captured traceback with an empty value.
- A dependent skip refers to its originating failure by ID. An origin must resolve in the accompanying failure set;
  origin links cannot cycle or cross an unrelated attempt. Parent/model/attempt/worker identities must agree wherever
  both records supply them; stage and report name may differ, and a shared error need not have the dependent result's
  narrower sample/class coordinates. No duplicate full errors are required merely to skip work.
- Capturing exceptions, assigning stages, writing logs, retrying, and deciding where to catch errors are producer
  responsibilities in later phases. These records support those boundaries without implementing them.

### 5.2 Defined, undefined, unavailable, and non-applicable values

Phase 1 does not define a `ValueStatus` enum. Generated value meaning is derived from the value, its owning schema, and
its aligned value-level failure references.

| Condition | Internal Python representation | Value-level failure references | Human-readable output |
| --- | --- | --- | --- |
| Finite, mathematically defined measure | `float` | Empty | Numeric value |
| Mathematically permitted positive infinity | `float("inf")` | Empty | `Inf` |
| Mathematically permitted negative infinity | `float("-inf")` | Empty | `-Inf` |
| Mathematically undefined measure | `float("nan")` | Empty | `NaN` |
| Computationally unavailable value | `None` | Nonempty | `NA` |
| Explicitly non-applicable value | `None` | Empty, only where the schema permits it | `NA` |

A required value cannot be `None` without an applicable failure. A present value cannot carry a value-level
unavailability failure. `NaN` is reserved for mathematical undefinedness and does not require a failure solely because
it is undefined. Exceptions and failed prerequisites produce `None` with failure provenance rather than `NaN`.

Exact identifiers, categorical labels, booleans, and integer counts retain their declared types. A generated
categorical prediction is either a valid label without a value-level failure or `None` with applicable failure
references. Report-level `ReportStatus` remains separate because it describes the outcome of a whole report rather
than the state of one value.

Infinity is accepted only where the corresponding measure's documented formula permits an extended-real result.
Standardized regression predictions and class probabilities must be finite when present. Invalid nonfinite model
outputs are computational failures represented as `None` with failure provenance.

Every class that owns generated scientific values or generated categorical predictions must explain its applicable
representation conditions in the `Notes` section of its NumPy-style docstring. The explanation must state, for every
condition accepted by that class:

- the internal value representation;
- whether value-level failure references must be empty or nonempty;
- the corresponding human-readable output;
- the meaning of `None` and `NaN`, when either can appear.

When a potentially confusing condition is forbidden by the class, its `Notes` section must say so explicitly. For
example, the prediction classes must state that present predictions are finite, that `None` means computationally
unavailable and requires failure references, and that `NaN`, infinity, and non-applicable `None` are not accepted
prediction values. Future concrete report-payload classes must document any permitted finite, infinite, `NaN`,
unavailable `None`, and non-applicable `None` cases in the same way.

`ReportOutcome` must document that its `status` is report-level and does not replace or reinterpret the payload
class's value/failure contract. Do not introduce base classes, helper objects, runtime descriptors, or duplicate status
fields solely to satisfy this documentation requirement. Review the required `Notes` content with each declaration;
do not add brittle tests that compare exact docstring wording.

Do not coerce `NaN` to zero, clip extreme values, fill missing probabilities, normalize model outputs, or aggregate
while skipping `NaN`. No arithmetic metric routine is implemented here. Result serialization is deferred, but its
later contract must preserve `float`, infinity, `NaN`, `None`, and failure-reference distinctions. Human-readable
output must use numeric values and the exact tokens `Inf`, `-Inf`, `NaN`, and `NA`.

| Signature | Responsibility | Direct tests |
| --- | --- | --- |
| `validate_trace_context(value: TraceContext) -> None` | Validate applicable identity/label fields and context dependencies. | Minimal stage-only context; complete attempt/fold/sample context; orphan model/attempt/sample; invalid stage. |
| `validate_failure_record(value: FailureRecord) -> None` | Check scope-specific context, error text, and non-self origin. | Each scope; operational scope without operation ID; report without name; sample influence without sample; worker failure without worker; captured text preserved. |
| `validate_failure_links(values: tuple[FailureRecord, ...]) -> None` | Check unique IDs, resolved origins, and acyclic same-context dependency references. | Dangling/self/cyclic origins; valid shared origin; unrelated attempt origin; duplicate failure ID. |

## 6. Standardized predictions and complete result contracts

### 6.1 Predictions

Prediction values and their position-aligned failure references use immutable tuples. A successful
value has an empty failure tuple; an unavailable value has the applicable failure IDs. This represents
independently unavailable labels and probability cells without coordinate lookup, mutable mappings,
or parallel status arrays.

The representation includes empty failure tuples for successful positions. This cost is accepted
for direct positional inspection and immutable ownership. Phase 1 defines no sparse association
type or alternative-representation adapter.

Preserve exact sample/class alignment and None-versus-NaN semantics. Concrete model adapters remain
in their assigned phases.

| Record | Fields |
| --- | --- |
| `RegressionPredictions` | `fold_id: FoldId`, `sample_ids: tuple[SampleId, ...]`, `values: tuple[float | np.floating | None, ...]`, `failure_ids: tuple[tuple[FailureId, ...], ...]` |
| `ClassificationPredictions` | `fold_id: FoldId`, `sample_ids: tuple[SampleId, ...]`, `class_labels: tuple[ScalarLabel, ...]`, `predicted_labels: tuple[ScalarLabel | None, ...]`, `label_failure_ids: tuple[tuple[FailureId, ...], ...]`, `probabilities: tuple[tuple[float | np.floating | None, ...], ...]`, `probability_failure_ids: tuple[tuple[tuple[FailureId, ...], ...], ...]` |

`PredictionBatch = RegressionPredictions | ClassificationPredictions` is a type alias, not another wrapper class.
Per-value and per-class failure references allow independently computed values to survive a failed component without
introducing parallel status vectors.

Prediction invariants:

- For `n` testing samples and `k` classes, regression and predicted-label values have length `n`, and probabilities have
  shape `(n, k)`. Failure associations identify the corresponding value positions exactly; their container need not
  mirror the value container.
- IDs exactly match the saved testing sequence, not merely its length or set. Adapters explicitly align their outputs;
  the validator rejects misalignment rather than silently reordering. Class labels exactly match parent class order.
- True targets, labels, groups, masks, and source references resolve through `SampleRecord`; do not duplicate a
  competing truth vector or different train/test sample schema inside predictions.
- A present predicted class label belongs to the parent class vocabulary and has no value-level failure reference.
  An unavailable predicted label is `None` and has at least one applicable failure ID. Non-applicability is not valid
  for a prediction corresponding to a saved testing sample.
- Present regression predictions and probabilities accept Python and NumPy floating-point scalars without
  conversion or precision restriction. Present regression predictions are finite; present probabilities are finite
  and within `[0, 1]`. Arrays and tensors remain model-adapter inputs. `NaN` and
  infinity are invalid standardized model outputs rather than formula-defined prediction values. An unavailable
  regression prediction or probability is `None` with at least one applicable failure ID.
- Present predictions have no unavailability failure association. Every `None` prediction has applicable failure
  references. Formula-defined `NaN` and mathematically permitted infinity belong to later report measures, not
  standardized predictions.
- Probability normalization and raw-model numerical admissibility belong to model adapters. This structural layer
  does not introduce a probability-sum tolerance, renormalization, clipping, or reconstructed missing probabilities.
- Empty testing sequences yield empty aligned rows. There is no fabricated prediction for an excluded sample.
  Predictions on training/removal-comparison populations used internally by future algorithms are not automatically
  validation records.

### 6.2 Report and result envelopes

A complete attempt result carries parent, model, attempt, and worker identities directly.
AttemptInputs serves allocation; ReceptionRecord serves reception and acceptance. Both identify their
logical unit through direct parent_id and model_id fields. Neither is embedded in the completed result.

A report's fixed result slot supplies its category. ReportOutcome contains status, payload,
and failure references. An independently consumed report receives its category explicitly from
its caller. Failure context includes ReportName when needed to identify the failed operation.

EvaluationOutcome is derived from a validated complete result. It is not a constructor field,
cached property, or independently maintained state. Report statuses remain authoritative because
payload presence alone cannot distinguish complete, failed, blocked, and unrequested reports.

| Record / enum | Fields / values |
| --- | --- |
| `ReportStatus` | `complete`, `failed`, `blocked`, `not_requested` |
| `ReportOutcome[T]` | `status: ReportStatus`, `payload: T | None`, `failure_ids: tuple[FailureId, ...]` |
| `EvaluationOutcome` | `completed`, `completed_with_errors`, `validation_failed`, `reconstruction_failed` |
| `AttemptResult[V, P, R, Z, I]` | `parent_id: ParentId`, `model_id: ModelId`, `attempt_id: AttemptId`, `worker_id: WorkerId`, `predictions: tuple[PredictionBatch, ...]`, `validation: ReportOutcome[V] | None`, `performance: ReportOutcome[P] | None`, `residual: ReportOutcome[R] | None`, `visualization: ReportOutcome[Z] | None`, `influence: ReportOutcome[I] | None`, `failures: tuple[FailureRecord, ...]` |

Generic type parameters reserve concrete report-data contracts for their producing phases without `Any`, an open
dictionary schema, placeholder report implementations, or a premature list of metric fields. A later concrete payload
must be an owned immutable data snapshot under that report's approved schema. Phase 1 validates the envelope, not an
unknown payload's internals; its tests use a small immutable typed payload fixture. Generic payload support is not
permission to send arbitrary executable objects through a result path.

The future validation/fold payload must carry the common sample metadata, true targets, user labels, and separate
training/testing source references required by architecture Section 4.4, associated with these predictions. Resolving
truth from the schedule during computation does not authorize omitting it from required user-facing fold results.
Concrete field layout and rendering belong to Phase 7; model labels in reports come from the fixed parent model
reference, not inferred descriptions.

Report outcome rules:

- `complete`: payload present, no operational failure references; formula-defined `NaN`, mathematically permitted
  infinities, and schema-defined non-applicable `None` values may still exist inside it.
- `failed`: one or more computational failures and nonempty failure references. Preserve successful and identified
  unavailable entries in the payload when available; otherwise the payload may be absent. Any report-level
  computational failure uses this status—there is no separate partial status.
- `blocked`: nonempty references to originating prerequisite failures; retain an unavailable result structure if
  available. Do not generate inputs or retrain to make a payload.
- `not_requested`: no payload or failures; allowed only for unrequested optional influence in the attempt envelope.
- Result-level validation resolves report/prediction failure references against the envelope's failure set. Detailed
  report value/failure consistency is checked by its later concrete payload validator.

Complete attempt rules:

- `reconstruction_failed` means initial model reconstruction failed before primary validation began. Predictions are
  empty, all report fields are `None`, and the envelope contains a `fit_prediction` failure with stage
  `reconstruction`. This is the only failure-only scientific outcome.
- `validation_failed` means reconstruction succeeded but at least one primary fold-model initialization, fit, or
  prediction failed. Preserve one prediction batch per schedule fold in schedule order; unavailable predictions in a
  failed fold are `None` with applicable failure references. Continue later independent folds where possible.
- A `validation_failed` envelope contains a failed validation report preserving available fold content. Performance,
  residual, and visualization reports are failed when they retain affected content or blocked when their prerequisites
  are unavailable. Requested influence is blocked because complete primary baseline predictions are unavailable;
  unrequested influence remains `not_requested`.
- `completed` means primary validation and every requested report completed without computational failure. It has no
  failed or blocked report and no unavailable prediction. Formula-defined `NaN`, permitted infinity, and
  schema-defined non-applicable `None` do not change this outcome.
- `completed_with_errors` requires completely successful primary validation and at least one failed or blocked ordinary
  or requested influence report. Preserve successful reports and successful values within failed reports.
- Optional influence uses stored successful primary-validation predictions as its fixed baseline. A removal-model
  initialization, fit, prediction, or calculation failure marks the influence report failed, preserves independent
  sample/class results, and contributes to `completed_with_errors`. No separate influence attempt outcome or
  influence evaluation-status field exists.
- Outcome precedence is `reconstruction_failed`, then `validation_failed`, then `completed_with_errors`, then
  `completed`. This precedence summarizes the required-stage result without replacing individual report statuses or
  failure records.

derive_evaluation_outcome() is a pure function consuming an already validated complete AttemptResult.
It applies the precedence above using reconstruction, primary-validation, and report failure facts.
It performs no mutation, caching, persistence, or recursive revalidation.

validate_attempt_result() checks completeness, stage consistency, and provenance directly. It does not
depend on a separately supplied outcome or use derivation as a substitute for those checks. Invalid or
incomplete records must not receive an outcome merely by applying precedence.

A Phase 12 outcome summary, if needed for independent consumption, is derived from the validated result
and checked against that result when loaded. It is not another authoritative state.

- The schedule's partition and dataset IDs must match the parent's binding. The result's parent_id
  must match that parent, and model_id must resolve to its models. Validate attempt/worker identities
  and matching failure context directly, without constructing allocation records solely for validation.
- Before any reconstruction-failure early return, validate every schedule target against parent.evaluation:
  regression requires finite floating-point scalars; classification requires strings belonging to the
  authoritative class vocabulary. These checks include masked samples. The prediction subtype must
  also match the parent's task kind.
- Failures and all fold/sample/class contexts resolve against the same parent, attempt, schedule, and class vocabulary.
  A sample-specific influence error names a training member of its stated fold; prediction errors naming a sample
  refer to that fold's testing population. A shared fold-level error need not invent a sample/class identity.
  The envelope's failure list describes scientific computation, not later data preparation, data synchronization,
  worker management, result transfer, result persistence, configuration, startup, or worker-process incidents.
  Those failures remain at their owning boundaries.
- Interruption/cancellation is not a fabricated reconstruction or primary-validation failure or complete result. A
  stopped attempt is represented by Section 7's state; this phase does not define a new cancelled-result file.
- One envelope describes one attempt; result files, dill bytes, completion publication, report loading, and user-facing
  rendering are later component responsibilities. At most one accepted result per unit is a separate root invariant.

### 6.3 Planned prediction/result validators and tests

| Signature | Responsibility | Direct tests |
| --- | --- | --- |
| `validate_regression_predictions(value: RegressionPredictions, *, fold: FoldRecord) -> None` | Validate sample alignment, vector lengths, finite present values, and value/failure-reference relationships. | Reordered/unknown/excluded IDs; length mismatch; finite value without failure; `None` with failure; reject `None` without failure, present value with failure, `NaN`, infinity, bool, and wrong numeric type; empty test set; no input changes. |
| `validate_classification_predictions(value: ClassificationPredictions, *, fold: FoldRecord, class_labels: tuple[ScalarLabel, ...]) -> None` | Validate class/sample alignment, matrix shapes, label membership, finite probabilities, and value/failure-reference relationships. | Swapped classes; ragged matrices; missing/unknown labels; independently unavailable label/class cells; reject `None` without failure, present value with failure, `NaN`, infinity, bool, and out-of-range probability; no probability filling/reordering. |
| `validate_report_outcome[T](value: ReportOutcome[T]) -> None` | Validate status/payload/failure relationships; report category comes from the owning slot or caller. | Every report status; complete immutable payload containing permitted NaN/infinity/None; failed with preserved or absent payload; blocked with preserved structure; required failure references; no stored report name. |
| `validate_attempt_result[V, P, R, Z, I](value: AttemptResult[V, P, R, Z, I], *, parent: ParentEvaluation, schedule: PartitionSchedule) -> None` | Validate direct routing, parent/schedule association, task-compatible targets, predictions, reports, provenance, and complete-stage rules. | Completed regression/classification; reconstruction failure without predictions/reports; primary-fold failure with preserved results and failed/blocked dependent reports; remaining-fold continuation; blocked influence; ordinary/influence failures after successful validation; unrequested influence; undefined-only completion; failure-only target checks; mismatched routing, task, vocabulary, schedule, dangling provenance, and excluded non-computational failure scopes. |
| `derive_evaluation_outcome[V, P, R, Z, I](value: AttemptResult[V, P, R, Z, I]) -> EvaluationOutcome` | Derive the summary of a validated complete result without mutation or revalidation. | All four outcomes and precedence; ordinary/influence failures after successful validation; formula-defined NaN does not create a failure; no caching. |

The type-parameter notation above uses Python 3.12 generic syntax. Declaration and validator units remain separate;
the notation does not authorize bundling helpers with their caller.

## 7. Orthogonal state and bounded-operation records

These are immutable observations owned by the indicated component. Phase 1 validators check internally consistent
snapshots; they do not implement events, permitted-transition tables, allocation policy, stop delivery, readiness
probes, result arbitration, retry loops, or recovery. Later components replace snapshots under their own authority.

Identify each observation’s owner and whether it is an independent fact or a derived summary, especially for parent
progress and worker availability. Derive convenience summaries where practical and validate intentionally stored
summaries against their source facts.

Preserve independent stop, process, readiness, selection, receipt, persistence, and acceptance observations. These
shared contracts remain in Phase 1; durable mechanisms belong to Phase 2, and runtime transitions and scheduling belong
to their later phases.

### 7.1 Vocabulary and fields

| Record / enum | Fields / values |
| --- | --- |
| `ExecutionStatus` | `not_started`, `running`, `interrupted`, `settled` |
| `ParentProgress` | `parent_id: ParentId`, `execution_status: ExecutionStatus` |
| `AttemptStatus` | `allocated`, `running`, `result_ready`, `stopped` |
| `StopStatus` | `none`, `requested`, `observed` |
| `StopReason` | `parent_interrupted`, `parent_cancelled`, `parent_superseded`, `duplicate_accepted` |
| `StopDisposition` | `status: StopStatus`, `reason: StopReason | None` |
| `AttemptProgress` | `inputs: AttemptInputs`, `status: AttemptStatus`, `stop: StopDisposition`, `first_started_at: float | None`, `result_artifact_id: ArtifactId | None` |
| `WorkerAvailability` | `unavailable`, `available`; usability is separate from occupied capacity. |
| `WorkerProcessStatus` | `unknown`, `running`, `unavailable` |
| `OwnershipStatus` | `unknown`, `owned_by_root`, `not_owned_by_root` |
| `CompatibilityStatus` | `unchecked`, `compatible`, `incompatible` |
| `ParentReadiness` | `parent_id: ParentId`, `dataset_ready: bool`, `evaluation_ready: bool`, `partition_ready: bool` |
| `WorkerObservation` | `worker_id: WorkerId`, `availability: WorkerAvailability`, `process_status: WorkerProcessStatus`, `ownership: OwnershipStatus`, `compatibility: CompatibilityStatus`, `parent_readiness: tuple[ParentReadiness, ...]`, `attempt_capacity: int | None`, `active_attempt_ids: tuple[AttemptId, ...]` |
| `OperationKind` | `connection_round`, `transfer`, `persistence` |
| `OperationStatus` | `pending`, `running`, `succeeded`, `failed`, `cancelled` |
| `BoundedOperation` | `operation_id: OperationId`, `kind: OperationKind`, `context: TraceContext`, `attempt_limit: int`, `attempts_made: int`, `status: OperationStatus`, `last_failure: FailureRecord | None` |
| `ReceiptStatus` | `not_received`, `incomplete`, `complete` |
| `PersistenceStatus` | `not_persisted`, `incomplete`, `complete` |
| `ReceptionRecord` | `parent_id: ParentId`, `model_id: ModelId`, `selected_attempt_id: AttemptId | None`, `selection_evidence_id: ArtifactId | None`, `receipt: ReceiptStatus`, `persistence: PersistenceStatus`, `result_artifact_id: ArtifactId | None`, `accepted: bool` |

`first_started_at` is a finite, nonnegative UTC Unix timestamp in seconds describing an observed execution start;
it is not used as a timer here. The coordinator later derives a unit's first execution start from its attempts for
oldest-running duplicate selection. Clock acquisition, ordering ties, and scheduling remain later algorithms.

Parent dataset/configuration/partition readiness and reusable root-side model-artifact preparation precede attempt
allocation and are outside the attempt lifecycle. `allocated` fixes the attempt identity, unit, and worker but has
no start timestamp because attempt-specific model synchronization or processing has not begun. `running` begins
when the worker starts attempt-specific model synchronization or local handoff; `first_started_at` records that
boundary. Synchronization may transfer the prepared model input, reuse an already available valid representation,
or establish applicable local access without retransmitting the parent dataset. Running continues through model
loading/reconstruction, evaluation, report and optional influence computation, and publication of the complete
worker-side result artifact. Publication changes the status to `result_ready`. Reception selection, delivery,
persistence acknowledgement, and bounded resend remain part of the attempt lifecycle but are represented by the
separate reception, persistence, and bounded-operation records rather than another attempt status.

The operation vocabulary includes only the architecturally required bounded operations. Add another kind only when
its owning phase needs it and approves that contract amendment; do not make scientific fitting retryable by treating
it as an operation of these kinds.

### 7.2 State invariants and ownership

- Root control owns consolidation and parent execution observations. `settled` describes execution handling, not an
  alternate acceptance bit; an exhausted delivery may leave an unaccepted unit. Worker inactivity is a separate fact.
- Worker control owns attempt execution/stop observations. Running requires a start timestamp; allocated has none.
  Result-ready requires a start timestamp and a completed result artifact reference. Stopped attempts may have
  started or may have been stopped before execution; stopping does not imply a result file or stopped worker process.
  Result artifacts are allowed only for result-ready or stopped attempts with an already completed result and
  a start timestamp.
- Stop `none` has no reason; requested/observed requires a reason. `stopped` requires observed stop. Result-ready may
  coexist with requested/observed stop because a terminal attempt needs no new evaluation loop to end unnecessary
  handling. Allocated/running attempts may have no stop or a requested stop, but not an observed stop. Active work can
  be marked requested while still running; edits elsewhere do not alter its inputs.
- Configured worker membership is separate from availability. Disconnection makes the worker
  unavailable for new allocations but preserves its assigned attempt identities. Unknown
  process status after lost contact is not proof of process failure.
- Available requires ownership by this root, compatible framework metadata, an observed running
  process, and known attempt capacity. It does not imply that any slot is free.
- attempt_capacity is a positive integer excluding bool, or None when unknown. For CPU workers,
  it is the effective_n_attempt supplied by worker control under architecture Section 8; for GPU
  workers, it is one. It is neither requested n_attempt nor a CPU count.
  Requested/effective limit reporting, CPU-demand accounting, the one-step adjustment or rejection,
  and capacity revalidation belong to worker control in the applicable later phases.
  Phase 1 does not detect CPUs, calculate capacity, or inspect model/framework parallelism settings.
- active_attempt_ids contains unique, valid IDs for allocated attempts whose slot reservations
  have not been released, including applicable pending-delivery handling. Attempt state and
  provenance remain in their own records.
- Admission requires known capacity and fewer occupied reservations than that capacity.
  Free slots are derived as max(0, attempt_capacity - len(active_attempt_ids)). A prospective
  lower limit may temporarily be below the number of earlier reservations; it permits no
  additional allocation and does not itself cancel those attempts.
- Parent readiness has separate dataset, evaluation, and partition facts, with unique parent IDs.
  Worker usability and free capacity do not establish readiness for a particular parent.
  Runtime allocation and reservation synchronization belong to later phases.
- Local and online workers use the same logical observation; there is no fake connection record for a local worker.
  Authentication and public allocation details are not encoded by `owned_by_root`.

Each reservation identifies one complete model-evaluation attempt. Cocoonet CPU multiprocessing is
across attempts, not folds or sample-removal evaluations. GPU execution has one Cocoonet attempt
slot and no Cocoonet-level multiprocessing. Model/framework loading and training parallelism
remains independently configured and passes unchanged to its owner.

Phase 1 records describe reservations and observations only. Admission synchronization, subprocess
supervision, isolation, and cooperative stopping belong to their runtime owners. A scientific result
or derived outcome does not itself release a slot or establish delivery or acceptance.

### 7.3 Bounded-operation semantics

`attempt_limit` is the **total** allowed attempts including the initial attempt, supplied by the future policy owner;
it is at least one. `attempts_made` is the count already begun and lies in `[0, attempt_limit]`. A future policy expressed
in retries must explicitly convert to this total; no default retry count or delay is selected here.

- Pending has zero attempts and no failure. Running has at least one attempt; it may preserve the previous failure.
- Succeeded has at least one attempt. Earlier failure information may remain; success does not mean no retry occurred.
- Failed has at least one attempt and a last failure. When its count reaches the limit it is exhausted. No extra
  exhausted status or independent flag may contradict the count/status pair.
- Cancelled may have zero or more attempts and does not imply success, acceptance, or worker-process unavailability.
- The last failure, when present, names the same operation ID and context and uses the applicable detailed operational
  scope. `connection_round` uses `worker_management`; `transfer` uses `data_sync` or `result_transfer` according to
  the transferred artifact; `persistence` uses `data_preparation` or `result_persistence` according to the persisted
  artifact. Its failure ID is local to this operation record, not a scientific error in the result envelope.
- Separate connection rounds use separate operation identities. This allows a later scheduled round without resetting
  an exhausted delivery operation. Records preserve counts; Phase 1 has no automatic retry or count-reset method.

### 7.4 Reception and acceptance semantics

Root coordination owns one `ReceptionRecord` per `(parent_id, model_id)` pair, covering local and online attempts equally:

- Before selection, both selected attempt and selection evidence are absent, receipt is not received, persistence is
  not persisted, result artifact is absent, and accepted is false.
- A selection names an attempt with the same parent_id and model_id and evidence that the future storage
  component has durably published. The artifact handle alone is not proof of durability; Phase 2 establishes
  and verifies that evidence. Production reception must not proceed without it. Selecting by receiving order
  belongs to later coordination.
- Incomplete receipt may coexist with incomplete persistence because bytes may be written as received. Complete
  persistence requires complete receipt and a result artifact. A result artifact may describe an incomplete write;
  presence alone is not completion. Incomplete/complete persistence requires an artifact reference.
- Acceptance requires selected attempt, durable selection evidence, complete receipt, complete persistence, and the
  corresponding result artifact. Complete persistence with `accepted=False` is representable for the crash/reconcile
  window. An acceptance validator checks structure; it never opens a file or verifies durability.
- A selected attempt remains selected even if delivery fails or is exhausted. Do not promote a competitor, erase
  selection, infer acceptance from a scientific result, or mark a unit complete because its worker became available.
- Duplicate files are not received as independent report deliveries. An unselected ready attempt can exist without
  any reception/operation record. Later arbitration stops duplicates only after acceptance.

### 7.5 Planned state validators and tests

| Signature | Responsibility | Direct tests |
| --- | --- | --- |
| `validate_parent_progress(value: ParentProgress) -> None` | Validate parent identity and execution tag, independently of consolidation. | All tags; invalid tag; no inferred acceptance/consolidation transition. |
| `validate_stop_disposition(value: StopDisposition) -> None` | Check stop status/reason pairing. | None without reason; requested/observed with each reason; invalid pairs. |
| `validate_attempt_progress(value: AttemptProgress, *, parent: ParentEvaluation) -> None` | Check attempt context, status/start/artifact/stop consistency. | Allocated/running/result-ready; stop while running; stopped before start; terminal handling stop; invalid timestamp; no result created. |
| `validate_worker_observation(value: WorkerObservation) -> None` | Check usability prerequisites, known/unknown positive capacity, unique valid reserved-attempt IDs, and unique parent readiness. | Available worker with zero, some, or all slots occupied; unavailable worker retaining multiple attempts; unknown capacity; invalid/bool capacity; duplicate attempt IDs; reservations exceeding a prospectively lowered limit; no scheduling or mutation. |
| `validate_bounded_operation(value: BoundedOperation) -> None` | Check count bounds, status, last-error context, and operation kind. | Zero/past-limit counts; bool rejected; success after retries; failed/exhausted; cancelled; distinct connection round identity; no budget reset or reevaluation. |
| `validate_reception_record(value: ReceptionRecord, *, attempts: tuple[AttemptInputs, ...]) -> None` | Validate parent/model identifiers, selected-attempt identity and parent/model agreement, and receipt/persistence/acceptance prerequisites. | Invalid parent/model IDs even before selection; unknown selected attempt; mismatched parent or model; partial file; complete persistence before acceptance; accepted failure outcome uses same facts; absent evidence; no file I/O. |
| `validate_reception_set(values: tuple[ReceptionRecord, ...], *, attempts: tuple[AttemptInputs, ...]) -> None` | Check unique parent/model pairs using a local tuple set and unique attempt IDs, then validate each record. | Reject duplicate parent/model pairs and duplicate attempt IDs; allow the same model ID under different parents and different models under one parent. |

Runtime enforcement against concurrent writes, historical selection replacement, or acceptance replacement cannot be
proved by a pure snapshot validator. Later storage/coordinator tests must exercise those behaviors. Phase 1 does not
claim that frozen records alone enforce distributed at-most-one acceptance.

## 8. Compatibility and component boundaries

### 8.1 Compatibility metadata and guard

`CompatibilityMetadata` contains:

- `cocoonet_version: str`: the complete nonempty installed Cocoonet version string.
- `python_version: tuple[int, int, int]`: major, minor, micro; nonnegative integers, excluding bool.
- `python_release_level: Literal["alpha", "beta", "candidate", "final"]`.

Plan `SUPPORTED_PYTHON_SERIES: tuple[tuple[int, int], ...] = ((3, 12), (3, 13), (3, 14))`, matching the inspected
`pyproject.toml` declarations. This is the scaffold's declared range, not a permanent ceiling on future releases.
The release owner updates metadata and validation together when support changes; tests detect drift. No runtime
network query, dependency negotiation, or automatic environment repair is introduced.

| Signature | Responsibility | Direct tests |
| --- | --- | --- |
| `validate_compatibility_metadata(value: CompatibilityMetadata) -> None` | Validate plain metadata shape and values without payload access. | Empty version, malformed/noninteger Python tuple, unknown release level; metadata has no executable payload field. |
| `require_compatible(root: CompatibilityMetadata, worker: CompatibilityMetadata) -> None` | Validate metadata, compare full Cocoonet strings exactly, and require both Python runtimes to be final releases within declared series. | Same complete version succeeds; patch/prerelease/local-suffix differences reject without normalization; every ordered pair of supported Python series succeeds; unsupported or nonfinal Python rejects. |

Use `ValueError` for incompatibility. A Cocoonet mismatch message includes both complete versions and suggests
installing the same Cocoonet version. Unsupported Python errors identify the unsupported runtime and supported series.
Check framework mismatch before Python compatibility to give the primary contract error consistently. Do not require
identical Python minor/micro versions, compare only major/minor Cocoonet versions, or equate versions through semantic
normalization. Cocoonet release identifiers themselves may include suffixes; exact equality still governs them.

Online consumers later run the guard before normal protocol operation, synchronization, assignment, or executable
loading. Local consumers use it without a network handshake. Metadata acquisition/transport, packaging/dill matrix
tests, and model dependency behavior remain later work; Phase 1 compatibility tests prove the guard, not serialization
interoperability or successful user-model execution.

### 8.2 Component input/output contracts

This table establishes logical boundaries only. It does not introduce executable abstract services or choose future
method signatures, IPC, or messages. Later component implementations call the appropriate concrete validators above.

| Owner / boundary | Consumes | Produces / obligation | Implementing phase |
| --- | --- | --- | --- |
| Root configuration | User inputs; later validated source ownership | `ComputationSnapshot`, fixed model/evaluation references; keep authoritative inputs behind immutable handles. | 3–4 |
| Dataset preparation | User source representation and split configuration or saved artifact | Validated `PartitionSchedule`, whose `PartitionReference` is the sole dataset/partition identity handle; exact identity/content verification and one authoritative durable schedule. | 2–4 |
| Root parent commitment | Complete validated configuration and durable partition reference | `ParentEvaluation` with fixed model references; each model defines one logical unit through the parent/model identity pair. Old snapshots remain fixed during prospective edits. | 3, 13 |
| Root coordination | Fixed parent, readiness, unit/attempt/reception observations | `AttemptInputs`, progress/selection/acceptance observations; allocation and duplicate policy remain separate from records. | 13–18 |
| Worker control/readiness | Applicable local/online parent context, owned worker | `WorkerObservation`, `AttemptProgress`; no model execution in the coordinator or control path. | 5, 13–17 |
| Model initialization/execution | Resolved fixed model input and framework-supplied sample data | Aligned standardized predictions or scoped failures; executable loading only inside enforced worker boundary. | 5–7, 10 |
| Validation/report components | Saved schedule, sample metadata, aligned predictions and valid prerequisites | Typed report payloads and `ReportOutcome`; preserve labels/classes, finite values, permitted infinities, `NaN`, `None`, and failures without redundant value statuses or independent file persistence. | 7–10 |
| Attempt workflow/packaging | Report outcomes and computational failure records for one attempt | `AttemptResult`; one complete file later, with a failure-only reconstruction outcome or a validation-failure envelope that preserves completed fold results. No report-specific delivery state. | 7–12 |
| Worker report persistence | Complete result envelope | Published worker artifact resolved by `ArtifactId`; preserve attempt identity. | 2, 12 |
| Root reception/persistence | Selected existing artifact and its unit/attempt identity | Durable selection/result evidence and acceptance/acknowledgement after complete persistence. No active-loop report loading. | 2, 13–18 |
| Delivery | Selected existing completed result, operation progress | Updated bounded-operation observations; resends never reconstruct/recompute results. | 14, 17 |
| Failure/log owner | Actual exception or detected failure plus precise context | `FailureRecord`, later structured log; no application error implies worker termination. | 2 and each producer |
| Compatibility boundary | Plain root/worker metadata | Compatibility decision before normal operation; no model/result payload loading. | 13, 16 |

No missing future component is filled with a fake implementation to complete Phase 1 tests.

## 9. Dependency-safe implementation units

**Contract design**

The shared contracts use direct source strings, parent-owned model references, direct attempt-result
routing fields, report identity supplied by its slot/caller, and a derived evaluation outcome.

PartitionReference binds a dataset to its realized partition. Report status/payload/failure records
distinguish report outcomes.
Trace context identifies failure provenance. Prediction failure associations use the position-aligned
representation specified in Section 6.1.

Before proposing each implementation unit, verify its producer, consumer, ownership, and required
invariants against this specification and perform the development reviews required by AGENTS.md.
Declarations, validators, consumers, tests, and component boundaries must agree on these contracts.

Implementation requires explicit unit approval. Preserve the root roadmap's phase boundaries,
independent operational states, and required persistence evidence.

Follow this order. Each named declaration is its own potential approval unit; each named validator is one production
function unit. A table row is an ordered inventory, **not** authorization to bundle its declarations or functions.
Implement declarations before the first validator that uses them, and finish its targeted tests/checks before the next
approved unit. Generated dataclass methods do not imply planned custom methods.

| Order | Declarations, in dependency order | Then implement these function units, in order | Direct test module |
| --- | --- | --- | --- |
| 1 | Scalar/target aliases; each ID alias; `ReportName` | `validate_identifier`; `validate_label` | `tests/contracts/test_identity.py` |
| 2 | `SourceReference` string alias; `SampleRecord`; `PartitionReference`; `FoldRecord`; `PartitionSchedule` | `validate_source_reference`; `validate_sample_record`; `validate_partition_schedule` | `tests/contracts/test_data.py` |
| 3 | `TaskKind`; `ModelFamily`; `ConsolidationState`; `ModelReference`; `EvaluationReference`; `ComputationSnapshot`; `ParentEvaluation`; `AttemptInputs` | `validate_model_reference`; `validate_evaluation_reference`; `validate_computation_snapshot`; `validate_parent_evaluation`; `validate_attempt_inputs` | `tests/contracts/test_execution.py` |
| 4 | `FailureScope`; `TraceContext`; `FailureRecord` | `validate_trace_context`; `validate_failure_record`; `validate_failure_links` | `tests/contracts/test_failure.py` |
| 5 | `RegressionPredictions`; `ClassificationPredictions`; `PredictionBatch` | `validate_regression_predictions`; `validate_classification_predictions` | `tests/contracts/test_prediction.py` |
| 6 | `ReportStatus`; `ReportOutcome`; `EvaluationOutcome`; `AttemptResult` | `validate_report_outcome`; `validate_attempt_result`; `derive_evaluation_outcome` | `tests/contracts/test_result.py` |
| 7 | Execution/attempt/stop enums; `ParentProgress`; `StopDisposition`; `AttemptProgress` | `validate_parent_progress`; `validate_stop_disposition`; `validate_attempt_progress` | `tests/contracts/test_state.py` |
| 8 | Worker/ownership/compatibility enums; `ParentReadiness`; `WorkerObservation` | `validate_worker_observation` | `tests/contracts/test_state.py` |
| 9 | Operation enums; `BoundedOperation` | `validate_bounded_operation` | `tests/contracts/test_state.py` |
| 10 | Receipt/persistence enums; `ReceptionRecord` | `validate_reception_record`; `validate_reception_set` | `tests/contracts/test_state.py` |
| 11 | `CompatibilityMetadata`; declared supported-series constant | `validate_compatibility_metadata`; `require_compatible` | `tests/contracts/test_compatibility.py` |
| 12 | No new production declaration/function | Verify agreement among schemas, validators, consumers, tests, and component boundaries; target-validation ownership; scalar preservation; bounded diagnostics; helper documentation; nonredundant composed validation; and complete-attempt reservation semantics. The universal validator and its tests are outside the development scope. Run applicable phase checks. | Targeted contract test modules. |

Place `ConsolidationState` in `execution.py` and `ReportName` in `identity.py`. No `ValueStatus` declaration is created.
`failure.py` has no import from prediction/result/state; `state.py` may consume execution/failure records. This prevents
cyclic dependencies while keeping the record definitions beside their responsibility.

Do not add an unplanned reusable helper silently. If a validator needs a substantial new helper or an approved
signature/data model must change, identify it and obtain the required approval before implementing the caller.
Small direct validation logic stays in its coherent validator; no generic validation framework is planned.

Declaration tests verify meaningful guarantees of the selected representation. Require keyword-only construction, frozen
fields, or tuple structure only where contractual. Preserve semantic validation and ownership tests; do not add tests
that merely repeat declarations.

## 10. Numerical behavior and validation requirements

Phase 1 implements **no scientific formula**, split randomness, CPU-capacity calculation, model invocation, or metric
aggregation. Architecture Sections 4.5–4.7 and 9.3 remain authoritative for later scientific phases. In particular,
the approved influence denominator, classwise/micro/macro accuracy distinctions, within-sample classification
residual axis, and formula-defined NaN behavior are not reopened here.

Numerical checks cover integer count bounds, standardized scalar admissibility, finite dataset
targets and timestamps, aligned dimensions, probability bounds, and value/failure consistency.
Present standardized predictions are finite. Use checks appropriate to the supplied scalar type;
do not narrow precision merely to test finiteness or identify NaN/infinity.

Formula-defined NaN and permitted infinity in report payloads retain their documented meanings.
The operation-count constraint is 0 <= attempts_made <= attempt_limit, with attempt_limit >= 1;
it is not a retry scheduling algorithm.

For each approved implementation unit:

- Run its directly corresponding pytest cases, including negative cases that could otherwise corrupt identity,
  hide a failure, or accept an incomplete outcome. Do not create unrelated test suites.
- Use typed fixtures matching the selected representations and test applicable ownership and aliasing guarantees. Keep
  sample IDs, sample labels, and class labels distinct, with nontrivial row/class ordering to expose alignment errors.
- Classification success fixtures use string targets from the evaluation's class vocabulary.
  Regression success fixtures use finite floating-point targets. Negative cases cover wrong
  task-target combinations, unknown class targets, nonfinite targets, and failure-only results
  with invalid target context.
- Contract tests cover supported Python/NumPy scalar families without conversion, including
  precision preservation and applicable string subclasses. Numeric and boolean standardized
  labels are rejected.
- Contract diagnostic assertions check the required field/context, expected condition, observed
  type/value, and bounded output. Exact complete wording is required only where contractual.
- Sample/class alignment diagnostics report lengths and a mismatch position or bounded preview,
  rather than complete sequences. Tests include long sequences and ordering mismatches.
- Test validators' lack of mutation and absence of operational side effects. Source-reference tests do not need real
  datasets; compatibility tests never deserialize a payload; state tests do not launch workers or run retry loops.
- Run mypy, Black `--check`, and Ruff on the applicable source/tests under `pyproject.toml`. Preserve its selected
  families, targets, exclusions, and narrow external-typing suppression policy. Do not silently add blanket ignores.
- Every class/function/method, including private helpers, has a NumPy-style source docstring.
  Document responsibility, applicable parameters and returns, validated-input preconditions,
  intentionally exposed exceptions, and any mutation of caller-owned accumulators. Distinguish
  local validation bookkeeping from mutation of scientific inputs or authoritative records.
  Explain non-obvious invariants and functional blocks in comments. Do not create helpers merely
  to satisfy documentation requirements or test exact docstring wording.
- At phase close, run the completed Phase 1 tests and configured project checks. Validate supported Python series
  against declared metadata; report which runtimes were actually exercised. Do not claim the future cross-version
  dill/model/result integration matrix was tested by plain record tests.
- Documentation rebuilds, if later authorized, require accessibility/version-switching checks. The inspected
  `pyproject.toml` declares documentation dependencies but no explicit build/accessibility/version-switching commands;
  do not claim configured commands ran or invent a documentation toolchain in Phase 1. This plan requires no rebuild.

The planning document itself is validated by scope review, relative-link checks, and a Git diff/status inspection.
No production tests are added or run merely to validate a documentation-only plan.

## 11. Resolution log, inconsistencies, and deferred decisions

### 11.1 Required Phase 1 decisions settled by this proposal

| Question | Resolution and source boundary |
| --- | --- |
| Are computation, parent, unit, and attempt the same identity? | Separate configuration snapshot, fixed parent, logical unit identified by `(parent_id, model_id)`, and attempt reference, following architecture 2.3/3.4. |
| Must Phase 1 build the public computation or lifecycle engine? | No. Phase 1 establishes records; Phase 3/13–18 own configuration and execution algorithms. |
| How can inputs be fixed before detailed settings/model schemas exist? | Immutable typed references to snapshots owned by the later configuration producer; no mutable/executable objects embedded or independently overrideable attempt fields. |
| How much dataset/report schema belongs here? | Sample/source/membership and prediction alignment plus generic report/envelope structure. Feature adapters and concrete metric payloads remain Phase 4/8–10. |
| Which fields/types are specified? | Sections 3–6 define scalar families, direct source strings, parent model references, direct result routing, report-slot identity, derived outcomes, and position-aligned prediction failure associations. Section 9 defines implementation units; Section 13 defines validation responsibilities. |
| Does undefined science mean an operation failed? | No. Formula-defined `NaN` has no failure solely for being undefined; computationally unavailable `None` has applicable failure references; schema-defined non-applicable `None` has none. No per-value status enum is needed. |
| Does a failure-only result finish delivery or terminate a worker? | No. It is a complete scientific outcome, independently selected/persisted/accepted later. |
| What does a retry count mean? | Total begun attempts including the first, bounded by a supplied total limit; no policy defaults or timers chosen. |
| Are frozen records proof of durable acceptance or isolation? | No. They describe evidence/references; storage, ownership, concurrency, and enforced execution boundaries remain later obligations. |
| Does Python mismatch imply framework incompatibility? | Not within the declared supported stable series; complete Cocoonet strings must match exactly. |

These contracts preserve the established scientific and lifecycle requirements. Concrete report
payloads, storage mechanisms, model adapters, and runtime algorithms belong to their assigned phases.
Plan approval and implementation-unit approval are separate requirements.

### 11.2 Inconsistencies found

1. **Roadmap location prose and links:** the actual file is `docs/roadmap.md`, consistent with root `AGENTS.md` and
   architecture references. Its final paragraph says it is intentionally at the repository root and calls references
   to `docs/roadmap.md` a mismatch. Its opening relative links, `AGENTS.md` and `docs/architecture.md`, also assume a
   root location. Record this as stale location prose/links, not a reason to move the file or change the phase scope.
   This plan uses correct links from `docs/roadmaps/`; the governing file remains untouched as requested.
2. **Documentation-check configuration gap:** root guidance refers to configured documentation build/accessibility/
   version-switching checks, but the inspected `pyproject.toml` has dependencies and quality settings without those
   explicit commands. This is not a contradiction in Phase 1's scientific or record contracts and does not block this
   no-rebuild plan. Resolve the actual commands before an authorized documentation rebuild.

No conflicting behavioral requirement was found among root `AGENTS.md`, the architecture, and the Phase 1 scope.
Their broad deferred-mechanism wording is compatible with choosing the minimal internal contracts here: Phase 1
specifies observations and associations; later phases implement their owners and behavior.

### 11.3 Explicit later-phase deferrals

| Decision | Earliest owning phase / boundary |
| --- | --- |
| Filesystem equality/containment, symlinks, overwrite safety, artifact layout, dill loading/publication, durable selection evidence and logs | Phase 2, before any file-changing consumer. |
| Public computation API, configuration containers/edits, settings payload schema, fixed-input ownership implementation, copy/exclusion mechanics, bulk commit behavior | Phase 3; recovery-specific mechanics Phase 15. |
| Dataset identity generation/content validation, sample-ID restoration, raw-input parsing and standardization, floating-point representation selection, mixed inline/file support, tensor shape encoding, splitter APIs and strategy coverage/empty-population decisions | Phase 4, before dataset consumers. Standardized scalar admissibility and contextual target compatibility are Phase 1 responsibilities. |
| Isolation enforcement and model/control execution boundary | Phase 5, before executable loading or user code. |
| Concrete model input and prediction adapters, class conversion, raw-output validation and numerical checks, fit-state reset | Phases 6 and 10. |
| Concrete fold/validation report payloads and baseline workflow | Phase 7. |
| Conventional metric/report conventions, concrete residual/visualization payloads and any actual unresolved custom formula | Phase 8; do not reopen established standard formulas or expected undefined values as ambiguities. |
| Concrete influence payload, exact comparison count, value/failure-reference representation, and removal workflow under architecture 4.7 | Phase 9; formula, populations, failure traceability, successful-result preservation, and absence of a separate evaluation-status field are already fixed. |
| PyTorch reconstruction/training/device failure details and process-pool/CPU-detection mechanics | Phases 10–11, preserving architecture 3.3/8. |
| Result serialization/publication and envelope payload loading restrictions | Phase 12 using Phase 2; no extra authoritative partition copy. |
| Local IPC, readiness verification, start/assignment/acceptance algorithms, process lifetime | Phase 13, with interruption/reconciliation in Phase 15. |
| Retry counts/delays, delivery and retention algorithms | Phase 14; separate connection-round timers in Phase 16. |
| Wire encodings, authentication, heartbeat/reconnect/probes, online transfer verification | Phases 16–17. |
| Duplicate arbitration/tie handling, concurrency enforcement, mixed-target integration | Phase 18 using existing unit/attempt/selection contracts. |
| Cleanup selection, deletion verification, public-service API and lifecycle policies | Phases 19–20; the public interface must be supplied externally. |
| Full supported-runtime serialization/model/result compatibility and lifecycle integration matrix | Incrementally with producers/consumers, completed in Phase 21. |

These deferrals do not weaken fixed parent inputs, shared saved partitions, source ownership, failure isolation,
topology-neutral acceptance, or the prohibition on resetting exhausted delivery budgets. Any later missing custom
definition or real architectural contradiction must be resolved before its dependent implementation, under the same
documentation and approval gates.

## 12. Review checkpoint

Implementation order: identity primitives → sample/source/schedule records → configuration/parent/attempt references
→ failures → predictions → report/result envelopes → execution/worker/operation/reception observations → compatibility
guard → contract coverage and quality checks.

Phase 1 completion requires conformance to the specified schemas, validation responsibilities,
component boundaries, documentation requirements, and tests. Unresolved requirements block their
dependent implementation units.

Plan approval does not authorize source implementation. Each unit requires explicit approval;
commits remain manual.

## 13. Runtime validation and diagnostics

The universal type validator in src/cocoonet/_type_validation.py and its tests in
tests/test_type_validation.py are established infrastructure outside Phase 1 contract-development
scope. Their implementation, interfaces, supported annotations, diagnostics, and dependencies
are not development tasks in this phase.

Its NumPy, pandas, and PyTorch imports support scientific-type validation. Python's module cache
serves normal repeated imports within a process; separate processes may initialize their own
dependencies. This overhead is acceptable under AGENTS.md and is not evidence of over-engineering.
Import-load reduction is not a Phase 1 completion requirement.

Contract validators own schema-specific value and association checks. They may use the universal
validator for applicable declared-type checks. Scientific scalar-family acceptance and numerical
conditions are checked explicitly where exact built-in type checks would be too restrictive.
Validation preserves scalar representations and boolean-versus-integer distinctions where required.

Each independent consuming boundary validates its complete inputs. Within one composed validation
path, assign each structural check to one owner: containing validators check their own fields and
cross-record relationships, while child validators check their records. Avoid whole-tree type
validation followed by another traversal performing the same structural checks.

Shared parent/schedule context is validated once within a composed boundary call. Internal helpers
may rely on documented validated inputs where ownership prevents intervening invalidation.
Do not introduce validation caches, persistent validity flags, skip-validation switches, or another
general annotation-validation framework.

Contract diagnostics follow Section 3.2, including bounded observed values and preservation of the
established mismatch when preview generation fails. Scientific dtype, shape, content, and alignment
checks belong to the components accepting those structures. Any new shared helper requires its
own explicit design and approval.
