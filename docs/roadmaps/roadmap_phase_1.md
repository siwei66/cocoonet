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

1. Inspect the applicable code and tests, and identify the next approved declaration, function, or method.
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

The repository is a release scaffold: `src/cocoonet/__init__.py` exposes `get_version()`; `pyproject.toml` provides
packaging and quality configuration. No existing domain implementation or tests were found in the inspected scaffold.
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
- A complete failure-only envelope and a complete envelope with partial report failures can both be represented;
  neither implies delivery, acceptance, or worker termination.
- Compatibility is decided using plain metadata; supported differing Python versions do not cause rejection.
- Component obligations are covered by record-level tests without fake workers, sockets, splitters, model execution,
  files, or simulated lifecycle engines.
- Applicable pytest, mypy, Black, and Ruff checks pass under the existing configuration. Unavailable checks are
  explicitly reported, not treated as passing.

Completion does not establish a usable evaluation workflow or permit beginning Phase 2 without its own planning and
implementation approvals.

## 3. Representation and validation conventions

### 3.1 Internal surface and file boundaries

Plan a private `src/cocoonet/_contracts/` package with these cohesive modules:

| Module | Responsibility |
| --- | --- |
| `identity.py` | Typed identifiers, unit key, shared scalar aliases, identity validation. |
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

- Use frozen, keyword-only, slotted dataclasses for records; use `StrEnum` for closed status vocabularies. Dataclasses
  contain no custom constructor, mutation method, transition method, or automatic filesystem action.
- Use tuples for ordered collections. Do not put mutable lists, dicts, arrays, arbitrary model objects, or callables
  into Phase 1 concrete records. Tuple nesting is recursive; freezing only the outer dataclass is insufficient.
- Use `NewType` over `str` for opaque identifiers. Their spelling is not a UUID, digest, filename, path, or ordering
  contract. Generation and storage/wire encoding remain deferred. Runtime validation requires nonempty strings
  without silently stripping, normalizing, or converting them. Separate types are a static safeguard; association
  validators enforce identity scope at runtime.
- `ScalarLabel = str | int | float | bool` is the normalized internal sample/model/group/class label vocabulary.
  Labels are not coerced to strings. Float labels must be finite; equality uses Python scalar equality, so `1`, `1.0`,
  and `True` cannot distinguish entries in the same unique-label population. Empty string labels are allowed; missing
  labels and non-scalar/custom objects are not internal label values. User-input conversion policy belongs to its
  later adapter, which must preserve meaning or reject unsupported input, not silently relabel it.
- `TargetValue = str | int | float | bool` describes a single target, not a multi-target or multilabel vector.
  Dataset input admissibility belongs to Phase 4. This alias does not authorize imputing missing/nonfinite targets.
- Counts are nonnegative integers, excluding `bool`; eligibility flags are integers exactly `0` or `1`, excluding
  `bool` in the normalized record. Adapters may later accept other input forms but must output this representation.
- `validate_*` functions return `None` and never mutate their arguments. They raise `TypeError` for incorrect runtime
  types and `ValueError` for invalid values, associations, or shapes. Error text names the field and relevant identity.
  Use these built-in exceptions rather than introducing a general exception hierarchy.
- Construction gives a typed candidate record, not a certificate of semantic validity. The producing component calls
  its validator before publishing it; receiving components validate at their boundary. Tests exercise validators
  directly. No hidden validation in `__post_init__` is planned, and no invalid candidate may enter a consuming workflow.
- Validators perform only the checks defined here. They do not execute or inspect user model behavior, determine
  scientific equivalence, inspect source contents, or claim to establish durable persistence or isolation.

The scalar label vocabulary and immutable containers are proposed internal representation choices, not new public
registration signatures. Approval settles them before later adapters depend on them.

### 3.3 Identifier inventory

Declare `DatasetId`, `SampleId`, `ModelId`, `ParentId`, `AttemptId`, `FoldId`, `WorkerId`, `PartitionId`,
`EvaluationConfigId`, `FailureId`, `OperationId`, and `ArtifactId` as distinct opaque string aliases.

| Identity | Scope and invariant |
| --- | --- |
| `DatasetId` | One exact representation, including sample associations, features, targets, and metadata. Changed representation needs a new identity. |
| `SampleId` | Unique only within its dataset; generated by Cocoonet, paired one-to-one with the supplied sample label. |
| `ModelId` | Cocoonet-assigned reference to the fixed supplied executable input. It is not a model-description hash. |
| `ParentId` | One fixed evaluation configuration and coherent result set. Copy/resume does not generate a replacement. |
| `UnitKey(parent_id: ParentId, model_id: ModelId)` | Logical unit identity; exactly one model in a parent. No independent dataset key in the unit. |
| `AttemptId` | One execution of a unit; duplicate attempts share `UnitKey` and fixed inputs. |
| `FoldId` | Unique within one realized partition schedule; tuple order is the authoritative schedule order. |
| `PartitionId` | Opaque handle for a complete realized schedule, not its filename or an additional authoritative copy. |
| `EvaluationConfigId` | Opaque handle for a fixed evaluation-settings snapshot; not a claim of content equivalence. |
| `WorkerId` | Execution target reference, independent of a parent and external public-pool identity. |
| `FailureId`, `OperationId` | Unique within the owning recorded execution/operation context; prevent ambiguous references. |
| `ArtifactId` | Logical handle resolved by the owning storage component; contains no Phase 1 path/publication mechanism. |

Do not introduce a computation ID simply to duplicate parent identity. A configuration snapshot may exist before any
parent. Root/user authentication identity and pool allocation IDs are not designed here.

Planned leaf functions:

| Signature | Responsibility | Direct tests |
| --- | --- | --- |
| `validate_identifier(value: str, *, field: str) -> None` | Enforce opaque nonempty string representation; no encoding inference. | Empty/non-string rejection; preserve spelling; unrelated ID types may have equal text without sharing scope. |
| `validate_label(value: ScalarLabel, *, field: str) -> None` | Enforce scalar label domain and finite float labels. | Each scalar type; empty string; NaN/infinity/custom object rejection; no conversion. |
| `validate_unit_key(value: UnitKey) -> None` | Validate the two required identity fields only. | Missing/invalid parent/model identities; no dataset, label, fold, or attempt in the key. |

## 4. Data, configuration, and fixed execution records

### 4.1 Dataset and partition records

All fields in the following tables are required unless an explicit default is shown. `tuple[T, ...]` preserves order.

| Record | Fields |
| --- | --- |
| `DatasetReference` | `dataset_id: DatasetId` |
| `SourceReference` | `relative_path: str` |
| `SampleRecord` | `sample_id: SampleId`, `sample_label: ScalarLabel`, `group: ScalarLabel`, `train_mask: int`, `test_mask: int`, `target: TargetValue`, `sources: tuple[SourceReference, ...]` |
| `PartitionReference` | `partition_id: PartitionId`, `dataset_id: DatasetId` |
| `FoldRecord` | `fold_id: FoldId`, `training_sample_ids: tuple[SampleId, ...]`, `testing_sample_ids: tuple[SampleId, ...]` |
| `PartitionSchedule` | `reference: PartitionReference`, `samples: tuple[SampleRecord, ...]`, `folds: tuple[FoldRecord, ...]`, `group_separated: bool` |

`SampleRecord` is common population metadata; it is neither an input dataframe nor a feature-data container. Features,
tensor layouts, and file readers remain in Phase 4. Both training and testing refer to these same records; predictions
are separate associated results. At least one source reference is required per sample, including inline samples that
originate in source dataset files. Several samples may share a file and one sample may reference several files.

Choose a platform-independent logical source syntax here: nonempty `/`-separated relative paths, with no empty, `.`
or `..` segment, leading slash, backslash, or drive prefix. This is an internal logical reference, not an OS path;
Phase 4 maps actual source paths to it. Reject invalid syntax rather than repairing it. This lexical check does not
resolve symlinks, establish containment, access a file, or replace Phase 2 ownership validation. Preserve spelling and
case; physical encoding and local storage layouts remain deferred.

Schedule invariants:

- Samples and folds are nonempty collections, with unique sample IDs, sample labels, and fold IDs in their scopes.
  Groups may repeat. IDs and labels stay paired; reordering a sample tuple does not change their association.
- Each fold references known sample IDs without repetition within either population; train/test IDs are disjoint.
- Training members have `train_mask == 1`; testing members have `test_mask == 1`.
- If `group_separated` is true, no group appears in both populations of one fold.
- Source-file overlap is allowed even when samples/groups must be disjoint. Source references within one sample are
  unique; this does not forbid reuse across samples or folds.
- Empty fold populations are structurally representable. Phase 1 does not invent an empty-population split policy or
  reject masked populations merely to manufacture a computable metric. Phase 4 defines strategy-level admissibility;
  later execution/report components apply their actual input requirements and defined failure/NaN rules.
- The structural validator does not infer a splitting method, regenerate a split, require all samples to be tested
  for every strategy, or reject repeat appearances across folds. Strategy-specific coverage belongs to Phase 4.
- `group_separated` records a schedule obligation; it is not a request to reassign group memberships.
- This is the complete logical schedule. Parent records hold a reference to it, not an embedded second authoritative
  storage copy. In-memory/worker inputs may contain it; durable single-file storage belongs to Phase 2/4.

### 4.2 Configuration and execution records

| Record / enum | Fields / values |
| --- | --- |
| `TaskKind` | `regression`, `classification` |
| `ModelFamily` | `sklearn`, `sklearn_like`, `pytorch` |
| `ModelReference` | `model_id: ModelId`, `model_label: ScalarLabel`, `family: ModelFamily`, `task_kind: TaskKind` |
| `EvaluationReference` | `configuration_id: EvaluationConfigId`, `task_kind: TaskKind`, `class_labels: tuple[ScalarLabel, ...]`, `influence_requested: bool` |
| `DatasetBinding` | `dataset: DatasetReference`, `partition: PartitionReference` |
| `ComputationSnapshot` | `root_dir: str`, `consolidation: ConsolidationState`, `dataset: DatasetBinding | None`, `evaluation: EvaluationReference | None`, `models: tuple[ModelReference, ...]`, `worker_ids: tuple[WorkerId, ...]`, `parent_id: ParentId | None` |
| `UnitTask` | `key: UnitKey`, `model: ModelReference` |
| `ParentEvaluation` | `parent_id: ParentId`, `dataset: DatasetBinding`, `evaluation: EvaluationReference`, `units: tuple[UnitTask, ...]` |
| `AttemptInputs` | `attempt_id: AttemptId`, `unit: UnitKey`, `worker_id: WorkerId` |

`ConsolidationState` has exactly `initialized`, `consolidated`, and `unconsolidated`.

`root_dir` is the user-selected location recorded as a nonempty string; no normalization or filesystem validation
happens in Phase 1. The configuration component later enforces ownership before initializing/reassigning it.

`ComputationSnapshot` is a read-only description, not the mutable public computation. It intentionally permits no
dataset, no settings, no models, and no workers in initialized state. `parent_id` is absent for initialized state and
required for consolidated/unconsolidated state; in unconsolidated state it identifies the previous execution, not a
new parent for the prospective configuration. Models/worker references are unique, including model labels within
the prospective configuration. An incomplete prospective configuration after a permitted removal is representable.
The validator does not compare changed configuration with an old parent or perform a state transition.

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

- A parent has one dataset/split binding and one evaluation reference, and at least one unit. Dataset IDs in the
  binding agree. Unit keys name that parent and match their model reference; model IDs and labels are unique there.
- The referenced configuration, dataset/split, and model inputs remain fixed throughout every attempt's lifetime.
  A worker-only configuration edit does not redefine them or create a parent.
- `AttemptInputs` resolves its model and all scientific inputs through its unit's parent. It has no independent
  dataset, settings, split, or model override. Duplicate attempts differ in attempt identity and potentially worker,
  while resolving the same scientific inputs.
- `ParentEvaluation` contains the known unit set. Attempts are separate records; a parent does not own worker lifetime.
- Copying a configuration does not create identities, runtime resources, or artifacts. Resumption and prospective
  edits follow architecture Section 2.3; transition/copy algorithms are Phase 3/15, not methods on these records.

### 4.3 Planned data/execution validators and tests

| Signature | Responsibility | Direct tests |
| --- | --- | --- |
| `validate_source_reference(value: SourceReference) -> None` | Validate logical relative syntax only. | Valid nested reference; empty/absolute/drive/backslash/dot/traversal forms rejected; no filesystem access. |
| `validate_sample_record(value: SampleRecord) -> None` | Validate identity/label/group, scalar target type, normalized masks, nonempty unique sources. | Both mask values, bad masks/types, several files, shared files across samples, missing source, no target repair. |
| `validate_partition_schedule(value: PartitionSchedule) -> None` | Validate complete logical associations and per-fold eligibility/disjointness/group rules. | Duplicate ID/label/fold; unknown ID; forbidden membership; group leakage; legal file overlap; reordered metadata; empty populations; repeated cross-fold use. |
| `validate_model_reference(value: ModelReference) -> None` | Check identity/label and declared family/task-kind tags. | Valid families/kinds; invalid tags; no model object/API inspection or derived descriptions. |
| `validate_evaluation_reference(value: EvaluationReference) -> None` | Check settings handle, task kind, class order, influence flag. | Regression/classification class rules; equality collisions; one class allowed; no sorting/coercion. |
| `validate_dataset_binding(value: DatasetBinding) -> None` | Check reference identities and matching dataset association. | Matching IDs; mismatched partition dataset; invalid identity. No file or content verification. |
| `validate_computation_snapshot(value: ComputationSnapshot) -> None` | Check structural consolidation/parent relationship, references, uniqueness; allow incomplete configuration. | All three states; zero models/workers/dataset; missing required parent; unchanged parent after worker-list edit; snapshot unchanged by validation. |
| `validate_parent_evaluation(value: ParentEvaluation) -> None` | Check fixed parent structure, known unit set, labels, model/task-kind agreement. | Empty units; mixed task kinds; duplicate IDs/labels; wrong unit parent/model; same label in different parents allowed. |
| `validate_attempt_inputs(value: AttemptInputs, *, parent: ParentEvaluation) -> None` | Resolve the exact unit/model context and validate attempt/worker identities. | Unknown/cross-parent unit; duplicates with different attempts/workers; no independent scientific override. |

Collection association checks use ID-indexed lookup and sets where possible; do not introduce quadratic pairwise
searches or caches. Runtime label comparisons follow Section 3.2. No identity generator is implemented in Phase 1.

## 5. Traceable failures and value semantics

### 5.1 Trace and failure records

| Record / enum | Fields / values |
| --- | --- |
| `FailureScope` | `configuration`, `startup`, `unit_evaluation`, `shared_intermediate`, `report`, `measure`, `influence_sample`, `operation`, `worker_process` |
| `TraceContext` | `stage: str`, `parent_id: ParentId | None = None`, `model_id: ModelId | None = None`, `attempt_id: AttemptId | None = None`, `worker_id: WorkerId | None = None`, `fold_id: FoldId | None = None`, `sample_id: SampleId | None = None`, `class_label: ScalarLabel | None = None`, `report_name: ReportName | None = None`, `operation_id: OperationId | None = None` |
| `FailureRecord` | `failure_id: FailureId`, `scope: FailureScope`, `context: TraceContext`, `exception_type: str`, `message: str`, `traceback: str`, `origin_failure_id: FailureId | None = None` |

`ReportName` is the shared enum `validation`, `performance`, `residual`, `visualization`, `influence`; place it in
`identity.py` to avoid a failure/result import cycle. Stage names are descriptive nonempty strings rather than a
premature exhaustive enum of future algorithms. Reconstruction and baseline fitting use `unit_evaluation` scope and
distinct stages such as `reconstruction` and `baseline_fit`.

- Scope describes the affected object, not the whole process. Startup may have no model/parent; it is not baseline
  failure. A worker-process failure identifies a worker and is not inferred from any other error.
- Model context requires parent context; attempt context requires parent/model/worker context; fold/sample context
  requires parent context. Class-specific context requires a class label. `None` means not applicable/known, not an
  anonymous replacement ID. A known class or sample must never be dropped from an error record.
- Report/measure failure requires `report_name`; influence-sample failure requires model, attempt, fold, sample, and
  influence report context. Operation failure requires an operation ID. Evaluation failures in attempt results carry
  that attempt's parent/model/worker context.
- Exception type and message are nonempty. Preserve the full supplied traceback as a string. It may be empty only
  for a failure detected without a caught exception; never replace a captured traceback with an empty value.
- A dependent skip refers to its originating failure by ID. An origin must resolve in the accompanying failure set;
  origin links cannot cycle or cross an unrelated attempt. Parent/model/attempt/worker identities must agree wherever
  both records supply them; stage and report name may differ, and a shared error need not have the dependent result's
  narrower sample/class coordinates. No duplicate full errors are required merely to skip work.
- Capturing exceptions, assigning stages, writing logs, retrying, and deciding where to catch errors are producer
  responsibilities in later phases. These records support those boundaries without implementing them.

### 5.2 Defined, undefined, and unavailable values

`ValueStatus` has `defined`, `undefined`, and `unavailable`.

- `defined`: an actual computed value, unchanged. A float marked defined cannot be NaN.
- `undefined`: the computation's formula has no defined value. Represent it as float NaN; no exception/failure record
  is required solely because the formula is undefined.
- `unavailable`: an operation or prerequisite failure prevented a value. Numeric slots contain NaN and reference the
  relevant failure. A missing predicted class label uses `None`, with status/failure, rather than inventing a class.
- Scientific `None`, such as macro-averaged confusion counts, is distinct from numeric NaN and missing predictions.
  Concrete report schemas define those fields in Phase 8; Phase 1 does not flatten them into a generic metric cell.
- Do not coerce NaN to zero, clip extreme values, fill missing probabilities, normalize model outputs, or aggregate
  while skipping NaN. No arithmetic metric routine is implemented here.

| Signature | Responsibility | Direct tests |
| --- | --- | --- |
| `validate_trace_context(value: TraceContext) -> None` | Validate applicable identity/label fields and context dependencies. | Startup without model; complete attempt/fold/sample context; orphan model/attempt/sample; invalid stage. |
| `validate_failure_record(value: FailureRecord) -> None` | Check scope-specific context, error text, and non-self origin. | Each scope; operation without operation ID; report without name; influence without sample; worker failure without worker; captured text preserved. |
| `validate_failure_links(values: tuple[FailureRecord, ...]) -> None` | Check unique IDs, resolved origins, and acyclic same-context dependency references. | Dangling/self/cyclic origins; valid shared origin; unrelated attempt origin; duplicate failure ID. |

## 6. Standardized predictions and complete result contracts

### 6.1 Predictions

Predictions are normalized component outputs, not raw model return values. Fields are immutable Python tuples;
adapters that use NumPy or PyTorch convert into this logical boundary in their own phases. Phase 1 adds neither
library nor adapter. One prediction record belongs to one fold and has the same row order as that fold's saved
testing IDs. Class columns use the parent evaluation's explicit order.

| Record | Fields |
| --- | --- |
| `RegressionPredictions` | `fold_id: FoldId`, `sample_ids: tuple[SampleId, ...]`, `values: tuple[float, ...]`, `statuses: tuple[ValueStatus, ...]`, `failure_ids: tuple[tuple[FailureId, ...], ...]` |
| `ClassificationPredictions` | `fold_id: FoldId`, `sample_ids: tuple[SampleId, ...]`, `class_labels: tuple[ScalarLabel, ...]`, `predicted_labels: tuple[ScalarLabel | None, ...]`, `label_statuses: tuple[ValueStatus, ...]`, `label_failure_ids: tuple[tuple[FailureId, ...], ...]`, `probabilities: tuple[tuple[float, ...], ...]`, `probability_statuses: tuple[tuple[ValueStatus, ...], ...]`, `probability_failure_ids: tuple[tuple[tuple[FailureId, ...], ...], ...]` |

`PredictionBatch = RegressionPredictions | ClassificationPredictions` is a type alias, not another wrapper class.
Per-class status allows an independently computable probability to survive a missing class or failed component.

Prediction invariants:

- For a fold with `n` saved testing IDs and `k` parent classes, regression vectors have length `n`; classification
  label vectors have length `n` and probability/status/failure matrices have shape `(n, k)`.
- IDs exactly match the saved testing sequence, not merely its length or set. Adapters explicitly align their outputs;
  the validator rejects misalignment rather than silently reordering. Class labels exactly match parent class order.
- True targets, labels, groups, masks, and source references resolve through `SampleRecord`; do not duplicate a
  competing truth vector or different train/test sample schema inside predictions.
- Defined predicted class labels belong to the parent class vocabulary; label status cannot be `undefined`.
  Unavailable labels are `None` and have at least one failure ID.
- Numeric undefined/unavailable entries are NaN. Defined probabilities are finite and in `[0, 1]`; defined regression
  values are floats other than NaN. Undefined entries carry no operational failure merely for being undefined;
  unavailable entries require a failure reference. Defined entries have no failure reference.
- Probability normalization and raw-model numerical admissibility belong to model adapters. This structural layer
  does not introduce a probability-sum tolerance, renormalization, clipping, or reconstructed missing probabilities.
- Empty testing sequences yield empty aligned rows. There is no fabricated prediction for an excluded sample.
  Predictions on training/removal-comparison populations used internally by future algorithms are not automatically
  validation records.

### 6.2 Report and result envelopes

| Record / enum | Fields / values |
| --- | --- |
| `ReportStatus` | `complete`, `partial`, `failed`, `blocked`, `not_requested` |
| `ReportOutcome[T]` | `name: ReportName`, `status: ReportStatus`, `payload: T | None`, `failure_ids: tuple[FailureId, ...]` |
| `EvaluationOutcome` | `completed`, `completed_with_errors`, `baseline_failed` |
| `AttemptResult[V, P, R, Z, I]` | `inputs: AttemptInputs`, `outcome: EvaluationOutcome`, `predictions: tuple[PredictionBatch, ...]`, `validation: ReportOutcome[V] | None`, `performance: ReportOutcome[P] | None`, `residual: ReportOutcome[R] | None`, `visualization: ReportOutcome[Z] | None`, `influence: ReportOutcome[I] | None`, `failures: tuple[FailureRecord, ...]` |

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

- `complete`: payload present, no operational failure references; formula-defined NaN values may still exist inside it.
- `partial`: payload present with preserved successful/identified unavailable entries and nonempty failure references.
- `failed`: nonempty failures; retain the identified result structure if available, otherwise payload may be absent.
- `blocked`: nonempty references to originating prerequisite failures; retain an unavailable result structure if
  available. Do not generate inputs or retrain to make a payload.
- `not_requested`: no payload or failures; allowed only for unrequested optional influence in the attempt envelope.
- Result-level validation resolves report/prediction failure references against the envelope's failure set. Detailed
  report field/status consistency is checked by its later concrete payload validator.

Complete attempt rules:

- `baseline_failed` covers essential reconstruction or any required baseline-fit failure. All report fields are
  `None`, predictions are empty, and there is a `unit_evaluation` failure with stage `reconstruction` or `baseline_fit`.
  This is a complete failure-only outcome eligible for the same later reception/acceptance process as success.
- For either completed outcome, the four required report fields exist with matching names; influence exists and
  matches `influence_requested` (`not_requested` when false, a requested status otherwise). A failed requested report
  remains an outcome record. Predictions contain one batch for each schedule fold in schedule order, including
  unavailable entries when prediction generation failed.
- `completed` has no failures and no partial/failed/blocked report or unavailable prediction. Scientific undefined
  values alone do not turn it into `completed_with_errors`.
- `completed_with_errors` has at least one applicable computational failure and at least one affected report or
  prediction. Required baseline failure cannot be downgraded into this category.
- The supplied schedule's partition and dataset IDs must exactly match the parent's binding. Failure-only outcomes
  obey this association too. The prediction subtype must match the parent's task kind.
- Failures and all fold/sample/class contexts resolve against the same parent, attempt, schedule, and class vocabulary.
  A sample-specific influence error names a training member of its stated fold; prediction errors naming a sample
  refer to that fold's testing population. A shared fold-level error need not invent a sample/class identity.
  The envelope's failure list describes computation, not later transfer, persistence, configuration, or worker-process
  incidents. Those operations retain their own records.
- Interruption/cancellation is not a fabricated baseline failure or complete result. A stopped attempt is represented
  by Section 7's state; this phase does not define a new cancelled-result file.
- One envelope describes one attempt; result files, dill bytes, completion publication, report loading, and user-facing
  rendering are later component responsibilities. At most one accepted result per unit is a separate root invariant.

### 6.3 Planned prediction/result validators and tests

| Signature | Responsibility | Direct tests |
| --- | --- | --- |
| `validate_regression_predictions(value: RegressionPredictions, *, fold: FoldRecord) -> None` | Validate aligned vectors, numeric statuses, failure-reference shape, and fold identity. | Reordered/unknown/excluded IDs; length mismatch; NaN undefined vs unavailable; defined NaN rejected; empty test set; no input changes. |
| `validate_classification_predictions(value: ClassificationPredictions, *, fold: FoldRecord, class_labels: tuple[ScalarLabel, ...]) -> None` | Validate class/sample alignment, matrix shapes, label membership, numeric/status/failure combinations. | Swapped classes; ragged matrices; missing/unknown class labels; independently unavailable class cells; bad probability range; no probability filling/reordering. |
| `validate_report_outcome[T](value: ReportOutcome[T]) -> None` | Validate generic outcome name/status/payload/failure relationship only. | Every status; complete immutable payload containing NaN; failed/blocked with preserved structure; empty failure references rejected where required. |
| `validate_attempt_result[V, P, R, Z, I](value: AttemptResult[V, P, R, Z, I], *, parent: ParentEvaluation, schedule: PartitionSchedule) -> None` | Compose prior validators; validate all contextual links and complete/failure-only result rules. | Regression/classification completion; baseline failure without reports; reject leaked reports; optional influence; partial prediction/report failures; undefined-only completion; wrong attempt/fold/class; dangling errors; ordinary operation failure excluded. |

The type-parameter notation above uses Python 3.12 generic syntax. Declaration and validator units remain separate;
the notation does not authorize bundling helpers with their caller.

## 7. Orthogonal state and bounded-operation records

These are immutable observations owned by the indicated component. Phase 1 validators check internally consistent
snapshots; they do not implement events, permitted-transition tables, allocation policy, stop delivery, readiness
probes, result arbitration, retry loops, or recovery. Later components replace snapshots under their own authority.

### 7.1 Vocabulary and fields

| Record / enum | Fields / values |
| --- | --- |
| `ExecutionStatus` | `not_started`, `running`, `interrupted`, `settled` |
| `ParentProgress` | `parent_id: ParentId`, `execution_status: ExecutionStatus` |
| `AttemptStatus` | `allocated`, `running`, `outcome_ready`, `stopped` |
| `StopStatus` | `none`, `requested`, `observed` |
| `StopReason` | `parent_interrupted`, `parent_cancelled`, `parent_superseded`, `duplicate_accepted` |
| `StopDisposition` | `status: StopStatus`, `reason: StopReason | None` |
| `AttemptProgress` | `inputs: AttemptInputs`, `status: AttemptStatus`, `stop: StopDisposition`, `first_started_at: float | None`, `result_artifact_id: ArtifactId | None` |
| `WorkerAvailability` | `unavailable`, `available`, `working` |
| `WorkerProcessStatus` | `unknown`, `running`, `unavailable` |
| `OwnershipStatus` | `unknown`, `owned_by_root`, `not_owned_by_root` |
| `CompatibilityStatus` | `unchecked`, `compatible`, `incompatible` |
| `ParentReadiness` | `parent_id: ParentId`, `dataset_ready: bool`, `evaluation_ready: bool`, `partition_ready: bool` |
| `WorkerObservation` | `worker_id: WorkerId`, `availability: WorkerAvailability`, `process_status: WorkerProcessStatus`, `ownership: OwnershipStatus`, `compatibility: CompatibilityStatus`, `parent_readiness: tuple[ParentReadiness, ...]`, `active_attempt_id: AttemptId | None` |
| `OperationKind` | `connection_round`, `transfer`, `persistence` |
| `OperationStatus` | `pending`, `running`, `succeeded`, `failed`, `cancelled` |
| `BoundedOperation` | `operation_id: OperationId`, `kind: OperationKind`, `context: TraceContext`, `attempt_limit: int`, `attempts_made: int`, `status: OperationStatus`, `last_failure: FailureRecord | None` |
| `ReceiptStatus` | `not_received`, `incomplete`, `complete` |
| `PersistenceStatus` | `not_persisted`, `incomplete`, `complete` |
| `ReceptionRecord` | `unit: UnitKey`, `selected_attempt_id: AttemptId | None`, `selection_evidence_id: ArtifactId | None`, `receipt: ReceiptStatus`, `persistence: PersistenceStatus`, `result_artifact_id: ArtifactId | None`, `accepted: bool` |

`first_started_at` is a finite, nonnegative UTC Unix timestamp in seconds describing an observed execution start;
it is not used as a timer here. The coordinator later derives a unit's first execution start from its attempts for
oldest-running duplicate selection. Clock acquisition, ordering ties, and scheduling remain later algorithms.

The operation vocabulary includes only the architecturally required bounded operations. Add another kind only when
its owning phase needs it and approves that contract amendment; do not make scientific fitting retryable by treating
it as an operation of these kinds.

### 7.2 State invariants and ownership

- Root control owns consolidation and parent execution observations. `settled` describes execution handling, not an
  alternate acceptance bit; an exhausted delivery may leave an unaccepted unit. Worker inactivity is a separate fact.
- Worker control owns attempt execution/stop observations. Running requires a start timestamp; allocated has none.
  Outcome-ready requires a start timestamp and a completed result artifact reference. Stopped attempts may have
  started or may have been stopped before execution; stopping does not imply a result file or stopped worker process.
  Result artifacts are allowed only for outcome-ready or stopped attempts with an already completed outcome and
  a start timestamp.
- Stop `none` has no reason; requested/observed requires a reason. `stopped` requires observed stop. Outcome-ready may
  coexist with requested/observed stop because a terminal attempt needs no new evaluation loop to end unnecessary
  handling. Allocated/running attempts may have no stop or a requested stop, but not an observed stop. Active work can
  be marked requested while still running; edits elsewhere do not alter its inputs.
- A worker's configured membership is recorded in configuration, not inferred from availability. A disconnected
  worker can be unavailable with a retained active attempt. Process status can remain unknown after lost contact;
  unavailable connection/eligibility is not proof of process failure.
- Available/working requires ownership by this root, compatible framework metadata, and observed running process
  status. Working requires an active attempt. Available has no active attempt blocking new work. Unavailable may
  retain an attempt; task outcome or delivery failure alone does not terminate the process.
- Parent readiness is per parent and has three separate facts. Reusing a ready dataset does not establish the next
  parent's evaluation/partition context. Availability alone does not prove readiness for any particular parent.
  Entries have unique parent IDs. Checking whether a worker can run a particular allocation is deferred.
- Local and online workers use the same logical observation; there is no fake connection record for a local worker.
  Authentication and public allocation details are not encoded by `owned_by_root`.

### 7.3 Bounded-operation semantics

`attempt_limit` is the **total** allowed attempts including the initial attempt, supplied by the future policy owner;
it is at least one. `attempts_made` is the count already begun and lies in `[0, attempt_limit]`. A future policy expressed
in retries must explicitly convert to this total; no default retry count or delay is selected here.

- Pending has zero attempts and no failure. Running has at least one attempt; it may preserve the previous failure.
- Succeeded has at least one attempt. Earlier failure information may remain; success does not mean no retry occurred.
- Failed has at least one attempt and a last failure. When its count reaches the limit it is exhausted. No extra
  exhausted status or independent flag may contradict the count/status pair.
- Cancelled may have zero or more attempts and does not imply success, acceptance, or worker-process unavailability.
- The last failure, when present, has operation scope and names the same operation ID/context. Its failure ID is local
  to this operation record, not a computational error in the result envelope.
- Separate connection rounds use separate operation identities. This allows a later scheduled round without resetting
  an exhausted delivery operation. Records preserve counts; Phase 1 has no automatic retry or count-reset method.

### 7.4 Reception and acceptance semantics

Root coordination owns one `ReceptionRecord` per `UnitKey`, covering local and online attempts equally:

- Before selection, both selected attempt and selection evidence are absent, receipt is not received, persistence is
  not persisted, result artifact is absent, and accepted is false.
- A selection names an attempt of this unit and evidence that the future storage component has durably published.
  The artifact handle alone is not proof of durability; Phase 2 establishes and verifies that evidence. Production
  reception must not proceed without it. Selecting by receiving order belongs to later coordination.
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
| `validate_attempt_progress(value: AttemptProgress, *, parent: ParentEvaluation) -> None` | Check attempt context, status/start/artifact/stop consistency. | Allocated/running/outcome-ready; stop while running; stopped before start; terminal handling stop; invalid timestamp; no result created. |
| `validate_worker_observation(value: WorkerObservation) -> None` | Check ownership/compatibility/availability combinations and unique parent readiness. | Disconnected active attempt retained; unknown vs failed process; available but unready parent; working without attempt; reused data without partition readiness; no configuration mutation. |
| `validate_bounded_operation(value: BoundedOperation) -> None` | Check count bounds, status, last-error context, and operation kind. | Zero/past-limit counts; bool rejected; success after retries; failed/exhausted; cancelled; distinct connection round identity; no budget reset or reevaluation. |
| `validate_reception_record(value: ReceptionRecord, *, attempts: tuple[AttemptInputs, ...]) -> None` | Validate selected attempt membership and receipt/persistence/acceptance prerequisites. | Unselected state; wrong unit/attempt; partial file; complete persistence before acceptance; accepted failure outcome uses same facts; absent evidence; no file I/O. |
| `validate_reception_set(values: tuple[ReceptionRecord, ...], *, attempts: tuple[AttemptInputs, ...]) -> None` | Check unique unit entries and unique attempt IDs, then validate each record. | Reject two records/accepted outcomes for one unit; independent units allowed; same attempt ID reused for different units rejected. |

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
| Dataset preparation | User source representation and split configuration or saved artifact | Validated `DatasetBinding` and `PartitionSchedule`; exact identity/content verification and one authoritative durable schedule. | 2–4 |
| Root parent commitment | Complete validated configuration and durable partition reference | `ParentEvaluation` and known unit set; never alter old snapshots for prospective edits. | 3, 13 |
| Root coordination | Fixed parent, readiness, unit/attempt/reception observations | `AttemptInputs`, progress/selection/acceptance observations; allocation and duplicate policy remain separate from records. | 13–18 |
| Worker control/readiness | Applicable local/online parent context, owned worker | `WorkerObservation`, `AttemptProgress`; no model execution in the coordinator or control path. | 5, 13–17 |
| Model initialization/execution | Resolved fixed model input and framework-supplied sample data | Aligned standardized predictions or scoped failures; executable loading only inside enforced worker boundary. | 5–7, 10 |
| Validation/report components | Saved schedule, sample metadata, aligned predictions and valid prerequisites | Typed report payloads and `ReportOutcome`; preserve labels/classes/NaN and failures, no independent file persistence. | 7–10 |
| Attempt workflow/packaging | Report outcomes and computational failure records for one attempt | `AttemptResult`; one complete file later, or failure-only baseline outcome. No report-specific delivery state. | 7–12 |
| Worker report persistence | Complete result envelope | Published worker artifact resolved by `ArtifactId`; preserve attempt identity. | 2, 12 |
| Root reception/persistence | Selected existing artifact and its unit/attempt identity | Durable selection/result evidence and acceptance/acknowledgement after complete persistence. No active-loop report loading. | 2, 13–18 |
| Delivery | Selected existing completed result, operation progress | Updated bounded-operation observations; resends never reconstruct/recompute results. | 14, 17 |
| Failure/log owner | Actual exception or detected failure plus precise context | `FailureRecord`, later structured log; no application error implies worker termination. | 2 and each producer |
| Compatibility boundary | Plain root/worker metadata | Compatibility decision before normal operation; no model/result payload loading. | 13, 16 |

No missing future component is filled with a fake implementation to complete Phase 1 tests.

## 9. Dependency-safe implementation units

Follow this order. Each named declaration is its own potential approval unit; each named validator is one production
function unit. A table row is an ordered inventory, **not** authorization to bundle its declarations or functions.
Implement declarations before the first validator that uses them, and finish its targeted tests/checks before the next
approved unit. Generated dataclass methods do not imply planned custom methods.

| Order | Declarations, in dependency order | Then implement these function units, in order | Direct test module |
| --- | --- | --- | --- |
| 1 | Scalar/target aliases; each ID alias; `ReportName`; `UnitKey` | `validate_identifier`; `validate_label`; `validate_unit_key` | `tests/contracts/test_identity.py` |
| 2 | `DatasetReference`; `SourceReference`; `SampleRecord`; `PartitionReference`; `FoldRecord`; `PartitionSchedule` | `validate_source_reference`; `validate_sample_record`; `validate_partition_schedule` | `tests/contracts/test_data.py` |
| 3 | `TaskKind`; `ModelFamily`; `ConsolidationState`; `ModelReference`; `EvaluationReference`; `DatasetBinding`; `ComputationSnapshot`; `UnitTask`; `ParentEvaluation`; `AttemptInputs` | `validate_model_reference`; `validate_evaluation_reference`; `validate_dataset_binding`; `validate_computation_snapshot`; `validate_parent_evaluation`; `validate_attempt_inputs` | `tests/contracts/test_execution.py` |
| 4 | `FailureScope`; `TraceContext`; `FailureRecord`; `ValueStatus` | `validate_trace_context`; `validate_failure_record`; `validate_failure_links` | `tests/contracts/test_failure.py` |
| 5 | `RegressionPredictions`; `ClassificationPredictions`; `PredictionBatch` | `validate_regression_predictions`; `validate_classification_predictions` | `tests/contracts/test_prediction.py` |
| 6 | `ReportStatus`; `ReportOutcome`; `EvaluationOutcome`; `AttemptResult` | `validate_report_outcome`; `validate_attempt_result` | `tests/contracts/test_result.py` |
| 7 | Execution/attempt/stop enums; `ParentProgress`; `StopDisposition`; `AttemptProgress` | `validate_parent_progress`; `validate_stop_disposition`; `validate_attempt_progress` | `tests/contracts/test_state.py` |
| 8 | Worker/ownership/compatibility enums; `ParentReadiness`; `WorkerObservation` | `validate_worker_observation` | `tests/contracts/test_state.py` |
| 9 | Operation enums; `BoundedOperation` | `validate_bounded_operation` | `tests/contracts/test_state.py` |
| 10 | Receipt/persistence enums; `ReceptionRecord` | `validate_reception_record`; `validate_reception_set` | `tests/contracts/test_state.py` |
| 11 | `CompatibilityMetadata`; declared supported-series constant | `validate_compatibility_metadata`; `require_compatible` | `tests/contracts/test_compatibility.py` |
| 12 | No new production declaration/function | Review completed contract coverage and run applicable phase checks; report approval status and deferred boundaries. | Existing targeted modules only. |

Place `ConsolidationState` in `execution.py`, `ValueStatus` in `prediction.py`, and `ReportName` in `identity.py`.
`failure.py` has no import from prediction/result/state; `state.py` may consume execution/failure records. This prevents
cyclic dependencies while keeping the record definitions beside their responsibility.

Do not add an unplanned reusable helper silently. If a validator needs a substantial new helper or an approved
signature/data model must change, identify it and obtain the required approval before implementing the caller.
Small direct validation logic stays in its coherent validator; no generic validation framework is planned.

Declaration-only tests verify field defaults, keyword-only construction, frozen/tuple structure, equality, and type
vocabulary as applicable. They do not pretend an unimplemented validator already rejects semantic errors. If a
declaration has no independently meaningful runtime behavior (for example a `NewType` alias), explain the test
deferral until its validator/consumer rather than adding a test that merely repeats the declaration.

## 10. Numerical behavior and validation requirements

Phase 1 implements **no scientific formula**, split randomness, CPU-capacity calculation, model invocation, or metric
aggregation. Architecture Sections 4.5–4.7 and 9.3 remain authoritative for later scientific phases. In particular,
the approved influence denominator, classwise/micro/macro accuracy distinctions, within-sample classification
residual axis, and formula-defined NaN behavior are not reopened here.

Numerical checks in this phase are structural: integer count bounds, finite label/timestamp metadata, aligned vector
dimensions, probability bounds, and status/NaN consistency. Compare NaN through `math.isnan`, not equality. Preserve
all supplied values; do not convert scientific outcomes to a preferred finite result. The operation count inequality
is `0 <= attempts_made <= attempt_limit`, with `attempt_limit >= 1`; it is not a retry scheduling algorithm.

For each approved implementation unit:

- Run its directly corresponding pytest cases, including negative cases that could otherwise corrupt identity,
  hide a failure, or accept an incomplete outcome. Do not create unrelated test suites.
- Use typed tests and immutable fixtures. Make sample labels deliberately different from internal sample IDs and
  class labels, and use nontrivial row/class order so accidental position-only alignment is detectable.
- Test validators' lack of mutation and absence of operational side effects. Source-reference tests do not need real
  datasets; compatibility tests never deserialize a payload; state tests do not launch workers or run retry loops.
- Run mypy, Black `--check`, and Ruff on the applicable source/tests under `pyproject.toml`. Preserve its selected
  families, targets, exclusions, and narrow external-typing suppression policy. Do not silently add blanket ignores.
- Add NumPy-style source docstrings to every class/function/method, including private validators. Use `Parameters`,
  applicable `Returns`/`Raises`, and useful examples. Explain non-obvious invariants and scaffolding logic in comments.
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
| Are computation, parent, unit, and attempt the same identity? | Separate configuration snapshot, fixed parent, `(parent, model)` unit key, and attempt reference, following architecture 2.3/3.4. |
| Must Phase 1 build the public computation or lifecycle engine? | No. Phase 1 establishes records; Phase 3/13–18 own configuration and execution algorithms. |
| How can inputs be fixed before detailed settings/model schemas exist? | Immutable typed references to snapshots owned by the later configuration producer; no mutable/executable objects embedded or independently overrideable attempt fields. |
| How much dataset/report schema belongs here? | Sample/source/membership and prediction alignment plus generic report/envelope structure. Feature adapters and concrete metric payloads remain Phase 4/8–10. |
| Which fields/types are selected now? | The explicit internal aliases, dataclasses, enums, tuple ordering, scalar label domain, logical source syntax, and validators above, subject to plan approval. |
| Does undefined science mean an operation failed? | No. Value status and failure references distinguish them; complete reports may contain formula-defined NaN. |
| Does a failure-only result finish delivery or terminate a worker? | No. It is a complete scientific outcome, independently selected/persisted/accepted later. |
| What does a retry count mean? | Total begun attempts including the first, bounded by a supplied total limit; no policy defaults or timers chosen. |
| Are frozen records proof of durable acceptance or isolation? | No. They describe evidence/references; storage, ownership, concurrency, and enforced execution boundaries remain later obligations. |
| Does Python mismatch imply framework incompatibility? | Not within the declared supported stable series; complete Cocoonet strings must match exactly. |

No required Phase 1 semantic ambiguity remains after these proposed representation choices. The governing documents
explicitly assign concrete containers/field names to Phase 1 and reserve the later mechanisms below. The plan does
not treat those reserved mechanisms as blockers or as implemented behavior. If the user rejects a proposed internal
representation, amend this plan before its affected declaration is implemented.

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
| Dataset identity generation/content validation, sample-ID restoration, dataframe/target admissibility, mixed inline/file support, tensor shape encoding, splitter APIs and strategy coverage/empty-population decisions | Phase 4, before dataset consumers. |
| Isolation enforcement and model/control execution boundary | Phase 5, before executable loading or user code. |
| Concrete model input and prediction adapters, class conversion, raw-output validation and numerical checks, fit-state reset | Phases 6 and 10. |
| Concrete fold/validation report payloads and baseline workflow | Phase 7. |
| Conventional metric/report conventions, concrete residual/visualization payloads and any actual unresolved custom formula | Phase 8; do not reopen established standard formulas or expected undefined values as ambiguities. |
| Concrete influence payload/count/status representation and removal workflow under architecture 4.7 | Phase 9; formula and populations already fixed. |
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

Remaining Phase 1 blocking ambiguity: **none identified**. All concrete choices in this proposal require plan approval;
that approval does not itself authorize implementing any production unit.

Stop here and wait for explicit user approval. No Phase 1 source code or tests have been implemented by this planning
task, and no automatic commit is authorized.
