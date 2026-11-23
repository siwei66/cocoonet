# Phase 3: Reusable EvaluationTask configuration and lifecycle foundations

Status: proposed; implementation requires separate approval.

## 1. Authority, goal, and approval boundary

This roadmap plans only Phase 3 of [the project roadmap](../roadmap.md). It applies the complete root
[AGENTS.md](../../AGENTS.md), the complete [architecture](../architecture.md), the previous
[Phase 1](roadmap_phase_1.md) and [Phase 2](roadmap_phase_2.md) roadmaps, and the implemented Phase 1 contract and
Phase 2 storage code and tests.

The phase goal is to introduce `EvaluationTask` as Cocoonet's core public API for reusable mutable root-side evaluation
configuration without launching a worker, running a model, creating a parent, or persisting an execution configuration.
The phase establishes:

- explicit construction and fresh initialization through the Phase 2 storage boundary;
- incomplete but internally coherent prospective configuration;
- ordered model and worker registries with singular and atomic bulk mutations;
- initialized, consolidated, and unconsolidated lifecycle semantics for later runtime consumers;
- one optional active prepared-dataset/evaluation slot for Phase 4 to populate;
- immutable inspection snapshots and fixed parent-snapshot preparation; and
- ordinary shallow-copy behavior that copies configuration containers without filesystem work or runtime cloning.

This phase concretizes the configuration boundary already required by the governing documents. It does not introduce
new architecture or a performance improvement. Approval of this roadmap does not authorize source or test changes.
Every implementation unit in Section 11 requires its own proposal, over-engineering review, explicit approval,
targeted implementation, verification, and stop. If an approved signature, field, invariant, or phase boundary must
change, implementation stops for an approved roadmap amendment.

This roadmap defines the Phase 3 implementation boundary; its governing naming and snapshot contracts are aligned with
the root development and architecture documents.

## 2. Scope

### 2.1 In scope

Phase 3 implements:

1. the public mutable `EvaluationTask` class and its top-level package export;
2. construction with a required user-provided root node working directory and `overwrite=False` by default;
3. explicit fresh `initialize()` using Phase 2 ownership and protected-root checks;
4. one optional dataset/partition/evaluation slot, initially empty and not publicly populated until Phase 4;
5. an ordered registry of opaque model inputs and their `ModelReference` metadata;
6. positive sequential model-ID assignment beginning at `1`, with no reuse after removal;
7. an ordered registry of root-local or online worker configuration records;
8. requested worker device text and positive requested `n_attempt`, defaulting to `1`;
9. public singular and bulk model/worker add, list, inspect, summarize, and remove operations;
10. all-or-nothing bulk validation and commit behavior;
11. model-evaluation edits changing consolidated state to unconsolidated, while worker-only edits preserve it;
12. a configuration-edit guard for later active-execution integration;
13. immutable `ModelingSnapshot` production and pure `ParentEvaluation` preparation; and
14. Python `copy.copy()` support with independent mutable registries and no constructor/filesystem/runtime action.

### 2.2 Out of scope

Phase 3 does not implement or expose:

- `add_dataset()` or another public dataset-installation method, dataset parsing, standardization, dataset/partition ID
  generation, split realization/reuse, workbook creation, transfer preparation, or dataset artifact copying;
- an independent general evaluation-settings API. Phase 4 installs the dataset, partition, and the first complete
  `EvaluationReference` together; later scientific phases add only the settings they own;
- model-file reads or deserialization, model API probing, `get_params`, `set_params`, deep copying, model
  serialization, declarative public-worker model schemas, reconstruction, fitting, prediction, or fixed model-transfer
  artifact preparation; extension-agnostic path admission remains in scope;
- CPU detection, model CPU-demand inspection, `effective_n_attempt` calculation, slot reservation, process creation,
  device availability validation, or GPU execution;
- private-worker endpoint/direction configuration, public-service allocation information, authentication, networking,
  readiness, connection rounds, or worker process management;
- `run()`, parent-ID generation, parent commitment, `execution_configuration.csv` or `execution_state.csv` writes,
  attempt creation, allocation, scheduling, result synchronization, or result acceptance;
- interruption, resume, recovery/open-existing execution, retry, retention, cleanup, or release APIs;
- deep-copy semantics for arbitrary model inputs, copied-instance runtime coordination, or concurrent controllers of
  one persisted execution; or
- new dependencies, generic registries, repositories, plugin systems, transactions, event buses, or persistence
  abstractions.

No placeholder method named `run`, `add_dataset`, `interrupt`, `recover`, `cleanup`, or `release` is added. Missing later
capabilities must remain visibly absent rather than silently doing nothing or raising a generic not-implemented error.

## 3. Entry and completion criteria

### 3.1 Entry criteria

Implementation may begin only when:

- this roadmap is explicitly approved;
- the exact implementation unit is separately approved;
- the Phase 1 and Phase 2 sources and tests remain available and their relevant tests pass in an environment containing
  the declared project dependencies;
- the working tree is reinspected for intervening changes and applicable nested instructions; and
- no source change is used to fold an inconsistency from Section 13 into Phase 3 without separate approval.

The phase adds no dependency. It uses the standard library, existing Phase 1 records/validators, and Phase 2 root
storage functions.

### 3.2 Completion criteria

Phase 3 is complete only when:

- construction prepares a safe fresh root and produces an initialized, empty configuration;
- incomplete configuration with no dataset, models, or workers is valid and inspectable;
- no local worker, online subsystem, dataset, model artifact, parent, attempt, or execution file is created implicitly;
- model IDs are assigned in fixed add order as canonical positive decimal text, begin at `1`, and are never reused after
  successful removal;
- model labels, worker labels, and worker IDs are unique in their owning active registries;
- all configured models have one task kind, and a later active evaluation reference must agree with it;
- in-memory model inputs remain opaque and are never called, copied, serialized, deserialized, or introspected in this
  phase; a file-backed input resolves to an existing canonical regular file with a nonempty extension, preserves its
  path without opening or reading it, and is not rejected because its extension is unfamiliar;
- root-local and online worker entries remain configuration only, and at most one root-local worker is configured;
- requested `n_attempt` is retained exactly as a positive integer excluding `bool`; no effective capacity is invented;
- every bulk mutation either commits the complete validated batch in supplied order or leaves configuration, lifecycle
  state, and the next model-ID counter unchanged;
- model add/remove and prepared-dataset/evaluation replacement use the evaluation-edit transition, while worker-only
  edits preserve consolidation and parent identity;
- fresh initialization validates every retained dataset source before filesystem mutation, retains prospective
  configuration, clears the prior parent association, and restores initialized state only after successful preparation;
- parent-snapshot preparation requires a complete scientific configuration but performs no persistence or lifecycle
  mutation and does not require a worker;
- `copy.copy()` performs no filesystem mutation, creates no identity, and gives the copy independent registry
  containers while retaining the same root, lifecycle state, parent identity, immutable references, and opaque model
  input object references;
- public inspection never exposes a mutable internal registry container;
- directly corresponding tests pass with real temporary directories and existing Phase 1/2 code paths; and
- applicable pytest, mypy, Black, and Ruff checks pass under `pyproject.toml` without a new suppression.

Completion does not make an `EvaluationTask` runnable. Phase 4 must populate and persist the dataset/partition slot, model
execution remains behind Phases 5–10, and Phase 13 owns the first parent commitment and runnable local path.

## 4. Governing architecture contracts and invariants

### 4.1 EvaluationTask, parent, and runtime separation

`EvaluationTask` is Cocoonet's core public API and mutable prospective root evaluation configuration. It is not a
committed parent evaluation task, model unit, attempt, worker, scheduler, or runtime handle. A `ParentEvaluation` is a
separate immutable scientific snapshot prepared from a complete `EvaluationTask` only for later commitment by `run()`.

The `EvaluationTask` owns one ordered model registry, one ordered worker registry, and at most one active prepared
dataset/evaluation association. Worker configuration is not included in `ParentEvaluation`; the parent fixes the
dataset/partition, evaluation reference, and ordered model references.

The `EvaluationTask` stores no attempt list, result-selection field, accepted-result bit, worker observation, connection,
process, or result bundle. Later runtime components own those facts.

### 4.2 Fresh initialization and root ownership

`EvaluationTask(work_dir, overwrite=False)` and `initialize(work_dir, overwrite=False)` use
`prepare_work_directory()` from Phase 2. They accept absolute and relative path-like specifications through that
boundary, use the canonical returned `RootStorageLayout.work_dir`, and never reproduce work-directory safety logic
locally.

Before mutation, initialization passes the complete currently configured physical dataset-source collection to Phase
2. In Phase 3 that collection is empty because no public dataset installation exists; the field and call path are
nevertheless part of the `EvaluationTask` so Phase 4 can populate the single slot without replacing initialization logic.

Construction or successful initialization sets:

```text
consolidation = initialized
parent_id = None
execution_status = not_started
```

Fresh initialization retains prospective model, worker, and future prepared-dataset configuration. It does not invent
or copy Phase 4 artifacts. Phase 4 must integrate artifact publication before it can install a nonempty dataset slot
that survives initialization as runnable configuration. Until then the slot remains empty in normal Phase 3 use.

If directory preparation fails, the `EvaluationTask` object's existing work directory, configuration, counters, and lifecycle fields
remain unchanged. Phase 2's documented operating-system failure behavior for an explicitly authorized filesystem clear
still applies to the target directory.

Opening or recovering an existing nonempty root is not construction or initialization and remains Phase 15 work.

### 4.3 Configuration edit categories

Every mutation first applies the future-compatible active-execution guard. `ExecutionStatus.RUNNING` rejects all model,
worker, dataset-slot, and initialization changes with `RuntimeError` before registry or filesystem mutation. The other
Phase 1 execution observations do not represent ordinary active execution:

- `NOT_STARTED`: edits are permitted;
- `INTERRUPTED`: prospective edits are permitted without claiming worker inactivity;
- `SETTLED`: edits are permitted.

The owning runtime phases later update this observation and validate their transition sequences. Phase 3 only defines
the mutation gate; it does not expose a public execution-status setter.

For a successful **model-evaluation edit** (model add/remove or future dataset/evaluation replacement):

```text
initialized    -> initialized
consolidated   -> unconsolidated
unconsolidated -> unconsolidated
```

For a successful **worker-only edit**, consolidation and `parent_id` are unchanged. A rejected or empty bulk operation
changes neither configuration nor lifecycle state. Inspection never changes state.

### 4.4 One active dataset/evaluation slot

The `EvaluationTask` carries these private prospective fields:

```text
_data_partition: PartitionReference | None
_evaluation_config: EvaluationReference | None
_dataset_sources: tuple[Path, ...]
```

Phase 3 initializes them to `None`, `None`, and `()`, includes the two references in `ModelingSnapshot`, passes the
sources into fresh root validation, and preserves them through ordinary copying. It does not expose a user mutation for
them. Phase 4 owns the one operation that validates/persists a prepared dataset and atomically replaces all three
fields, plus dataset inspection/removal APIs.

The two references are either both absent or both present. No model may disagree with a present evaluation task kind.
These invariants are checked by snapshot construction and parent preparation even though normal Phase 3 public use
leaves the slot empty.

This boundary deliberately does not choose mixed inline/file rows, tensor-shape representation, splitter arguments,
source workbook schemas, or artifact carry-forward. Those are Phase 4 decisions.

### 4.5 Model registry

Model registration stores an opaque prospective input and an immutable Phase 1 `ModelReference`. It does not establish
that the input can be reconstructed or executed. Registration rules are:

- the user supplies `model_label`, `ModelFamily`, and `TaskKind` explicitly;
- Cocoonet assigns `ModelId(str(next_model_number))` only on successful commit;
- the first committed model receives `"1"`; each later committed model consumes the next integer;
- removal never decrements the counter or frees an ID for reuse;
- labels are unique in the active registry using the existing standardized `ScalarLabel` equality semantics;
- every active model has the same `TaskKind`;
- if an evaluation reference is present, every model task kind equals it;
- registry and summary order is successful registration order; removal preserves survivor order; and
- replacing a model requires explicit remove followed by add and therefore receives a new model ID.

A file-backed input may be supplied as text or `Path`. Resolve relative input from the current console working
directory, retain the canonical absolute `Path`, and require an existing regular file with a nonempty complete suffix.
Phase 3 accepts arbitrary suffixes, preserves their spelling in the stored path, imposes no suffix/family association,
and never selects or invokes a loader. Phase 6 selects an available loader from the complete lowercase extension after
the Phase 5 isolation boundary exists. The user owns the risk of later replacement or mutation of the referenced file;
Phase 3 adds no lock, digest, or metadata-change check.

Phase 3 stores the exact in-memory `model_input` object reference or the normalized canonical `Path` for a file-backed
input. It does not call `copy.copy`, `copy.deepcopy`, a model loader, model methods, `repr`, `get_params`, or another
user-defined hook. Registry metadata and replacement are controlled, but Phase 3 does not claim to make an arbitrary
caller-owned object deeply immutable. Phase 6 must prepare the fixed executable/declarative model input before Phase 13
can commit a parent. An already committed parent's prepared input, not a later mutation of the prospective Python
object or source file, is authoritative for its attempts.

### 4.6 Worker registry

Worker registration is configuration only. A worker record identifies a user-defined worker label, a user-configured
worker ID, execution topology, device text, and requested attempt limit. It does not prove process availability,
ownership, compatibility, readiness, capacity, endpoint validity, or public allocation.

Phase 3 uses exactly two topology values:

```text
root-local
online
```

Public/private online ownership is intentionally not another topology value. Phase 16 adds private connection details;
Phase 20 adds externally allocated public-worker information under the supplied service interface. Both remain online
workers in the root/local-versus-online execution boundary.

Rules:

- `worker_label` is a valid user-facing `ScalarLabel` and unique in the active registry;
- `worker_id` is a nonempty opaque string and independently unique in the active registry;
- public worker registry operations identify a worker by `worker_label`; runtime and persistence use `worker_id`;
- at most one `root-local` worker may be configured because it represents the `EvaluationTask`'s single locally managed
  worker service with internal attempt capacity;
- any number of distinct online configurations may be retained;
- `device` is nonempty text retained without normalization or availability probing;
- `n_attempt` is a positive integer excluding `bool`, defaults to `1`, and is the requested value only;
- removal and re-addition of the same user-supplied worker label or ID is allowed after successful removal; and
- worker removal has no stop, disconnect, cleanup, reset, release, filesystem, or lifecycle side effect.

The public `WorkerConfiguration` docstring must include the Phase 11 capacity contract. For CPU workers, later worker
control derives `C = N - 1`, `J = max(per-attempt demand)`, keeps `A = n_attempt` when `A * J <= C`, reduces exactly
once to `A - 1` when permitted, and otherwise rejects without changing model/framework settings. GPU execution later
uses one Cocoonet attempt slot. Phase 3 performs none of those calculations.

### 4.7 Bulk atomicity

Bulk methods materialize their iterable once, reject a path/string/bytes object used as the collection, validate every
entry and every collision against a staged copy, and commit one final registry/counter/state replacement only after all
validation succeeds.

For bulk add, assigned model IDs follow supplied order and are consumed only at commit. For bulk remove, requested keys
must be unique and every key must exist before any removal. Empty bulk inputs are successful no-ops returning `()` and
do not change lifecycle state.

There is no rollback framework: all validation occurs before in-memory commit, and the commit is a bounded assignment
of already validated Python containers. These methods perform no persistent I/O.

### 4.8 Copy semantics

`copy.copy(evaluation_task)` is the only Phase 3 copy API. `EvaluationTask.__copy__()` bypasses `__init__`, performs no root
validation or preparation, and creates no parent/model/worker identity. The copy receives:

- the same immutable root layout value and root node working directory;
- the same consolidation state, parent identity, execution observation, partition/evaluation references, and source
  path tuple;
- new model and worker registry containers preserving order;
- the same next model-ID counter; and
- the same opaque model input object references.

Later registry edits to either object do not mutate the other's containers. Copying does not clone or attach live
processes, connections, attempts, locks owned by a runtime, or result bundles. Phase 15 must integrate recovery/runtime
ownership and coordination for multiple objects referring to one persisted execution. `copy.deepcopy()` receives no
special support in this phase because recursively copying user model inputs could execute user-defined behavior and
would not represent the required ordinary copy.

## 5. Planned data models and validation

### 5.1 Public records in `src/cocoonet/configuration.py`

```python
class WorkerKind(StrEnum):
    ROOT_LOCAL = "root-local"
    ONLINE = "online"


@dataclass(frozen=True, kw_only=True, slots=True)
class ModelRegistration:
    model_input: object
    model_label: ScalarLabel
    family: ModelFamily
    task_kind: TaskKind


@dataclass(frozen=True, kw_only=True, slots=True)
class WorkerConfiguration:
    worker_label: ScalarLabel
    worker_id: WorkerId
    kind: WorkerKind
    device: str = "cpu"
    n_attempt: int = 1
```

`ModelRegistration` exists primarily for atomic bulk input. The opaque input is not validated beyond rejecting
`None`; family-specific validity belongs to Phases 6 and 10. Construction alone does not register anything.

`WorkerConfiguration` is both the singular result and the bulk input record. Future connection/device-specific phases
must add their own associated records rather than overloading `device` or `kind` with endpoints, credentials, public
allocation, or effective capacity.

Planned validators:

```python
def validate_model_registration(value: ModelRegistration, /) -> None
def validate_worker_configuration(value: WorkerConfiguration, /) -> None
```

They validate only their declared representation and local values. Registry-scoped uniqueness and cross-record task
agreement belong to `EvaluationTask` mutations.

### 5.2 Private registered model

In `src/cocoonet/evaluation_task.py`:

```python
@dataclass(frozen=True, kw_only=True, slots=True)
class _RegisteredModel:
    reference: ModelReference
    model_input: object

    @property
    def model_id(self) -> ModelId

    @property
    def model_label(self) -> ScalarLabel
```

The private record keeps the authoritative Phase 3 association once. Public list/summary methods return immutable
metadata and never return the internal registry list. The two properties expose the corresponding immutable
`ModelReference` fields without storing duplicate independently maintained identity or label state. The future
model-preparation component defines its own narrow opaque-input retrieval boundary in Phase 6.

### 5.3 Mutable EvaluationTask fields

`EvaluationTask` is an ordinary mutable class, not a dataclass. It retains:

```text
_work_dir: Path
_consolidation_state: ConsolidationState
_parent_id: ParentId | None
_execution_status: ExecutionStatus
_data_partition: PartitionReference | None
_evaluation_config: EvaluationReference | None
_dataset_sources: tuple[Path, ...]
_models: list[_RegisteredModel]
_workers: list[WorkerConfiguration]
_next_model_number: int
```

The class stores the canonical `RootStorageLayout.work_dir` returned by Phase 2 as `_work_dir`; it does not retain a second
root representation. It does not use `slots`, allowing later approved phases to add owned runtime/configuration fields
without a storage-layout migration. It does not expose the mutable lists. No duplicate mapping is maintained beside
each list; `snapshot()` derives immutable mappings on demand. The small ordered registries use direct scans, avoiding
independently synchronized order and index state. Later performance evidence may justify a separately approved index.

### 5.4 Pure lifecycle and parent helpers

In `src/cocoonet/evaluation_task.py`:

```python
def _consolidation_after_evaluation_edit(
    value: ConsolidationState,
    /,
) -> ConsolidationState


def _require_configuration_editable(
    value: ExecutionStatus,
    /,
) -> None


def prepare_parent_evaluation(
    snapshot: ModelingSnapshot,
    /,
    *,
    parent_id: ParentId,
) -> ParentEvaluation
```

The first function implements only the three-state edit table in Section 4.3. The second rejects `RUNNING` and has no
side effect. `prepare_parent_evaluation()` validates its complete input, requires initialized state with no existing
parent, one partition/evaluation pair, and at least one model, constructs the existing Phase 1 `ParentEvaluation`, and
passes `tuple(snapshot.models.values())` as the parent's ordered model references. It validates the parent before
return. It does not check workers, generate an ID, persist files, change the `EvaluationTask`, or claim that a parent
was committed.

`prepare_parent_evaluation()` remains private to the package and is not exported from `cocoonet`; Phase 13 will call it
after its run prerequisites and parent-ID generation are satisfied.

## 6. Public APIs and exact signatures

### 6.1 Construction, initialization, and inspection

In `src/cocoonet/evaluation_task.py`:

```python
class EvaluationTask:
    def __init__(
        self,
        work_dir: PathInput,
        /,
        *,
        overwrite: bool = False,
    ) -> None

    @property
    def work_dir(self) -> Path

    @property
    def state(self) -> ConsolidationState

    @property
    def parent_id(self) -> ParentId | None

    def snapshot(self) -> ModelingSnapshot

    def initialize(
        self,
        work_dir: PathInput,
        /,
        *,
        overwrite: bool = False,
    ) -> None

    def __copy__(self) -> EvaluationTask
```

Constructor behavior is real fresh preparation, not recovery. `initialize()` requires an explicit directory argument;
reinitializing the current work directory therefore remains an explicit call with the same path and, when nonempty, explicit
`overwrite=True`. There is no implicit current-directory fallback.

`snapshot()` constructs and validates exactly:

```python
ModelingSnapshot(
    work_dir=str(self._work_dir),
    consolidation_state=self._consolidation_state,
    data_partition=self._data_partition,
    evaluation_config=self._evaluation_config,
    models=MappingProxyType(
        dict(
            (registered.model_label, registered.reference)
            for registered in self._models
        )
    ),
    worker_ids=MappingProxyType(
        dict(
            (worker.worker_label, worker.worker_id)
            for worker in self._workers
        )
    ),
    parent_id=self._parent_id,
)
```

The newly built dictionaries preserve registration order; `MappingProxyType` prevents mutation of the snapshot's
nested mappings. `work_dir` returns `_work_dir`, the canonical `Path` returned by Phase 2. `state` and `parent_id`
expose lifecycle facts without permitting assignment.

### 6.2 Model registry API

```python
class EvaluationTask:
    def add_model(
        self,
        model_input: object,
        /,
        *,
        model_label: ScalarLabel,
        family: ModelFamily,
        task_kind: TaskKind,
    ) -> ModelReference

    def add_models(
        self,
        registrations: Iterable[ModelRegistration],
        /,
    ) -> tuple[ModelReference, ...]

    def list_models(self) -> tuple[ScalarLabel, ...]

    def inspect_model(
        self,
        model_label: ScalarLabel,
        /,
    ) -> ModelReference

    def summarize_models(self) -> tuple[ModelReference, ...]

    def remove_model(
        self,
        model_label: ScalarLabel,
        /,
    ) -> ModelReference

    def remove_models(
        self,
        model_labels: Iterable[ScalarLabel],
        /,
    ) -> tuple[ModelReference, ...]
```

`list_models()` returns user-facing labels in registry order. `inspect_model()` returns one immutable metadata reference.
`summarize_models()` returns every immutable metadata reference in registry order. The distinction avoids returning
opaque executable inputs in a summary while providing both compact listing and complete configured metadata.

Model lookup/removal uses the user-facing unique label. Phase 3 retains the exact opaque input inside
`_RegisteredModel` but adds no retrieval API solely for a later consumer. Phase 6 owns the narrow private retrieval or
preparation boundary required by its model-adapter implementation. Missing labels raise `KeyError` with the requested
key; type/value errors are raised before lookup.

### 6.3 Worker registry API

```python
class EvaluationTask:
    def add_worker(
        self,
        worker_id: WorkerId,
        /,
        *,
        worker_label: ScalarLabel,
        kind: WorkerKind,
        device: str = "cpu",
        n_attempt: int = 1,
    ) -> WorkerConfiguration

    def add_workers(
        self,
        workers: Iterable[WorkerConfiguration],
        /,
    ) -> tuple[WorkerConfiguration, ...]

    def list_workers(self) -> tuple[ScalarLabel, ...]

    def inspect_worker(
        self,
        worker_label: ScalarLabel,
        /,
    ) -> WorkerConfiguration

    def summarize_workers(self) -> tuple[WorkerConfiguration, ...]

    def remove_worker(
        self,
        worker_label: ScalarLabel,
        /,
    ) -> WorkerConfiguration

    def remove_workers(
        self,
        worker_labels: Iterable[ScalarLabel],
        /,
    ) -> tuple[WorkerConfiguration, ...]
```

`list_workers()` returns user-facing labels in registry order; `inspect_worker()` returns one immutable configuration;
and `summarize_workers()` returns every immutable configuration, including label and ID, in registry order. Missing
labels raise `KeyError` after label validation. Worker labels select registry entries; worker IDs remain the routing and
persistence identities.

No method contacts a worker. `remove_worker()` and `remove_workers()` only change the root-side registry.

### 6.4 Public package exports

`src/cocoonet/__init__.py` exports exactly these new Phase 3 names in addition to `get_version`:

```text
EvaluationTask
ConsolidationState
ModelFamily
ModelReference
ModelRegistration
TaskKind
WorkerConfiguration
WorkerKind
```

The existing Phase 1 types become public only through these explicit exports. Other `_contracts`, `_storage`, helper,
and registry implementation names remain private. User-facing documentation covers these public names; internal
helpers retain source docstrings without becoming public API.

## 7. Method responsibilities and mutation behavior

### 7.1 `EvaluationTask.__init__`

- Validate `overwrite` through the Phase 2 call path.
- Call `prepare_work_directory(work_dir, dataset_sources=(), overwrite=overwrite)` exactly once.
- Assign object fields only after successful preparation.
- Set empty registries, no dataset/evaluation, model counter `1`, initialized state, no parent, and `NOT_STARTED`.
- Do not call another public mutation method or create a worker/model/dataset artifact.

### 7.2 `EvaluationTask.initialize`

- Apply the active-execution guard before filesystem work.
- Snapshot the current prospective configuration fields and full dataset source tuple.
- Call `prepare_work_directory()` with the proposed root node working directory and retained sources.
- Only after success, replace `_work_dir`, set initialized/no-parent/not-started, and retain prospective configuration,
  registry order, and next model counter.
- Do not open/recover old state, assign model IDs again, or alter source files.

### 7.3 Read-only inspection

- Properties return immutable/scalar owned values.
- `snapshot()` builds directly from current authoritative fields and validates the result once.
- List/summary methods return new tuples.
- Inspect methods validate the requested key and return the immutable stored record, not a registry container.
- No inspection reads or writes persistent files, probes a worker, or validates a model API.

### 7.4 Singular model mutations

- `add_model()` is the primitive model-add semantic operation.
- Validate a temporary `ModelRegistration`, active edit permission, label/task agreement, and current evaluation
  agreement before assigning the next ID or changing state.
- Append one `_RegisteredModel`, increment the counter once, and then apply the evaluation-edit transition.
- `remove_model()` validates and resolves the label before mutation, removes exactly one record, preserves survivor
  order and the counter, then applies the evaluation-edit transition.
- Neither operation modifies the opaque model input.

### 7.5 Bulk model mutations

- Materialize and validate the entire collection.
- Stage against copies of the registry and counter using the singular semantic checks without calling the mutating
  public singular method.
- Detect duplicates within the batch and against existing labels.
- For removal, resolve every unique label before staging survivors.
- Commit registry, counter, and one lifecycle transition only after all checks succeed.
- Return records in supplied request order.

### 7.6 Singular and bulk worker mutations

- Apply the same validation-first/commit-last structure as model operations.
- Enforce worker-label uniqueness, worker-ID uniqueness, and the single root-local invariant.
- Worker operations never apply the evaluation-edit transition.
- Removal performs no compensating action even if a later runtime worker is disconnected, active, or retains data;
  the active-execution gate is the only Phase 3 runtime-related admission check.

### 7.7 `EvaluationTask.__copy__`

- Allocate with `type(self).__new__(type(self))`; do not invoke construction.
- Copy mutable registry containers with `list(...)`.
- Assign immutable/scalar fields and the model counter directly.
- Share opaque model inputs and immutable references.
- Do not inspect the filesystem or create/delete/copy artifacts.

## 8. Numerical behavior

Phase 3 implements no scientific formula, split, random generation, timestamp, parent-ID hash, metric, prediction,
influence value, retry count, or effective worker-capacity calculation.

Its numeric rules are limited to:

- `_next_model_number` starts at integer `1` and increases by one per successfully committed model registration;
- public `model_id` text is `str(number)` and therefore positive, canonical, and unpadded;
- failed and empty bulk additions consume no number;
- removals never reduce or reuse a number; and
- `n_attempt` is an exact positive built-in integer, excluding `bool`, retained without adaptation.

No Phase 3 operation changes model/framework parallelism. The Phase 11 formula documented in worker configuration is
descriptive here and is not executed.

## 9. Validation and error behavior

- Wrong Python argument/record types raise `TypeError` with field path, expected type, observed type, and bounded value
  information under root `AGENTS.md`.
- Invalid values, duplicate keys, mixed model task kinds, conflicting evaluation task kind, more than one root-local
  worker, nonpositive `n_attempt`, and inconsistent dataset/evaluation presence raise `ValueError` with expected and
  observed conditions.
- Missing model labels and worker IDs after valid key validation raise `KeyError`.
- A mutation attempted during `ExecutionStatus.RUNNING` raises `RuntimeError` before any in-memory or filesystem change.
- Phase 2 `StorageError` propagates from fresh directory preparation with its chained filesystem cause.
- A missing file-backed model input raises `FileNotFoundError`; a directory, non-regular file, or extensionless file
  raises `ValueError`. Phase 3 does not raise an unsupported-suffix or suffix/family-mismatch error.
- Validators and failed mutations do not normalize labels, device text, worker IDs, or in-memory model inputs. Resolving
  a file-backed model reference to its canonical absolute `Path` is the explicit model-input exception.
- Diagnostics show bounded registry counts/keys rather than complete model objects, large batches, or payloads.

All public and internal classes/functions/methods receive NumPy-style source docstrings. Every non-helper class and
function ends its `Notes` section with one standardized category prefix -- `Node usage scope: Root -`,
`Node usage scope: Worker -`, or `Node usage scope: Shared -` -- followed by a concise functionality-specific
clarification. Phase 3 components are root-only except reused shared Phase 1 records. The public
`WorkerConfiguration` notes contain the requested/effective capacity distinction and formula described in Section 4.6.

## 10. Directly corresponding tests

Tests use real Phase 1 records, real Phase 2 directory preparation, and `tmp_path`. There is no network boundary in
Phase 3, so tests use no mocks, stubs, fakes, or monkeypatching. No test runs model code or constructs a fake worker.

### 10.1 `tests/configuration/test_records.py`

Test:

- exact `WorkerKind` values;
- frozen/slotted keyword-only record behavior;
- valid opaque model registration without input inspection;
- `None` model input rejection;
- model label/family/task type validation;
- valid local/online worker records;
- default and retained requested `n_attempt`;
- nonempty device/worker ID requirements;
- zero, negative, boolean, float, and string `n_attempt` rejection; and
- no effective-capacity or device-availability field.

### 10.2 `tests/configuration/test_lifecycle.py`

Test:

- the exact evaluation-edit transition for all three consolidation states;
- worker edits having no state-transition helper;
- `RUNNING` rejecting configuration edits;
- `NOT_STARTED`, `INTERRUPTED`, and `SETTLED` permitting prospective edits;
- complete parent preparation from a manually constructed valid snapshot;
- rejection of non-initialized, already-parented, missing partition, missing evaluation, empty-model, mixed-task, and
  invalid parent-ID cases;
- parent preparation not requiring a worker and not mutating its snapshot; and
- no parent ID generation, persistence, or state commitment.

### 10.3 `tests/configuration/test_evaluation_task.py`

Test:

- construction with a missing and existing empty root;
- explicit overwrite behavior delegated to real Phase 2 preparation;
- canonical absolute `work_dir` after relative input;
- initial state, parent, snapshot, counter behavior, and empty registries;
- exact Phase 2 initial filesystem entries and absence of dataset/execution/worker artifacts;
- construction failure leaving no partially returned object;
- fresh initialization to a new root;
- failed initialization leaving object fields unchanged;
- successful initialization retaining prospective registries/counter and resetting state/parent/execution observation;
- no open/recovery behavior for a nonempty existing work directory without overwrite;
- `snapshot()` returning a new validated immutable record; and
- exact top-level public exports with no private helper export.

### 10.4 `tests/configuration/test_models.py`

Test:

- model IDs `1`, `2`, ... in successful registration order;
- no ID reuse after removal and no ID consumption on failure/no-op;
- `_RegisteredModel.model_label` and `_RegisteredModel.model_id` exposing the matching immutable reference fields;
- exact model-label ordering from list and metadata ordering from summary;
- inspect by user-facing model label;
- duplicate label, mixed task kind, present-evaluation mismatch, invalid enum, and `None` input rejection;
- canonical absolute model paths, relative-path resolution, arbitrary-extension acceptance, case and compound suffix
  preservation, and directory, missing-path, and extensionless-path rejection;
- invalid model-file bytes accepted without being read, with no loader import or call and no suffix/family association;
- singular removal, missing-label behavior, survivor order, and incomplete zero-model configuration;
- atomic bulk add with within-batch and existing collisions;
- atomic bulk remove with duplicate/missing labels;
- generator materialization exactly once and string/bytes collection rejection;
- failed bulk operations preserving registry, state, and counter;
- model edits applying the evaluation-edit transition; and
- `snapshot().models` preserving registration order and mapping each label to its matching reference; and
- no model method/API/repr/copy/loader invocation during registration, inspection, summary, removal, or copy.

### 10.5 `tests/configuration/test_workers.py`

Test:

- explicit root-local and online registration with no runtime or filesystem side effect;
- at most one root-local worker, including within an atomic batch;
- independently unique active worker labels and worker IDs, with re-add after removal;
- list/inspect/summary by worker label, in registry order, with immutable returned configurations;
- exact preservation of device and requested `n_attempt`;
- `snapshot().worker_ids` preserving registry order and mapping each worker label to its worker ID;
- atomic bulk add/remove and no-op behavior;
- missing/duplicate removal keys;
- all worker mutations preserving each consolidation state and parent identity;
- removal sending no stop/disconnect/cleanup/reset/release action; and
- no endpoint, connection, availability, effective-capacity, or public-allocation claim.

### 10.6 `tests/configuration/test_copying.py`

Test:

- `copy.copy()` performs no root preparation or filesystem mutation;
- copied root/state/parent/snapshot values agree;
- registry containers are independent and preserve order/counter;
- later add/remove operations affect only the selected object's prospective configuration;
- immutable references and opaque model input object identity are shared without invoking copy hooks;
- copying creates no new parent/model/worker identity and no artifacts; and
- no promise or test of `copy.deepcopy()` or live-runtime coordination.

### 10.7 Existing regression coverage and checks

- Run the directly affected Phase 1 execution/data/state contract tests.
- Run the Phase 2 path/work-directory tests used by construction and initialization.
- Run the full configured pytest suite in an environment containing declared dependencies.
- Run mypy, Black `--check`, and Ruff under `pyproject.toml`.
- Add no suppression, dependency, documentation-build claim, or unrelated test suite.

## 11. Dependency-safe implementation order

Each numbered item is one proposed approval unit unless the user explicitly approves an exact smaller unit or tightly
coupled helper bundle. Declaration approval does not authorize its methods, and a test unit remains directly tied to
its production unit.

1. Add `configuration.py` and declare `WorkerKind` with its exact two-value vocabulary.
2. Add `ModelRegistration` and its declaration tests.
3. Implement `validate_model_registration()` and direct tests.
4. Add `WorkerConfiguration` with the required capacity docstring and declaration tests.
5. Implement `validate_worker_configuration()` and direct tests.
6. Add `_RegisteredModel` in `evaluation_task.py`.
7. Implement `_consolidation_after_evaluation_edit()` and direct all-state tests.
8. Implement `_require_configuration_editable()` and direct status tests.
9. Implement `prepare_parent_evaluation()` and complete/incomplete snapshot tests.
10. Declare the mutable `EvaluationTask` class and owned field annotations without behavior.
11. Implement `EvaluationTask.__init__()` with real Phase 2 fresh preparation and direct tests.
12. Implement `work_dir`, `state`, and `parent_id` read-only properties as an explicitly approved inspection bundle, or
    separately if bundle approval is not given.
13. Implement `snapshot()` and direct validation/immutability tests.
14. Implement `initialize()` and direct safe-fresh-initialization tests.
15. Implement `add_model()` and direct ID/task/lifecycle tests.
16. Implement `list_models()`.
17. Implement `inspect_model()`.
18. Implement `summarize_models()`.
19. Implement `remove_model()` and direct no-reuse/lifecycle tests.
20. Implement atomic `add_models()` and direct all-or-nothing tests.
21. Implement atomic `remove_models()` and direct all-or-nothing tests.
22. Implement `add_worker()` and direct configuration-only tests.
23. Implement `list_workers()`.
24. Implement `inspect_worker()`.
25. Implement `summarize_workers()`.
26. Implement `remove_worker()` and direct no-side-effect tests.
27. Implement atomic `add_workers()` and direct all-or-nothing tests.
28. Implement atomic `remove_workers()` and direct all-or-nothing tests.
29. Implement `EvaluationTask.__copy__()` and direct shallow-copy/no-I/O tests.
30. Add the exact public exports in `cocoonet.__init__` and API-surface tests.
31. Run targeted tests, the full available suite, static/format/lint checks, and a final scope, docstring, mutation,
    side-effect, and redundancy audit.

This order establishes records before validators, lifecycle rules before mutable consumers, storage-backed construction
before registry operations, singular primitive semantics before bulk conveniences, and all configuration behavior before
public export. It does not jump into Phase 4 dataset preparation or Phase 13 runtime commitment.

## 12. Decisions explicitly deferred

| Decision | Owning phase |
| --- | --- |
| Public `add_dataset()`, dataset inspection/removal, normalized input records, dataset/partition/evaluation-reference generation, split API, workbook schemas, artifact carry-forward across initialization, mixed inline/file representation, tensor-shape encoding, and transfer-preparation reuse | Phase 4 |
| Technical model isolation and responsive worker-control boundary | Phase 5 |
| Fixed sklearn/custom model input capture, model API validation, the narrow private model-input retrieval/preparation boundary, trusted/declarative representation, and mutation isolation at parent commitment | Phase 6 |
| Validation/report payloads, scientific formulas, visualization data, and influence settings/workflows | Phases 7–9 |
| PyTorch configuration and model-specific device/training interpretation | Phase 10 |
| CPU detection, per-model demand, effective `n_attempt`, GPU slot enforcement, reservations, and subprocess supervision | Phase 11 |
| Attempt-result bundle contents and worker publication | Phase 12 |
| `run()`, parent timestamp/ID generation, complete `execution_configuration.csv` schema/codec/write, execution-state initialization, local worker startup, and first root result acceptance | Phase 13 |
| Retry/delivery policy and result retention | Phase 14 |
| Interruption, recovery/open-existing execution, copied-instance runtime coordination, and exact recovery API | Phase 15 |
| Private online endpoints, direction, authentication, heartbeat, connection rounds, and control protocol | Phase 16 |
| Dataset/model online synchronization, archive/tiling/compression, HMAC bytes, acknowledgements, and online delivery | Phase 17 |
| Mixed scheduling, duplicate eligibility/arbitration, and connection-loss replacement attempts | Phase 18 |
| Worker dataset/all-data cleanup selection and verification | Phase 19 |
| Public-service allocation/configuration and release integration | Phase 20 |
| Full supported-runtime and end-to-end lifecycle matrix | Phase 21 |

The deferral of executable input capture is not permission to mutate a committed parent's input. No parent can be
committed in Phase 3. Phases 6 and 13 must establish and consume a fixed prepared input before execution.

## 13. Consistency and ambiguity report

### 13.1 Resolved Phase 3 decisions

- Cocoonet's core public API and mutable prospective configuration object is named `EvaluationTask`; construction
  always means fresh preparation and does not create a committed parent evaluation task.
- `initialize()` requires an explicit root node working directory path and is also fresh preparation, never recovery.
- The immutable Phase 1 root-side modeling record is named `ModelingSnapshot` with fields `work_dir`,
  `consolidation_state`, `data_partition`, `evaluation_config`, `models`, `worker_ids`, and `parent_id`.
  `models` is a fresh immutable insertion-ordered `MappingProxyType` from `model_label` to `ModelReference`, and
  `worker_ids` is the corresponding mapping from `worker_label` to `WorkerId`.
- `_RegisteredModel.model_label` and `_RegisteredModel.model_id` are read-only properties derived from its authoritative
  `ModelReference`, so the registry exposes the required identities without duplicate state.
- Every worker has a user-defined `worker_label` unique within its `EvaluationTask`; public registry operations use that
  label, while the independently unique `worker_id` remains the runtime and persistence identity.
- Model IDs are assigned internally, sequentially, and never reused after removal.
- Model inputs remain opaque and are not copied or inspected in the coordinator during this phase.
- Phase 3 retains opaque model inputs internally; Phase 6 defines their private retrieval/preparation boundary.
- Worker topology is only root-local versus online; private/public online ownership is added by its owning phases.
- Worker device text is retained but not interpreted; requested capacity is retained but not adapted.
- The active dataset/evaluation slot is represented once but has no public Phase 3 installer.
- Parent preparation is pure and private; `run()` later owns durable commitment.
- The complete `execution_configuration.csv` schema, codec, and persistence belong exclusively to Phase 13; Phase 3
  owns only prospective in-memory configuration and snapshot preparation.
- Node-scope docstrings use a standardized Root/Worker/Shared prefix followed by a functionality-specific clarification.

These resolved choices require no further Phase 4 data-schema, Phase 5 isolation, Phase 6 model-adapter, Phase 11
capacity-control, or Phase 13 runtime-persistence design. Full Phase 3 approval still depends on the two current-phase
questions in Section 13.2.

### 13.2 Remaining blocking ambiguities

The following two questions remain deliberately unresolved pending separate user decisions:

1. the exact shared validation/staging/commit mechanism for singular and bulk additions; and
2. the final shallow-copy, deep-copy, warning, and independent-work-directory policy.

These unresolved contracts block approval and implementation of their affected Phase 3 units. This revision does not
choose among their alternatives or modify their existing design sections.

### 13.3 Inconsistencies found

1. **`execution_configuration.csv` ownership is described differently.** The Phase 2 roadmap says its fields are owned
   by Phase 3, while the project-wide roadmap assigns persistence of the complete execution configuration to Phase 13.
   The approved ownership decision is deterministic: Phase 3 owns only in-memory configuration and snapshot
   preparation, while Phase 13 exclusively owns the complete persisted schema, codec, and write. The Phase 2 sentence
   remains stale documentation but does not transfer implementation scope to Phase 3.
2. **The root docstring rule and the approved Phase 3 convention differ.** Root `AGENTS.md` names three exact complete
   node-usage sentences. The approved Phase 3 convention standardizes only `Node usage scope: Root -`,
   `Node usage scope: Worker -`, or `Node usage scope: Shared -`, followed by a functionality-specific clarification.
   Phase 3 follows that approved convention; aligning the governing prose is a separate documentation revision.

Neither remaining documentation inconsistency changes a Phase 3 runtime contract. They must not be silently repaired
outside the separately authorized documentation scope.

## 14. Approval checkpoint

Implementation order, in summary:

```text
configuration records and validators
-> lifecycle/edit and parent-snapshot helpers
-> storage-backed EvaluationTask construction and inspection
-> explicit fresh initialization
-> singular model registry operations
-> atomic bulk model operations
-> singular worker registry operations
-> atomic bulk worker operations
-> ordinary copy semantics
-> public exports and full phase verification
```

No source code or tests may be implemented until this roadmap is approved. Approval of the roadmap still does not
authorize more than the next separately approved implementation unit. Commits remain manual.
