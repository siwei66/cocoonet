# Cocoonet Architecture

## 1. Purpose and Design Status

This document describes the initial architecture of Cocoonet, a distributed engine for scientific evaluation of
classification and regression models. It establishes the worker task boundary and the requested worker input/output
contract, and provides a skeleton for the remaining system design.

The repository is at the scaffold stage. Documented requirements do not imply that implementations exist. This
document distinguishes resolved behavior from implementation details to specify before developing the affected unit.
Standard metrics use established conventional definitions unless an explicit Cocoonet variation is specified;
missing formula detail alone does not make a standard metric unresolved. Cocoonet-specific measures require explicit
definitions, including the approved influence measure in Section 4.7.

Root [`AGENTS.md`](../AGENTS.md) governs the development workflow. [`roadmap.md`](roadmap.md) is the location for
sequencing implementation work; this architecture document does not assign milestones or authorize implementation.

**Implementation Style**

Follow the development philosophy in `AGENTS.md`. Scientific functions receive the data and settings needed for their
calculation and return results with provenance. Coordination, communication, persistence, and resource management own
their operational state.

Model mutation here means changes made by fitting a scikit-learn-like model or training a PyTorch model, such as
updating learned parameters and training state. These changes belong to the execution-owned model and must preserve the
required fresh-model boundaries between folds and removal evaluations. They do not authorize changes to fixed
configuration, dataset, partition, or pristine reconstruction state.

The component map describes responsibilities and dependencies, not a required class hierarchy.

## 2. System Context and Responsibilities

| Component | Initial responsibility |
| --- | --- |
| User | Supply the dataset, sample metadata, model inputs, and evaluation directives. |
| Root/coordinator | Own computation/task coordination; prepare local data access or online synchronization, assign tasks, and accept persisted results. |
| Worker | Serve as an explicitly configured root-local or online execution target; evaluate complete unit-task attempts, durably publish standardized result bundles, and deliver only the selected bundle. |
| External public-worker service | Own public discovery, worker selection, reservation/allocation, and availability. |

Workers execute independently without inter-worker communication or dependence on shared mutable worker state.
Internal evaluation state remains local. The coordinator may maintain scheduling, assignment, lifecycle, failure,
and result-acceptance state; this does not make workers depend on each other's execution state.

Make the exact dataset ready before tasks use it, through online synchronization or local artifact access as applicable;
then supply model inputs and execution directives for subsequent tasks.
Each completed attempt produces one immutable result bundle. Duplicate attempts remain distinct execution outcomes,
but the root durably accepts at most one selected bundle for each model/unit under Section 5.

### 2.1 Internal component and responsibility map

## Root Node
Root control component
    Owns the reusable computation configuration and the lifecycle of each
    parent model-evaluation operation created from it.
    Coordinates the root components.
    Owns batch-level running logs and error logs.
Root coordinator component
    Owns logical task state and scheduling.
    Owns batch/model progress tracking.
    Preserves each parent task's fixed evaluation context, including its
    complete realized partition schedule, for dispatch and recovery.
    Owns recovery/resume state and scheduling decisions after reconciliation.
Data splitting component
    Constructs and validates the complete realized train/test partition
    schedule within root-side `add_dataset()` preparation. Its complete set is
    durably persisted once before any unit-task allocation.
    Provides the supported regression and classification splitting methods,
    including grouped, stratified, K-fold, LOO, and LOGO variants as
    applicable.
    Resolves splitting randomness during dataset/partition preparation;
    parents bind the current validated schedule without regenerating it.
    Does not train models or compute evaluation reports.
Root connection component
    Applies only when online workers are used; local dispatch uses the
    root control component's direct local execution/control boundary.
    Owns online-worker connection-information retrieval.
    Owns connection establishment, authentication support, reconnection,
    and connection state.
Root communication component
    Owns network protocol communication with connected online workers.
    Owns task delivery, heartbeat/status exchange, control instructions,
    result reception, persistence acknowledgements, and resend requests.
Root result persistence component
    Owns durable local persistence of selected result bundles.
    Determines whether persistence succeeded.
    A result becomes accepted only after atomic publication of the complete bundle.
    Persistence failure is reported through local control or the online
    communication component so the applicable retry lifecycle can proceed.

## Worker Node
Worker communication component
    Owns communication through the applicable local or online root boundary.
    Owns status exchange and receipt of tasks and control instructions;
    network heartbeat/reconnection applies only to online workers.
    Remains responsive independently of model-evaluation execution.
Worker control component
    Owns admission and lifecycle tracking for assigned model-evaluation attempts.
    Owns Cocoonet attempt-slot reservations and evaluation subprocess supervision.
    Coordinates each attempt's synchronization, evaluation, cancellation,
    result availability, and delivery independently.
    Owns worker-side running logs and error logs.
    Evaluation/operation failures must not terminate the worker control
    process.
Dataset synchronization component
    Owns dataset identity validation, local artifact readiness, storage,
    invalidation, and cleanup; online workers additionally require network
    synchronization.
Model synchronization component
    Owns artifact preparation and local storage of model reconstruction data,
    supplied model data, configuration, and initial weights as applicable.
    Online workers additionally require network synchronization.
    Owns synchronized-model readiness, invalidation, and cleanup.
    Model-type-specific semantic validation belongs to model initialization,
    not synchronization.
Workers consume the parent's supplied realized partitions. They may validate
sample references against the ready dataset but must not independently
regenerate, reshuffle, or alter fold memberships.

## Model Initialization
sklearn model initialization component
    Owns sklearn reconstruction-metadata validation, model reconstruction,
    and initialization.
sklearn-like model initialization component
    Owns validation of the required sklearn-like model interface and
    initialization of supplied custom models.
PyTorch model initialization component
    Owns PyTorch model reconstruction and initial-weight initialization.
    Owns initialization of the optimizer, scheduler, loss/training
    configuration, and other required PyTorch training state.

## Model Execution
Model execution component
    Owns fitting an initialized model on an assigned training set and
    producing standardized predictions for an assigned evaluation set.
    Does not own evaluation metrics or report construction.
    Preserves user-configured model-internal parallelism, including supported
    n_jobs settings, under the execution and resource policy in Section 8.

## Evaluation Components
Validation result component
    Owns standardized validation/fold result construction.
Performance report component
    Owns performance-metric computation and standardized performance
    reports.
Residual analysis component
    Owns residual-analysis computation and standardized residual reports.
Influence analysis component
    Owns the defined influence-analysis computation.
    May invoke repeated model execution where required by the influence
    definition.
Visualization data component
    Owns generation of standardized plotting data, including ROC-curve
    data for classification and scatter/residual-plot data for regression.
    Plot-data generation is separated from general model execution.
Worker result persistence component
    Owns construction, validation, and durable publication of one immutable
    result bundle for an attempt. Evaluation components do not independently
    own filesystem persistence.
Result bundle component
    Collects the applicable XLSX result workbooks, visualization-data dill
    files, model reference, error log, and manifest for one attempt.
    Individual bundle members have no independent delivery, retry,
    acceptance, or recovery lifecycle.

## Model Evaluation Workflow
sklearn evaluation workflow component
    Orchestrates the complete evaluation of one sklearn model using the
    corresponding synchronization, initialization, execution, evaluation,
    persistence, and packaging components and the parent's realized partitions.
sklearn-like evaluation workflow component
    Orchestrates the complete evaluation of one sklearn-like custom model
    using the corresponding components.
PyTorch evaluation workflow component
    Orchestrates the complete evaluation of one PyTorch model using the
    corresponding components.
    CPU attempts may run concurrently through Cocoonet; GPU attempts do not
    use Cocoonet-level multiprocessing. PyTorch-managed loading/training
    parallelism remains independently configurable under Section 8.

## Delivery
Delivery component
    Owns delivery and retry of the selected immutable bundle. Online delivery
    reuses the same `.tar.zst` package; root-local delivery reuses the same
    worker-published directory. Delivery never recomputes or reconstructs
    scientific results. Delivery failure is contained within the defined
    delivery/attempt lifecycle and does not terminate the worker process.

### 2.2 Private and public worker connections

Every online worker accepts only the versioned Cocoonet protocol and serves at most one authenticated root/user at a
time. Connection direction never transfers scheduling or coordination authority from the root.

For a private/non-public worker, users configure both root and worker endpoints and select `auto`, `root_connects`, or
`worker_connects`. Initial connection and reconnection follow that policy. `auto` attempts both directions. The root
accepts the first authenticated, exact-version-compatible connection for the configured `worker_id` and closes
duplicates. If no permitted direction succeeds, the worker remains unavailable; `run()` fails startup when no usable
configured execution target remains. Private workers never appear in public discovery and remain under explicit user
lifecycle control.

Once a connection is established, the root initiates reconciliation and all task and lifecycle instructions.

For a public worker, the root requests an allocation from the external service, receives connection information, and
initiates the connection. Public discovery, reservation, allocation, authentication, and release policies remain owned
by that service. The public-worker service is outside Cocoonet's scope, for example a separately planned
`CocoonService`. Pool state remains internal to that service; Cocoonet receives connection information rather than
managing the pool.

Only authenticated, authorized CocoonService developers or operators may provision or register public-pool workers.
Ordinary users and ordinary worker credentials cannot add workers to the public pool. A locally configured worker
cannot make itself public through a configuration flag. CocoonService is authoritative for public-pool membership;
users may only request and use workers allocated by that service.

Cocoonet may send lifecycle notifications required by the supplied interface, such as release notices. Reservation
durations, timeout-based release, and credential-renewal workflows for the public pool are not Cocoonet policies. Do not
invent the endpoint, authentication, request/response format, connection-information format, or notification protocol
before the external interface is supplied for implementation.

Private and public workers use the same direct root-worker evaluation protocol. The public service does not proxy
datasets, models, tasks, or results. Workers never communicate directly with other workers. Availability for another
task under the owning root does not itself release a public allocation.

Transport is defined in Section 7; worker-side cleanup and its service boundary are defined in Section 5.3.

### 2.3 Reusable computation, initialization, and user workflow

A **computation instance** holds root-side dataset, model, worker, and evaluation configuration; it is not a parent
task. The first evaluation establishes a parent. Thereafter, changed model evaluation configuration requires a new
parent, while unchanged model evaluation configuration resumes the existing parent, with or without worker
configuration changes. An ordinary copy preserves those same identity rules.

#### Root directory and initialization

Construction requires a user-provided root working directory. `overwrite` defaults to `False`. The root working
directory is Cocoonet-managed persistent storage. User-provided dataset source paths are user-owned and must remain
outside it.

Before creating, changing, or deleting persistent files, resolve the proposed root to its canonical physical location
and reject it if it equals, contains, or is contained within a filesystem anchor, a Python installation or
package-installation directory, the installed Cocoonet package directory, or the complete Cocoonet source repository
tree.

Enforce the following ownership invariant for every configured dataset source path:

```text
dataset_path == root_dir                   -> fail explicitly
dataset_path is contained within root_dir  -> fail explicitly
dataset_path is outside root_dir           -> passes the ownership check
```

Validate this invariant in both configuration directions:

- When adding a dataset or changing a source path, validate it against the current root working directory.
- When initializing, changing, or reassigning the root directory, validate the proposed directory against every
  existing configured dataset source path before modifying persistent files. This includes dataset configuration
  retained by a copied instance.

A conflicting root-directory configuration fails explicitly without creating, changing, or deleting persistent
files. A conflicting dataset configuration also fails explicitly. This is a computation-configuration invariant,
not a check confined to `add_dataset()` or `initialize()`. Outside paths must still satisfy other dataset requirements.
Equality and containment concern canonical physical filesystem locations.

For example, if a copied instance retains source `E:/data/source.csv`, assigning `E:/data` as its root must fail before
directory preparation, even with `overwrite=True`.

After ownership validation succeeds, apply the existing directory preparation rules:

| Selected directory | Constructor behavior |
| --- | --- |
| Does not exist | Create/prepare it; state becomes `initialized`. |
| Exists and is empty | Accept it; state becomes `initialized`. |
| Exists and is nonempty, `overwrite=False` | Reject initialization. |
| Exists and is nonempty, `overwrite=True` | Explicitly replace its contents and prepare it; state becomes `initialized`. |

Overwrite is an explicit opt-in to destructive fresh-directory preparation. `initialize(...)` similarly establishes
the filesystem boundary for a fresh execution, possibly using a new root directory. It must not silently adopt old
persisted execution state as belonging to changed configuration. Exact initialization arguments remain deferred.
Recovery never clears or freshly initializes the existing execution's directory. Because configured dataset sources
are outside the managed root, clearing/replacement needs no dataset-path exclusions. Existing partition persistence
and recovery protections still apply.

#### Configuration and consolidation states

Ordinary configuration operations modify instance attributes in memory without creating, changing, or deleting files.
Incomplete configuration is permitted: dataset registration may be pending and model/worker counts may be zero.
`add_dataset()` is an explicit artifact-preparation exception: it generates/validates and persists a new authoritative
split, or loads/validates an existing one. This is partition artifact preparation, not persistence of the parent's
execution configuration. `run()` alone commits that execution configuration and makes the instance consolidated.

The dataset artifact-preparation exception also covers managed `root_data_reference.xlsx` generation and reusable
compressed or sharded transfer representations. These operations do not consolidate execution configuration.

`run()` from `initialized` validates prerequisites, atomically persists the fixed execution configuration in
`execution_configuration.csv`, establishes a fresh parent, and begins the workflow. After configuration is committed
to execution, state becomes `consolidated`. Durable mutable observations are stored in `execution_state.csv`. `run()`
is the sole boundary converting mutable configuration into persisted execution configuration; it is not the sole
writer of execution artifacts. Partition information, lifecycle/recovery state, results, reports, and failure
information are persisted by their respective workflow operations.

Model evaluation configuration comprises the dataset and realized partition, supplied models/model inputs, and
evaluation settings. A permitted change to that configuration after consolidation makes the instance
`unconsolidated`, whether constructed, copied, or reused, and requires a new parent. `run()` rejects unconsolidated
state: explicitly initialize a directory for fresh execution first. An empty directory does not implicitly change
the object's state. Edits before first run leave the instance initialized.

Worker configuration, including additions, removals, and local/online topology, is separate. Worker-only changes
neither make the instance unconsolidated nor change parent identity. With unchanged model evaluation configuration,
`run()` resumes the existing parent and preserves its fixed scientific inputs, progress, results, and retry limits,
with or without worker changes. Calling `run()` again does not itself establish a new evaluation. Worker configuration
changes remain subject to the ordinary execution lock and post-interruption rules; runtime use still requires
availability, ownership, readiness, and applicable cleanup.

```text
construct / initialize(root_dir, ...) -> initialized
initialized + configure in memory    -> initialized
initialized + run()                  -> persist configuration -> fresh parent -> consolidated
consolidated + unchanged evaluation  -> same parent / resume, with or without worker changes
consolidated + permitted evaluation edit -> unconsolidated
unconsolidated + run()               -> reject
unconsolidated + initialize(...)     -> initialized
```

Each parent is one fixed scientific execution configuration and one coherent result set. Its allocated units and
attempts retain that configuration for their lifetime. Prospective changes cannot propagate into old work or mix
configurations in one parent's results. Ordinary active execution keeps configuration changes locked. After root
interruption, prospective edits are permitted without waiting for old workers to reach their cooperative stop
boundaries. Their allocated work retains its fixed configuration. Configuration consolidation, execution settlement,
and worker inactivity are distinct: permission to edit is not permission to reuse an active worker or clear its data.

#### Copying, reuse, and recovery

An ordinary copy reproduces instance attributes, including current configuration, root directory, and lifecycle state.
It does not allocate a new directory or parent, clear files, reset state, or clone live workers, connections, attempts,
or accepted result bundles. A consolidated copy remains consolidated and refers to the same persisted execution.
Model evaluation configuration edits to either copy or original follow the same transition to unconsolidated;
worker-only edits preserve the existing parent. Copying alone creates no transition.

Unchanged model evaluation configuration has resume semantics, including after worker changes. Recovery preserves
existing parent identity, scientific configuration, partition association, results, and lifecycle evidence.
Fresh initialization for changed model evaluation configuration prepares a separate fresh-execution
filesystem boundary; it is not a recovery operation. Exact recovery APIs, copy mechanics, and history presentation
remain implementation details. Copies referring to the same persisted execution remain subject to the same
coordinator ownership, task identity, and at-most-one acceptance invariants; copying does not create an independent
execution or authorize uncoordinated controllers of the same state.

Dataset registration and splitting use `add_dataset()` as specified in Section 3.4. One current dataset/split
association is active at a time. Re-registration may replace it when configuration is mutable, but may not overwrite
an artifact still required by an existing parent. Later instances and parents can reuse valid persisted partitions
through `split_path`. Model-only changes do not invalidate the dataset/split itself, even though they require
reinitialization before a fresh execution under the consolidation lifecycle.

Each computation working directory represents one parent execution and contains one current managed
`root_data_reference.xlsx`. When users provide changed source tables, Cocoonet generates a new reference file for the
changed dataset and its new parent.
Do not replace a reference file still required by a resumable parent. Use a fresh working directory when the
previous execution must remain resumable; explicit fresh initialization and overwrite retain their existing
managed-directory semantics.

The normal workflow is construction/initialization, in-memory configuration and dataset/split preparation, fresh run,
execution, and subsequent permitted reuse or recovery. No exact method ordering is required beyond dependencies.

| Area | Required capabilities |
| --- | --- |
| Initialization | Prepare a root directory explicitly for fresh execution, with overwrite behavior. |
| Dataset | Add/set with partition preparation or validated reuse, inspect, remove configuration references. |
| Models | Add one/multiple, list/inspect, summary, remove one/multiple. |
| Workers | Add one/multiple, list/inspect, summary, configuration-only remove one/multiple; separate dataset/all-user-data cleanup and public release. |
| Execution | Fresh run, break/cancel, resume/recovery of existing persisted execution. |
| Reuse | Permitted configuration edits, ordinary copying, explicit reinitialization for changed model evaluation configuration; worker-only changes preserve the parent. |

Singular mutations are primitive semantic operations; bulk forms are explicit conveniences. Define commit/failure
behavior before implementation so validation failure cannot silently leave undocumented partial configuration.
A general root persistent-file clear API is not required; file management remains separate, with ownership boundaries
in Section 6.1. Exact unestablished signatures, aliases, containers, and exceptions remain implementation details.

### 2.4 Execution targets and local/online boundaries

Networking is required only for online workers. Local-only execution must work without initializing the online-worker
network subsystem. Online authentication, heartbeat/reconnection, network synchronization, and result transfer apply
only to online targets. The task, scientific, isolation, persistence, and acceptance contracts remain shared.

The root always owns coordinator responsibilities. It does not automatically act as a worker. An explicitly configured
root-local worker and configured online workers are independently optional; supported topologies are local-only,
online-only, and mixed local/online execution. Root coordination itself does not provide an implicit execution target.

Zero configured workers is valid before run. `run()` requires a dataset, one or more models, and at least one usable
execution target before evaluation begins. With no configured target, fail explicitly; do not run models in the
coordinator, automatically enable a local worker, construct a fake network worker, or wait indefinitely for an
unconfigured worker. This is configuration/startup failure, not baseline model-training failure.

A configured worker is a configuration entry; a runnable/available worker also meets applicable connection, readiness,
and ownership conditions. `run()` uses currently runnable targets without waiting for all configured online workers.
If none is runnable, fail startup explicitly. Configured disconnected workers remain under the independent periodic
connection policy in Section 7.4; a later run may succeed after one becomes runnable. An unavailable worker does not
prevent other ready workers from receiving tasks.

When preflight is requested, `run()` may create a temporary root-local worker process for a real trial on selected
configured inputs. Preflight does not execute user model code in the coordinator and does not make that worker an
implicit full-evaluation target.

The root-local worker is a separate process started with the standard-library multiprocessing `spawn` context. Root
and worker use one duplex `Pipe`; bounded control messages use `Connection.send_bytes()` and `recv_bytes()` with
explicit dill protocol 5 serialization. User model code must not run inside the root coordinator. Do not use WSS
loopback or another fake network path to imitate an online worker.

Dataset files and result bundles never travel through the pipe. Large trusted model payloads may be persisted once as
designated dill files and referenced through managed logical paths. Root coordinator storage and root-local worker
storage remain distinct ownership scopes. The worker uses an explicitly configured work directory or defaults to a
managed `cocoonet/` child of the worker service's console working directory.

The local worker is an execution target using the same initialization, execution, scientific-reporting, and failure
contracts as online workers, not a second evaluation implementation. It consumes the same parent configuration,
exact dataset identity, persisted partition memberships, unit/attempt identities, and model execution restrictions.
Cocoonet supplies local data/model artifacts through the framework boundary; models still have no direct filesystem
or network permission. Local artifact access must resolve the same logical source/sample associations as online
synchronization.

The worker validates and atomically publishes its result bundle in worker storage, then reports its identity and
managed path. The root copies only the selected bundle into root staging and atomically publishes it. Copy failure
retries that same published bundle without reevaluation. Root-local acceptance still requires successful durable root
publication; the root does not repeat worker-side scientific or result validation.

Root-local execution uses no networking, authentication, archive compression, HMAC, or online reconnection. Orderly
root shutdown stops and joins the worker. Unexpected pipe closure causes it to exit at the next cooperative boundary.
After root-process termination, a local worker is recreated rather than reconnected. The enforced model restrictions
in Section 7.3 remain necessary in addition to the separate process.

In mixed execution, one coordinator schedules both local and online targets through the same logical task system.
A local attempt and an online attempt of the same unit obey identical reception selection, at-most-one acceptance,
persistence confirmation, and duplicate-stop rules. Topology does not grant priority or create a separate result
lifecycle. The control/handoff mechanism differs; scientific execution and logical outcome semantics do not.

## 3. Dataset and Model Contracts

### 3.1 Dataset and sample identity

A configured computation has exactly one active dataset, shared by all unit tasks of each parent created from it.
It may have no active dataset while configuration is incomplete, but cannot run a parent without one. A valid
retained representation may serve successive parents under the same `dataset_id`; see Section 3.4 for parent scope.
Worker evaluation requires feature data (`X`), true target values, module-generated sample IDs, and root-assigned
sample labels supplied by the user. Group-based splitting also requires sample group assignments. Classification
requires class identities and an explicit association between classes and probability columns.

Within the active dataset, each `sample_id` uniquely identifies a sample for internal computation, alignment, and
fold membership. Each sample's `label` is its unique user-facing identifier. Preserve the one-to-one mapping between
these identifiers through evaluation and reporting, independently of row ordering. User-facing sample reports use
`label`; internal records retain `sample_id` for alignment and traceability. Neither identifier establishes a global
or cross-dataset sample identity. A sample label is distinct from its target/class label: for example, sample
`sample_id = 42` may have user label `specimen-A` and true class `positive`.

The local module assigns a unique `dataset_id` to each exact dataset representation, including the row-wise
association of sample IDs, labels, features, targets, and related metadata. This key identifies synchronized data for
worker-side synchronization, cache reuse, and retention. A changed representation requires a new key regardless of
its descriptive label. A dataset label is not a substitute for `dataset_id`.

Generate `dataset_id` as exactly 12 lowercase hexadecimal characters from a cryptographically secure random source
equivalent to `secrets.token_hex(6)`. Check for collision at registration. Reject any later association of the same
identifier with different dataset, sample, or source metadata. Do not derive it from dataset size or row count.

Before model evaluation begins on a participating worker, establish the parent's dataset readiness through online
synchronization or the local artifact boundary as applicable.
All unit tasks of that parent share this ready dataset. Subsequent transfers carry model information and evaluation
directives in the established parent context, without retransferring or separately identifying the dataset in each
unit task. A newly participating worker must satisfy the same readiness requirement before executing unit tasks.

According to the configured dataset-retention policy, a worker may retain the dataset after the parent task finishes.
A later model evaluation task requiring the same `dataset_id` may reuse that ready representation without another
synchronization. If data has been cleared, make it ready again through the applicable online/local boundary before use.
Mandatory release cleanup still
applies before serving another user, as specified in Section 5.3.

The canonical dataset abstraction is sample-oriented and tabular. The user supplies the unique sample label;
Cocoonet generates `sample_id`. `sample_label` below denotes the same user-facing `label` described above, not a third
identifier. Training and testing populations use the same underlying sample structure. Conceptual input forms are:

```text
Ungrouped, unmasked:
  Inline features: sample_label | y | X0 | X1 | ... | Xn
  Inline tensor:   sample_label | y | shape | X0 | X1 | ... | Xn
  File-backed:     sample_label | y | X_file

Ungrouped, masked:
  Inline features: sample_label | train_mask | test_mask | y | X0 | X1 | ... | Xn
  Inline tensor:   sample_label | train_mask | test_mask | y | shape | X0 | X1 | ... | Xn
  File-backed:     sample_label | train_mask | test_mask | y | X_file

Grouped, unmasked:
  Inline features: sample_label | group | y | X0 | X1 | ... | Xn
  Inline tensor:   sample_label | group | y | shape | X0 | X1 | ... | Xn
  File-backed:     sample_label | group | y | X_file

Grouped, masked:
  Inline features: sample_label | group | train_mask | test_mask | y | X0 | X1 | ... | Xn
  Inline tensor:   sample_label | group | train_mask | test_mask | y | shape | X0 | X1 | ... | Xn
  File-backed:     sample_label | group | train_mask | test_mask | y | X_file

product(shape) == number of flattened X values
```

The in-memory and persisted dataset dataframe use the same normalized grouped, masked structure, plus the internal
generated `sample_id`. Ungrouped input receives a distinct default group identity for each sample; unmasked input
receives `train_mask = 1` and `test_mask = 1`. These input defaults establish declared eligibility and do not repair
scientific metric values. Preserve the same normalized sample/label/group/mask associations in storage and execution.

`group` supplies group membership for grouped train-test splitting, grouped k-fold, and LOGO. The two masks are
independent eligibility constraints:
- `train_mask == 1` permits training selection; `train_mask == 0` excludes the sample from every training population.
- `test_mask == 1` permits testing selection; `test_mask == 0` excludes the sample from every testing population.

A permission is not a membership assignment or an instruction to include the row in every fold. Root split preparation
combines the configured splitting method with these constraints and records the resulting realized memberships.
Preserve train/test disjointness and applicable group separation. For example, augmented rows can permit training
but forbid testing, while reserved validation rows can permit testing but forbid training; their realized fold
placement accepts the configured splitting randomness. Workers consume the saved outcome rather than resplitting.
Groups and masks belong to the exact dataset representation and must agree with any reused partition's memberships.

Group separation applies to every realized fold, regardless of splitting method: no group may occur in both
training and testing populations. Every normalized sample has a group value. When groups are absent, distinct
per-sample groups make this rule equivalent to sample disjointness. Preserve supplied groups; do not replace
repeated groups to permit a split. Apply the same rule to newly generated and reused partitions. References
to "applicable group separation" and "group-based validation" do not provide an exemption for other methods.
Reject configurations that cannot satisfy both the selected splitting method and this invariant; do not
silently substitute another method.

Tensor reconstruction must be deterministic. Reject inconsistent shape/value information rather than silently
padding, truncating, repairing, or inferring replacement data. `X_file` denotes logical/dataset-relative source
reference(s), resolved through the representation's `dataset_id` and framework-managed local storage; a root-machine
absolute path is not presumed to exist on workers. The source-file and sample associations in Section 3.5 also apply
to inline representations. These examples describe input capabilities, not exact column names or Python containers.

Whether inline and file-backed rows may coexist in one dataset, and whether shape metadata is encoded per sample or
once for a uniform dataset, may be decided during dataset-schema/input-adapter design before dependent consumers.
Neither choice changes sample identity, saved partition membership, deterministic tensor reconstruction, or framework
ownership of data access. Mixed-row support is not implicitly promised; either representation must resolve the shape
and feature data of each sample unambiguously.

#### Dataset input and standardization

Users supply source input tables, not Cocoonet's managed `root_data_reference.xlsx`. Inline-feature and
inline-tensor datasets accept one or more input tables; file-backed datasets accept exactly one input
reference table.

Supported formats are CSV (`.csv`, `.csv.gz`, `.csv.zst`), XLS, XLSX, ODS, Parquet, ORC, and Feather.
Pickle/dill dataset inputs are prohibited. Read only the first worksheet of XLS, XLSX, and ODS files.
Automatically detect CSV encoding from decompressed CSV content.

Multiple inline inputs divide samples only. Each file contains complete, distinct sample records.
Column-wise assembly of one sample from multiple files is forbidden.

Initial ingestion identifies rows by `sample_label`, which must be unique across the complete dataset.
Reject a present label column containing any missing value. When the column is entirely absent, generate:

```text
#%_<logical-source-reference>_<one-based-initial-row-number>
```

Reserve `#%_` for generated labels and reject supplied labels using that prefix. Generate `sample_id` after
initial identification and preserve its association with the standardized label. Row ordering has no identity
semantics after this step.

The standardized field contracts are:

- Identifier-like values are text. Preserve text; convert integers directly to decimal text; reject floats
  and booleans. Validate uniqueness and associations after conversion and reject collisions.
- Classification targets are standardized as text. Preserve text and convert integers directly to decimal text; reject
  floating-point and boolean targets. After conversion, every distinct class label must match
  `^[A-Za-z0-9_-]{1,64}$`: only ASCII digits, uppercase and lowercase letters, `_`, and `-` are permitted. Reject
  invalid labels during initial dataset loading without stripping, escaping, replacing, or otherwise changing them.
  Reserve `micro-average` and `macro-average` for classification-performance aggregate rows and reject them as class
  labels.
  Reject classification input during initial dataset loading if its class count would make any required wide XLSX
  classification-result schema exceed the XLSX column limit. Do not silently switch to another report layout.
- Regression targets and inline X values are finite floating-point numeric data. Reject nonnumeric values,
  booleans, NaN, and infinities. The dataset API selects the floating-point representation.
- Standardized masks are integers exactly `0` or `1`, excluding `bool`. Each missing mask independently defaults to `1`;
  `1` permits the corresponding training or testing use and `0` prohibits it.
- Tensor shapes are ordered positive integer dimensions. Accept tuple, list, array, and textual forms and
  normalize them to the same shape semantics. Reject booleans, malformed text, noninteger dimensions,
  nonpositive dimensions, and unsupported structures.

Cocoonet does not impute, drop, replace, clip, or otherwise cleanse invalid or nonfinite dataset values.
Users must provide deterministic, valid input.

Dataset preparation generates one managed `<root_dir>/root_data_reference.xlsx` for the current dataset. It records
generated IDs, labels, groups, masks, targets, source associations, and tensor metadata. It is not user input.

For file-backed samples, source references identify the actual X-bearing files named by the input table.
For inline samples, the applicable input table is the source. Managed reference or metadata files are not
sample sources.

SourceReference contains a logical path only. Initial source-row matching uses the standardized sample label;
subsequent internal associations use sample_id. SourceReference contains no row locator.

Detailed data schemas, key encoding, and synchronization verification belong in the corresponding implementation
design. The sample-identity scope is the active dataset; cross-dataset identity tracking is not required.
Multiple-target and multilabel extensions are not established by this document; the influence formulas below
explicitly cover single-target regression.

### 3.2 Model identity and supported inputs

The root generates each `model_id` as a positive sequential decimal integer beginning at `1`, represented as text where
required by existing identity contracts, and associates it with a user-provided unique `model_label`. Do not zero-pad
or reuse a model identifier within the parent's persistent scope. Labels distinguish models within their parent model
evaluation task. Cocoonet
uses `model_id` internally for model references, task association, execution tracking, duplicate handling, and
provenance. User-facing reports identify the model by `model_label`; internal records may retain `model_id` for
traceability. Model identity is assigned, not derived from descriptive metadata or inferred equivalence of models.

The supplied model or model-construction information is the authoritative executable input. Cocoonet does not
derive, catalogue, or maintain separate user-level model metadata such as hyperparameter descriptions, preprocessing
provenance, or experiment lineage. Recording and interpreting that metadata remains the user's responsibility.
Construction and training inputs needed to execute the supplied model remain governed by the contracts below;
using parameters for reconstruction does not establish a separate descriptive metadata catalogue.

The model input associated with a unit task is fixed. Every execution attempt of that unit task uses the same
supplied model or construction information; an attempt must not substitute or redefine it. Fitting and fold-local
initialization operate under that input and do not redefine the unit task's model.

| Model family | Required input contract |
| --- | --- |
| Scikit-learn-like regressor | `fit` and `predict`. |
| Scikit-learn-like classifier | `fit`, `predict`, `predict_proba`, and required class information. |
| PyTorch model | Complete reconstruction/training inputs and the user's initialization choice; see Section 3.3. |

For standard scikit-learn models, identify the supported model type, obtain parameters with `get_params`, and
reconstruct/configure the corresponding worker-side model using `set_params` as appropriate. These parameters are
execution inputs. Custom scikit-learn-like models receive only the required API validation for their category;
do not inspect their deeper dependencies, implementation, or execution compatibility.

The custom-model path applies to root-local and private workers. Public workers accept only declarative
supported-component structures. Public restrictions do not remove private/local custom-model or trusted
dill-deserialization support.

Supported public-worker scikit-learn composition may include Pipeline, ensemble, boosting, meta-estimator,
and Cocoonet-native structures when fully declarative. Public workers reject custom Python classes,
arbitrary imports, callables, lambdas, and executable model objects.

These paths establish the information needed to attempt evaluation, not a guarantee that every supplied model will
execute successfully. Execute directly after applicable validation and reconstruction. Do not add compatibility
prediction, dependency negotiation or inspection, environment repair, fallback reconstruction, model substitution,
or corrective result manipulation. Actual reconstruction, fit, prediction, and other failures follow Section 9.

Detailed method argument/return shapes, payload schemas, and mechanisms for resetting fit state are specified with
the affected units. Preserve the configured preprocessing procedure and independence of learned state between folds.

### 3.3 PyTorch reconstruction and training

Supply all information needed to reconstruct and train the configured model. These are executable construction and
training inputs, not a requirement for Cocoonet to derive or maintain user-level model descriptions:

1. The model definition or resolvable reference, version, constructor arguments, architecture parameters, and custom
   layers/components.
2. Input and target schemas, tensor shapes and dtypes, preprocessing/encoding configuration, and output interpretation.
   Classification also supplies class identities/order and the conversion of model outputs to class probabilities.
3. The training procedure or resolvable reference, loss/criterion and settings, optimizer and settings, batch size,
   training duration/stopping rules, data ordering, and any scheduler or other procedure settings.
4. The user's initialization choice: supplied initial weights or a specified pattern, including random initialization.
   Include the pattern's parameters and seed when randomness is used. Transfer weight values only for supplied-weight
   initialization; otherwise reconstruct the chosen initialization on the worker.
5. Training/data-order randomness controls and the dependency/version information and custom definitions needed to
   resolve the model and training procedure. This metadata does not introduce dependency negotiation or repair.

#### Public-worker weight representation

For public workers, supplied weight values are transferred as non-executable Parquet model-data artifacts.
These artifacts belong to the applicable model/unit input. They are not dataset samples, do not participate
in `dataset_id`, and do not alter the parent's dataset or partition.

The Parquet representation must retain each tensor's declarative identity, ordered shape, dtype, element
ordering, and values so the supported worker-local model can reconstruct the supplied initialization exactly.
The exact column/container schema is defined with the Phase 10 model adapter.

Apply the tabular transfer thresholds to the original prepared weight artifact's stored size:

- At or below 16 MB, Parquet compression is optional.
- Above 16 MB, use ZSTD-compressed Parquet.
- Above 1 GB, calculate shard count using the configurable target-size policy, defaulting to 512 MB, and
  write multiple Parquet shards.

Weight sharding occurs only between complete tensor records unless the later schema explicitly defines a
deterministic tensor-chunk representation with complete reconstruction metadata. It must never change tensor
values or ordering.

Store prepared weights in a model-specific Cocoonet-managed directory on the root. Prepare them once for the
fixed unit-task model input and reuse them across attempts and worker transfers. Public workers validate the
declarative association before applying the weights.

Private and root-local workers retain the existing supplied-weight and serialized/custom-model paths and are
not required to use this public-worker Parquet representation.

For public workers, PyTorch definitions, components, training procedures, losses, optimizers, schedulers,
and initialization patterns must resolve through supported Cocoonet-controlled component identities and
declarative parameters. Public workers reject custom classes, arbitrary import references, custom callables,
and serialized Python objects. Root-local and private workers retain the broader trusted reconstruction
contract above.

Device selection belongs to worker configuration. Workers may use different supported device types. Shared seeds and
recorded configuration support auditability without promising identical numerical results across devices.

### 3.4 Model evaluation tasks, unit tasks, and execution attempts

Capture `created_at_utc_ns` once as a nonnegative UTC Unix timestamp in integer nanoseconds when the parent is created.
Generate:

```text
parent_id = sha256(
    b"cocoonet-parent-v1\0"
    + dataset_id.encode("ascii")
    + b"\0"
    + str(created_at_utc_ns).encode("ascii")
).hexdigest()[:16]
```

Persist the timestamp with the parent. Reject a colliding worker path associated with different parent metadata.

A **parent model evaluation task** is one fresh evaluation run created by `run()` from initialized state. It contains
exactly
one dataset, one fixed task-wide evaluation method/configuration, and one or more models. The parent establishes
data-splitting,
validation, and other task-wide evaluation settings once. Individual models cannot override them: all models are
evaluated against the same dataset under the same evaluation configuration. Required model-specific reconstruction
and training inputs remain part of each supplied model and do not override the parent's evaluation settings.

Root-side `add_dataset()` establishes the dataset and its complete realized partition schedule together. Validate
dataset source paths against the current root working directory under Section 2.3 before accepting the configuration
or persisting a new partition:

- With an existing `split_path`, load the persisted partition and validate it against the supplied dataset. Do not
  rerun the splitter.
- Without an existing split, require a splitting method and its required configuration. Generate and validate the
  complete realized partition set on the root and persist it before `add_dataset()` successfully completes. Its
  destination may be supplied explicitly; otherwise derive it from
  the computation's working directory, which is required at initialization.
- Before accepting the dataset, validate its exact representation and the paired internal `sample_id` and
  user-facing `sample_label` identities against the persisted/supplied partition. A matching ID with a different label,
  or a matching label with a different ID, is not the same association. Validate all training/testing memberships.
  Memberships must obey the normalized training/testing masks and applicable group separation in Section 3.1.
  Exact validation, serialization, and identifier-restoration mechanisms remain implementation details.

`run()` consumes the current validated association; neither a separate explicit split call nor automatic splitting at
run is required. Every unit and attempt within that parent uses the same realized memberships. Workers and models
must not regenerate, reshuffle, or alter them. Root-side partition randomness is accepted when the set is generated;
reuse preserves the actual memberships rather than regenerating them from a seed.

Immediately after generating the complete partition set, persist it once in the authoritative root
`partition_schedules.xlsx` before unit allocation. Worker copies are execution inputs and never competing authority.
The schedule table has:

```text
partition_id | dataset_id | fold_id | population | position | sample_id | sample_label
```

`population` is `training` or `testing`. `position` is the one-based position within that fold population. All rows in
the workbook belong to one `partition_id` and one `dataset_id`. Serialize folds in their established order; the order
of first occurrence of each `fold_id` is the persisted fold order. Within a fold, serialize training before testing and
preserve each population through `position`. Thus no separate `fold_index` is required and arbitrary opaque `fold_id`
values remain compatible with the existing contract.

Validate exact sample-ID/label associations, unique fold identities, population membership, masks, train/test
disjointness, and group separation before publication and after loading. Persistence must complete before any
unit-task allocation. Do not divide or duplicate the authoritative schedule or repeatedly persist it during normal
execution. Parent execution state references the dataset and this artifact; the artifact need not be rewritten to
attach every later parent.

A later `add_dataset()` may select an existing valid artifact or create another one. Replacing the computation's current
split association does not authorize overwriting a file still required by an existing parent. Parents preserve their
original dataset/partition associations for their lifetime. Another computation may reuse `split_path` only after the
same validation. The exact signature, default filename, in-file container, and validation mechanism remain deferred.

Supply a copy of the applicable indices together with online dataset synchronization, or local artifact preparation,
during parent-context setup. Workers use those indices to select data from the exact ready representation; they never
invoke the splitter. Worker copies are execution inputs, not additional authoritative root copies. Reusing a retained
ready dataset does not require retransmitting it, but the applicable parent partition context must still be supplied.

After a root restart following successful partition persistence, load and validate the existing
`partition_schedules.xlsx`. Do not rerun the splitter or create a replacement set. Keep the workbook associated with
the exact dataset representation for which the memberships were generated. The single authoritative workbook and
pre-allocation persistence gate are resolved architectural requirements.

A **unit task** evaluates exactly one model within that parent. Each supplied model produces one unit task,
distinguished within the parent by its unique `model_label`. Dataset identity and evaluation configuration belong to
the parent and are not independent parts of unit-task identity. Models intended as different evaluations within the
same parent must have different `model_label` values.

The worker establishes the parent's dataset and evaluation context before unit-task execution. Unit-task transfers
provide only the model and required model-specific information, together with lifecycle identifiers where needed.
They neither resend nor separately identify the dataset and cannot replace the parent evaluation configuration.
The parent association determines the active dataset and settings; a unit task cannot execute against another
parent's context.

Once established, the parent's dataset, evaluation configuration, realized partitions, and each unit task's model
input remain fixed for that model evaluation task. Changing the dataset/partition, supplied models/model inputs, or
task-wide evaluation settings requires a separate parent under Section 2.3, not mutation of existing unit tasks.
Worker changes, including addition/removal, do not change this evaluation configuration or parent identity; unchanged
evaluation resumes the existing parent.

An **execution attempt** is one worker execution of a unit task. Duplicate allocation creates another attempt of
that same unit task using the same model input, parent dataset, evaluation configuration, and realized partitions.
It creates neither a new unit task nor a new model evaluation task. Internal lifecycle identifiers may support
scheduling, attempt tracking, failures, recovery, and result acceptance; they do not require dataset identity to be
repeated in individual unit tasks.

Parent dataset/configuration/partition readiness and reusable root-side model-artifact preparation precede attempt
allocation and are outside the attempt lifecycle. After allocation, attempt processing begins when the worker starts
attempt-specific model synchronization or local handoff. The attempt then includes initial-model loading or
reconstruction with its reproducible construction state, evaluation and report generation, publication of the
complete worker-side result artifact, and delivery under the reception and persistence workflow in Section 5.

Fold computations and sample-removal evaluations remain internal to an execution attempt. Their partition definitions
come from the shared parent schedule. Fold results retain their producing attempt context and their association with
the corresponding parent-defined fold and training/test sample identities. A completed attempt produces one result
file associated with its unit task and producing attempt; at most one result is accepted per unit task under Section 5.
The existing one-model worker-task and logical-task lifecycle rules apply to unit tasks in this hierarchy. Initial
reconstruction failure ends that unit task's current attempt before primary validation. After successful
reconstruction, failure of any primary-fold model initialization, fit, or prediction produces a
`validation_failed` attempt while later independent folds continue where possible. Neither outcome automatically
fails the parent task or worker process.

### 3.5 Dataset source files and fold references

Every dataset representation contains at least one source dataset file. A source-file reference is a path relative
to that representation's dataset root. Together, `dataset_id` and the relative path identify the source file
independently of any root or worker filesystem location. The key can be resolved through the parent dataset context;
this reference relationship does not require a dataset identifier in each unit-task transfer.

A sample may derive from one file, multiple files, or a file shared with other samples. Preserve the association
between samples and their source files without assuming a one-to-one relationship.

Each fold separately provides one or more source-file references for its training population and one or more for its
testing population. These reference sets may overlap when different training and testing samples originate from the
same source file. Fold membership is determined by the evaluation split and internal sample identities, not file
membership. Referencing a file does not assign all samples contained in that file to the corresponding population.
For example, two samples from `source.csv` may belong to different partitions, so both reference sets contain
`source.csv` while their sample memberships remain disjoint.

Workers may map these logical references to arbitrary local paths for synchronization and execution. Reports
preserve the dataset-root-relative references so they remain interpretable by the root and user after worker
cleanup, without exposing or depending on machine-specific paths. Concrete path encoding and local storage layouts
belong in implementation design.

#### Reusable tabular transfer representations

Cocoonet generates worker-local `worker_data_reference.xlsx` using synchronized paths selected by Cocoonet.
Root-machine physical paths do not become worker physical paths.

For tabular transfer preparation, measure each original table independently by stored file size. Use decimal
MB and GB. Do not use decompressed size, memory use, or combined dataset size.

Convert tables larger than 16 MB to ZSTD-compressed Parquet unless already ZSTD-compressed Parquet.
For an unsharded conversion, remove the recognized suffix or compound suffix and append
`_compressed.parquet`.

For an original table larger than 1 GB, calculate:

```text
shard_count = max(2, ceil(original_stored_size / target_shard_size))
```

The configurable target defaults to 512 MB and determines shard count rather than a strict generated-file
limit. Divide complete rows among shards as evenly as possible.

Each source remains an independent preparation unit. Read and prepare one source at a time. Dataset-global
shard numbering does not authorize combining or simultaneously materializing multiple source files.

Store prepared files in a dataset-specific Cocoonet-managed root-node preparation directory. Name shards
`cocoonet_dataset_shard_<zero-padded-one-based-number>.parquet`, using numbering unique across the prepared
dataset.

Preparation operates on standardized data and must preserve its content and semantics exactly. Never modify
the original source. Establish a parent's prepared references before allocation and retain them unchanged.

Within a parent, reuse prepared files without repeating metadata checks. Across executions, automatic reuse
compares persisted source path, stored size, modification time, and preparation settings. Matching state
permits reuse; changed state requires regeneration. This is intentionally metadata-based rather than
content-digest verification.

A new computation instance without persisted source state regenerates by default. Explicit reuse and
regeneration remain later dataset-API choices. Prepared root-node files persist until manual OS removal and
are not worker-cleanup targets.

## 4. Worker Evaluation

### 4.1 Minimum task unit

One worker task is a complete evaluation process using the dataset and one supplied classification or regression model.
The worker runs all required folds or splits and generates the required reports, returning its immutable result bundle
when selected for reception by the root. Optional sample influence analysis belongs to that same evaluation when
requested.

A fold, a held-out sample, or an individual influence refit is not a separate distributed task in the initial design.
One completed attempt produces one immutable result bundle, including its failure information. Duplicate attempts
belong to the same logical task, with at most one accepted result under Section 5.2. Delivery retries resend the
existing package or reuse the published local bundle; they do not rerun the evaluation. Initial model reconstruction
failure produces a `reconstruction_failed` failure-only bundle without predictions or reports. Primary-validation failure
produces a complete `validation_failed` bundle containing preserved fold results, unavailable predictions, dependent
failed or blocked reports, and failure provenance.

### 4.2 Worker inputs

- The parent's ready dataset and sample/target metadata described in Section 3.1, resolved through the
  established parent context rather than a repeated dataset identifier in each unit-task transfer.
- One model input matching Section 3.2.
- The parent's fixed evaluation directives and complete realized partition schedule, including the training and
  testing sample memberships of every fold/split. All unit tasks and attempts use this same schedule.
- The parent-configured request for optional sample influence analysis, if enabled.

Supported strategies are k-fold, grouped k-fold, Leave-One-Out (LOO), Leave-One-Group-Out (LOGO), and train-test splits.
The root applies the selected strategy, required group/mask inputs, and supplied reproducibility settings during
`add_dataset()` when a new partition is required; an existing supplied split is validated and reused without splitting.
Workers resolve and validate the supplied sample memberships against
the ready dataset; they do not invoke split generation again. Influence analysis reuses those exact validation
partitions. Compatibility, including the online handshake, is defined in Section 7.2.

Specify the dataset-registration/split interface, schedule representation, and parent-context transfer schema
during the relevant design work. These details must preserve the actual partitions and supplied model/preprocessing
configuration without introducing extra determinism controls or a separate influence split schedule.

### 4.3 Evaluation flow and correctness constraints

The conceptual sequence is to resolve the parent dataset and realized partition schedule together with the unit's
model input, train and predict for each supplied split, assemble validation results, compute reports and optional
influence results, and construct one result bundle. Dataset/partition preparation has already completed before parent
allocation.
Deliver that bundle when selected for reception by the root. This sequence does not prescribe function boundaries
or authorize implementing multiple production units.

- Sample IDs must preserve the association between features, true targets, predictions, and every sample-level report.
- Held-out samples must not influence fitting or learned preprocessing for their validation fold. Learned model state
  must not leak between folds; the reset mechanism implements the reconstruction contracts in Section 3.
- Group-based validation must respect group separation between training and held-out samples.
- Probability columns must have explicit class identities and consistent alignment across folds and reports.
- Without masks, k-fold, grouped k-fold, LOO, and LOGO validation results cover all samples; train-test results cover
  all test samples. With masks, report the realized testing memberships permitted by `test_mask`, never fabricated
  predictions for testing-excluded rows. Training memberships obey `train_mask`. Fitted training predictions cannot
  substitute for held-out validation predictions.

- Reconstruct the initial model and its reproducible construction state before primary validation. For each fold,
  create a fresh untrained model, fit it on that fold's training population, predict its testing population, retain
  the aligned results and provenance, and release the fitted fold model. Learned state must not pass between folds.
- Initial reconstruction failure produces `reconstruction_failed` and prevents every scientific branch from
  starting. After successful reconstruction, failure of fold-model initialization, fitting, or prediction in any
  primary fold produces `validation_failed`. Preserve completed folds and continue later independent folds when the
  model-construction mechanism remains usable; do not retry failed folds.
- Primary validation must succeed without fold failures before ordinary reports and optional influence analysis begin.
  Ordinary reports and influence may then run independently or concurrently. A failure in either branch does not stop
  another branch whose prerequisites remain valid.
- Preserve identifiers and successful values. Formula-defined undefined values remain `NaN`; computationally
  unavailable values are `None` with applicable failure references. No application-level evaluation failure
  terminates the worker process.

Section 9 defines failure and formula fidelity rules. Specify output ordering, concrete fold representation, and
report aggregation conventions during the corresponding schema/report design using conventional metric definitions
and the explicit project variations below.

### 4.4 Fold outputs shared by both task types

Every result table is a separate XLSX workbook with exactly two worksheets. The first contains result data. The second
is a schema-controlled key/value metadata table containing at least `creation_datetime`, `parent_id`, `attempt_id`,
`model_id`, `model_label`, and `worker_id`, plus applicable dataset, partition, fold, artifact-kind, and schema
identities. Finite values are numeric cells. `Inf`, `-Inf`, `NaN`, and `NA` are deliberate text tokens under
Section 9.3;
spreadsheet-library missing-value inference is not authoritative.

Classification result workbooks use the applicable model's fixed class-label order. Preserve that exact order in every
class-dependent column group and record it in workbook metadata. Validate every fold and report against that same order.
Use each exact standardized class label as the column prefix and append `_probability`, `_residual`, `_z_score`, or
`_influence`. Class labels may themselves contain underscore-separated or suffix-like text. Column names must not be
parsed to recover class identity; the metadata mapping is authoritative.

For each classification workbook, metadata records the applicable model's exact ordered class labels and their
corresponding class-dependent column names. Validate the exact model class order, class-label set, expected columns and
their order, applicable suffix for each report field, and consistency of the same class mapping across validation,
residual, influence, performance, and visualization artifacts. Do not infer or reconstruct class identity by removing
or parsing a column-name suffix.

Sample-oriented result rows correspond to `sample_id`, but `sample_id` remains an explicit ordinary column and must not
be persisted only as a dataframe or worksheet row index.

Training and testing populations use one logical sample-data structure. Each record contains the internal
`sample_id`, the user-provided sample label, group, training/testing eligibility masks, dataset-relative source-file
reference(s), and the true target value or class label. A source reference describes the sample's underlying data in
either population; calling it a training
data filename does not restrict that sample to training. Preserve the one-to-many and shared-file relationships in
Section 3.5. Saved fold/split membership determines whether the sample is in training or testing for that fold;
do not define structurally different training-sample and testing-sample records.

- Fold `X` results provide source references separately for both populations, rather than returning feature arrays.
  Resolve those references through the parent dataset context and preserve the dataset-relative paths.
- Classification fold `y` results associate held-out samples with predicted class labels and probabilities for every
  class with explicit class identities, alongside their true class labels.
- Regression fold `y` results associate held-out samples with predicted target values alongside their true targets.

Predictions and other computed fields are evaluation results associated with the common sample records; they do not
redefine the source-data structure. Internal records retain `sample_id` for alignment, partition membership, and
provenance. User-facing reports identify samples by their user-provided labels. This distinction applies to all
sample-level reports in Sections 4.5 and 4.6. Keep sample/class associations intact within the immutable result bundle.

### 4.5 Classification reports

#### Validation results

Classification validation columns, in order, are:

```text
sample_id | sample_label | y_true | y_pred
| <class label 0>_probability
| <class label 1>_probability
| ...
| <class label n>_probability
```

`y_true` and `y_pred` contain categorical class labels. Each probability column contains the predicted probability for
its named class. Probability columns follow the exact model class-label order recorded in metadata.

#### Performance report

Classification performance columns, in order, are:

```text
class | TP | TN | FP | FN | precision | recall | f1_score | accuracy | AUC
```

Class rows follow the exact model class-label order, followed by `micro-average` and `macro-average`. Aggregate rows
contain averages only for `precision`, `recall`, `f1_score`, `accuracy`, and `AUC`; `TP`, `TN`, `FP`, and `FN` are
non-applicable `NA`. Do not store a separate duplicate confusion-matrix table. The complete confusion matrix is
derivable from the authoritative validation rows.

Accuracy has three explicitly defined scopes:

- Classwise overall accuracy: `(TP + TN) / (TP + TN + FP + FN)`, using each class's one-versus-rest confusion counts.
  `TN` denotes true negatives; `TF` was an earlier notation error.
- Micro-average overall accuracy: `number of true predictions / sample size`, counting correctly predicted class
  labels in the evaluated sample population.
- Macro-average overall accuracy: the arithmetic mean of the classwise overall accuracies of all classes.

Different values across these scopes in multiclass classification are expected. Preserve each definition rather than
replacing it with another scope's quantity. Other standard metrics use their established conventional definitions
unless an explicit Cocoonet variation applies. Detailed formulas and report conventions can be recorded during the
corresponding planning/implementation work; missing detail alone is not an unresolved standard-metric definition.
Undefined values remain `NaN`, without repairs or NaN-skipping aggregation, under Section 9.3.

#### Residual report

Classification residual-analysis columns, in order, are:

```text
sample_id
| sample_label
| y_true
| <class label 0>_probability
| ...
| <class label n>_probability
| <class label 0>_residual
| ...
| <class label n>_residual
| <class label 0>_z_score
| ...
| <class label n>_z_score
| sum_of_absolute_residual
| root_sum_of_squared_residual
| Shannon_entropy_of_absolute_residual
| Mahalanobis_distance_of_residual
```
Each probability, residual, and z-score column group follows the model’s exact class-label order recorded in workbook metadata.

For sample `i` and class `c`:

- `q_{i,c}` is the predicted probability.
- `t_{i,c}` is the one-hot true-class indicator.
- The signed probability residual is:

```text
r_{i,c} = t_{i,c} - q_{i,c}
```

Therefore:

- `<class label c>_probability` contains `q_{i,c}`.
- `<class label c>_residual` contains `r_{i,c}`.
- `<class label c>_z_score` contains the z-score of `r_{i,c}`, calculated across all evaluated samples for class `c`.

##### Probability-residual z-score

For the `N` evaluated samples:

```text
mu_c = (1/N) * sum_{i=1}^{N}(r_{i,c})
```

```text
sigma_c = sqrt((1/N) * sum_{i=1}^{N}((r_{i,c} - mu_c)^2))
```

```text
z_{i,c} = (r_{i,c} - mu_c) / sigma_c
```

This uses population standard deviation (`ddof=0`). If `sigma_c = 0`, the z-scores for that class are `NaN`.

The user-configurable missing-value policy has two behaviors:

- `propagate`, the default: if any residual in the class vector is `NaN`, every z-score derived from that vector is `NaN`.
- `omit`: exclude `NaN` residuals when calculating `mu_c` and `sigma_c`. A sample whose own residual is `NaN` still receives a `NaN` z-score. If insufficient valid values remain or the resulting standard deviation is zero, the affected z-scores are `NaN`.

##### Sample-level residual summaries

For each sample:

```text
sum_of_absolute_residual_i = sum_{c=1}^{C}(|r_{i,c}|)
```

```text
root_sum_of_squared_residual_i = sqrt(sum_{c=1}^{C}(r_{i,c}^2))
```

These calculations propagate `NaN`.

##### Normalized Shannon entropy of absolute residuals

For sample `i`:

```text
p_{i,c} = |r_{i,c}| / sum_{k=1}^{C}(|r_{i,k}|)
```

```text
H_{norm,i} = -(1 / log_b(C)) * sum_{c=1}^{C}(p_{i,c} * log_b(p_{i,c}))
```

Use `0 * log_b(0) = 0`. The logarithm base does not affect the normalized result when the same base is used throughout.

If all residuals are zero, define:

```text
H_{norm,i} = 1
```

This explicitly represents the equal distribution of absolute residual magnitude as maximum normalized entropy. If any residual required by the calculation is `NaN`, the entropy is `NaN`.

##### Mahalanobis distance of signed probability residuals

Let `N` be the complete evaluated sample population represented in the residual report. For sample `i`:

```text
r_i = t_i - q_i, where r_i is in R^C
```

Define the mean residual vector:

```text
r_bar = (1/N) * sum_{j=1}^{N}(r_j)
```

and the sample covariance matrix:

```text
S = (1/(N-1)) * sum_{j=1}^{N}((r_j - r_bar) * (r_j - r_bar)^T)
```

The reported distance is:

```text
d_M(r_i) = sqrt((r_i - r_bar)^T * S^+ * (r_i - r_bar))
```

where `S^+` is the Moore–Penrose pseudoinverse.

The following behavior applies:

- When `N < 2`, every Mahalanobis distance is `NaN`.
- A sample containing any `NaN` residual receives a `NaN` distance.
- With the default `propagate` policy, any invalid residual vector prevents covariance estimation and all distances are `NaN`.
- With `omit`, exclude complete sample residual vectors containing `NaN` from mean and covariance estimation. Do not omit individual class entries from a vector.
- If fewer than two complete residual vectors remain after omission, all distances are `NaN`.
- If covariance or pseudoinverse computation fails, all Mahalanobis values are computationally unavailable and carry the applicable failure information.
- A singular covariance matrix is not itself a failure; use its Moore–Penrose pseudoinverse.

Use established numerical implementations such as SciPy where applicable, while preserving these formulas and edge-case rules.

##### Reference computation

The reference code illustrates the computation and uses the required persisted column names:

```python
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import entropy

# Inputs use shape (samples, classes).
residuals = y_true_proba - y_pred_proba

report = pd.DataFrame(
    {
        "sample_id": sample_ids,
        "sample_label": sample_labels,
        "y_true": y_true,
    }
)

# Probability, residual, and z-score groups preserve class-label order.
for class_index, class_label in enumerate(class_labels):
    report[f"{class_label}_probability"] = y_pred_proba[:, class_index]

for class_index, class_label in enumerate(class_labels):
    report[f"{class_label}_residual"] = residuals[:, class_index]

for class_index, class_label in enumerate(class_labels):
    class_residuals = residuals[:, class_index]
    report[f"{class_label}_z_score"] = (
        class_residuals - np.mean(class_residuals)
    ) / np.std(class_residuals, ddof=0)

# Sample-level residual summaries.
report["sum_of_absolute_residual"] = np.sum(np.abs(residuals), axis=1)
report["root_sum_of_squared_residual"] = np.sqrt(
    np.sum(residuals**2, axis=1)
)

maximum_entropy = np.log2(len(class_labels))
report["Shannon_entropy_of_absolute_residual"] = [
    1.0
    if np.all(row == 0)
    else entropy(np.abs(row), base=2) / maximum_entropy
    for row in residuals
]

# Mahalanobis distance across sample residual vectors.
mean_residual = np.mean(residuals, axis=0)
covariance = np.cov(residuals, rowvar=False, ddof=1)
inverse_covariance = np.linalg.pinv(covariance)

report["Mahalanobis_distance_of_residual"] = [
    mahalanobis(row, mean_residual, inverse_covariance)
    for row in residuals
]
```

The capitalization of `Shannon_entropy_of_absolute_residual` and `Mahalanobis_distance_of_residual` is intentional for these user-facing report columns. Internal Python variables and functions continue to use lowercase snake case.

Document the conventional per-measure formulas, encoding, sign, and report conventions with the relevant report
implementation. Any project-specific variation requires an explicit definition. Formula-defined undefined values
remain `NaN`; report their risks without adding fallback calculations.

#### Visualization data

Return data for plotting ROC curves, including AUC for each class. Return plotting data rather than rendered plots to
reduce transfer load. Specify ROC array layouts and references to corresponding performance-report fields during
report-schema implementation.

#### Optional sample influence report

Classification sample-influence columns, in order, are:

```text
sample_id
| sample_label
| prediction_comparison_count
| <class label 0>_influence
| <class label 1>_influence
| ...
| <class label n>_influence
| sum_of_influence
| root_sum_of_squared_influence
```

`prediction_comparison_count` is the exact per-sample comparison count required by the approved influence aggregation
formula. It appears immediately after `sample_label`, before every influence-measure column. Influence columns follow
the exact model class-label order.

Report influence for training samples only, using the same sample structure as testing samples. Associate each omitted
sample's internal `sample_id`, user-facing label, source references, and true class label with one generalized Cook's
distance-like influence value per class. The score
describes the normalized effect of removing that training sample on fixed test-set probability predictions.
The predictions used for this computation belong to the test samples; no prediction for the omitted training sample
itself is required. Test samples receive no influence score from a fold in which they are held out. In cross-validation,
a sample can receive contributions from other folds in which it belongs to the training partition.

Apply the formulas in Section 4.7 with the following requirements. Retain each sample's exact integer
prediction-comparison count, influence values, and applicable value-level failure references; their field encoding
belongs to the Phase 9 report schema. Do not add a separate per-value evaluation-status field.

- Reuse the exact validation training and test partitions. Within each fold, omit each training sample once, refit
  the configured model, and compare baseline and removal-model probabilities on that fold's unchanged test set.
- Set `p` to the number of actual predictors in input `X` supplied for fitting. Do not substitute a parameter count
  or an alternative inferred feature count.
- Calculate baseline MSE against true class indicators on the test set, separately for every class. Keep class
  identities aligned. This average across test samples is distinct from the residual report's calculations.
- Aggregate using the approved prediction-comparison count `M_i = sum_{f in F_i} |V_f|`, including comparisons whose
  prediction change is zero. Retain the count with the report. Only folds in which the sample is a training candidate
  contribute; samples that are exclusively held out have no influence result.
- Leave mathematically undefined influence values as `NaN`, including undefined division at zero MSE. Leave small
  positive MSE and the resulting defined values unchanged. Do not add epsilon, thresholds, clipping, imputation, or
  other adjustments.
- If a missing class or failed fit/prediction prevents evaluation, represent the affected computationally unavailable
  influence values as `None` and record a clear failure identifying the fold, sample, and class where applicable.
  Do not fabricate missing probabilities, introduce fallback fits, remove comparisons, or shrink the divisor to obtain
  a numeric result. Formula-defined undefined contributions propagate as `NaN`; computationally unavailable
  contributions propagate as `None` with their failure provenance. Isolate removal-evaluation errors at their
  sample/class boundary and continue other independent computations. Initial reconstruction or primary-validation
  failure prevents influence analysis, as specified in Section 9.2.
- Use the model's configured fitting and preprocessing procedure for baseline and removal fits. Accept any randomness
  influence from that procedure. Do not introduce extra seed coupling, repeated fits to select a preferred result,
  or post-computation adjustments to remove stochastic effects.

These are defined computation outcomes and reporting rules. Report their edge cases and risks without reopening
formula-defined undefined values or accepted randomness effects as design ambiguities. Keep the computation direct.

### 4.6 Regression reports

#### Validation results

Regression validation columns, in order, are:

```text
sample_id | sample_label | fold_id | y_true | y_pred
```

#### Performance report

The regression performance workbook contains one row with columns, in order:

```text
mean_error | std_of_error | MAE | normalized_MAE | MSE | RMSE
| normalized_RMSE | CV_RMSE | RPD | R2
```

Use established conventional definitions unless Cocoonet explicitly specifies a variation. Record detailed formulas,
normalization, sign, standard-deviation, and report aggregation conventions during the corresponding planning and
implementation steps. Do not treat absent formula detail alone as an ambiguity or invent a project-specific measure.
Undefined results, including formula-defined constant-target cases, remain `NaN` without corrective manipulation.

#### Residual report

Regression residual-analysis columns, in order, are:

```text
sample_id | sample_label | y_true | y_pred | residual | standardized_residual
```

Use conventional residual and standardization definitions, with their sign/population documented alongside the report
implementation. Preserve undefined results as `NaN` and apply the same scoped failure rules as other reports.

#### Visualization data

- Data for a scatter plot of true target values and predicted target values.
- Data for a residual plot of predicted target values and residuals.

Return plotting data rather than rendered plots. Specify its representation and reuse of validation/residual report
fields alongside the report-schema implementation.

#### Optional sample influence report

Regression sample-influence columns, in order, are:

```text
sample_id | sample_label | influence | prediction_comparison_count
```

For single-target regression, use the common sample information in Section 4.4 for each omitted training sample:
internal `sample_id`, user-facing label, source references, and true target value. Associate one generalized Cook's
distance-like influence value per sample using Section 4.7, accompanied by its exact integer prediction-comparison
count and applicable value-level failure references. Do not add a separate per-value evaluation-status field. This
measures the mean normalized change in held-out target predictions caused by removing that sample from eligible
training partitions.

Baseline MSE compares the fold's baseline target predictions with its true held-out targets. Both baseline and removal
fits predict on the fixed test set; a prediction for the omitted training sample itself is not required. Test samples
receive no influence metric from a fold in which they are held out. A sample may contribute through other folds where
it is in training; a sample never in training has no influence metric, rather than an observed zero.

Keep formula-defined undefined contributions as `NaN`, preserve the comparison population, and isolate
removal-evaluation failures under Section 9.2. Computationally unavailable contributions are `None` with applicable
failure references. Count and failure-reference field encoding belongs to the Phase 9 report schema; it does not
change the approved formula.

### 4.7 Optional generalized Cook's distance-like influence

#### Meaning and evaluation populations

The approved measure is the average normalized effect of removing a training sample on held-out predictions.
Classification produces one value per sample per class; single-target regression produces one value per sample.
The measure describes the magnitude of prediction changes, not whether removal improves predictive performance.

Influence analysis exactly reuses the parent schedule's realized validation partitions, including their established
training and test sample memberships. It does not generate a separate schedule for any model or execution attempt.
Each validation fold `f` has a training partition `T_f` and a disjoint held-out partition `V_f`. Do not generate a
separate influence split schedule. The baseline fit uses the complete training partition for that fold. Each training
sample `i` is omitted once within that fold, and the model is refitted on `T_f` without `i`.

Both fits predict on exactly the same fixed test rows `j` in `V_f`, aligned by sample ID and, for classification, class.
The numerator sums prediction changes on that test partition. The predictor count `p` is the number of actual
predictors in input `X`; for a two-dimensional feature matrix, this is its number of predictor columns. Use the actual
input feature dimension, not the number of model parameters or a hypothetical alternative feature representation.

Influence metrics belong only to training samples in each fold. If `i` is already held out, that fold supplies no removal
evaluation for `i`; it does not supply an observed zero-valued influence. Test samples do not participate in that fold's model
construction and receive no influence metrics for that fold. A sample may be a training sample in other validation
folds, where its removal is evaluated normally. No equality of predictions from differently
fitted models is assumed. Removing individual training samples is distinct from choosing LOO as the outer validation
strategy, and these refits remain internal to the complete worker task for one model.

#### Computation steps

This computation requires the complete stored predictions from successful primary validation. It uses those predictions
and their fold-specific baseline MSE as the fixed comparison baseline. Influence analysis may run independently or
concurrently with ordinary reports after primary validation, but it cannot begin when primary validation is
`validation_failed`; a requested influence report is then `blocked`.

For each eligible training sample, removal-model initialization, fitting, prediction, and influence calculation form
the sample's influence evaluation. A failure is local to the affected sample, fold, and class contribution. Preserve
successful contributions and continue independent removal evaluations without stopping the worker process.

1. Reuse the exact model validation splits and their training and test sample identities.
2. Use the baseline fit on each fold's full training partition and its held-out predictions from model validation.
3. Calculate baseline MSE against the true held-out targets. For classification, use true class indicators and compute
   a separate MSE for each class. Keep this baseline MSE fixed across all removal fits within that fold.
4. Omit each training sample `i` once per fold, refit using the configured model and training/preprocessing procedure,
   and predict on the unchanged test partition. Accept any randomness effects in the resulting predictions; do not
   alter the results or introduce extra fitting procedures to compensate for them.
5. Sum squared changes between baseline and removal predictions over held-out rows, then divide by the predictor
   count multiplied by baseline MSE to obtain the fold contribution defined below.
6. Sum the fold contributions for each removed sample and divide by its total held-out prediction-comparison count.
   Count a comparison even when its prediction change is zero. Count repeated evaluations separately.
7. Associate each influence result and its comparison count with the omitted training sample's ID. The predictions
   used in the calculation belong to the fixed test set; the formula requires no prediction for the omitted sample
   itself. Test samples receive no influence metrics from that fold.

Evaluate the defined formulas directly. Represent undefined influence values as `NaN`, including undefined ratios
when baseline MSE is zero. Leave small positive MSE unchanged and accept the resulting defined values. The behavior
specified at the end of this section is part of the computation contract, not an unresolved design question.

#### Classification formula

Let `t_{j,c}` be 1 when sample `j` truly belongs to class `c`, and 0 otherwise. Let `q_{f,j,c}` be the baseline class
probability and `q_{f,-i,j,c}` the probability after fitting without training sample `i`. The baseline error is:

```text
MSE_{f,c} = (sum_{j in V_f} (t_{j,c} - q_{f,j,c})^2) / |V_f|
```

For each eligible training sample, the fold contribution is:

```text
D_{f,i,c} = (sum_{j in V_f} (q_{f,j,c} - q_{f,-i,j,c})^2) / (p * MSE_{f,c})
```

The numerator compares the two models' predictions. The denominator compares baseline predictions against truth;
it is not the mean squared difference between the two models' predictions. The cancellation to a constant discussed
for that earlier denominator does not apply to this definition.

#### Regression formula

Let `y_j` be the true target, `y_hat_{f,j}` the baseline target prediction, and `y_hat_{f,-i,j}` the prediction after
fitting without training sample `i`. For single-target regression:

```text
MSE_f = (sum_{j in V_f} (y_j - y_hat_{f,j})^2) / |V_f|
D_{f,i} = (sum_{j in V_f} (y_hat_{f,j} - y_hat_{f,-i,j})^2) / (p * MSE_f)
```

Baseline MSE measures target prediction error on the held-out partition. It is fixed for the fold and is not
recomputed against removal-model predictions.

#### Approved aggregation and coverage

Let `F_i` contain the validation folds in which sample `i` belongs to the training partition. This membership is
determined by the validation splits, not by whether the resulting contribution is finite. Define the comparison count:

```text
M_i = sum_{f in F_i} |V_f|
```

The reported classification and regression influence values are, respectively:

```text
D_bar_{i,c} = (sum_{f in F_i} D_{f,i,c}) / M_i
D_bar_i = (sum_{f in F_i} D_{f,i}) / M_i
```

This is the arithmetic mean of individual normalized squared prediction changes, with each held-out comparison
receiving equal weight. Equivalently, average each fold's per-prediction score `D_{f,i} / |V_f|` with weight `|V_f|`;
apply the same rule separately for every classification class. This interpretation follows the definitions of the
[arithmetic and weighted means](https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weigmean.pdf).

The divisor is neither the number of contributing folds nor the final fold's training size minus one. The per-fold
MSE normalization remains in each contribution; aggregation does not replace it with a pooled MSE. The count is of
evaluations, not distinct sample IDs: if the same held-out sample is evaluated in repeated folds, each comparison
counts. Repeating an identical evaluation increases its accumulated sum and count proportionally.

Only samples that participate in training receive influence metrics. For a single train-test split, omit each training
sample once and evaluate its removal on the fixed test partition; the test samples receive no influence metrics.
Across validation folds, aggregate each sample's training-fold contributions only. A sample that is never in training
receives no influence metric, rather than an invented zero-valued result.

Preserve `NaN` contributions in ordinary aggregation. An undefined contribution makes the corresponding aggregate
undefined (`NaN`); for classification this applies separately to each affected class. Do not skip undefined values,
replace them with zero, or remove their comparisons from `M_i` to manufacture a defined result.

#### Worked examples

With 100 samples, two equal folds, and 10 predictors, each sample is a training candidate in one fold and its removal
is compared on 50 held-out samples. Consequently, `M_i = 50`.

| Quantity | Classification: one class | Single-target regression |
| --- | ---: | ---: |
| Baseline held-out MSE | 0.04 | 4 |
| Sum of squared prediction changes | 0.20 | 80 |
| Fold contribution, dividing by `p * MSE` | 0.50 | 2 |
| Approved reported value, dividing by 50 comparisons | 0.010000 | 0.040000 |
| Superseded value, dividing by training size minus one (49) | 0.010204 | 0.040816 |

Other complete evaluation schedules give the following counts for both task types:

| Influence evaluation schedule | Prediction-comparison divisor `M_i` |
| --- | --- |
| 100 samples, equal 5-fold | 80: four eligible folds, each with 20 held-out samples. |
| 100 samples, LOO | 99: 99 eligible folds, each with one held-out sample. |
| Single split: 70 training / 30 test samples | 30 for training samples; no influence metrics for test samples. |

For unequal folds, calculate `M_i` from the actual eligible held-out partitions. No property of the last processed
fold determines this count. Different splits can still yield different values because models and baseline errors
change, even though the reported quantity remains a mean per prediction comparison.

#### Defined edge behavior and direct computation

- **Undefined values:** Leave undefined influence values as `NaN`. For example, perfect baseline predictions yield
  zero MSE and an undefined normalized influence. This is an expected outcome of the defined measure.
- **Small MSE:** Use small positive MSE as it is. Do not add epsilon, impose a floor, clip a large influence value,
  or otherwise change the formula to obtain a preferred numerical result.
- **Missing classes and removal-evaluation failures:** If a missing class or failed removal fit/prediction prevents a
  contribution from being computed, represent the affected influence values as `None` with applicable failure
  references. Record a clear failure identifying the fold, training sample, and class where applicable. Preserve
  independently computable class values and continue with other samples. Removing the only training example of a
  class must not trigger invented probabilities, zero filling, fallback fits, or formula-defined `NaN` unless the
  formula itself is mathematically undefined. A failed primary validation prevents influence analysis because the
  complete stored comparison baseline is unavailable. A removal-evaluation failure after successful primary validation
  remains local to the influence report and does not invalidate ordinary reports.
- **Randomness:** Apply the configured training and preprocessing procedure directly, including its configured
  randomness settings. Accept randomness effects on predictions and influence values. Do not add corrective refits,
  select favorable runs, or adjust computed values to suppress those effects.
- **Aggregation:** Keep the specified comparison population and denominator. Undefined contributions remain `NaN`
  through aggregation; they are not grounds for dropping observations, changing weights, or reporting a repaired mean.
- **Scientific interpretation:** Report edge cases and risks clearly, without treating expected `NaN`, small MSE,
  large defined values, or randomness effects as ambiguities in this defined formula. Do not request a new edge-case
  policy for these specified outcomes. Computation must remain simple, clear, direct, and efficient.

## 5. Coordinator and Task Lifecycle

### 5.1 Assignment and task identity

The root tracks its configured execution targets and their current availability separately. It assigns a unit task
to a runnable target only after compatibility, dataset readiness, and the parent's fixed evaluation context and
persisted partition schedule have been established. It dispatches directly to the separate local worker or initiates
connections to online workers. Online assignment, status, results, and subsequent tasks share the existing
bidirectional connection; local operations use the local control boundary. A worker's availability to its owning root is
distinct from public-pool availability,
which belongs to the external service.

When tasks are ready, check configured online workers with free eligible attempt slots asynchronously.
A worker may already have active attempts. Among responsive eligible workers, allocate first to the
fastest connection response observed for that opportunity, then continue while tasks and eligible slots
remain. Root and worker admission must account for reserved slots so concurrent assignments cannot
overbook capacity. No full batch is required before execution starts.

Response speed is an allocation-time observation, not permanent priority. Section 5.2 determines
duplicate eligibility. Local workers require no network probe; acceptance remains topology-neutral.

Distinguish the logical task from its execution attempts. Each accepted result bundle has a stable identity
associated with the logical task and identifies its producing attempt. Identifier encoding belongs in the protocol
design. Simultaneous arrivals and reconnects must not create more than one accepted result for a logical task.

### 5.2 Duplicate execution and ordered reception

These rules apply across local and online attempts of the same unit. Local handoff and online transfer feed the same
root selection and persistence boundary; neither topology receives a separate acceptance rule or priority.

- Allocate a duplicate only when no unallocated tasks remain, every remaining task is finished or
  already assigned, and an eligible attempt slot would otherwise be unused. Choose an unfinished,
  currently running task, starting with its oldest first execution time. The duplicate is another
  attempt of that logical task, not a new logical result.
- Serialize result reception for each logical task. When duplicate attempts are ready, select one according to
  receiving order. Online reception receives only that bundle's `.tar.zst` package; root-local reception copies only
  that selected published bundle. Reception for different logical tasks remains independent.
- Selection does not depend on whether evaluation succeeded or failed. An earlier received failure outcome takes
  precedence over a later success; defined `NaN` and recorded computational failures do not invalidate the outcome.
- Complete the logical task only after the selected bundle is durably published and accepted. Then send stop
  instructions to all remaining duplicate attempts without receiving or handing off their bundles.
- Computing workers check stop instructions before each evaluation loop and stop at those cooperative boundaries;
  immediate interruption of arbitrary model code is not required. Ready duplicate attempts await reception selection
  or a stop instruction instead of sending unselected bundles. Stopping an attempt does not stop the worker process.
- Persistence waiting and retries apply only to the selected bundle. Online retry resends the same package; root-local
  retry reuses the same published bundle. Other duplicates' unreceived bundles do not enter retry and are not delivery
  failures. Failure of the selected transfer/persistence operation does not promote a later duplicate result; apply
  the bounded retry rules in Section 6.
- Duplicate execution reduces latency, not recovery from computation failure. Do not automatically retry/reallocate
  a terminal reconstruction or primary-validation failure or start another duplicate while an existing completed
  result awaits delivery.

Specify attempt identifiers, stop messages, and cancellation acknowledgements later without changing these scopes
or moving duplicate cancellation ahead of successful result acceptance.

### 5.3 Worker ownership, release, and cleanup

Workers are reusable resources of the root/computation configuration, not objects whose lifetime is exclusively owned
by a single parent task. Successive parents may reuse configured workers, online connections, a local worker process,
and retained valid datasets according to their applicable ownership and retention rules. This does not establish exact
online process/connection lifetimes, runtime sharing between copied computations, or automatic release on parent
completion. The root-local startup, shutdown, and root-restart lifetime follows Section 2.4. Public allocation ownership
remains external. Configuration removal has the following resolved scope.

`remove_worker` only modifies root-side worker configuration. It has no direct worker-side action: no stop,
disconnection, cleanup, reset, or resource release. If root configuration modification fails, report it explicitly;
do not compensate with worker-control actions. Worker configuration is mutable before ordinary execution begins,
after it finishes, or after root interruption even while old workers reach their stop boundaries; it is locked
during ordinary active execution. Worker changes do not change model evaluation configuration, make the instance
unconsolidated, or create a parent. Unchanged evaluation resumes its existing parent with the revised worker
configuration, subject to normal runtime eligibility.

A worker re-added/reconnected after interruption or recovery may be removed from mutable configuration under the
same rule, whether it is working, idle, or holding retained data. Any accompanying disconnection follows Section 7.4
and the existing ownership/service lifecycle; removal creates no new disconnection, timeout, or scientific policy.

| Operation | Scope |
| --- | --- |
| `remove_worker` | Root configuration modification only. |
| Connection/disconnection | Applicable connection, attempt, and ownership lifecycle. |
| `cleanup_dataset` | Selected retained worker dataset copies; Section 6.1's inactivity gate applies. |
| `cleanup` | All retained worker user data; preserve execution-stop and applicable cleanup safety boundaries. |
| Public-worker `release` | Clean retained user state and perform the supplied external resource/state-release operation. |
| Private-worker `release` | No resource-release action; warn that the private runtime lifecycle remains user controlled. |
| Run/cancel/recovery | Execution lifecycle, independent of configuration removal. |

Removing a public worker does not implicitly clean or release its resource. Removing a private worker does not clean,
stop, or reset its independent runtime. Those actions require applicable explicit management operations. A public
worker disconnected without explicit release remains governed by the supplied service policies; do not invent pool
timers or a new allocation policy. Exact method names, selection arguments, and management message forms are deferred.

Private workers remain under explicit user control through the root or worker module. Connection loss alone does not
release a private worker. Public allocation, reservation durations, timeout-based release, and availability decisions
belong to the external service; Cocoonet does not implement its own pool-release timers or allocation policy.

For online workers, maintain a root-worker heartbeat/status exchange for connection health. Cocoonet must not treat
connection loss itself
as permission to serve another user. Follow the supplied service interface for public lifecycle coordination.

During release, stop previous user execution and remove all Cocoonet-managed datasets, task state, models, reports,
and other user state before serving another user. Prior execution must not continue or recreate state after cleanup.
Require confirmed cleanup before accepting another user's work or reporting cleanup success. If cleanup fails or
cannot be confirmed, keep the worker locally unavailable to another user and report the failure through the defined
interface. The service retains ownership of its pool's availability state. This local restriction does not imply
that the worker process has terminated.

Specify cleanup verification and root-worker reconnection authentication before implementing those boundaries.
Service notifications follow the supplied external interface rather than an invented Cocoonet service protocol.

### 5.4 Root restart and progress reconciliation

Resume/recovery preserves the existing parent, root directory, scientific configuration, authoritative
`partition_schedules.xlsx`, accepted model-keyed bundles, selected reception, and remaining retry budgets. It does not
invoke fresh initialization or create a new parent.

Reconstruct progress from `execution_configuration.csv`, `execution_state.csv`, the authoritative partition workbook,
and complete final root bundles. Staging directories, partial packages, and incomplete workbooks are not completion
evidence. Recovery does not rely solely on pre-termination memory.

Durably record the selected attempt in `execution_state.csv` before online package receipt or root-local handoff.
Selection alone is neither acceptance nor logical-task completion.

- **Still computing:** If no accepted bundle exists and the parent has not been stopped, cancelled, or superseded,
  restore the assignment and let it continue. Otherwise preserve its stop disposition under Section 5.5. Apply
  duplicate-stop rules once another attempt's bundle is accepted.
- **Complete bundle already persisted:** Reconstruct the model/unit as complete and confirm acceptance as needed. Do not
  reevaluate or retransmit it. Stop remaining duplicates without receiving their bundles.
- **Selected bundle pending delivery:** If the selected bundle is absent and its worker retains it with retries
  remaining, resume delivery of that same bundle within its remaining budget. Never substitute an unselected duplicate.
- **Delivery retries exhausted:** That operation remains failed through reconnection or restart. Do not silently reset
  its budget or restart it merely because a retained bundle remains.

Re-establish private online connections under `auto`, `root_connects`, or `worker_connects`; `auto` retries both
directions. After establishment, the root initiates reconciliation. Public reconnection remains root-initiated through
the supplied service interface. A root-local worker from a terminated root process is recreated, not reconnected;
retained managed artifacts are reconciled through the recreated local boundary. Keep root recovery distinct from
exceptional worker-process failure and restart, described in Section 9.2.

### 5.5 Parent interruption and prospective execution

Stopping, cancelling, or superseding a parent sends stop instructions to its executing workers. Executing model work
stops at the next defined cooperative boundary. Already-terminal attempts and accepted, rejected, or otherwise
unneeded result handling do not wait for another evaluation boundary; end their remaining unnecessary handling
immediately. This does not require forcibly interrupting arbitrary model code or deleting an accepted result.

Root interruption permits future configuration edits immediately, without waiting for old attempts to finish
stopping. Each old attempt retains its fixed inputs and stop disposition. A model evaluation configuration change
requires a new parent after the Section 2.3 initialization gate; worker-only changes preserve the parent, and unchanged
evaluation resumes it. Editing or resuming does not reset settled results or exhausted operation retry limits.

Late outcomes remain associated with their original parent and cannot be accepted into another parent. A new parent
uses its own fixed scientific configuration and separate result round; changed evaluation cannot extend an old result
set. Workers must satisfy availability, ownership, and the applicable parent's dataset/configuration/partition
readiness before any allocation. A stop request does not itself prove a worker is inactive or ready for dataset cleanup.

Duplicate cancellation following acceptance remains governed by Section 5.2. User-requested parent interruption has
a different trigger and does not require waiting for a result to be accepted. Configuration unlocking after root
interruption and the all-target inactivity gate for cleanup are separate requirements.

## 6. Dataset Distribution and Report Persistence

### 6.1 Dataset synchronization and reuse

Establish readiness of the exact representation identified by `dataset_id` before executing tasks that use it.
For online workers, synchronization must complete; for the local worker, framework-managed local artifact access
must be ready. Reuse that local dataset for subsequent
tasks; per-task transfers carry model inputs and directives rather than another dataset copy. A changed representation
requires a new `dataset_id`, and cleared data must be made ready again through the applicable boundary before reuse.
Public release cleanup removes
prior user data and task state before reassignment.

Removing/replacing the active dataset changes in-memory configuration references, not arbitrary persistent files.
No general persistent-file clear API is required. Users may manage files directly outside the module; a future
convenience API is optional. Explicit initialization/overwrite prepares the selected directory for a fresh execution
under Section 2.3. It must remain distinct from recovery and from worker-side dataset cleanup.

Root-side user-provided dataset files are user-owned and their configured source paths must remain outside the
Cocoonet-managed root working directory. Enforce Section 2.3's ownership invariant for both dataset-path and
root-directory configuration changes, including copied configurations, before accepting the proposed configuration.
Cocoonet stores references and its own dataset identity, generated sample IDs, configuration, task state, and partition
information. It does not delete, move, rename, garbage-collect, or otherwise modify the underlying user source data.
Dataset removal removes references/configuration only. Managed root contents may be cleared or replaced under the
existing persistence/overwrite rules without selective dataset-path preservation.
The authoritative partition workbook is a Cocoonet-generated artifact under the partition/recovery contract, not an
object of worker-dataset cleanup; it is not a user-provided dataset source path.

Synchronized worker copies are Cocoonet-managed and may be retained for reuse. Successful dataset cleanup invalidates
readiness for that worker/dataset, so later use requires synchronization again. Private copies remain under user
control; public workers must clean retained user state before ownership release and another user's admission.
For a root-local worker, shared access to user source files does not make those files cleanup targets.

Dataset cleanup may target one worker, a selected set, or all applicable workers. It normally follows parent completion;
an interruption may leave workers active. The admission rule is all-or-nothing across the requested set: all targeted
workers must actually be inactive. If any targeted worker is active, reject the entire request before deleting any
dataset copy anywhere in that set. Do not partially execute or defer a rejected cleanup; the user may request it again
after all targets are inactive. A parent interruption alone does not satisfy this gate. Explicit cleanup is distinct
from worker removal, scientific failure, result persistence, and public-resource release.

The exact selection API, readiness verification, and deletion implementation remain deferred and must preserve
source-file ownership, all-target inactivity before cleanup, and invalidation after successful deletion.
Operational cleanup failures remain subject to Section 9; public release still requires confirmed cleanup success.

Key encoding and synchronization verification are implementation details that must preserve the row/sample associations
and readiness invariant.

### 6.2 Managed persistence layout and authority

One root working directory represents one parent execution. Accepted results are keyed by model; attempt identity
remains in execution state, manifests, workbook metadata, failures, and protocol messages.

Use this root layout:

```text
<parent_evaluation_instance_work_dir>/
  cocoonet_metadata.csv
  root_data_reference.xlsx
  partition_schedules.xlsx
  execution_configuration.csv
  execution_state.csv
  models_for_transfer/
    model_<model_id>.dill
    models_parent_<parent_id>.<archive_extension>
    models_parent_<parent_id>_<tile_id>.<archive_extension>
  attempt_results/
    model_<model_id>/
      root_model_<model_id>_reference.csv
      worker_model_<model_id>_reference.csv
      bundle_manifest.xlsx
      attempt_metadata.xlsx
      fold_sources.xlsx
      validation_result.xlsx
      classification_performance.xlsx | regression_performance.xlsx
      residual_analysis.xlsx
      sample_influence.xlsx
      failures.xlsx
      visualizations/
        roc_curve.dill
        scatter_plot.dill
        residual_plot.dill
        rendered_plots/
      worker_error.log
  logs/
    root_running.log
    root_error.log
```

Store dataset-transfer preparation under the configured source dataset root:

```text
<source_dataset_root_dir>/
  dataset_for_transfer/
    dataset_<dataset_id>.<archive_extension>
    dataset_<dataset_id>_<tile_id>.<archive_extension>
```

Use this worker layout:

```text
<worker_work_dir>/
  datasets/
    dataset_<dataset_id>/
  models/
    parent_<parent_id>/
      model_<model_id>.dill
  results/
    parent_<parent_id>/
      cocoonet_metadata.csv
      worker_data_reference.xlsx
      partition_schedules.xlsx
      execution_configuration.csv
      execution_state.csv
      attempt_results/
        model_<model_id>/
          worker_model_<model_id>_reference.csv
          bundle_manifest.xlsx
          attempt_metadata.xlsx
          fold_sources.xlsx
          validation_result.xlsx
          classification_performance.xlsx | regression_performance.xlsx
          residual_analysis.xlsx
          sample_influence.xlsx
          failures.xlsx
          visualizations/
            roc_curve.dill
            scatter_plot.dill
            residual_plot.dill
          logs/
            worker_running.log
            worker_error.log
```

Root `root_data_reference.xlsx`, `partition_schedules.xlsx`, `execution_configuration.csv`, and `execution_state.csv`
are authoritative. Worker copies are execution inputs and worker-local observations.

`root_model_<model_id>_reference.csv` and `worker_model_<model_id>_reference.csv` contain at least `model_id`,
`model_label`, and the logical owner-relative `model_path`. The worker reference is copied unchanged with the selected
bundle; the root independently adds its reference and final inventory.

The root publishes only the selected bundle for a model. Late or unselected duplicates never create or overwrite the
model directory. A worker must not admit a second attempt for the same parent/model while its staging, published, or
retained model-keyed result path is occupied.

Other generated identifiers used in managed paths, including tile identifiers, are positive sequential decimal
integers beginning at `1`, without zero padding or reuse in their owning persistent scope. Paths are semantic locators,
not authoritative identity evidence; exact identities remain stored and validated inside artifacts.

Exact archive selection, tiling threshold, and compression settings for dataset/model transfer remain deferred. This
does not affect the fixed `.tar.zst` result-package contract.

### 6.3 One immutable bundle per evaluation attempt

A completed attempt produces one immutable result bundle containing all applicable reports, results, and failure
information. Independent computational failures preserve successful results alongside unavailable `None` values and
logged errors; formula-defined undefined results remain `NaN`. A report with any computational failure has report
status `failed`, even when its payload preserves successful or identified unavailable results; there is no separate
partial-report status. Individual bundle members have no separate delivery, retry, persistence, acceptance, or recovery
state.

Initial reconstruction failure produces a `reconstruction_failed` failure-only bundle without predictions or reports.
Primary-validation failure produces a `validation_failed` bundle containing preserved primary-fold results, failed or
blocked dependent reports, and complete failure provenance. Report or influence failures after successful primary
validation produce `completed_with_errors`.

Package receipt and root-side content processing must not block the active network event loop. Never deserialize
visualization dill data there. The complete attempt bundle preserves its parent, unit/model, attempt, and worker
associations, scientific outputs, and failure provenance. Prefer directly inspectable payloads with explicit
associations. Avoid duplicate report identities and independently maintained summaries; derive summaries from
authoritative facts, or validate their consistency when stored.

### 6.4 Persistence, acknowledgement, integrity, and bounded retries

Root persistence and outcome acceptance apply to both local and online execution. Online workers package the selected
bundle once as `.tar.zst`. Compression parameters are user-configurable; defaults favor best performance for result
bundles of approximately 1 MB or less.

Every package carries its exact byte length and SHA-256 digest. For private online delivery, root and worker receive the
same random key of at least 32 bytes through keyword-only runtime API parameters. Cocoonet never transmits, persists,
internally copies, displays, or logs the key.

The worker computes HMAC-SHA-256 over canonical metadata containing protocol version, parent/model/attempt identities,
package length, and package SHA-256. The root stages the package and verifies length, digest, and HMAC before safe
extraction, manifest/schema validation, and publication. Public-worker authentication remains governed by the external
service.

A root-local worker instead exposes its already validated published bundle. The root copies its contents, adds only the
root model reference and final root inventory, and publishes it without archive creation, HMAC, unpacking, or duplicate
scientific/result validation.

- Accept the selected result and mark its logical task complete only after the complete bundle has been durably
  published in root storage. Confirm acceptance through the applicable online or local control boundary. Receiving
  bytes, incomplete writes, staging content, or transient memory does not establish completion.
- Never overwrite an accepted bundle with a later duplicate.
- Integrity, extraction, validation, delivery, or persistence failure leaves the logical task unfinished. Retry the
  same package or local bundle within its remaining budget without reevaluation or per-file retry. A computational
  failure recorded inside a complete bundle neither requires retransmission nor prevents normal acceptance.
- Until acceptance or delivery-retry exhaustion, retain the selected worker bundle and keep its producing attempt slot
  occupied. Other eligible attempt slots remain usable.
- Retry exhaustion fails only that delivery operation. It does not accept or complete the logical task, authorize
  reevaluation or duplicate substitution, or terminate the worker.
- Unselected duplicate attempts do not enter this delivery lifecycle. After successful acceptance, stop the remaining
  attempts without receiving or handing off their bundles.

Exact retry timing, canonical HMAC metadata byte encoding, compression parameter names, and acknowledgement encoding
remain deferred. They must implement these acceptance semantics rather than redefine them.

### 6.5 Atomic publication

Authoritative XLSX files are written as same-directory `<filename>.partial` files. Validate before writing, flush and
synchronize the completed file, then atomically rename immutable files. Publish mutable `execution_state.csv` through
same-directory staging and atomic replacement.

A worker constructs a complete result in `model_<model_id>.staging` beside its final model directory and publishes it
with one atomic directory rename after validation.

The root constructs the selected result in `attempt_results/model_<model_id>.staging`, copies worker-produced files,
adds its root model reference and final inventory, and publishes the final model directory with one atomic rename. It
does not recompute or reinterpret scientific results.

Staging content is never authoritative and is never automatically promoted. Recovery or retry removes only the exact
stale staging entry and restarts the complete write. Logs append directly and may retain an incomplete final line after
a crash.

Implementation uses one focused file-publication helper and one focused bundle-publication helper, not a general
transaction, repository, or artifact framework.

### 6.6 Worker-side retention

Configure bundle retention as deletion after confirmed acceptance, retention until all applicable root tasks finish,
or no automatic deletion. Retain pending-delivery bundles and online packages as required by the delivery rules.
Mandatory public release cleanup overrides local retention, including pending delivery: no previous user's artifact may
survive reassignment. Outside that release cleanup, do not prematurely delete a pending-delivery bundle. Cleanup of an
undelivered bundle does not finish its logical task. Retention conditions, operation failure, result acceptance, and
worker-process lifetime are distinct scopes.

Deletion on transition to a different parent task is an additional worker-side retention mode. It is configurable for
private and root-local workers and mandatory for public workers, overriding a longer or no-automatic-deletion setting.
Before a public worker begins an attempt for a different parent, remove worker-side result artifacts belonging to
earlier parents. Any selected delivery still required by the lifecycle must first complete or reach its established
terminal operation state. This cleanup does not affect root-persisted results or mark an undelivered logical task
finished or accepted.

## 7. Transport, Serialization, and Trust

### 7.1 Runtime and connections

Support Python 3.12 through the latest stable Python release, excluding prereleases. Use `asyncio` and `aiofiles`.
Online communication uses one versioned Cocoonet protocol over TLS-protected HTTPS/WSS. Bounded control messages and
streamed dataset, model, and result-package payloads share the established bidirectional connection. Local-only
execution does not initialize this subsystem.

Private establishment and reconnection follow `auto`, `root_connects`, or `worker_connects`. Public establishment
follows the supplied service interface and is root-initiated. Connection direction does not change root scheduling
authority. After establishment, reconciliation and task/control instructions are root-initiated. Workers never
communicate directly with other workers and do not poll for work or use separate task-submission endpoints.

Root-local execution uses the multiprocessing pipe boundary in Section 2.4 and does not initialize online networking.

Sending a result alone does not release its attempt slot. Apply that attempt's acknowledgement,
pending-delivery, retry-limit, stop, and recovery rules. Other eligible free slots remain available.

### 7.2 Serialization ownership and compatibility

- Use schema-controlled UTF-8 CSV for Cocoonet metadata, model-reference files, execution configuration, and execution
  state. Use schema-controlled XLSX for root/worker data references, authoritative/worker partition schedules, and
  two-sheet result workbooks. Do not rely on automatic type or missing-value inference.
- Use dill only for trusted root-local/private model payloads, trusted root-local IPC messages, and standardized
  visualization data; do not introduce a parallel direct `pickle` path.
- Dataset table inputs and generated CSV/Parquet transfer artifacts follow Sections 3.1 and 3.5. Their formats are
  separate from model, result, protocol, and partition persistence.
- For root-local and private workers, only the root creates serialized model payloads, and only those workers
  deserialize them within the enforced execution boundary.
- Public workers do not accept or deserialize Python model payloads. The root sends a declarative model specification
  and applicable non-executable model-data artifacts, and the public worker reconstructs the model from supported
  worker-local components.
- Workers construct immutable result bundles. Online workers transfer the selected bundle as `.tar.zst`; root-local
  workers use direct directory handoff. Partition memberships are loaded from `partition_schedules.xlsx`.
- Preserve finite values and the explicit `Inf`, `-Inf`, `NaN`, and `NA` representations required by the owning schema.
  Required XLSX workbooks write finite values as numeric cells and the special tokens deliberately as text cells.
- Require identical complete Cocoonet version identifiers for root and worker. Online workers check during the
  handshake before normal protocol operation, dataset synchronization, assignment, or model loading; local checking
  requires no network handshake. For example: `1.2.3` accepts `1.2.3` and rejects `1.2.4`.
- Read version metadata without loading model/report payloads. Reject an incompatible worker immediately, identify
  both versions, and suggest installing the same Cocoonet version. No compatibility fallback or model deserialization
  follows rejection.
- Each release must operate across its declared supported Python versions. Root and worker Python versions may
  differ within that range. Dependency constraints and release validation must uphold this runtime/serialization
  contract; do not introduce dependency or environment negotiation into the worker protocol.

Framework compatibility does not guarantee execution of every user-supplied model. Model-specific failures follow
Section 9, without deeper custom-model dependency inspection, environment repair, or substitution.

### 7.3 Enforced model execution boundary

Private online workers accept serialized model payloads only from their authenticated owning root. The
root-local worker receives model inputs only through its owning root's local boundary. Both retain trusted
custom-model and dill-deserialization support; the user is responsible for trusting that content.

Public workers reject pickle/dill and equivalent serialized Python model objects. They accept only validated
declarative specifications and non-executable model-data artifacts whose component identities resolve through
Cocoonet-controlled supported components. No worker exposes general-purpose remote execution.

User-provided model code has no direct filesystem read/write or network permission. Cocoonet manages dataset
references, reads, and worker copies without taking ownership of user-provided root source files. It supplies in-memory
training/evaluation data through the supported interface, validates the returned
model outputs, and constructs and persists standardized reports. Models return only the interface's required outputs.

Framework networking is limited to defined authenticated root-worker operations and external discovery/lifecycle
operations required by the supplied service interface. The model itself has no network capability.

Specify technical enforcement of execution, filesystem, and network isolation before implementing this boundary,
covering executable deserialization, reconstruction, and model operations. Enforcement cannot rely on Python module
conventions or cooperative model behavior. The concrete mechanism, wire schemas, authentication/reconnection details,
and heartbeat settings are subsequent implementation design, not permission to weaken these restrictions.

### 7.4 Online connection rounds and reconnection

Use the same periodic, configurable connection policy before run, during active evaluations, and after disconnect.
Each worker has a retry period and a configured sequence of delays within a connection round. For a period of
60 seconds and intervals `[1, 2, 4]`, a round begins periodically; if its initial attempt fails, further attempts follow
after those delays. Exhausting the round marks that worker unavailable for allocation until the next scheduled round.

A configured unavailable worker remains configured. Runtime availability changes alone do not change execution
configuration or the instance's consolidation state. A failed round does not fail other tasks or workers and does not
block the coordinator. When a usable connection is established and normal readiness checks succeed, that worker
automatically becomes eligible for subsequent allocations. Connection checks, retries, and response measurement are
asynchronous and isolated per worker. Authentication, exact-version compatibility, and ownership rules still apply.

Disconnection during an assigned attempt marks the worker unavailable for new allocations and activates the same
reconnection policy. Existing work follows the established attempt, delivery, duplicate, and recovery rules; connection
loss is not a scientific/model-training failure and does not authorize uncontrolled reevaluation or public release.

A completed unsuccessful round is a failed bounded operation; the next periodic round is a separately scheduled
connection operation. This policy does not restart an exhausted result-delivery operation or reset its budget.
`run()` does not wait for every worker or for an all-unavailable set to recover: it requires a currently runnable
target, otherwise startup fails while connection management continues independently. Timers, exact parameter names,
response measurement/probes, and concrete connection-state representation remain implementation details.

## 8. Execution and Resource Management

Worker communication/control execution and model-evaluation execution are separate execution responsibilities.
Long-running model reconstruction, fitting, prediction, influence analysis, report computation, or result packaging must
not block the worker's ability to maintain applicable local/online root communication, exchange status information and
online heartbeats, receive control instructions, or manage the current task lifecycle.
Resource management must preserve this control-path responsiveness while allowing model evaluation to use the worker's available computational resources efficiently.

### 8.1 Worker execution boundary
A root-local worker runs in a separate locally managed process under the direct local control boundary in Section 2.4;
online workers use the connection boundary in Section 7. The same worker evaluation contracts apply to both.
The worker communication and control path remains responsive independently of model-evaluation execution. Computationally intensive evaluation work executes through the model-evaluation workflow rather than directly in the communication/control execution path.
Worker control owns admission, slot reservations, and supervision of Cocoonet model-evaluation
processes. Each model-evaluation workflow owns one attempt's scientific execution. User-configured
model/framework parallelism operates inside that attempt under the same isolation requirements.
Failure of an attempt or its subprocesses must not terminate the long-running worker or unrelated
attempts.
Stop and cancellation instructions received through the communication/control path are applied at the cooperative boundaries defined by the task lifecycle. This execution separation does not require immediate interruption of arbitrary user model code.

### 8.2 Concurrent CPU model attempts

For CPU execution, each worker has a requested positive integer concurrency setting
`n_attempt`, defaulting to `1`. It is the maximum number of complete model-evaluation
attempts requested concurrently on that worker. `n_attempt` is worker configuration,
not part of a parent's fixed scientific model input.

Let `N` be the CPU count currently available to the worker and let:

```text
C = N - 1
```

`C` is the worker's accounted CPU capacity for model evaluation. One available CPU is
reserved for worker communication and control. A worker with `N < 2` cannot accept CPU
evaluation work under this policy.

Each CPU model input has a positive accounted per-attempt CPU demand `j`. For models
with an applicable `n_jobs` setting, `j` is its worker-local resolved positive value.
A model without applicable internal CPU parallelism has `j = 1`. Symbolic or negative
framework values must be resolved according to their owning API before capacity
validation. The corresponding PyTorch demand is derived from its supplied CPU
loading/training configuration under Section 3.3.

When multiple models eligible for the worker have different demands, use:

```text
J = max(j for every eligible CPU model)
```

The maximum is required because duplicate attempts may concurrently execute the same
highest-demand model.

For requested `A = n_attempt`, derive the effective attempt limit as follows:

```text
if A * J <= C:
    effective_n_attempt = A
elif A > 1 and (A - 1) * J <= C:
    effective_n_attempt = A - 1
else:
    reject the configuration
```

Adaptation reduces the requested value by exactly one. Do not repeatedly decrement it
and do not produce a zero attempt limit. Rejection leaves the existing configuration
unchanged.

Incomplete computation configuration remains valid. Perform this compatibility check
when both applicable model and worker configurations are available, and repeat it at
run or worker readiness before allocation. A changed available CPU count requires
revalidation. Preserve the requested value and report the effective value, `N`, and
`J` for inspection and auditability.

Each effective slot belongs to one complete model-evaluation attempt, including its
allocation, execution, and outstanding terminal handling. Reservation begins at
allocation and is released only under the applicable completion, acknowledgement,
failure, or cooperative-stop rule. Lower prospective capacity does not cancel existing
attempts; it prevents additional allocation until occupancy is within the effective
limit.

### 8.3 Model and framework parallelism

Model-level settings, including supported `n_jobs` parameters, belong to the fixed
model input. PyTorch loading/training multiprocessing settings, including data-loader
workers, belong to the fixed loading/training configuration. Pass these settings
unchanged to their owning model, framework API, or training procedure.

Capacity enforcement changes only Cocoonet's effective outer attempt limit or rejects
the incompatible configuration. It must not silently cap, divide, or override supplied
model/framework settings.

For example, with `N = 16`, `n_attempt = 4`, and `J = 4`, the requested accounted
demand is 16 against `C = 15`. Reducing the attempt limit once gives an effective limit
of three and an accounted demand of 12.

With `N = 12` and the same configuration, the reduced demand would be 12 against
`C = 11`. The configuration is therefore rejected; it is not reduced further.

The calculated demand is a resource-accounting value, not a guarantee that every
claimed physical CPU is continuously occupied. Exact utilization may depend on the
model, backend, numerical libraries, and workload. Any framework parallelism not
represented by `n_jobs` or an equivalent setting must be included in the model
adapter's accounted demand.

GPU execution has one Cocoonet model-attempt slot and no Cocoonet-level
multiprocessing. Independently configured framework-managed loading or training
parallelism remains permitted and is passed unchanged.

Choose an execution mechanism that supports the permitted nested model/framework
subprocesses. All child execution remains subject to the applicable isolation,
failure, cancellation, and cleanup requirements.


## 9. Reproducibility, Auditability, and Failure Reporting

### 9.1 Scientific provenance

The framework does not guarantee deterministic model results. Accept randomness and nondeterminism from models,
dependencies, environments, and hardware without corrective manipulation. Pass supplied reproducibility settings
through where applicable; do not invent extra determinism controls or select favorable runs. User-defined model and
dependency determinism remains the user's responsibility.

Partition-generation randomness is resolved when `add_dataset()` prepares a new split. Preserve the realized artifact
and each parent's fixed reference to it; link unit-task/attempt fold results to that schedule. Identical realized
partitions within a parent
are required even when model outputs are nondeterministic; separate parents may establish different partitions.
Record actual training/test sample memberships, not only the splitting method or seed.

Trace dataset, model, logical-task, and attempt identities, validation splits, supplied settings, and relevant
environment information. Preserve actual results, including defined `NaN` and failure outcomes. Use structured logs
for assignment, lifecycle transitions, acceptance, and failures. Concrete provenance fields and event schemas belong
in the relevant design; distinguish operation, task, result, and worker-process scope in every record.

Cocoonet-generated runtime type-validation errors identify the affected field path, expected type, observed type, and
observed value. Short strings and scalar values may be shown directly. Potentially large values use bounded,
type-aware previews that preserve useful information such as type, shape, size, dtype, and representative values
without rendering the complete object. Diagnostic generation must not conceal an already established type mismatch.

Contract validators reject semantically invalid candidate records with built-in `ValueError`. Their diagnostics
identify the affected field, record, relationship, or relevant identity; the expected value or condition; and the
observed value or condition. Simple violations may use a concise sentence:

```text
Invalid fold_id: expected a nonempty string; got an empty string.
```

Validators use a structured multiline form when several context fields are needed:

```text
Invalid report outcome:
  report: performance
  status: complete
  expected: failure_ids to be empty
  got: 2 failure references
```

Diagnostics include only the context needed to locate and understand the violation. Large payloads and data structures
are represented by bounded identities, indices, counts, shapes, ranges, or short summaries rather than complete
values. These rules do not replace persisted failure provenance, operational failure reporting, or exceptions
propagated from Python, user code, or third-party libraries.

### 9.2 Failure scope and worker lifetime

Configuration/startup failures precede scientific evaluation. Missing execution targets or other run prerequisites
must not be reported as model-reconstruction or primary-validation failure, report failure, influence failure, or
worker-process failure.
The new computation lifecycle and local execution path do not alter the scientific failure-isolation rules below.

The worker is a long-running execution service expected to operate continuously, potentially for months. Handle
normal application-level failures at the smallest applicable scope, respecting actual computational dependencies:

- **Measure/report:** Represent formula-defined undefined values as `NaN` and computationally unavailable values as
  `None` with applicable failure references. Preserve successful results and identifiers, log the failure, and continue
  reports with valid inputs. Ordinary reports do not retrain a model to recover.
- **Shared predictions/intermediates:** Represent only dependent unavailable results as `None` and identify the
  originating failure when dependent work is skipped. Independence does not authorize fabricated inputs or repetition
  of failed work.
- **Initial reconstruction:** Failure prevents primary validation and every report branch from starting. Return a
  `reconstruction_failed` failure-only bundle without predictions or reports. Do not automatically retry reconstruction.
- **Primary validation:** After successful reconstruction, failure to initialize, fit, or predict any primary fold
  produces `validation_failed`. Preserve completed folds and unavailable predictions with failure provenance, continue
  later independent folds where possible, and do not retry the failed fold. Dependent reports are failed or blocked.
- **Influence analysis:** Influence requires complete successful primary-validation predictions but is independent
  from ordinary report computation afterward. A removal-model initialization, fit, prediction, or calculation failure
  is local to the affected sample, fold, and class contribution. Preserve independently computed influence values,
  continue other removal evaluations, mark the influence report `failed`, and use `completed_with_errors` when primary
  validation otherwise succeeded.
- **Bounded operation:** Exhausting a transfer, persistence, connection, or other configured retry limit fails only
  that operation. Stop retries and log its identity, limit, attempts made, and final error. Follow its defined scope's
  normal failure policy without additional recovery, fallback, recomputation, or special action solely for exhaustion.
  A configured limit does not authorize retries where the underlying operation's failure policy prohibits them.
  Section 7.4 explicitly schedules new connection rounds after a round fails; this is not a retry-budget reset for
  an exhausted delivery operation.

These failures do not terminate or restart the worker process. Task-level failure produces its defined outcome while
the worker continues with subsequent work; logical-result acceptance follows the separate persistence lifecycle.
Catch errors at the applicable boundary so clear, descriptive errors remain visible without escaping into termination
of the long-running worker service.

Log the evaluation/model identity, computational stage, report name, applicable fold/sample/class identifiers,
exception type, descriptive message, and traceback. Keep handling simple, explicit, and proportional; avoid cascading
retries and unnecessary recomputation. A failure-only result bundle is a complete failure outcome for reception and
arbitration, even though it contains no evaluation reports.

**Worker failure** specifically means the worker process itself is unavailable, such as after an unrecoverable crash,
machine shutdown, external termination, or comparable infrastructure/runtime failure. This is exceptional. Restarting
the worker is not normal task/error recovery; external intervention may be needed to start the worker module again.
Root restart reconciliation, a failed delivery, or local unavailability to another user does not mean worker-process
failure. Always state the actor, object, lifecycle stage, and scope of failure or termination.

### 9.3 Formula fidelity and generated scientific values

Implement metrics directly using their defined formulas, populations, and normalization. Standard metrics use
established conventional definitions unless a project-specific variation is explicit; record detailed formulas/report
conventions during relevant planning and implementation without reopening standard metrics solely for absent detail.
Cocoonet-specific/recomposed measures require explicit definitions, including Section 4.7's influence calculation.

Generated scalar scientific measure values use these internal representations:

| Condition | Internal Python representation | Value-level failure references | Human-readable output |
| --- | --- | --- | --- |
| Finite and mathematically defined | `float` | Empty | Numeric value |
| Mathematically permitted positive infinity | `float("inf")` | Empty | `Inf` |
| Mathematically permitted negative infinity | `float("-inf")` | Empty | `-Inf` |
| Mathematically undefined | `float("nan")` | Empty | `NaN` |
| Computationally unavailable | `None` | Nonempty | `NA` |
| Explicitly non-applicable | `None` | Empty, only when permitted by the owning schema | `NA` |

Do not add a parallel per-value status enum. Derive the condition from the value, owning schema, and value-level failure
references. Exact identifiers, categorical labels, booleans, and integer counts retain their declared types. Generated
categorical predictions use a present label without a value-level failure or `None` with applicable failure references.

A required scientific value cannot be `None` without applicable failure provenance. A concrete schema may permit
`None` without failures only for a non-applicable field. Formula-defined `NaN` does not acquire a failure solely because
it is undefined. A computational exception or failed prerequisite produces `None` with failure provenance rather than
`NaN`.

Accept infinity only where the measure's documented mathematical definition permits an extended-real result.
Prediction adapters require finite regression predictions and finite class probabilities. An invalid nonfinite model
output or floating-point overflow where infinity is not mathematically permitted is computational unavailability and
must be traced as a failure.

Small denominators and the resulting defined values remain unchanged. Do not introduce epsilon, clipping, imputation,
invented probabilities, fallback calculations, or NaN-skipping aggregation to obtain a preferred result. Report
formula-defined edge cases and risks, including accepted randomness, without treating them as ambiguities or reasons
to alter the computation. Seek clarification only for an actually missing custom definition or contradiction after
accounting for scope and context. Keep module computations simple, clear, direct, and efficient.

## 10. Subsequent Design Boundaries

The established invariants above are requirements: computation and parent identities are distinct; ordinary active
execution locks configuration changes, and root interruption permits prospective edits while old work stops.
Changed model evaluation configuration requires a new parent; worker-only changes preserve the parent. Dataset and
worker reuse preserve fixed scientific state; local and online targets share task and scientific contracts. Saved
partitions, classification-column schemas, persistence layouts, connection direction, package integrity, publication,
acceptance, failure scopes, and CPU capacity follow their explicit rules. Questions explicitly marked open are not
silently resolved by this section.

The following remain deferred to their responsible design or implementation phase:

- Exact online control-message schemas and framing.
- Canonical HMAC metadata byte encoding.
- Public-service authentication and certificate provisioning.
- Retry timing and acknowledgement encoding.
- Result compression parameter names.
- Exact recovery API signatures.
- Exact worker-service API signatures.
- Model/dataset transfer archive selection, size thresholds, tiling, and compression settings.
- Technical enforcement of model isolation and public cleanup verification.
- Whether inline/file-backed rows may coexist and whether uniform tensor shape is represented per sample or once.
- Conventional metric details required only by their later report implementation phases, and explicit definitions
  before implementation for any additional project-specific measure.

These decisions must preserve the fixed identifiers, layouts, ownership, partition, connection, integrity,
publication, classification-column, acceptance, and recovery contracts above.

These are implementation design boundaries, not a reopening of resolved ownership, acceptance, failure scopes, or
formula-defined `NaN` behavior. Record relevant designs before implementing the affected units, following the exact
documentation-first gate and authorized scope in root `AGENTS.md`.

The mandatory one-change-unit approval gate remains in force: one production function/method per approved step by
default, with only explicitly approved bounded exceptions, targeted tests, and approval before further production
work. This document authorizes no code changes or automatic commits. Root `AGENTS.md` and `pyproject.toml` govern
Ruff, mypy, Black, pytest, source documentation, and the configured documentation checks; preserve those requirements.

The initial worker contract does not establish distributed fold execution, communication between workers, rendered
visualizations as worker outputs, or an implementation roadmap. Extensions remain subject to the documentation and
approval workflow in root `AGENTS.md`.

### 10.1 Resolved clarifications and deferred implementation decisions

The dataset/partition association, startup availability, copying/consolidation, worker removal, and worker-dataset
cleanup questions from the prior review are resolved by Sections 2.3, 3.4, 5.3, 6.1, and 7.4. `add_dataset()` is the
confirmed artifact-preparation exception to memory-only configuration edits; `run()` alone persists execution
configuration. Root interruption permits prospective edits while old work stops with fixed configuration.
Model evaluation configuration changes require a new parent; worker-only changes preserve the existing parent,
which resumes when evaluation configuration is unchanged. Blank advice entries do not introduce alternatives to
those resolved contracts.

The root-directory overwrite question is resolved by the ownership invariant in Sections 2.3 and 6.1: configured
user dataset source paths cannot equal or lie within the managed root. Validate both dataset-path and root-directory
changes; reject a conflicting root before modifying persistent files. Managed-root overwrite therefore needs no
dataset-path exclusions. Dataset removal remains reference-only, and existing partition/recovery protections apply.

The deferred decisions listed in Section 10 remain implementation boundaries, not permission to alter the resolved
contracts. Object-copy mechanics, history presentation, connection-probe mechanics, timer representation, and
coordination of references to the same persisted execution also remain implementation details under those contracts.
The separately supplied external public-service interface remains outside Cocoonet; public pool policy is not inferred
or implemented here.

Expected undefined scientific values, masked-out populations, and ordinary operation failures do not themselves
reopen the metric, retry, or failure contracts. No mechanism may change an existing parent's configuration, reinterpret
a late result as belonging to a new parent, silently regenerate a saved split, or reset an exhausted delivery budget.
