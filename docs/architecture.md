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

## 2. System Context and Responsibilities

| Component | Initial responsibility |
| --- | --- |
| User | Supply the dataset, sample metadata, model inputs, and evaluation directives. |
| Root/coordinator | Own computation/task coordination; prepare local data access or online synchronization, assign tasks, and accept persisted results. |
| Worker | Serve as an explicitly configured local or online execution target; evaluate complete unit tasks and return selected result files. |
| External public-worker service | Own public discovery, worker selection, reservation/allocation, and availability. |

Workers execute independently without inter-worker communication or dependence on shared mutable worker state.
Internal evaluation state remains local. The coordinator may maintain scheduling, assignment, lifecycle, failure,
and result-acceptance state; this does not make workers depend on each other's execution state.

Make the exact dataset ready before tasks use it, through online synchronization or local artifact access as applicable;
then supply model inputs and execution directives for subsequent tasks.
Each completed attempt produces one serialized evaluation-result file. The root receives only selected files through
the reception process in Section 5.

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
    Owns durable local persistence of received evaluation-result files.
    Determines whether persistence succeeded.
    A result becomes accepted only after successful persistence.
    Persistence failure is reported through local control or the online
    communication component so the applicable retry lifecycle can proceed.

## Worker Node
Worker communication component
    Owns communication through the applicable local or online root boundary.
    Owns status exchange and receipt of tasks and control instructions;
    network heartbeat/reconnection applies only to online workers.
    Remains responsive independently of model-evaluation execution.
Worker control component
    Owns the lifecycle of the currently assigned model-evaluation attempt.
    Coordinates synchronization, evaluation workflow, cancellation, result
    availability, and delivery.
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
    sklearn and sklearn-like models must not create their own
    multiprocessing execution pools. Multiprocessing for these workflows
    is owned and controlled by Cocoonet.

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
Worker report persistence component
    Owns local persistence of standardized evaluation/report outputs.
    Evaluation components themselves do not independently own filesystem
    persistence.
Result packaging component
    Collects the standardized outputs belonging to one model-evaluation
    attempt and constructs the single standardized Cocoonet result dill
    file used for transfer.

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
    PyTorch-specific training/device behavior remains under the PyTorch
    execution contract rather than being assumed identical to sklearn
    multiprocessing.

## Delivery
Delivery component
    Owns the result transfer and resend lifecycle after a complete result
    file becomes available.
    Does not perform model evaluation or reconstruct evaluation results.
    Delivery failure is contained within the defined delivery/attempt
    lifecycle and does not terminate the worker process.

### 2.2 Private and public worker connections

This section describes online workers. The separate root-local boundary is defined in Section 2.4.

- Every online worker serves at most one authenticated root/user at a time through Cocoonet's internal protocol.
- Private-worker registration returns connection information and a unique credential directly to its user through
  the worker module, for example as a dictionary for root configuration. The root initiates the connection using that
  information. Private workers never appear in public discovery and remain under explicit user lifecycle control.
- For public workers, the root requests an idle worker from the external service, receives the selected worker's
  connection information, and initiates the connection. Workers do not connect back to the root or to other workers.
- The public-worker service is outside Cocoonet's scope, for example a separately planned `CocoonService`. It owns
  availability, idle-worker selection, reservation/allocation state, and decisions about availability after release.
  Pool state remains internal to that service; Cocoonet receives connection information rather than managing the pool.
- Cocoonet may send lifecycle notifications required by the supplied interface, such as release notices. Reservation
  durations, timeout-based release, and credential-renewal workflows for the public pool are not Cocoonet policies.
  Do not invent the endpoint, authentication, request/response format, connection-information format, or notification
  protocol before the external interface is supplied for implementation.
- Public and private workers use the same direct authenticated evaluation protocol. The external service does not
  proxy datasets, models, tasks, reports, or evaluation traffic. Availability for another task under the owning root
  does not itself release a public allocation.

Transport is defined in Section 7; worker-side cleanup and its service boundary are defined in Section 5.3.

### 2.3 Reusable computation, initialization, and user workflow

A **computation instance** holds root-side dataset, model, worker, and evaluation configuration; it is not a parent
task. The first evaluation establishes a parent. Thereafter, changed model evaluation configuration requires a new
parent, while unchanged model evaluation configuration resumes the existing parent, with or without worker
configuration changes. An ordinary copy preserves those same identity rules.

#### Root directory and initialization

Construction requires a user-provided root working directory and an `overwrite` choice. The root working directory
is Cocoonet-managed persistent storage. User-provided dataset source paths are user-owned and must remain outside it.

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
Equality and containment concern filesystem locations; the concrete path-validation mechanism remains deferred.

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
execution configuration. Start alone commits that execution configuration and makes the instance consolidated.

`start()` from `initialized` validates the current configuration, persists it in the root working directory,
establishes a fresh parent, and begins the workflow. After configuration is committed to execution, state becomes
`consolidated`. Start is the sole boundary converting mutable configuration into persisted execution configuration;
it is not the sole writer of execution artifacts. Partition information, lifecycle/recovery state, results, reports,
and failure information are persisted by their respective workflow operations.

Model evaluation configuration comprises the dataset and realized partition, supplied models/model inputs, and
evaluation settings. A permitted change to that configuration after consolidation makes the instance
`unconsolidated`, whether constructed, copied, or reused, and requires a new parent. Start rejects unconsolidated
state: explicitly initialize a directory for fresh execution first. An empty directory does not implicitly change
the object's state. Edits before first start leave the instance initialized.

Worker configuration, including additions, removals, and local/online topology, is separate. Worker-only changes
neither make the instance unconsolidated nor change parent identity. With unchanged model evaluation configuration,
start resumes the existing parent and preserves its fixed scientific inputs, progress, results, and retry limits,
with or without worker changes. Calling start again does not itself establish a new evaluation. Worker configuration
changes remain subject to the ordinary execution lock and post-interruption rules; runtime use still requires
availability, ownership, readiness, and applicable cleanup.

```text
construct / initialize(root_dir, ...) -> initialized
initialized + configure in memory    -> initialized
initialized + start()                -> persist configuration -> fresh parent -> consolidated
consolidated + unchanged evaluation  -> same parent / resume, with or without worker changes
consolidated + permitted evaluation edit -> unconsolidated
unconsolidated + start()             -> reject
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
or accepted result files. A consolidated copy remains consolidated and refers to the same persisted execution.
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

The normal workflow is construction/initialization, in-memory configuration and dataset/split preparation, fresh start,
execution, and subsequent permitted reuse or recovery. No exact method ordering is required beyond dependencies.

| Area | Required capabilities |
| --- | --- |
| Initialization | Prepare a root directory explicitly for fresh execution, with overwrite behavior. |
| Dataset | Add/set with partition preparation or validated reuse, inspect, remove configuration references. |
| Models | Add one/multiple, list/inspect, summary, remove one/multiple. |
| Workers | Add one/multiple, list/inspect, summary, configuration-only remove one/multiple; separate dataset/all-user-data cleanup and public release. |
| Execution | Fresh start, break/cancel, resume/recovery of existing persisted execution. |
| Reuse | Permitted configuration edits, ordinary copying, explicit reinitialization for changed model evaluation configuration; worker-only changes preserve the parent. |

Singular mutations are primitive semantic operations; bulk forms are explicit conveniences. Define commit/failure
behavior before implementation so validation failure cannot silently leave undocumented partial configuration.
A general root persistent-file clear API is not required; file management remains separate, with ownership boundaries
in Section 6.1. Exact signatures, aliases, containers, exceptions, and storage layouts remain implementation details.

### 2.4 Execution targets and local/online boundaries

Networking is required only for online workers. Local-only execution must work without initializing the online-worker
network subsystem. Online authentication, heartbeat/reconnection, network synchronization, and result transfer apply
only to online targets. The task, scientific, isolation, persistence, and acceptance contracts remain shared.

The root always owns coordinator responsibilities. It does not automatically act as a worker. An explicitly configured
root-local worker and configured online workers are independently optional; supported topologies are local-only,
online-only, and mixed local/online execution. Root coordination itself does not provide an implicit execution target.

Zero configured workers is valid before start. Start requires a dataset, one or more models, and at least one usable
execution target before evaluation begins. With no configured target, fail explicitly; do not run models in the
coordinator, automatically enable a local worker, construct a fake network worker, or wait indefinitely for an
unconfigured worker. This is configuration/startup failure, not baseline model-training failure.

A configured worker is a configuration entry; a runnable/available worker also meets applicable connection, readiness,
and ownership conditions. Start uses currently runnable targets without waiting for all configured online workers.
If none is runnable, fail startup explicitly. Configured disconnected workers remain under the independent periodic
connection policy in Section 7.4; a later start may succeed after one becomes runnable. An unavailable worker does not
prevent other ready workers from receiving tasks.

The root-local worker is a separate locally managed worker process, reached through a direct local execution/control
boundary. User model code must not run inside the root coordinator. Do not use WSS loopback or another fake network
path to imitate an online worker. Local execution does not require network authentication, heartbeat/reconnect,
network dataset synchronization, or network result delivery.

The local worker is an execution target using the same initialization, execution, scientific-reporting, and failure
contracts as online workers, not a second evaluation implementation. It consumes the same parent configuration,
exact dataset identity, persisted partition memberships, unit/attempt identities, and model execution restrictions.
Cocoonet supplies local data/model artifacts through the framework boundary; models still have no direct filesystem
or network permission. Local artifact access must resolve the same logical source/sample associations as online
synchronization. Whether artifacts are shared or copied locally is an implementation choice, not a new dataset identity.

The local control boundary must support assignment, status, stop/cancel control, result availability/selection, and
confirmation of root persistence while the coordinator and worker control remain responsive. Complete standardized
result files pass through the same root acceptance contract; a local outcome is not accepted solely because evaluation
has finished. Local handoff and any persistence retry use the existing completed outcome rather than recomputation.
Exact IPC, artifact-access mechanisms, and message representation remain deferred. The enforced model restrictions in
Section 7.3 remain necessary in addition to the separate process.

In mixed execution, one coordinator schedules both local and online targets through the same logical task system.
A local attempt and an online attempt of the same unit obey identical reception selection, at-most-one acceptance,
persistence confirmation, and duplicate-stop rules. Topology does not grant priority or create a separate result
lifecycle. The control/handoff mechanism differs; scientific execution and logical outcome semantics do not.

## 3. Dataset and Model Contracts

### 3.1 Dataset and sample identity

A configured computation has exactly one active dataset, shared by all unit tasks of each parent created from it.
It may have no active dataset while configuration is incomplete, but cannot start a parent without one. A valid
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

Detailed data schemas, key encoding, and synchronization verification belong in the corresponding implementation
design. The sample-identity scope is the active dataset; cross-dataset identity tracking is not required.
Multiple-target and multilabel extensions are not established by this document; the influence formulas below
explicitly cover single-target regression.

### 3.2 Model identity and supported inputs

The root assigns each supplied model an arbitrary, module-generated unique `model_id` and associates it with a
user-provided unique `model_label`. Labels distinguish models within their parent model evaluation task. Cocoonet
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

Device selection belongs to worker configuration. Workers may use different supported device types. Shared seeds and
recorded configuration support auditability without promising identical numerical results across devices.

### 3.4 Model evaluation tasks, unit tasks, and execution attempts

A **parent model evaluation task** is one fresh evaluation run created by `start()` from initialized state. It contains
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

Start consumes the current validated association; neither a separate explicit split call nor automatic splitting at
start is required. Every unit and attempt within that parent uses the same realized memberships. Workers and models
must not regenerate, reshuffle, or alter them. Root-side partition randomness is accepted when the set is generated;
reuse preserves the actual memberships rather than regenerating them from a seed.

Immediately after generating the complete partition set, persist it once in one authoritative `dill` file on the root.
Preserve the dataset identity, sample-ID/label association, and fold/split memberships needed to validate reuse.
Persistence must complete before any unit-task allocation. Do not divide or duplicate the authoritative set across
files or repeatedly persist it during normal execution. Worker copies are execution inputs. Parent execution state
references the dataset and this artifact; the artifact need not be rewritten to attach every later parent.

A later `add_dataset()` may select an existing valid artifact or create another one. Replacing the computation's current
split association does not authorize overwriting a file still required by an existing parent. Parents preserve their
original dataset/partition associations for their lifetime. Another computation may reuse `split_path` only after the
same validation. The exact signature, default filename, in-file container, and validation mechanism remain deferred.

Supply a copy of the applicable indices together with online dataset synchronization, or local artifact preparation,
during parent-context setup. Workers use those indices to select data from the exact ready representation; they never
invoke the splitter. Worker copies are execution inputs, not additional authoritative root copies. Reusing a retained
ready dataset does not require retransmitting it, but the applicable parent partition context must still be supplied.

After a root restart following successful partition persistence, load the indices directly from the existing `dill`
file. Do not rerun the splitter or create a replacement set. Keep the file associated with the exact dataset
representation for which the indices were generated. The single-file lifecycle and pre-allocation persistence gate
are resolved architectural requirements; in-file containers, filenames, directories, and the mechanism that ensures
durable completion remain implementation details.

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

Fold computations and sample-removal evaluations remain internal to an execution attempt. Their partition definitions
come from the shared parent schedule. Fold results retain their producing attempt context and their association with
the corresponding parent-defined fold and training/test sample identities. A completed attempt produces one result
file associated with its unit task and producing attempt; at most one result is accepted per unit task under Section 5.
The existing one-model worker-task and logical-task lifecycle rules apply to unit tasks in this hierarchy. In
particular, baseline-training failure ends that unit task's current evaluation, not automatically the entire parent
model evaluation task or the worker process.

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

## 4. Worker Evaluation

### 4.1 Minimum task unit

One worker task is a complete evaluation process using the dataset and one supplied classification or regression model.
The worker runs all required folds or splits and generates the required reports, returning its result file when
selected for reception by the root. Optional sample influence analysis belongs to that same evaluation when requested.

A fold, a held-out sample, or an individual influence refit is not a separate distributed task in the initial design.
One completed attempt produces one evaluation-result file, including its failure information. Duplicate attempts belong
to the same logical task, with at most one accepted result under Section 5.2. Delivery retries resend the existing file;
they do not rerun the evaluation. Essential baseline-training failure produces a failure-only file without reports.

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
influence results, and construct one result file. Dataset/partition preparation has already completed before parent
allocation.
Return that file when selected for reception by the root. This sequence does not prescribe function boundaries
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

- If required reconstruction or any baseline fit fails, stop the entire current model evaluation, including influence
  analysis, and produce the failure-only outcome. Do not return evaluation reports or retry the failed training.
- After required baseline training succeeds, isolate report/measure failures and continue computations with valid
  prerequisites. Shared prediction failures affect only their dependent results. Ordinary reports do not retrain
  models to recover from errors; influence removal refits are part of their specified computation.
- Preserve identifiers and successful values. Undefined/unavailable values remain `NaN` as defined, with failures
  logged at their applicable scope. No application-level evaluation failure terminates the worker process.

Section 9 defines failure and formula fidelity rules. Specify output ordering, concrete fold representation, and
report aggregation conventions during the corresponding schema/report design using conventional metric definitions
and the explicit project variations below.

### 4.4 Fold outputs shared by both task types

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
sample-level reports in Sections 4.5 and 4.6. Concrete schemas, field names, and ordering remain implementation details;
keep sample/class associations intact within the single evaluation-result file.

### 4.5 Classification reports

#### Validation results

For the sample coverage defined in Section 4.3, use the common sample information in Section 4.4: internal
`sample_id`, user-facing sample label, source references, and true class label. Associate predicted class labels and
probabilities for every explicitly identified class with those samples. Preserve internal alignment while using the
sample label as the user-facing identifier.

#### Performance report

Include the confusion matrix and class names, precision, recall, F1-score, accuracy, and AUC (area under the curve).
Include TP, TN, FP, and FN counts associated with the class-level evaluation. Macro- and micro-averages are requested
for the performance metrics. Macro-averages of TP, TN, FP, and FN are explicitly `None`.

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

Associate the common sample information in Section 4.4 with predicted probabilities of every class and the following
computed fields. Retain internal `sample_id` alignment and identify samples to users by their labels:

- Probability residual for every class.
- Z-score of probabilities.
- Sum of absolute residuals and root sum of squared residuals.
- Shannon entropy of absolute residuals.
- Mahalanobis distance of the residual vector.

Compute classification residual measures across the class entries within each individual sample, independently of
other samples. This axis applies to probability z-scores and residual summaries. Do not substitute normalization or
covariance estimates computed across samples.

Document the conventional per-measure formulas, encoding, sign, and report conventions with the relevant report
implementation, preserving this explicit axis requirement. Any project-specific variation requires an explicit
definition. Formula-defined undefined values remain `NaN`; report their risks without adding fallback calculations.

#### Visualization data

Return data for plotting ROC curves, including AUC for each class. Return plotting data rather than rendered plots to
reduce transfer load. Specify ROC array layouts and references to corresponding performance-report fields during
report-schema implementation.

#### Optional sample influence report

Report influence for training samples only, using the same sample structure as testing samples. Associate each omitted
sample's internal `sample_id`, user-facing label, source references, and true class label with one generalized Cook's
distance-like influence value per class. The score
describes the normalized effect of removing that training sample on fixed test-set probability predictions.
The predictions used for this computation belong to the test samples; no prediction for the omitted training sample
itself is required. Test samples receive no influence score from a fold in which they are held out. In cross-validation,
a sample can receive contributions from other folds in which it belongs to the training partition.

Apply the formulas in Section 4.7 with the following requirements. Retain each sample's prediction-comparison count
and evaluation status with its influence values; their field encoding belongs to the report schema.

- Reuse the exact validation training and test partitions. Within each fold, omit each training sample once, refit
  the configured model, and compare baseline and removal-model probabilities on that fold's unchanged test set.
- Set `p` to the number of actual predictors in input `X` supplied for fitting. Do not substitute a parameter count
  or an alternative inferred feature count.
- Calculate baseline MSE against true class indicators on the test set, separately for every class. Keep class
  identities aligned. This average across test samples is distinct from the residual report's within-sample summaries.
- Aggregate using the approved prediction-comparison count `M_i = sum_{f in F_i} |V_f|`, including comparisons whose
  prediction change is zero. Retain the count with the report. Only folds in which the sample is a training candidate
  contribute; samples that are exclusively held out have no influence result.
- Leave undefined influence values as `NaN`, including undefined division at zero MSE. Leave small positive MSE and
  the resulting defined values unchanged. Do not add epsilon, thresholds, clipping, imputation, or other adjustments.
- If a missing class or a failed fit/prediction prevents evaluation, leave the affected influence values `NaN` and
  raise a clear, descriptive error identifying the affected fold, sample, and class where applicable. Do not fabricate
  missing probabilities, introduce fallback fits, skip undefined contributions, or shrink the divisor to obtain a
  numeric result. Undefined contributions propagate as `NaN` in the affected aggregate. Catch and log removal-evaluation
  errors at their sample/class boundary and continue other independent computations. Essential baseline-fit failure
  instead stops the entire model evaluation and produces no evaluation reports, as specified in Section 9.2.
- Use the model's configured fitting and preprocessing procedure for baseline and removal fits. Accept any randomness
  influence from that procedure. Do not introduce extra seed coupling, repeated fits to select a preferred result,
  or post-computation adjustments to remove stochastic effects.

These are defined computation outcomes and reporting rules. Report their edge cases and risks without reopening
formula-defined undefined values or accepted randomness effects as design ambiguities. Keep the computation direct.

### 4.6 Regression reports

#### Validation results

For the sample coverage defined in Section 4.3, use the common sample information in Section 4.4: internal
`sample_id`, user-facing sample label, source references, and true target value (`true y`). Associate the predicted
target value (`predicted y`) with each applicable sample without changing that underlying sample structure.

#### Performance report

Include `Mean_Error`, `SDE`, `MAE`, `Normalized_MAE`, `CV_MAE`, `MSE`, `RMSE`, `Normalized_RMSE`, `CV_RMSE`, `RPD`,
and `R2`.
Use established conventional definitions unless Cocoonet explicitly specifies a variation. Record detailed formulas,
normalization, sign, standard-deviation, and report aggregation conventions during the corresponding planning and
implementation steps. Do not treat absent formula detail alone as an ambiguity or invent a project-specific measure.
Undefined results, including formula-defined constant-target cases, remain `NaN` without corrective manipulation.

#### Residual report

Associate the common sample information in Section 4.4 with predicted `y`, residual, and standardized residual.
Retain internal `sample_id` alignment and identify samples to users by their labels.
Use conventional residual and standardization definitions, with their sign/population documented alongside the report
implementation. Preserve undefined results as `NaN` and apply the same scoped failure rules as other reports.

#### Visualization data

- Data for a scatter plot of true target values and predicted target values.
- Data for a residual plot of predicted target values and residuals.

Return plotting data rather than rendered plots. Specify its representation and reuse of validation/residual report
fields alongside the report-schema implementation.

#### Optional sample influence report

For single-target regression, use the common sample information in Section 4.4 for each omitted training sample:
internal `sample_id`, user-facing label, source references, and true target value. Associate one generalized Cook's
distance-like influence value per sample using Section 4.7, accompanied by its prediction-comparison count and
evaluation status. This measures the mean normalized change in held-out target
predictions caused by removing that sample from eligible training partitions.

Baseline MSE compares the fold's baseline target predictions with its true held-out targets. Both baseline and removal
fits predict on the fixed test set; a prediction for the omitted training sample itself is not required. Test samples
receive no influence metric from a fold in which they are held out. A sample may contribute through other folds where
it is in training; a sample never in training has no influence metric, rather than an observed zero.

Keep undefined contributions as `NaN`, preserve the comparison population, and isolate removal-evaluation failures
under Section 9.2. Count/status field encoding belongs to the report schema; it does not change the approved formula.

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

This computation uses the validation baseline fits and predictions. If a required baseline fit fails, stop the whole
model evaluation and return its failure-only outcome, including no influence report. A removal refit or calculation
failure is local to the affected sample/class contribution and is caught and logged without stopping other independent
work or the worker process. Shared prediction failures affect only their dependent results; see Section 9.2.

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
- **Missing classes and removal-evaluation failures:** If a missing class or failed removal fit/prediction makes a
  contribution undefined, leave the related influence values `NaN` and raise a clear, explicit error identifying the
  fold, training sample, and class where applicable. Catch and log it at that boundary, preserve independently
  computable class values, and continue with other samples. Removing the only training example of a class must not
  trigger invented probabilities, zero filling, or fallback fits. Essential baseline-fit failure instead follows the
  task-wide failure rule in Section 9.2; it produces no evaluation reports.
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

When tasks are ready, check configured idle online worker candidates asynchronously. Skip unavailable candidates.
Among responsive eligible idle online workers, allocate first to the fastest connection response observed for that
opportunity, then continue assigning tasks to the remaining eligible idle workers until tasks or targets are exhausted.
Response speed is an allocation-time observation, not a permanent priority. This chooses online execution targets;
Section 5.2 still determines which logical task is eligible for duplicate execution. Local workers require no
artificial network probe, and result acceptance remains topology-neutral.

Distinguish the logical task from its execution attempts. Each accepted evaluation-result file has a stable identity
associated with the logical task and identifies its producing attempt. Identifier encoding belongs in the protocol
design. Simultaneous arrivals and reconnects must not create more than one accepted result for a logical task.

### 5.2 Duplicate execution and ordered reception

These rules apply across local and online attempts of the same unit. Local handoff and online transfer feed the same
root selection and persistence boundary; neither topology receives a separate acceptance rule or priority.

- Allocate a duplicate only when no unallocated tasks remain, every remaining task is finished or already assigned,
  and a worker would otherwise be idle. Choose an unfinished, currently running task, starting with the oldest by
  its first execution start time. The duplicate is another attempt at that logical task, not a new logical result.
- Serialize result reception for each logical task. When duplicate attempts are ready, select one according to
  receiving order and receive only that file at a time. Reception for different logical tasks remains independent.
- Selection does not depend on whether evaluation succeeded or failed. An earlier received failure outcome takes
  precedence over a later success; defined `NaN` and recorded computational failures do not invalidate the outcome.
- Complete the logical task only after the selected file is entirely received, successfully persisted, and accepted.
  Then send stop instructions to all remaining duplicate attempts without receiving their result files.
- Computing workers check stop instructions before each evaluation loop and stop at those cooperative boundaries;
  immediate interruption of arbitrary model code is not required. Ready duplicate attempts await reception selection
  or a stop instruction instead of sending unselected files. Stopping an attempt does not stop the worker process.
- Persistence waiting and resends apply only to the selected file. Other duplicates' unreceived files do not enter
  the resend process and are not delivery failures. Failure of the selected transfer/persistence operation does not
  promote a later duplicate result; apply the bounded retry rules in Section 6.3.
- Duplicate execution reduces latency, not recovery from computation failure. Do not automatically retry/reallocate
  a terminal baseline-training failure or start another duplicate while an existing completed result awaits delivery.

Specify attempt identifiers, stop messages, and cancellation acknowledgements later without changing these scopes
or moving duplicate cancellation ahead of successful result acceptance.

### 5.3 Worker ownership, release, and cleanup

Workers are reusable resources of the root/computation configuration, not objects whose lifetime is exclusively owned
by a single parent task. Successive parents may reuse configured workers, online connections, a local worker process,
and retained valid datasets according to their applicable ownership and retention rules. This does not establish exact
process/connection lifetimes, runtime sharing between copied computations, or automatic release on parent completion.
Exact local-worker startup/shutdown and root-restart lifetime remain deferred. Public allocation ownership remains
external. Configuration removal has the following resolved scope.

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
| Start/cancel/recovery | Execution lifecycle, independent of configuration removal. |

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

Resume/recovery uses the existing parent's unchanged model evaluation configuration, preserving parent identity and
root directory even when worker configuration changes. It does not invoke fresh initialization or create a new parent.
Reconstruct progress from the known task set and
complete root-persisted result files, then
reconcile worker state through the applicable boundary. Online reconnection is root-initiated; local process
reconciliation uses local control. Recovery does not rely solely on pre-crash memory.

Before resuming dispatch, restore each parent's dataset association and evaluation configuration from its committed
execution state, then load its schedule from the referenced authoritative root-side `dill` file. That file was durably
written once, immediately after the complete partition set was generated and before any unit-task allocation.
Existing computations, resumed work, and duplicate attempts retain those same fold/sample memberships. Do not rerun
the splitter, create replacement partitions, or duplicate the authoritative persisted set during recovery.

- **Still computing:** If no accepted result exists and the parent has not been stopped/cancelled/superseded, restore
  the assignment and let it continue. Otherwise preserve its stop disposition under Section 5.5. Apply duplicate-stop
  rules once another attempt's result is accepted.
- **Complete file already persisted:** Reconstruct the task as finished and confirm delivery as needed. No
  reevaluation or retransmission is required, including after a crash between persistence and an in-memory update or
  acknowledgement. Stop remaining duplicates without receiving their files; workers can resume their normal task flow.
- **Selected file pending delivery:** If the root lacks the selected file and its worker retains the completed result
  with retries remaining, resume that delivery within its remaining limit before assigning the worker another task.
  Preserve the reception selection; do not receive an unselected duplicate's file in place of it.
- **Delivery retries exhausted:** That operation remains failed through reconnection or restart. Do not silently reset
  its limit or restart it merely because a local file remains. Normal operation failure handling applies.

Preserve the receiving-order selection even if its file was not fully persisted before the crash. Define durable
evidence for that pending selection before implementing recovery. Selection alone is not acceptance or completion;
an incomplete file is not evidence of a completed task. Keep root recovery distinct from exceptional worker-process
failure and restart, described in Section 9.2.

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
The authoritative partition file is a Cocoonet-generated artifact under the partition/recovery contract, not an object
of worker-dataset cleanup; it is not a user-provided dataset source path.

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

The exact selection API, storage layout, readiness verification, and deletion implementation remain deferred and must
preserve source-file ownership, all-target inactivity before cleanup, and invalidation after successful deletion.
Operational cleanup failures remain subject to Section 9; public release still requires confirmed cleanup success.

The key encoding, synchronization verification, and filename-reference layout are implementation details that must
preserve the row/sample associations and readiness invariant.

### 6.2 One file per evaluation attempt

A completed attempt produces one evaluation-result file containing all its reports, results, and failure information.
Independent computational failures preserve successful results alongside unavailable/`NaN` values and logged errors.
The file represents the complete attempt outcome, including partial computation failures; individual reports do not
have separate network-transfer or persistence states. Essential baseline failure produces a failure-only file with
status/error information and no evaluation reports.

Result files are expected to be small. Use asynchronous I/O to write received serialized files promptly to local
storage. Bounded-memory streaming and backpressure are not required by the initial design. Never deserialize a report
file in the active network event loop; root-side loading and content processing occur separately from that path.

### 6.3 Persistence, acknowledgement, and bounded resends

Root persistence and outcome acceptance apply to both local and online execution. Network transfer/resend applies to
online workers; local handoff and persistence confirmation use the local boundary. Retry handling for an applicable
local persistence/handoff operation also reuses the existing outcome and cannot authorize reevaluation.

- The root receives only the file selected under Section 5.2. Accept it and mark the logical task finished only after
  the complete file has been received and successfully written to persistent local storage. Confirm persistence to
  the selected worker. Receiving bytes, incomplete writes, or transient memory state alone do not establish completion.
- Never overwrite an already accepted result with a later duplicate. Stable logical-task/attempt identity and a
  distinction between complete and incomplete files support restart reconciliation.
- Transfer or root-persistence failure for the selected file leaves the logical task unfinished. Log the operation
  failure and request resending that same existing worker file within its configured retry limit; do not recompute
  the evaluation. Report-computation failures inside a complete file require no resend and do not prevent acceptance.
- After sending the selected file, its worker waits for persistence confirmation or a resend request. Until root
  persistence succeeds or the retry limit is exhausted, retain that file and do not assign a new task to its worker.
- At exhaustion, fail only the delivery operation, end its wait, and log the operation, configured limit, attempts
  made, and final error. Return to ordinary task handling with the worker process still running; new work requires
  applicable local-control/online-connection, ownership, and readiness conditions. The result remains unaccepted and
  the logical task is not
  finished by that delivery failure. Exhaustion authorizes neither recomputation nor additional special recovery.
- Unselected duplicate attempts do not enter this persistence/retry sequence. After successful acceptance of the
  selected file, stop the remaining attempts without receiving their files or treating them as failed deliveries.

Exact retry counts/timing, integrity checks, file publication mechanisms, naming, and acknowledgement encoding belong
in the protocol/implementation design. They implement these acceptance semantics rather than redefine them.

### 6.4 Worker-side retention

Configure retention as deletion after confirmed delivery, retention until all root tasks finish, or no automatic
deletion. Retain pending-delivery files as required by the delivery rules. Mandatory public release cleanup overrides
local retention, including pending delivery: no previous user's file may survive reassignment. Outside that release
cleanup, do not prematurely delete a pending-delivery file. Cleanup of an undelivered file does not finish its logical
task. Retention conditions, operation failure, result acceptance, and worker-process lifetime are distinct scopes.

## 7. Transport, Serialization, and Trust

### 7.1 Runtime and connections

Support Python 3.12 through the latest stable Python release, excluding prereleases. Use `asyncio` and `aiofiles`.
The following network requirements apply only to online workers; local-only execution does not initialize this
subsystem. The root initiates secure WebSocket (`WSS`) connections to online workers over port 443 using configured
private-worker
information or public connection information supplied by the external service. Workers expose only the designated
Cocoonet endpoint reachable from the root. They never initiate connections to the root or to other workers.

Keep the bidirectional connection open for assignments, status, dataset/model/configuration transfers, selected
results, and subsequent tasks. Reconnection is also root-initiated. Do not replace this model with worker polling or
separate task-submission endpoints. Sending a result alone does not make a worker available for another task; apply
the acknowledgement, pending-delivery, retry-limit, and restart rules above.

### 7.2 Serialization ownership and compatibility

- Use `dill` exclusively for Cocoonet serialized payloads; do not introduce a parallel `pickle` path.
- Only the root creates model payloads for workers, following Section 3's model preparation contracts. Only workers
  deserialize those payloads, within the technically enforced model-execution boundary.
- Workers construct and serialize standardized evaluation-result files. Only the root deserializes those files,
  outside the active receive event loop. Restrict model/result loading to their designated paths. Separately, the
  root loads its authoritative partition `dill` file through the designated dataset/partition preparation or recovery
  path, validating the dataset association before reuse.
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

Online workers accept model payloads only from the authenticated owning root through the designated transfer path.
The root-local worker receives models only through its owning root's designated local boundary. Authentication and
required-API validation do not make executable model content safe. Do not execute arbitrary received payloads or
expose a general-purpose remote-execution interface.

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

Use the same periodic, configurable connection policy before start, during active evaluations, and after disconnect.
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
Start does not wait for every worker or for an all-unavailable set to recover: it requires a currently runnable target,
otherwise startup fails while connection management continues independently. Timers, exact parameter names, response
measurement/probes, and concrete connection-state representation remain implementation details.

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
The worker control component owns the lifecycle of the current model-evaluation attempt. Model-evaluation workflow components own their evaluation execution and any Cocoonet-managed evaluation subprocesses. Application-level evaluation or subprocess failure follows the scoped failure rules in Section 9 and must not terminate the long-running worker process.
Stop and cancellation instructions received through the communication/control path are applied at the cooperative boundaries defined by the task lifecycle. This execution separation does not require immediate interruption of arbitrary user model code.

### 8.2 Parallel execution policy

Each local or online worker has a user-configurable CPU budget `n_cpu`, defaulting to that machine's available CPU
count `N`. Its resulting evaluation CPU capacity is:

```text
nCPU = max(1, min(n_cpu, N) - 1)

n_cpu = user-configured CPU budget (default: N)
N     = available CPU count
nCPU  = resulting evaluation CPU capacity
```

`nCPU = 1` means serial evaluation; parallel evaluation occurs only when `nCPU > 1`. Both `n_cpu = 1` and `n_cpu = 2`
intentionally produce serial evaluation. Apply the same conservative rule to online workers. The effective CPU
capacity must not exceed the formula's result. The minimum capacity of one takes precedence: this formula does not
guarantee that one physical CPU remains unused. Coordinator and worker-control responsiveness are independent
requirements, including during serial evaluation.

Cocoonet owns multiprocessing for scikit-learn and scikit-learn-like evaluation workflows. Individual scikit-learn models and custom scikit-learn-like models executed through these workflows must not independently create or manage multiprocessing pools. This prevents nested or model-dependent multiprocessing behavior from bypassing Cocoonet's worker-level resource allocation.
The corresponding model-evaluation workflow component owns creation, supervision, and termination of Cocoonet-managed evaluation subprocesses. Parallel work remains internal to the complete worker task for one model; it does not turn folds, influence refits, or other internal computations into separate distributed root-worker tasks.

### 8.3 Model-specific execution
Scikit-learn and scikit-learn-like workflows follow the Cocoonet-managed multiprocessing policy in Section 8.2. Their parallel execution is controlled at the evaluation-workflow level rather than independently by individual models.
PyTorch execution follows its separate reconstruction and training contract in Section 3.3. Device selection remains worker configuration. Do not assume that PyTorch CPU/GPU execution follows the scikit-learn multiprocessing model solely because both execute within the same worker architecture.
The multiprocessing restriction in Section 8.2 concerns process-level parallelism. Thread-level parallelism used internally by numerical libraries, model implementations, or framework runtimes is a distinct resource-management concern. No additional thread-level restriction is established here unless explicitly defined in subsequent design work.
Concrete subprocess implementation, process-pool structure, scheduling mechanics, CPU detection, configuration fields, and termination mechanisms are implementation details. They must preserve the execution boundaries, user-selected resource limit, Cocoonet-owned multiprocessing, and worker responsiveness established by this section.


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

### 9.2 Failure scope and worker lifetime

Configuration/startup failures precede scientific evaluation. Missing execution targets or other start prerequisites
must not be reported as baseline-training failure, report failure, influence failure, or worker-process failure.
The new computation lifecycle and local execution path do not alter the scientific failure-isolation rules below.

The worker is a long-running execution service expected to operate continuously, potentially for months. Handle
normal application-level failures at the smallest applicable scope, respecting actual computational dependencies:

- **Measure/report:** Leave affected values unavailable or `NaN` as defined, log the failure, preserve successful
  results and identifiers, and continue reports with valid inputs. Ordinary reports do not retrain a model to recover.
- **Shared predictions/intermediates:** Mark only dependent results unavailable and identify the originating error
  when dependent work is skipped. Independence does not authorize fabricated inputs or repetition of failed work.
- **Required reconstruction or baseline fit:** Failure that prevents baseline training, including any required fold's
  baseline-fit failure, stops the entire current model evaluation and influence analysis. Return a failure-only file
  without evaluation reports. Do not automatically retry that training or perform downstream recovery fits.
- **Influence removal evaluation:** A sample-removal refit/calculation failure is local to that sample's affected
  contribution. Keep its undefined values `NaN`, preserve independently computable class values, log the failure,
  and continue other samples. It is not failure of the original baseline model or unrelated reports.
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
retries and unnecessary recomputation. A failure-only evaluation file is a complete failure outcome for reception and
arbitration, even though it contains no evaluation reports.

**Worker failure** specifically means the worker process itself is unavailable, such as after an unrecoverable crash,
machine shutdown, external termination, or comparable infrastructure/runtime failure. This is exceptional. Restarting
the worker is not normal task/error recovery; external intervention may be needed to start the worker module again.
Root restart reconciliation, a failed delivery, or local unavailability to another user does not mean worker-process
failure. Always state the actor, object, lifecycle stage, and scope of failure or termination.

### 9.3 Formula fidelity

Implement metrics directly using their defined formulas, populations, and normalization. Standard metrics use
established conventional definitions unless a project-specific variation is explicit; record detailed formulas/report
conventions during relevant planning and implementation without reopening standard metrics solely for absent detail.
Cocoonet-specific/recomposed measures require explicit definitions, including Section 4.7's influence calculation.

Undefined metrics remain `NaN`; small denominators and the resulting defined values remain unchanged. Do not introduce
epsilon, clipping, imputation, invented probabilities, fallback calculations, or NaN-skipping aggregation to obtain a
preferred result. Report formula-defined edge cases and risks, including accepted randomness, without treating them
as ambiguities or reasons to alter the computation. Seek clarification only for an actually missing custom definition
or contradiction after accounting for scope and context. Keep module computations simple, clear, direct, and efficient.

## 10. Subsequent Design Boundaries

The established invariants above are requirements: computation and parent identities are distinct; ordinary active
execution locks configuration changes, and root interruption permits prospective edits while old work stops.
Changed model evaluation configuration requires a new parent; worker-only changes preserve the parent. Dataset and
worker reuse preserve fixed scientific state; local and online targets share task and scientific contracts. Saved
partitions, acceptance, failure scopes, and CPU capacity follow their explicit rules. Questions explicitly marked open are not silently resolved by this section. Subsequent work may specify the
following mechanisms without changing those invariants:

- The external public-service interface, supplied by its separate project rather than invented within Cocoonet.
- Message/identifier formats, authentication and reconnection mechanics, heartbeat timing, stop acknowledgements,
  transfer integrity, file publication, retry counts/timing, and durable evidence for reception selection and recovery.
- Technical enforcement of model isolation and verification of cleanup before another user's work is admitted.
- Detailed dataset, parent partition-schedule, model-input, and report schemas and fit-state reset mechanisms,
  preserving the specified identities, reconstruction inputs, output fields, shared realized partitions, and
  independence of learned state between validation folds. Partition indices are durably written once to one
  authoritative root-side `dill` file before any unit-task allocation, distributed with parent-context preparation,
  and loaded unchanged for recovery. Only the in-file structure, filename/directory layout, transfer representation,
  and mechanism ensuring durable completion remain implementation design; the persistence lifecycle is resolved.
- Detailed conventional metric formulas/report conventions during their corresponding planning and implementation;
  explicit definitions before implementation for any additional project-specific measure.

Exact API signatures, return types, exception hierarchies, local IPC, and other concrete containers remain deferred.
Persistence format/layout flexibility does not reopen the required single authoritative partition `dill` file.

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
confirmed artifact-preparation exception to memory-only configuration edits; start alone persists execution
configuration. Root interruption permits prospective edits while old work stops with fixed configuration.
Model evaluation configuration changes require a new parent; worker-only changes preserve the existing parent,
which resumes when evaluation configuration is unchanged. Blank advice entries do not introduce alternatives to
those resolved contracts.

The root-directory overwrite question is resolved by the ownership invariant in Sections 2.3 and 6.1: configured
user dataset source paths cannot equal or lie within the managed root. Validate both dataset-path and root-directory
changes; reject a conflicting root before modifying persistent files. Managed-root overwrite therefore needs no
dataset-path exclusions. Dataset removal remains reference-only, and existing partition/recovery protections apply.

The following remain safely deferred to the corresponding implementation work:
- Exact APIs and in-file containers, default partition paths, durable-write/publication mechanisms, validation and
  sample-identity restoration, and recovery-state encoding.
- Object-copy mechanics, history presentation, and coordination/exclusion mechanisms for references to the same
  persisted execution. Parent identity and at-most-one acceptance remain mandatory.
- Whether inline/file-backed rows may coexist and whether uniform tensor shape is represented per sample or once;
  group/mask normalization, sample identity, source references, and deterministic reconstruction are fixed.
- Connection probe/response measurement, timers and round scheduling mechanics, state representation, and configured
  parameter names/values. Independent periodic rounds and fail-startup-with-no-runnable-target behavior are fixed.
- Local IPC, artifact access, process-lifetime mechanisms, cleanup verification/deletion mechanics, and model-isolation
  enforcement. These must implement the existing ownership, activity, readiness, and security requirements.
- The separately supplied external public-service interface; public pool policy remains outside Cocoonet.

Expected undefined scientific values, masked-out populations, and ordinary operation failures do not themselves
reopen the metric, retry, or failure contracts. No mechanism may change an existing parent's configuration, reinterpret
a late result as belonging to a new parent, silently regenerate a saved split, or reset an exhausted delivery budget.
