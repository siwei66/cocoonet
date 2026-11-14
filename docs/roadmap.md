# Cocoonet Initial Development Roadmap

This roadmap uses root [`AGENTS.md`](../AGENTS.md) and [`architecture.md`](architecture.md) as its architectural
basis. It sequences capabilities; it neither authorizes implementation nor replaces their contracts. A phase is a
milestone that will later be divided into small, approved changes, usually one production function or method at a time.
Intermediate phases are not promises of a complete public execution mode.

## Development discipline and sequencing

Implement the smallest useful foundations first, then grow dependencies in one direction:

```text
shared contracts -> durable storage -> configuration -> dataset and saved partitions
    -> enforced execution boundary -> model execution -> baseline results -> reports
    -> influence -> PyTorch integration -> bounded worker parallelism -> complete attempt bundles
    -> local coordination and acceptance -> bounded delivery -> interruption and recovery
    -> private online connections -> online synchronization and execution
    -> mixed scheduling and duplicates -> explicit cleanup -> public-service integration
    -> complete compatibility and lifecycle verification
```

Shared contracts establish required information, identity scopes, ownership, alignment, and scientific and lifecycle
semantics. Choose concrete representations before their first dependent implementation, using producer and consumer
examples and the development philosophy in `AGENTS.md`. Do not predefine unnecessary object hierarchies for later
phases.

Component ownership follows architecture Section 2.1; responsibilities do not require one file or class per component.

Each approved implementation step should complete one coherent unit and its applicable tests before moving on.
Necessary helpers are approved and implemented separately unless the user explicitly approves a bounded set.
Document architectural impact and follow the documentation-first gate before affected implementation. Preserve
established semantics; representation changes remain possible through the same documented approval process.

Throughout every phase:

- Follow Python 3.12 through the latest stable release, excluding prereleases, and the existing dependency/tool
  configuration. Use asyncio/aiofiles for asynchronous I/O. Use schema-controlled UTF-8 CSV for Cocoonet metadata,
  model references, execution configuration, and execution state. Their metadata-oriented tables use exactly two
  columns named `field` and `value`; nested and repeated values use schema-defined hierarchical field keys and
  zero-based indices rather than additional columns. Use schema-controlled XLSX for root/worker data references,
  partition schedules, and two-sheet result workbooks; every result metadata worksheet uses the same exact
  `field,value` contract. Dill remains limited to designated trusted model payloads, root-local IPC messages, and
  standardized visualization data. Online selected-result delivery uses `.tar.zst`; dataset and model transfer
  preparation follows its separate Parquet and archive contracts. Applicable pytest, mypy, Black, and Ruff checks must
  pass for approved implementation work; use targeted tests, not unrelated suites.
  Validate documentation accessibility and version switching when documentation is rebuilt.
- Provide NumPy-style source docstrings for public and internal components, public-only API documentation, useful
  examples, and comments explaining non-obvious logic. Preserve the typing, narrow suppression, decomposition,
  simplicity, efficiency, and line-length rules in AGENTS.md.
- Preserve defined formulas, populations, class/sample alignment, and normalization. Formula-defined undefined values
  remain `NaN`; mathematically permitted infinities remain infinite; computationally unavailable values are `None`
  with applicable failure provenance; and schema-defined non-applicable values are `None` without failures. Do not add
  a redundant per-value status enum. Do not add epsilon, clipping, imputation, fallback probabilities, NaN-skipping
  aggregation, favorable-run selection, or corrective fitting.
- Keep configuration/startup, scientific computation, bounded operation, logical-result acceptance, and worker-process
  availability distinct. Errors must be descriptive and traceable at the smallest applicable failure boundary.
- Do not execute user model code in the coordinator or use an unenforced temporary model boundary. Do not introduce
  implicit local workers, fake network workers, general remote execution, dependency repair, or environment negotiation.
- Do not commit automatically. Supply suggested commit messages for the user's manual use after implementation steps.
  This roadmap itself introduces no code, new dependencies, or permission to batch production changes.

## Ordered phases

### 1. Shared identities, records, and component contracts

Start with the small contracts that every later subsystem must agree on. Define the minimal representations and
validation responsibilities for a reusable computation, parent evaluation, unit task, execution attempt, fold,
dataset/model reference, prediction, report outcome, and failure. These contracts are the first dependency; they do
not require a worker, filesystem workflow, or scheduler.

Shared family metadata distinguishes standard scikit-learn models, custom scikit-learn-like models,
PyTorch models, and Cocoonet-native compositions. Phase 1 records identities and family tags;
construction and evaluation behavior belong to the later model-adapter phases.

Keep module-generated dataset_id, sample_id, and model_id distinct from user-facing sample/model labels. Sample
identity and its one-to-one label mapping are dataset-scoped; model labels distinguish unit tasks within their parent.
A parent fixes one dataset, realized partition schedule, task-wide evaluation configuration, and one or more model
inputs. Duplicate attempts do not create another unit or parent. Establish the framework compatibility contract:
complete Cocoonet versions must match exactly, while Python versions may differ within the release's supported range.
Compatibility metadata must be readable without loading executable model or report payloads.

Establish the logical state needed for configuration consolidation, worker readiness/availability, immutable attempt
inputs, stop disposition, report status, result selection, durable acceptance, and bounded-operation progress.
Any report containing a computational failure has `failed` report status, even when successful results remain in its
payload; no separate partial-report status is used.
Define standardized prediction/class alignment and the logical one-attempt `AttemptResult` contract, including the
outcome, before their producers and consumers. Distinguish formula-defined `NaN`, unavailable `None` with failure
references, and schema-defined non-applicable `None` without introducing a parallel per-value status enum. Define trace
context for stage, model, fold, sample/class, exception, message, and traceback.

Before dependent implementation expands, review the shared contracts for unnecessary nesting, duplicate facts, derived
summaries, container restrictions, and validation machinery. Select minimal representations using concrete production,
inspection, and consumption examples, including partial failures.

Phase 1 settles the shared identities, associations, prediction/result structure, and operational evidence needed by
dependent phases. Concrete metric payload schemas and producers, model adapters, persistence implementation, wire
encoding, and lifecycle algorithms remain in their assigned phases.

Reconcile affected schemas, validators, tests, and component boundaries before implementing their consumers. Preserve
scientific and lifecycle semantics; this review does not itself authorize code changes.

### 2. Filesystem ownership and durable persistence foundations

Build on Phase 1 to establish the storage boundary before any configuration operation can prepare files. Implement
root-directory preparation and bidirectional ownership validation: dataset source paths must neither equal nor lie
inside the managed root, and a proposed root must be checked against every configured source path, including paths
retained by a copied instance. Accept absolute and relative root specifications: resolve absolute paths directly and
relative paths inside `<current console working directory>/cocoonet/<parent_id>/`. Resolve canonical physical
locations, reject a filesystem anchor itself, and reject a root that equals, contains, or is contained within a Python
installation or package-installation directory, installed Cocoonet package directory, or the complete Cocoonet source
repository before modifying persistent files. Do not reject an otherwise safe path merely because it is absolute.

Apply the missing, empty, nonempty, and `overwrite` rules without changing user source files. Fresh initialization and
recovery remain separate operations; no general root-file clearing API is required. Managed paths must follow the
semantic root/worker layouts and identifier contracts in Architecture Section 6.2. Paths are locators rather than
authoritative identity evidence.

Establish schema-controlled CSV and XLSX persistence boundaries and structured running/error logs. Implement the
focused atomic file-publication foundation required by Architecture Section 6.5: authoritative XLSX files use
same-directory `<filename>.partial` staging, mutable `execution_state.csv` uses staged atomic replacement, and staging
content is never authoritative or automatically promoted. Establish the focused directory-publication foundation for
later immutable result bundles without requiring future metric payloads.

Define durable evidence for fixed parent context, the known task set, selected reception, acceptance, remaining retry
budgets, and complete-versus-incomplete publication. Preserve asynchronous I/O and keep later online receive-path
content processing outside the active event loop.

The exit condition is a reliable, format-specific persistence interface with stable recovery evidence and focused file
and bundle publication helpers. Phase 2 does not generate partitions, schedule tasks, accept results, run recovery
transitions, perform retention cleanup, or transfer network payloads. Later phases integrate their concrete schemas
without changing these ownership and publication contracts.

### 3. Reusable computation configuration and lifecycle foundations

With identities and storage available, build root configuration and initialization without launching scientific work.
Support incomplete configuration, one active dataset slot, model and worker registries, and model/worker
add/list/inspect/summary/remove capabilities with singular and bulk operations. Define bulk commit/failure behavior
before implementing bulk conveniences; do not leave undocumented partial configuration.
Assign registered models positive sequential decimal `model_id` values beginning at `1`, without zero padding or reuse
within their owning persistent scope.

Retain worker device configuration and the requested positive integer `n_attempt`,
defaulting to `1`, without starting runtime services. The worker-configuration API
docstring must distinguish requested from effective attempt capacity, explain that
one CPU is reserved, identify the possible single-step reduction, and document
configuration rejection. Runtime CPU detection and model/worker capacity validation
remain Phase 11 responsibilities.
Represent initialized, consolidated, and unconsolidated states and their permitted transitions. Ordinary configuration
edits are in-memory operations; Phase 4 supplies the explicit dataset/partition preparation exception. Preserve
ordinary active-execution locks and prospective edits after root interruption. Model evaluation configuration changes
require a new parent and explicit reinitialization; worker-only changes preserve the existing parent.

Ordinary copying preserves configuration, root directory, lifecycle state, and existing parent identity without cloning
runtime resources, duplicating accepted bundles, or creating another parent. Dataset removal removes references only.
Worker removal is configuration-only
and sends no stop, disconnect, cleanup, reset, or release instruction. Root reassignment must use Phase 2 validation.

Before proceeding, configuration validation, copy semantics, mutation boundaries, and parent-snapshot preparation must
be stable. Dataset registration becomes complete in Phase 4. Runnable `run()`, cancellation, resume, worker runtime
management, and external service operations remain deferred; no placeholder may silently perform those operations.

### 4. Dataset normalization and authoritative realized partitions

Complete dataset registration using Phases 1–3. Support ordinary inline features, reconstructible inline tensors,
and file-backed samples with dataset-relative source references. Normalize grouped/ungrouped and masked/unmasked
inputs into the same in-memory and stored sample structure: generated sample_id, user sample label, group, independent
train/test masks, target, and applicable feature/shape/source fields. Default absent groups to a unique group per
sample and absent masks to 1. Reject inconsistent tensor shapes rather than repairing them.

Preserve sample-to-source relationships: multiple files per sample and files shared across populations are valid.
Train/test rows have the same logical structure, and file overlap does not imply sample overlap. Grouped train-test,
grouped k-fold, LOGO, k-fold, LOO, train-test, and applicable classification stratification follow the configured
splitting contract, group separation, and mask eligibility.

Apply one group-separation rule to every generated or reused partition: training and testing groups must be
disjoint within each fold, regardless of splitting method. Missing groups receive distinct per-sample values
during normalization; supplied groups remain unchanged. Reject incompatible splitting configurations explicitly,
without changing the selected method or group values.

Complete `add_dataset()` as dataset-plus-partition preparation. Load and validate a supplied `split_path` without
running the splitter; otherwise require the splitting method and configuration, realize the complete schedule, validate
it, and persist it before successful completion. Generate `dataset_id` as exactly 12 lowercase hexadecimal characters
from a cryptographically secure random source, with the collision handling required by Architecture Section 3.1.

Persist the normalized dataset reference as managed `root_data_reference.xlsx` and the single authoritative schedule
as managed `partition_schedules.xlsx`, using Phase 2 atomic workbook publication. The schedule columns are
`partition_id`, `dataset_id`, `fold_id`, `population`, `position`, `sample_id`, and `sample_label`. Preserve fold order,
write training rows before testing rows within each fold, and preserve one-based population position. Validate exact
dataset identity, sample-ID/label associations, masks, disjoint memberships, and group separation before publication
and after loading. No separate `fold_index` or duplicate authoritative schedule is required.

Provide dataset and partition inspection alongside registration. Re-registration cannot overwrite an artifact still
required by an existing parent. Resolve mixed inline/file-backed rows and per-sample versus dataset-global tensor-shape
encoding here before dependent consumers; neither capability is implied until that decision is documented. Worker
synchronization, model fitting, and runtime dispatch remain later work.

Implement the dataset contracts from architecture Sections 3.1 and 3.5. Support CSV and compressed CSV with
encoding detection, first-sheet XLS/XLSX/ODS, Parquet, ORC, and Feather. Reject pickle/dill dataset input.

Accept multiple inline inputs only as sample-wise divisions. Reject column-wise assembly. Initially identify
rows by globally unique sample labels; reject partial missing labels and generate reserved labels only when
the column is entirely absent. Generate sample IDs after initial identification.

Implement text identifier conversion, configurable finite floating-point regression/X values, integer `0`/`1` masks
excluding booleans, and positive-integer tensor-shape parsing. Standardized classification labels must match
`^[A-Za-z0-9_-]{1,64}$`; reserve `micro-average` and `macro-average`, and reject a class count that would exceed the XLSX
column limit for any required wide result schema. Reject inadmissible types, nonfinite numeric input, conversion
collisions, malformed shapes, and invalid class labels without cleansing or rewriting them.

Generate one managed `<root_dir>/root_data_reference.xlsx`. For file-backed samples, sources are the actual X-bearing
files; for inline samples, sources are the input tables. Managed metadata and reference workbooks are not sample
sources.

Implement reusable transfer preparation: ZSTD Parquet above 16 MB and sample-row sharding above 1 GB.
Calculate shard count from original stored size and the configurable target, defaulting to 512 MB. Process
sources independently, use dataset-global shard numbering, and store prepared representations in
`<source_dataset_root_dir>/dataset_for_transfer/`, using the established dataset and shard naming contracts.
Dataset/model archive wrapping, tiling thresholds, and archive compression remain Phase 17 decisions.

Implement persisted metadata-based reuse state using source path, stored size, modification time, and
preparation settings. Do not compute content digests solely for reuse decisions.

### 5. Technically enforced model execution boundary

Before any executable model loading or invocation, establish and verify the worker-side isolation boundary using
the contracts and data access from Phases 1–4. Choose the concrete enforcement mechanism before implementing it;
authentication, Python conventions, API validation, and process separation alone do not satisfy the architecture.

Cover executable deserialization, reconstruction, and model operations. User model code has no direct filesystem or
network permission. Framework components resolve source references and supply in-memory data, validate outputs, and
own persistence. Establish the separation between responsive worker control and model evaluation, and the local
execution/control interface needed by later orchestration; do not introduce WSS loopback.

Begin with one complete model attempt and explicit worker-control ownership of its execution boundary.
The boundary must support user-configured model/framework child processes while preserving isolation
and scoped failures. Architecture Section 8 defines CPU attempt concurrency, GPU restrictions, and
independent model/framework parallelism. Phase 11 supplies concurrent model-attempt execution.

Before proceeding, model isolation, control responsiveness, framework data access, and operation/failure return
contracts must be verified. The chosen boundary must accommodate all supported model families without a later
security redesign. Complete workflows, public start, and remote communication remain deferred. Phase 11 completes
bounded internal parallelism; optimization beyond the architectural resource policy is not required for this boundary.

### 6. Scikit-learn and custom-model preparation and execution

Build model adapters on Phase 5 and the stable Phase 1 prediction contracts. For root-local and private
workers, the root prepares authoritative model inputs and may create dill model payloads; those workers may
deserialize trusted custom models at the enforced boundary. Trust in private/local serialized model content
belongs to the user.

Implement declarative model-description support needed by public-worker integration. Supported scikit-learn
models and compositions reconstruct from controlled component identities, hyperparameters, and declarative
structure. Public workers receive no serialized Python model objects.

Implement initialization, training, and standardized prediction as separate responsibilities. Preserve supplied
preprocessing, independent learned state between folds, sample ordering/alignment, and explicit probability-class
identities. Do not derive a user-level metadata catalogue, inspect or negotiate custom dependencies, repair an
environment, substitute a model, or promise deterministic outcomes.

The exit condition is a tested serial execution path for both supported scikit-learn categories with descriptive
reconstruction, fit, and prediction failures. Output contracts must work for later reports and PyTorch adapters.
No report generation, complete unit-task scheduling, or model-training recovery retries belong here.

### 7. Baseline fold evaluation and validation results

Use Phase 4's saved memberships and Phase 6's execution adapters to implement initial model reconstruction followed by
primary validation for one model attempt. Successful reconstruction establishes the pristine model-construction state.
For every fold, create a fresh untrained fold model, fit it on that fold's training population, predict its held-out
population, retain the aligned predictions and provenance, and release the fitted fold model. Learned state must not
pass between folds.

Produce common fold/sample records, separate train/test source-file references rather than returned feature arrays,
true and predicted targets or class labels, and class-aligned probabilities. Validation covers the configured testing
populations, including mask exclusions. Retain the completed primary predictions required by ordinary reports and
optional influence analysis.

Initial reconstruction failure produces `reconstruction_failed` and prevents primary validation. After successful
reconstruction, failure to initialize, fit, or predict any primary fold produces `validation_failed`. Preserve
successful folds and continue later independent folds where possible; do not retry failed folds. The validation report
is failed, and dependent reports are failed or blocked according to their available prerequisites.

Before proceeding, reconstruction output, fresh fold-model state, primary-validation alignment, failure provenance,
and the `reconstruction_failed`/`validation_failed` boundaries must be stable. This remains one worker attempt rather
than separately distributed fold tasks.

### 8. Ordinary reports and visualization data

Build independent report components from Phase 7's standardized results. Settle concrete report schemas, conventional
formula/report conventions, and reuse of shared intermediates immediately before their consumers. Their structured
outputs must match the exact column order and metadata contracts in Architecture Sections 4.4–4.6; Phase 12 owns
workbook persistence.

Classification validation, performance, and residual outputs preserve exact model class order and use the exact class
label followed by `_probability`, `_residual`, or `_z_score`. Workbook metadata is authoritative for class identity;
do not parse column suffixes. The performance table contains class rows followed by `micro-average` and
`macro-average`, with non-applicable aggregate confusion counts represented as `NA`. Do not create a duplicate
confusion-matrix table. Preserve the three specified accuracy definitions.

Regression performance contains, in order, `mean_error`, `std_of_error`, `MAE`, `normalized_MAE`, `MSE`, `RMSE`,
`normalized_RMSE`, `CV_RMSE`, `RPD`, and `R2`. Implement the corresponding validation and residual schemas from
Architecture Section 4.6. Conventional metric details are documented with their implementation; explicit Cocoonet
variations remain unchanged.

Every result table is represented for persistence as a separate XLSX workbook with one result-data worksheet and one
schema-controlled metadata worksheet. The metadata worksheet contains exactly two columns named `field` and `value`.
Finite values remain numeric; `Inf`, `-Inf`, `NaN`, and `NA` are deliberate text tokens. Visualization components
return standardized rendering data rather than plotting objects or rendered images.

Each report runs only when its prerequisites are available. A report failure preserves other reports, identities,
successful values, and detailed failure provenance. Components return report data and do not independently persist
files or retrain models. Rendered plots, influence computation, and result-bundle delivery remain out of scope.

### 9. Optional sample influence

Add influence as an optional branch of the same complete attempt after successful primary validation. Reuse the exact
parent schedule and the stored primary-validation predictions as the fixed baseline. Influence may run independently
or concurrently with ordinary report generation, but a failed primary validation blocks requested influence analysis.

For each eligible training sample and each fold where it belongs to the training population, create a fresh removal
model, fit without that sample, predict the unchanged held-out population, compare those predictions with the stored
primary predictions, compute the defined influence contribution, and release the fitted removal model. Do not perform
a removal evaluation in folds where the sample is held out.

Implement architecture Section 4.7 exactly: baseline held-out MSE against truth, actual input predictor count `p`,
class-specific probability influence or single-target regression influence, and aggregation divided by
`M_i = sum_{f in F_i} |V_f|`. Count zero changes and repeated comparisons, retain exact comparison counts and
applicable value-level failure references, and propagate formula-defined undefined contributions through ordinary
aggregation. Do not add a separate influence evaluation-status field or replace the divisor with fold count or
training size. Persist the exact influence schemas defined in Architecture Sections 4.5 and 4.6. Classification places
`prediction_comparison_count` immediately after `sample_label`, before the class-ordered `_influence` columns;
regression records `sample_id`, `sample_label`, `influence`, and `prediction_comparison_count`.

Contain removal-model initialization, fit, prediction, and calculation failures at the sample/fold/class boundary.
Preserve independently computable values and continue other removal evaluations. Any such computational failure marks
the influence report `failed`; when primary validation succeeded, it contributes to `completed_with_errors`. Do not
add a separate influence evaluation-status field.
Accept configured fitting randomness without extra coupling or corrective runs.

Before proceeding, influence meaning, populations, alignment, failure scope, and integration with ordinary reports
must be verified with the architecture's examples. This capability is optional per parent configuration, not a
separate distributed task. Additional influence variants remain outside the initial design.

### 10. PyTorch reconstruction and evaluation integration

Extend the established execution interface after shared reports are stable, using Phases 4–9 without introducing a
second report system. Implement the separate PyTorch initialization and evaluation workflow: resolvable definition
and version, constructor/architecture/custom components, input/target shapes and dtypes, preprocessing/encoding,
output interpretation, and class identities/probability conversion.

Consume the supplied training procedure, loss, optimizer, applicable scheduler, batching, duration/stopping rules,
data ordering, randomness settings, and dependency/custom-definition information. Support user-selected supplied
weights or initialization patterns; transfer weights only when supplied weights are selected. Reconstruct other
initialization on the worker and preserve supplied seeds without promising cross-device numerical identity.

Implement the public-worker non-executable weight representation. Encode supplied public weights as Parquet
model-data artifacts with explicit tensor identities, shapes, dtypes, order, and values. Apply the same 16 MB
ZSTD-compression and 1 GB configurable-sharding thresholds used for tabular transfer preparation.

Keep weight artifacts model-specific. They do not become part of the parent dataset or `dataset_id`. Prepare
them once and reuse them across the unit task's attempts and worker transfers.

Retain the existing custom model, dill payload, and supplied-weight paths for root-local and private workers.
Those trusted paths remain the user's responsibility.

Define the declarative supported-component representation needed for public-worker PyTorch reconstruction.
Custom Python classes, arbitrary import paths, callables, and serialized model objects remain outside the
public-worker contract. Private/local reconstruction retains its existing trusted-input scope.

Device selection belongs to worker configuration. Pass supplied PyTorch loading and
training multiprocessing settings, including data-loader workers, unchanged to their
owning APIs or training procedure. Define how these settings produce the positive
per-attempt CPU demand consumed by Architecture Section 8. CPU PyTorch attempts use
the effective outer concurrency implemented in Phase 11; GPU attempts have one
Cocoonet slot and no Cocoonet-level multiprocessing. Preserve isolation and report
device failures through the established boundaries without environment repair or
fallback models.

The exit condition is all supported model families producing the same validation/report/result contracts, including
influence when requested, under the enforced boundary. Network integration, device-specific optimization, and
unestablished multi-target/multilabel extensions remain deferred or outside scope.

### 11. Bounded parallel evaluation and worker resource control

Complete worker execution resource management after the individual model workflows
and adapters in Phases 5–10 are stable. Worker control owns CPU detection, capacity
validation, attempt-slot reservation, process creation, supervision, and termination.

Implement Architecture Section 8 using `n_attempt`; do not introduce `n_cpu`. Resolve
each eligible CPU model's positive accounted demand, use the maximum demand for
worker-level validation, reserve one available CPU, and derive
`effective_n_attempt`. Apply at most the specified one-step reduction. Reject a
configuration when the requested limit and its one-step reduction both exceed
capacity.

Preserve user-supplied model `n_jobs` and PyTorch loading/training multiprocessing
settings. Capacity enforcement changes only Cocoonet's effective attempt limit or
rejects the configuration. Keep Cocoonet parallelism across complete attempts rather
than adding a pool across folds or removal evaluations.

Expose requested and effective attempt limits and the capacity inputs through worker
inspection and diagnostic output. Complete the public worker-configuration docstrings
with the formula, adaptation behavior, rejection conditions, and examples.

Verify at least:

- unchanged acceptance when `n_attempt * J <= N - 1`;
- the `N=16`, `n_attempt=4`, `J=4` reduction to three attempts;
- rejection for `N=12`, `n_attempt=4`, `J=4`;
- rejection when `n_attempt=1` already exceeds capacity;
- `N < 2`;
- heterogeneous model demands and duplicate high-demand attempts;
- revalidation after a changed worker CPU count;
- preservation of supplied model/framework settings;
- GPU capacity of one; and
- independent stop, failure, reservation, and release behavior.

Keep communication/control responsive and contain failures without stopping unrelated
attempts or the long-running worker. Root allocation and result acceptance are
integrated in the following phases.

### 12. Complete attempt result bundles and worker persistence

Once evaluation producers and Phase 11 resource control are stable, construct one immutable result bundle per completed
attempt. Persist every applicable result table as its own schema-controlled two-sheet XLSX workbook; include applicable
visualization-data dill files, model reference, worker error log, attempt metadata, fold/source associations, failures,
and a manifest defining the complete inventory. No bundle member has an independent delivery, retry, acceptance, or
recovery lifecycle.

Initial reconstruction failure produces a complete `reconstruction_failed` failure-only bundle. Primary-validation
failure produces a complete `validation_failed` bundle containing preserved fold results, failed or blocked dependent
reports, and failure provenance. Successful validation with report or influence failures produces
`completed_with_errors`.

Construct the bundle in worker `model_<model_id>.staging`, validate the complete inventory and schemas, and publish it
with one atomic directory rename to `attempt_results/model_<model_id>/`. Staging is never authoritative or promoted
during recovery. Preserve parent, model/unit, attempt, worker, fold, sample, and class associations.

Before proceeding, every attempt outcome must be durably publishable and distinguishable from an incomplete write.
Online `.tar.zst` creation and delivery belong to Phase 17; root-local copying and acceptance belong to Phase 13.
Do not introduce per-file delivery or streaming/backpressure machinery for these small bundles.

### 13. Local coordinator, worker control, and result acceptance

Integrate Phases 1–12 into the first complete normal execution path with an explicitly configured,
separate root-local worker. Start it with the multiprocessing `spawn` context and use one duplex `Pipe`. Serialize
bounded control messages explicitly with dill protocol 5 through `send_bytes()` and `recv_bytes()`. Dataset files and
result bundles do not travel through the pipe, and root and worker managed storage remain separate ownership scopes.

Complete `run()` prerequisite checks and persist `execution_configuration.csv` and mutable `execution_state.csv`.
Require one active dataset, at least one model, and a usable configured target. Capture `created_at_utc_ns` and generate
`parent_id` exactly as specified by Architecture Section 3.4. Bind the validated saved partition before allocating work.

Durably record the selected attempt before local handoff. The worker reports its validated published-bundle identity
and managed path; the root copies only that selected bundle into root staging, adds its root model reference and final
inventory, and publishes the model-keyed root directory atomically. Root-local handoff uses no archive, compression,
HMAC, unpacking, or duplicate scientific/result validation. A scientific failure-only bundle is accepted on the same
persistence basis as a successful bundle.

Before proceeding, a full local evaluation must yield an atomically published root bundle and stable task/attempt
progress. Worker completion alone does not establish acceptance, and sending the bundle identity does not release the
producing attempt slot. Bounded retries, interruption/restart, multiple-target arbitration, and online execution remain
in their assigned later phases.

### 14. Bounded operations, delivery waits, and result retention

Build normal-operation failure handling on Phase 13's persisted selection/acceptance state and Phase 12's completed
worker bundle. Implement acknowledgement, retry, and wait behavior through the local boundary; online transport later
uses the same operation semantics.

A selected handoff or root-persistence failure retries the same published bundle within its remaining budget, without
reevaluation or per-file retry. Retain that bundle and keep its producing attempt slot occupied until acceptance or
retry exhaustion; other eligible slots remain usable. Exhaustion ends only that delivery operation. It does not accept
or complete the logical task, authorize reevaluation or duplicate substitution, or terminate the worker. Persist enough
state to prevent a restart from resetting the budget.

Implement bundle-retention choices: deletion after confirmed acceptance, retention until all applicable root tasks
finish, or no automatic deletion. Add configurable deletion on transition to a different parent for private and
root-local workers; Phase 20 makes it mandatory for public workers. Never delete a bundle still required by selected
delivery except under mandatory public-release cleanup.

Before proceeding, delivery status, scientific outcome, acceptance, and worker availability must be independently
observable. Counts/timing and operation interfaces are settled here. Reconnection and periodic connection rounds do not
exist yet and must not be simulated by resetting delivery budgets.

### 15. Cooperative interruption, local reconciliation, and reuse

With durable progress and bounded operations stable, complete break/cancel, prospective reconfiguration, and local
resume/root-restart recovery. Workers stop model work at defined cooperative boundaries; already-terminal or unnecessary
handling does not wait for another model loop. Stopping an attempt does not terminate the worker service.

After root interruption, permit future configuration edits while old attempts retain fixed inputs and their stop
disposition. Changed model evaluation configuration requires explicit reinitialization and a new parent; unchanged
evaluation, with or without worker edits, resumes the existing parent. Late results cannot enter another parent.
Configuration unlocking is distinct from worker inactivity and cleanup eligibility.

Recover the known task set and fixed parent from `execution_configuration.csv`, `execution_state.csv`, the authoritative
`partition_schedules.xlsx`, and complete model-keyed root result bundles. Recognize an atomically published accepted
bundle without reevaluation or retransmission; never infer completion from a package, staging directory, or incomplete
bundle. Preserve selected reception, stop dispositions, accepted outcomes, and exhausted retry budgets.

Explicit interruption and unexpected root-process termination use the same persisted loading and continuation
foundation but have different runtime consequences. The existing instance may continue after interruption. After
termination, the user recreates it against the existing working directory and the root-local worker is recreated rather
than reconnected. Settle the exact recovery API here.

Before proceeding, interruption and restart must preserve identity and at-most-one acceptance across the
publication-before-acknowledgement window. Online reconnection and multi-attempt recovery in Phases 16–18 reuse these
records rather than replacing the recovery model.

### 16. Private online-worker connection and control

Introduce networking only after the complete local path and lifecycle are stable. Extend the root connection and
communication components. Private workers use configured endpoints and one of `auto`, `root_connects`, or
`worker_connects`; initial establishment and reconnection follow the same policy. `auto` attempts both directions. The
root accepts the first authenticated, exact-version-compatible connection for the configured worker identity and closes
duplicates. Connection direction never transfers scheduling or coordination authority: after establishment, the root
initiates reconciliation and all task and lifecycle instructions.

Choose the exact control-message framing, connection authentication, heartbeat/disconnect detection, reconnection
mechanism, and stop/status acknowledgement details before implementing their consumers. Enforce exclusive single-root
ownership and exactly matching complete Cocoonet versions before normal protocol operations, synchronization,
assignment, or model loading. Private workers never enter public discovery.

Implement asynchronous per-worker connection rounds before `run()`, during execution, and after disconnect, using the
configured period and intra-round delays. Exhaustion makes only that worker unavailable until the next round and does
not reset a delivery budget or create a scientific failure.

Before proceeding, authenticated control, version rejection, direction-aware reconnection, root-initiated
reconciliation, readiness observations, and independent connection rounds must be stable. `run()` uses currently
runnable targets and fails when none are usable while connection management continues independently.

### 17. Online synchronization, execution, and delivery

Use Phase 16's connection and Phases 4–15's existing contracts to add the online artifact path. Dataset synchronization
owns exact representation validation, storage, readiness, and invalidation. Model synchronization owns transport/local
artifact readiness; model initialization retains semantic validation and reconstruction responsibilities.

Transfer Phase 4's fixed prepared representations and generate worker-local reference metadata. Reuse each
parent's prepared files without per-transfer checks, recompression, or resharding. Root-side prepared files
are not worker-cleanup targets.

Synchronize dataset plus applicable saved partition/parent context before unit execution. Apply the configured dataset
retention policy, subject to mandatory public-release cleanup in Phase 20. Reuse a retained ready
dataset_id without retransmission, but always establish the applicable parent configuration and partition context.
A changed representation or cleared copy requires preparation again. Unit-task messages carry the model and required
model-specific/lifecycle information, not another dataset or an independent dataset identity. After allocation, the
online attempt begins with this attempt-specific model synchronization and continues through model reconstruction,
evaluation, result publication, and delivery.

For the selected result, package the already published immutable bundle once as `.tar.zst` and reuse those same package
bytes for every retry. Record and verify exact byte length and SHA-256. Private delivery also uses the shared runtime
key and HMAC-SHA-256 metadata contract from Architecture Section 6.4. Resolve canonical HMAC metadata encoding, result
compression parameter names, acknowledgement encoding, and applicable retry timing before implementing this path.

The root stages the package and verifies length, digest, HMAC, safe extraction, manifest inventory, workbook schemas,
and identities before atomic root publication. Package receipt and content processing must not block the active network
event loop, and visualization dill data must not be deserialized there. Public authentication remains governed by the
external service interface.

Before proceeding, one private online worker must support complete normal, scientific-failure, package-failure, retry,
and reconnect paths with dataset reuse and provenance intact. Resolve dataset/model transfer archive selection, tiling,
thresholds, and compression here. Cleanup remains Phase 19; duplicate arbitration remains Phase 18.

### 18. Mixed scheduling and duplicate-attempt arbitration

Extend the coordinator across explicitly configured local and online targets. Allocate against
free eligible attempt slots, including slots on workers already handling other attempts.
For online candidates, use current response speed while preserving parent readiness, capacity
reservations, and topology-neutral result acceptance.

Allocate only against the worker's validated `effective_n_attempt` and current
reservations. A worker whose CPU availability changed must complete Phase 11
revalidation before receiving another attempt.

Duplicates are eligible only when no unallocated units remain, every remaining unit is finished
or assigned, and an eligible slot would otherwise be unused. Choose an unfinished running unit,
oldest by its first execution start. A completed result awaiting delivery is not a reason to
launch a duplicate; terminal reconstruction or primary-validation failures are not retried.

Exercise receiving-order selection, serialized reception per unit, and at-most-one durable acceptance across attempts.
An earlier failure outcome can win over later success. Selection or delivery failure cannot promote another attempt.
Only after the selected bundle is atomically published and accepted may remaining duplicates be stopped without
receiving their packages or handing off their local bundles. Unselected bundles do not enter the retry lifecycle.
Different units remain independent.

Before proceeding, mixed execution, simultaneous ready results, duplicate cancellation, reconnects, and root restart
must all use the existing identities, persisted selection state, and remaining budgets. This phase adds scheduling policy and
multi-attempt integration, not new scientific semantics or fold/sample-level distributed tasks. Public-pool allocation
and worker data/resource management remain separate later capabilities.

### 19. Explicit worker data management and cleanup

Implement cleanup against Phases 15–18's trustworthy worker activity and ownership state. Provide selected-dataset
cleanup and all-retained-user-data cleanup independently from configuration-only worker removal, disconnection,
scientific failures, and result acceptance. Support one worker, a selected group, or all applicable workers.

For dataset cleanup, admit the request only when every targeted worker is inactive. If any is active, reject the entire
request before deleting any copy, with no partial cleanup or queued retry of the rejected request. Root interruption
alone does not prove inactivity. Keep this admission rule distinct from handling an operational deletion failure.

Successful worker dataset cleanup invalidates readiness; later use requires the applicable synchronization/preparation.
Private copies may be retained under user control. Protect user source files outside the root; local shared access
does not turn source files into cleanup targets. Root partition artifacts retain their separate persistence contract.

Before proceeding, stop/cleanup coordination, deletion verification, readiness invalidation, and retained-data reporting
must be stable. Specify selection/deletion mechanisms without adding a general root-file clearing requirement.
Private release performs no resource-release action and warns that runtime lifecycle is user-controlled. Public release
combines these verified cleanup capabilities with the externally supplied interface in Phase 20.

### 20. External public-worker service integration

Proceed only when the separate public service supplies its interface. Use Phases 16–19's authenticated worker protocol
and verified cleanup rather than implementing another evaluation transport or reproducing the service's pool.

Integrate requesting an idle worker, receiving connection information, root-initiated connection, and required
lifecycle/release notifications. The service owns selection, reservation/allocation, availability, timeout-based pool
release, and applicable credential lifecycle. It does not proxy dataset/model/task/report traffic. Worker availability
to its owning root does not release the allocation.

Restrict public-pool registration and provisioning to authenticated, authorized CocoonService developers or
operators. Normal users may request allocated public workers but cannot register workers into the pool, and
a local worker cannot become public through ordinary configuration.

For public execution, transfer only declarative model specifications and applicable non-executable Parquet
weight artifacts. Public workers resolve supported component identities through Cocoonet-controlled
worker-local components and reject pickle/dill model payloads, custom Python classes, arbitrary imports, and
user callables.

These restrictions apply only to public workers. Public integration must not remove or narrow the existing
private/root-local trusted custom-model and deserialization capabilities.

During release, stop prior user execution and clean all managed user data, models, task state, and reports before
another user can be admitted; old work must not recreate state. Confirm cleanup or keep the worker locally unavailable
to another user and report failure through the service interface. Mandatory release cleanup overrides local
retention, including
pending result delivery, without marking an undelivered logical result accepted.

Force-enable parent-transition result cleanup for public workers. Before a public worker begins an attempt for a
different parent, delete worker-side result artifacts belonging to earlier parents, regardless of a longer configured
retention choice. First complete any selected delivery still required by the lifecycle or allow it to reach its
established terminal operation state. This parent-transition cleanup remains separate from the stronger public-release
cleanup, which may override pending delivery before reassignment to another user.

Before proceeding, exclusive access, cleanup failure, release, and disconnection behavior must match the supplied
service contract. Worker removal still does not imply release. Do not invent service endpoints, reservation timers,
first-connection allocation, pool-state ownership, or credential-renewal policy. The external service's own development
is outside this roadmap.

### 21. Complete compatibility and lifecycle verification

After the capabilities exist, close the integration verification for supported model families and local-only,
online-only, and mixed operation. Earlier phases already require applicable checks; this milestone is not permission
to postpone correctness, security, or failure isolation.

Verify the declared Python support range and cross-supported-version behavior for trusted dill model/visualization
payloads, CSV state, XLSX workbooks, and `.tar.zst` result packages under exactly matching Cocoonet releases. Exercise
resource limits and responsiveness, fixed partitions and identifiers, report/failure outcomes, duplicate acceptance,
atomic publication and acknowledgement interruption, exhausted retry preservation, connection rounds, data
retention/reuse, configuration copying/reuse, and public cleanup through the supplied integration. Verify that XLSX
result round trips preserve finite values and the distinct `Inf`, `-Inf`, `NaN`, and `NA` tokens.

Complete public workflow documentation and examples using the implemented APIs, keeping internal details in source
documentation. Confirm asynchronous receipt does not deserialize in the active event loop and that worker application
errors do not terminate the long-running service. Keep tests and fixes decomposed into approved units.

The exit condition is the initial architectural contract demonstrated end to end, with documented supported behavior.
Performance measurements may guide later work, but optimizations must preserve formulas, isolation, CPU ownership,
persistence, and acceptance semantics and follow the architecture/documentation and approval gates. Additional model
categories, distributed folds, rendered worker plots, and a public-service implementation are not implied follow-ons.

## Dependency decisions and roadmap risks

This ordering places difficult boundaries before their consumers:

- **Shared evidence before behavior:** Phase 1 defines minimal parent/unit/attempt associations and independent stop,
  selection, acceptance, and operation observations. Phase 2 establishes their required durable evidence. Later phases
  implement transitions, scheduling, delivery, and recovery. Derive convenience summaries where practical; do not
  collapse independently meaningful observations or require a nested runtime object hierarchy.
- **Data and results before orchestration:** Dataset/sample/source and prediction/report contracts precede model
  workflows and packaging. Model-family adapters consume one report interface. Local control and online transport
  differ at their boundaries while preserving task, scientific, persistence, and acceptance semantics.
- **Isolation is an early implementation prerequisite:** The concrete enforcement mechanism is deferred by the
  architecture but must be selected and verified before Phase 5 executes any model payload. Confirm it supports the
  intended reconstruction, local/online ownership, and PyTorch needs. If it cannot enforce the documented restrictions,
  pause that dependent work and seek an architectural decision; do not implement an unsafe temporary path.
- **Durable publication and resource lifetimes are established before use:** Phase 2 implements the fixed protected-root,
  semantic-path, and atomic-publication contracts. Phase 4 implements the fixed partition workbook. Phase 13 implements
  the fixed `spawn`/duplex-`Pipe` local boundary. Later phases add lifecycle behavior without redefining these mechanisms.
- **Protocol and service boundaries remain staged dependencies:** Phase 16 settles online framing, authentication,
  heartbeat, and direction-aware reconnection. Phase 17 settles canonical HMAC bytes, acknowledgements, result
  compression parameters, and dataset/model transfer archives. The supplied external service interface remains a
  dependency only for Phase 20.
- **Scientific and platform compatibility require early checks:** Set report conventions at Phase 8 and test model
  output and serialization contracts as their adapters are introduced. Phase 21 broadens integration coverage rather
  than discovering incompatible schemas or unsupported Python behavior for the first time.

Resolve the remaining Architecture Section 10 decisions only in their owning phases: mixed-row/tensor-shape
representation in Phase 4; technical isolation in Phase 5; conventional report details in Phase 8; recovery and
worker-service APIs in Phases 13–15; online protocol, integrity, retry, and compression details in Phases 16–17; and
cleanup verification and public-service integration in Phases 19–20.
