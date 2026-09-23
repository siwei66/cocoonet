# Cocoonet Initial Development Roadmap

This roadmap uses root [AGENTS.md](AGENTS.md) and [docs/architecture.md](docs/architecture.md) as its architectural
basis. It sequences capabilities; it neither authorizes implementation nor replaces their contracts. A phase is a
milestone that will later be divided into small, approved changes, usually one production function or method at a time.
Intermediate phases are not promises of a complete public execution mode.

## Development discipline and sequencing

Implement the smallest useful foundations first, then grow dependencies in one direction:

```text
shared contracts -> durable storage -> configuration -> dataset and saved partitions
    -> enforced execution boundary -> model execution -> baseline results -> reports
    -> influence -> PyTorch integration -> bounded worker parallelism -> complete attempt files
    -> local coordination and acceptance -> bounded delivery -> interruption and recovery
    -> private online connections -> online synchronization and execution
    -> mixed scheduling and duplicates -> explicit cleanup -> public-service integration
    -> complete compatibility and lifecycle verification
```

The shared contracts are logical records and component interfaces required by the architecture, not a speculative
framework. Their concrete representation is settled before the first dependent component. Component ownership follows
architecture Section 2.1: root control and coordination, connection and communication, synchronization, model
initialization and execution, evaluation components, report persistence, packaging, and delivery retain distinct jobs.
A component map does not require one file or class per component.

Each approved implementation step should complete one coherent unit and its applicable tests before moving on.
Necessary helpers are approved and implemented separately unless the user explicitly approves a bounded set.
Document architectural impact and follow the documentation-first gate before affected implementation. Preserve
finished contracts; later phases add consumers and adapters rather than replacing earlier semantics.

Throughout every phase:

- Follow Python 3.12 through the latest stable release, excluding prereleases, and the existing dependency/tool
  configuration. Use asyncio/aiofiles for asynchronous I/O and dill exclusively for Cocoonet serialized payloads.
  Applicable pytest, mypy, Black, and Ruff checks must pass for approved implementation work; use targeted tests, not
  unrelated suites. Validate documentation accessibility and version switching when documentation is rebuilt.
- Provide NumPy-style source docstrings for public and internal components, public-only API documentation, useful
  examples, and comments explaining non-obvious logic. Preserve the typing, narrow suppression, decomposition,
  simplicity, efficiency, and line-length rules in AGENTS.md.
- Preserve defined formulas, populations, class/sample alignment, and normalization. Undefined values remain NaN;
  small denominators and randomness effects remain as computed. Do not add epsilon, clipping, imputation, fallback
  probabilities, NaN-skipping aggregation, favorable-run selection, or corrective fitting.
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

Keep module-generated dataset_id, sample_id, and model_id distinct from user-facing sample/model labels. Sample
identity and its one-to-one label mapping are dataset-scoped; model labels distinguish unit tasks within their parent.
A parent fixes one dataset, realized partition schedule, task-wide evaluation configuration, and one or more model
inputs. Duplicate attempts do not create another unit or parent. Establish the framework compatibility contract:
complete Cocoonet versions must match exactly, while Python versions may differ within the release's supported range.
Compatibility metadata must be readable without loading executable model or report payloads.

Establish the logical state needed for configuration consolidation, worker readiness/availability, immutable attempt
inputs, stop disposition, report status, result selection, durable acceptance, and bounded-operation progress.
Define standardized prediction/class alignment and the single-attempt result envelope, including the failure-only
outcome, before their producers and consumers. Distinguish scientific undefined values from operational errors in
these records and define trace context for stage, model, fold, sample/class, exception, message, and traceback.

Before proceeding, identity scopes, ownership, required associations, state distinctions, and component input/output
contracts must be stable. Exact containers and field names are chosen here where needed; wire encodings, lifecycle
algorithms, metric calculations, and model execution remain for later phases.

### 2. Filesystem ownership and durable persistence foundations

Build on Phase 1 to establish the storage boundary before any configuration operation can prepare files. Implement
root-directory preparation and the bidirectional ownership validation: dataset source paths must neither equal nor
lie inside the managed root, and a proposed root must be checked against all configured source paths, including those
retained by a copied instance. Reject conflicting root configuration before any persistent-file modification.

Keep user source files outside managed storage and unchanged by Cocoonet. Apply the missing/empty/nonempty directory
and overwrite rules, while preserving existing parent partition/recovery protections. Fresh initialization and recovery
are separate operations; a general root-file clearing API is not required.

Establish designated dill persistence/loading paths and complete-versus-incomplete publication for partition artifacts,
execution configuration/state, and result files. Provide the durable evidence needed later for the known task set,
fixed parent context, pending result selection, acceptance, and remaining retry budgets. Establish structured
root/worker
running and error logs, failure/provenance context, and explicit persistence errors. Result persistence must support
asynchronous I/O and
content processing outside the active receive path.

The exit condition is a reliable storage interface and stable recovery evidence, not an implemented recovery engine.
Do not duplicate the authoritative partition storage. Split generation, execution-state transitions, resend loops,
retention actions, and network transfer are introduced by their owning later phases.

### 3. Reusable computation configuration and lifecycle foundations

With identities and storage available, build root configuration and initialization without launching scientific work.
Support incomplete configuration, one active dataset slot, model and worker registries, and model/worker
add/list/inspect/summary/remove capabilities with singular and bulk operations. Define bulk commit/failure behavior
before implementing bulk conveniences; do not leave undocumented partial configuration.

Retain worker CPU/device and applicable retention/operation policy configuration without starting runtime services.
Represent initialized, consolidated, and unconsolidated states and their permitted transitions. Ordinary configuration
edits are in-memory operations; Phase 4 supplies the explicit dataset/partition preparation exception. Preserve
ordinary active-execution locks and prospective edits after root interruption. Model evaluation configuration changes
require a new parent and explicit reinitialization; worker-only changes preserve the existing parent.

Ordinary copying preserves configuration, root directory, lifecycle state, and existing parent identity without cloning
runtime resources, duplicating accepted files, or creating another parent. Dataset removal removes references only.
Worker removal is configuration-only
and sends no stop, disconnect, cleanup, reset, or release instruction. Root reassignment must use Phase 2 validation.

Before proceeding, configuration validation, copy semantics, mutation boundaries, and parent-snapshot preparation must
be stable. Dataset registration becomes complete in Phase 4. Runnable start, cancellation, resume, worker runtime
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

Complete add_dataset() as dataset-plus-partition preparation. Load and validate a supplied split_path without running
the splitter; otherwise require the splitting method/configuration, realize the whole schedule, validate it, and
persist it once in one authoritative dill file before successful completion. Validate exact dataset identity,
sample_id/sample_label associations, and memberships. Resolve splitting randomness here.

Provide dataset and partition inspection alongside registration, exposing the normalized sample associations and saved
memberships. The stable output is an evaluation-ready dataset/partition association that parents can bind and later
instances can reuse after validation. Re-registration cannot overwrite an artifact still required by an existing parent.
Define
default/explicit partition paths and identity restoration before consumers depend on them. Decide mixed-row and
tensor-shape encoding support here without promising unsupported variants. Worker synchronization, model fitting, and
runtime dispatch remain later work.

### 5. Technically enforced model execution boundary

Before any executable model loading or invocation, establish and verify the worker-side isolation boundary using
the contracts and data access from Phases 1–4. Choose the concrete enforcement mechanism before implementing it;
authentication, Python conventions, API validation, and process separation alone do not satisfy the architecture.

Cover executable deserialization, reconstruction, and model operations. User model code has no direct filesystem or
network permission. Framework components resolve source references and supply in-memory data, validate outputs, and
own persistence. Establish the separation between responsive worker control and model evaluation, and the local
execution/control interface needed by later orchestration; do not introduce WSS loopback.

Begin with serial execution and explicit ownership of evaluation subprocesses. Preserve the worker budget
nCPU = max(1, min(n_cpu, N) - 1), with n_cpu defaulting to available N; capacity 1 is serial, and parallel evaluation is
allowed only above 1. Cocoonet owns scikit-learn/scikit-learn-like multiprocessing; models may not create their own
pools. Thread-level behavior and PyTorch device execution retain their distinct scopes.

Before proceeding, model isolation, control responsiveness, framework data access, and operation/failure return
contracts must be verified. The chosen boundary must accommodate all supported model families without a later
security redesign. Complete workflows, public start, and remote communication remain deferred. Phase 11 completes
bounded internal parallelism; optimization beyond the architectural resource policy is not required for this boundary.

### 6. Scikit-learn and custom-model preparation and execution

Build model adapters on Phase 5 and the stable Phase 1 prediction contracts. The root prepares authoritative
model inputs and creates serialized model payloads; only workers deserialize them at the enforced boundary.
For standard scikit-learn models, identify the supported type and reconstruct/configure through get_params/set_params
as appropriate. Custom models receive only required-interface validation: fit/predict for regression and
fit/predict/predict_proba with required class information for classification.

Implement initialization, training, and standardized prediction as separate responsibilities. Preserve supplied
preprocessing, independent learned state between folds, sample ordering/alignment, and explicit probability-class
identities. Do not derive a user-level metadata catalogue, inspect or negotiate custom dependencies, repair an
environment, substitute a model, or promise deterministic outcomes.

The exit condition is a tested serial execution path for both supported scikit-learn categories with descriptive
reconstruction, fit, and prediction failures. Output contracts must work for later reports and PyTorch adapters.
No report generation, complete unit-task scheduling, or model-training recovery retries belong here.

### 7. Baseline fold evaluation and validation results

Use Phase 4's saved memberships and Phase 6's execution adapters to build the baseline stage of a complete
one-model attempt. Establish framework-managed local dataset/model readiness and consume the fixed parent context;
workers validate references but never split, shuffle, or alter memberships.

Run every required baseline fold fit and held-out prediction. Produce common fold/sample records, separate train/test
source-file references rather than returned feature arrays, true/predicted targets or class labels, and class-aligned
probabilities. Validation covers the specified testing populations, including mask exclusions. Retain baseline
predictions and inputs required by reports
and influence so neither requires recovery fits or redundant baseline training.

Required reconstruction or any baseline-fit failure stops the entire current unit evaluation and suppresses evaluation
reports, producing the failure-only outcome for later packaging. Prediction/shared-intermediate failures mark only
dependent values unavailable. Preserve successful independent computations and trace the originating error.

Before proceeding, validation outputs, baseline/shared-intermediate failure boundaries, and complete required-fit
status must be stable. This is a workflow stage, not a separately distributed fold task. Ordinary metrics, influence,
packaging/delivery, and coordinator execution remain later capabilities.

### 8. Ordinary reports and visualization data

Build independent report components from Phase 7's standardized results. Settle concrete report schemas, conventional
formula/report conventions, and reuse of shared intermediates immediately before their consumers; preserve explicit
project-specific definitions rather than inventing variants.

Classification includes confusion matrices and class identities, TP/TN/FP/FN, precision, recall, F1, accuracy, AUC,
and the requested micro/macro scopes. Preserve the three specified accuracy definitions and None for macro-averaged
confusion counts. Residual reports include per-class probability residuals, probability z-scores, absolute/root-squared
summaries, Shannon entropy of absolute residuals, and Mahalanobis distance under the prescribed within-sample class
axis. Return class-specific ROC/AUC plotting data.

Regression includes Mean_Error, SDE, MAE, Normalized_MAE, CV_MAE, MSE, RMSE, Normalized_RMSE, CV_RMSE, RPD, and R2,
plus residual/standardized-residual reports and scatter/residual plotting data. User-facing reports identify samples
and models by their user labels while internal records preserve identities and dataset-relative sources.

Each report runs when its actual prerequisites are available. Its failure preserves other reports, identifiers, and
successful values, with unavailable values and detailed errors represented as specified. Components return report
data; they do not independently own filesystem persistence or retrain models. Before proceeding, report consumers
must need no model-specific orchestration. Rendered plots, influence, and result-file delivery remain out of scope.

### 9. Optional sample influence

Add influence as an internal stage of the same complete attempt, using the baseline data and report/failure contracts
from Phases 7–8. Reuse the exact parent validation schedule. Omit each eligible training sample once per fold, refit,
and compare predictions on that fold's unchanged test set. Held-out samples receive no removal evaluation or invented
zero for that fold; no prediction of the omitted sample itself is required.

Implement architecture Section 4.7 exactly: baseline held-out MSE against truth, actual input predictor count p,
class-specific probability influence or single-target regression influence, and aggregation divided by
M_i = sum_{f in F_i} |V_f|. Count zero changes and repeated comparisons, retain comparison counts/status, and propagate
undefined contributions through ordinary aggregation. Do not replace the divisor with fold count or training size.

Contain removal-refit/calculation failures at the sample/class boundary, preserve independently computable class
values, and continue independent samples. Baseline-fit failure still suppresses the entire unit's evaluation reports.
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

Device selection belongs to worker configuration. Keep device/training behavior separate from scikit-learn's
multiprocessing policy. Resolve concrete device failure handling before implementing those operations within the
existing failure boundaries; no environment repair or fallback model is introduced.

The exit condition is all supported model families producing the same validation/report/result contracts, including
influence when requested, under the enforced boundary. Network integration, device-specific optimization, and
unestablished multi-target/multilabel extensions remain deferred or outside scope.

### 11. Bounded parallel evaluation and worker resource control

Complete execution resource management after the serial workflows and supported adapters in Phases 5–10 are stable.
Implement Cocoonet-owned creation, supervision, and termination of evaluation subprocesses for scikit-learn and
scikit-learn-like workflows. Parallel work stays inside one complete model attempt; folds and removal refits do not
become distributed tasks.

Apply the configured CPU-capacity formula across the worker's evaluation work. Capacity 1 remains serial, including
budgets of one or two; above that, supported internal parallel work must stay within the effective capacity.
Keep worker communication/control responsive and prevent nested model-owned multiprocessing pools. Do not infer
thread restrictions or PyTorch device behavior from this process-level contract.

Preserve independent learned state, fixed partitions, sample/class alignment, failure isolation, and direct scientific
calculations when scheduling internal work. Subprocess or evaluation failure must be contained at its applicable
scope without terminating or restarting the long-running worker service. Preserve cooperative stop boundaries so
Phase 15 can integrate cancellation without changing execution ownership.

Before proceeding, resource limits, serial/parallel output contracts, supervision, and control responsiveness must
be stable under the existing nondeterminism policy. Choose concrete process/pool mechanics within this phase.
Root task scheduling, result delivery, and optimization beyond the defined resource policy remain later work.

### 12. Complete attempt packaging and worker persistence

Once evaluation producers and Phase 11 resource control are stable, complete worker report persistence and result
packaging using Phase 2. Construct exactly one standardized dill result file per completed execution attempt,
containing its reports,
successful/unavailable values, statuses, and traceable failures. Required baseline failure instead creates the complete
failure-only file without evaluation reports.

Preserve parent/unit/attempt associations, model identity, fold/sample/class alignment, and the context needed for root
report processing and provenance. Evaluation components remain separate from persistence; delivery receives a completed
existing file and does not reconstruct evaluation results. Keep persistence failures at their bounded-operation scope
and the long-running worker alive.

Before proceeding, normal, partial-report-failure, and baseline-failure outcomes must be publishable and distinguishable
from incomplete writes. No report gets an independent delivery lifecycle. File construction must be stable before
local acceptance, bounded resends, or networking; do not add streaming/backpressure machinery for these small files.

### 13. Local coordinator, worker control, and result acceptance

Integrate Phases 1–12 into the first complete normal execution path, with an explicitly configured separate root-local
worker process. Root control owns the reusable computation; the coordinator owns unit-task assignment and progress;
worker control owns an attempt and remains responsive while its evaluation workflow runs. Evaluation state stays
local to the executing worker, without inter-worker communication or shared mutable execution state.

Apply Phase 1's exact-version compatibility rule through local control without a network handshake. Complete start
prerequisite checks and execution-configuration persistence. Require one active dataset, at least one
model, and a usable configured target; never execute in the coordinator or silently create a worker. Bind the parent's
validated saved partitions and confirm durable partition persistence before allocation. Establish local artifacts and
parent context, then dispatch only model-specific input and necessary lifecycle references in that context.

Implement root selection and acceptance using the stable attempt records even though this milestone needs only the
local normal path. Persist pending selection evidence, receive/handoff the selected complete file, durably persist it,
then accept and acknowledge it. A scientific failure-only outcome is eligible on the same basis as success; incomplete
writes are not completion. Root deserialization/content processing stays outside active reception. Worker completion
alone does not establish availability or result acceptance.

Before proceeding, a full local evaluation must yield the standardized root-persisted outcome and stable task/attempt
progress. Choose local IPC/artifact access and process-lifetime mechanisms before this integration. Runtime scheduling
must fit the shared local/online contracts without artificial network behavior. Bounded delivery, cancellation/restart,
multiple-target arbitration, and remote execution are completed in the following phases.

### 14. Bounded operations, delivery waits, and result retention

Build normal-operation failure handling on Phase 13's persisted selection/acceptance state and Phase 12's completed
worker file. Implement the delivery component's acknowledgement/resend/wait lifecycle through the local boundary;
online transport later uses the same operation semantics.

A selected transfer or root-persistence failure requests the same existing file within its configured retry budget,
without reevaluation. Retain it and keep its worker unavailable for new tasks until confirmation or exhaustion.
At exhaustion, end only that operation, log its limit/attempts/final error, and allow normal worker handling when
eligible. The logical task remains unfinished and unaccepted; exhaustion authorizes neither recomputation nor promotion
of a competing result. Persist enough state to prevent budget resets after restart.

Implement result-retention choices: deletion after confirmed delivery, retention until all root tasks finish, or no
automatic deletion. Preserve pending delivery except for the mandatory public-release cleanup introduced in Phase 20.
Apply configured bounded-operation handling to persistence and subsequent transfer consumers without allowing retries
where scientific failure policy prohibits them.

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

Recover the known task set, committed parent configuration, authoritative partition file, reception selection,
accepted files, and retry progress. Reconcile continuing work; recognize complete persisted files without reevaluation
or retransmission; continue eligible pending selected delivery within its remaining budget. Preserve stop dispositions,
accepted outcomes, and exhausted budgets. Do not resplit, infer completion from partial files, or invent automatic
training recovery after a worker/process failure.

Before proceeding, local interruption and crash/restart cases must preserve identity and at-most-one acceptance,
including the persistence-before-acknowledgement window. Settle concrete durable-state and local-process reconciliation
mechanisms here. Online reconnection and multi-attempt recovery will exercise these same records in Phases 16–18,
not replace the recovery model.

### 16. Private online-worker connection and control

Introduce networking only after the complete local path and lifecycle are stable. Extend the root connection and
communication components with private registration credentials, root-initiated authenticated WSS over port 443,
exclusive single-root ownership, and a persistent bidirectional control/status connection. Private registration
generates a unique credential and returns connection information directly to its user; private workers never appear
in public discovery. Workers never connect back or communicate with one another. Local-only execution still
initializes no online subsystem.

Choose handshake/control encoding, authentication/reconnection mechanisms, heartbeat/disconnect detection, and
stop/status acknowledgement details before dependent messages. Enforce exactly matching complete Cocoonet versions
before normal protocol operations, synchronization, assignment, or model loading. Read compatibility metadata without
loading model/report payloads; support differing declared Python versions without dependency negotiation.

Implement asynchronous per-worker connection rounds before start, during execution, and after disconnect, using the
configured period and intra-round delays. Exhaustion makes only that worker unavailable until the next scheduled round;
configuration remains. Connection failure is not scientific failure, worker release, or delivery-budget reset.
Measure allocation-time responsiveness without permanent priority or blocking other workers.

Before proceeding, authenticated control, version rejection, readiness/status observations, and independent reconnection
must be stable. Start uses currently runnable targets and fails if none exist while connection management continues.
Model/dataset transfer, remote evaluation, and public discovery are not enabled by this connection milestone alone.

### 17. Online synchronization, execution, and delivery

Use Phase 16's connection and Phases 4–15's existing contracts to add the online artifact path. Dataset synchronization
owns exact representation validation, storage, readiness, and invalidation. Model synchronization owns transport/local
artifact readiness; model initialization retains semantic validation and reconstruction responsibilities.

Synchronize dataset plus applicable saved partition/parent context before unit execution. Apply the configured dataset
retention policy, subject to mandatory public-release cleanup in Phase 20. Reuse a retained ready
dataset_id without retransmission, but always establish the applicable parent configuration and partition context.
A changed representation or cleared copy requires preparation again. Unit-task messages carry the model and required
model-specific/lifecycle information, not another dataset or an independent dataset identity.

Dispatch through the established connection and run the same evaluation components. Return only the selected complete
file over that connection; persist promptly with asynchronous I/O, acknowledge, and apply the existing bounded delivery
and retention rules. Reconnection reconciles running, stopped, pending-delivery, and already-persisted outcomes using
Phase 15's durable evidence, without scientific retries or resetting exhausted operations.

Before proceeding, one private online worker must support complete normal, failure, and reconnect paths, with dataset
reuse and provenance intact. Synchronization verification and concrete transfer schemas are resolved here. Dataset
cleanup operations remain in Phase 19; multi-worker/duplicate arbitration is the next milestone, and public allocation
remains outside this phase.

### 18. Mixed scheduling and duplicate-attempt arbitration

Extend the now-stable coordinator across explicitly configured local and online targets. At each allocation opportunity,
skip unavailable online workers and use current response speed among eligible idle online candidates, then continue
allocating while units and eligible workers remain. Do not give local/online results different acceptance priority.

Introduce duplicate scheduling only when no unallocated units remain, every remaining unit is finished or assigned,
and a worker would otherwise be idle. Choose an unfinished currently running unit, oldest by its first execution start.
A completed outcome awaiting delivery is not a reason to launch a duplicate; terminal baseline failure is not retried.

Exercise receiving-order selection, serialized reception per unit, and at-most-one durable acceptance across attempts.
An earlier failure outcome can win over later success. Selection or transfer failure cannot promote another result.
Only after selected-file persistence and acceptance, stop remaining duplicates without receiving their files.
Unselected files do not enter the resend lifecycle. Different units remain independent.

Before proceeding, mixed execution, simultaneous ready results, duplicate cancellation, reconnects, and root restart
must all use the existing identities, selection evidence, and remaining budgets. This phase adds scheduling policy and
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

During release, stop prior user execution and clean all managed user data, models, task state, and reports before
another user can be admitted; old work must not recreate state. Confirm cleanup or keep the worker locally unavailable
to another user and report failure through the service interface. Mandatory release cleanup overrides local
retention, including
pending result delivery, without marking an undelivered logical result accepted.

Before proceeding, exclusive access, cleanup failure, release, and disconnection behavior must match the supplied
service contract. Worker removal still does not imply release. Do not invent service endpoints, reservation timers,
first-connection allocation, pool-state ownership, or credential-renewal policy. The external service's own development
is outside this roadmap.

### 21. Complete compatibility and lifecycle verification

After the capabilities exist, close the integration verification for supported model families and local-only,
online-only, and mixed operation. Earlier phases already require applicable checks; this milestone is not permission
to postpone correctness, security, or failure isolation.

Verify the declared Python support range and cross-supported-version dill/model/result behavior for exactly matching
Cocoonet releases. Exercise resource limits and responsiveness, fixed partitions and identifiers, all report/failure
outcomes, duplicate acceptance, persistence/acknowledgement interruption, exhausted retry preservation, connection
rounds, data retention/reuse, configuration copying/reuse, and public cleanup through the supplied integration.

Complete public workflow documentation and examples using the implemented APIs, keeping internal details in source
documentation. Confirm asynchronous receipt does not deserialize in the active event loop and that worker application
errors do not terminate the long-running service. Keep tests and fixes decomposed into approved units.

The exit condition is the initial architectural contract demonstrated end to end, with documented supported behavior.
Performance measurements may guide later work, but optimizations must preserve formulas, isolation, CPU ownership,
persistence, and acceptance semantics and follow the architecture/documentation and approval gates. Additional model
categories, distributed folds, rendered worker plots, and a public-service implementation are not implied follow-ons.

## Dependency decisions and roadmap risks

No unresolved architectural contradiction currently prevents this ordering. The difficult boundaries are handled
before their consumers, rather than postponed until they force changes across working subsystems:

- **State before behavior:** Phase 1 defines parent/unit/attempt, selection, acceptance, operation, and stop records;
  Phase 2 makes required evidence durable. Local scheduling, delivery, recovery, and distributed arbitration then
  implement those records in that order. Do not equate a failed operation with a stopped worker or a completed unit.
- **Data and results before orchestration:** Dataset/sample/source and prediction/report contracts precede model
  workflows and packaging. Model-family adapters consume one report interface. Local control and online transport
  differ at their boundaries while preserving task, scientific, persistence, and acceptance semantics.
- **Isolation is an early implementation prerequisite:** The concrete enforcement mechanism is deferred by the
  architecture but must be selected and verified before Phase 5 executes any model payload. Confirm it supports the
  intended reconstruction, local/online ownership, and PyTorch needs. If it cannot enforce the documented restrictions,
  pause that dependent work and seek an architectural decision; do not implement an unsafe temporary path.
- **Durable publication and resource lifetimes need timely decisions:** Choose atomic/complete publication evidence,
  source-path equality/containment validation, and partition identity restoration before Phases 2–4 depend on them.
  Choose local IPC/process lifetime and reconciliation mechanisms before Phases 13–15. These are implementation
  decisions constrained by existing invariants, not separate architecture-repair phases.
- **Protocol and service boundaries are staged dependencies:** Concrete authenticated messages, heartbeat detection,
  transfer validation, and timer mechanisms are decided immediately before Phases 16–17. The external public-service
  interface is a real dependency for Phase 20 only; private/local implementation can proceed without inventing it.
- **Scientific and platform compatibility require early checks:** Set report conventions at Phase 8 and test model
  output and serialization contracts as their adapters are introduced. Phase 21 broadens integration coverage rather
  than discovering incompatible schemas or unsupported Python behavior for the first time.

No new architectural decision is required solely to begin the foundations. Deferred API names, containers, file/wire
encodings, timer values, and local algorithms should be settled with their first dependent implementation unit, under
the existing documentation gate, without reopening defined NaN behavior or adding speculative architecture.

This roadmap is intentionally at the repository root as requested. Existing references to docs/roadmap.md in the
architectural documents are a documentation-location mismatch; those references were not changed by this task.
Resolve their location when documentation updates are next authorized. This does not alter the dependency sequence.
