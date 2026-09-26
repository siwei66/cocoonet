# AGENT.md

## Project Overview

Cocoonet evaluates machine learning models using an explicitly configured root-local worker, independent heterogeneous
online workers, or both. The root (user) node always owns coordination; it does not automatically act as a worker.
It prepares dataset access, supplies models and evaluation directives, and receives serialized reports. Online
workers use synchronized datasets. Scientific auditability, transparency, and reproducibility are core requirements.

**Reusable Computation and User Workflow:**

- A computation instance is reusable root-side configuration, not a parent evaluation task. Construction requires a
  user-provided working directory. Prepare a missing directory or accept an empty one; reject a nonempty directory
  with `overwrite=False`. With explicit `overwrite=True`, clear its contents and prepare it for fresh execution.
  Successful construction/initialization sets `state = initialized`. Initialization is distinct from recovery.
  The root working directory is Cocoonet-managed persistent storage; configured user-owned dataset source paths must
  remain outside it. Validate this ownership boundary before preparing or clearing the directory.
- Reject a dataset source path equal to or contained within the root working directory. Enforce this configuration
  invariant in both directions: validate dataset addition/path changes against the current root, and validate root
  initialization/change/reassignment against every existing configured dataset source path, including paths retained
  by a copied instance. A conflicting root configuration fails explicitly without modifying persistent files;
  a conflicting dataset configuration also fails explicitly. An outside path satisfies this ownership check.
  Managed root contents may be cleared under the existing persistence/overwrite rules without dataset-path exclusions.
- Incomplete configuration is valid before start: dataset setup may be pending and model/worker counts may be zero.
  Ordinary configuration edits change in-memory attributes only; they do not create, modify, or delete files.
  `add_dataset()` is the explicit artifact-preparation exception: it realizes/validates and persists a new split,
  or loads/validates an existing split. That artifact write does not consolidate execution configuration; only start
  performs that configuration-persistence transition.
- `start()` from `initialized` validates prerequisites, persists current execution configuration in the root working
  directory, establishes a fresh parent, and begins the workflow. Once committed, the instance is `consolidated`.
  This is the configuration-persistence boundary; execution artifacts, lifecycle state, results, and failures may
  also be written during the workflow. No unit allocation precedes durable partition persistence.
- Each parent represents one fixed execution configuration and coherent result set. Units and attempts retain their
  parent's configuration for their lifetime. Later edits are prospective only, never propagated into allocated work.
  Ordinary active execution locks configuration changes. Root interruption permits prospective edits without waiting
  for old workers to reach their stop boundaries; their allocated work retains its fixed configuration.
- Model evaluation configuration comprises the dataset and realized split, supplied models/model inputs, and
  evaluation settings. A permitted change to it makes a consolidated instance `unconsolidated` and requires a new
  parent. Start rejects that state, even with an empty directory; explicit `initialize(...)` must prepare a fresh
  execution directory and restore `initialized` before start commits the new parent. Edits before first start retain
  initialized state.
- Worker configuration, including additions, removals, and execution topology, is separate from model evaluation
  configuration. Worker-only changes neither make the instance unconsolidated nor create a parent. With unchanged
  model evaluation configuration, start resumes the existing parent, with or without worker changes, preserving
  its fixed scientific inputs, progress, results, and retry limits. Worker changes still obey configuration locks,
  readiness, ownership, and cleanup rules.
- An ordinary copy reproduces attributes, including configuration, working directory, and lifecycle state. Copying
  does not create a parent, choose/clear a directory, reset state, or clone live processes, connections, attempts,
  or result files. An unchanged consolidated copy refers to the same persisted execution and has resume semantics.
  A later model evaluation configuration edit makes either original or copy unconsolidated under the same rule;
  worker-only changes preserve the existing parent and consolidation state.
- Unchanged consolidated state represents an existing execution: resume/recovery preserves its parent identity and
  persisted state and never clears or freshly initializes its directory. A new parent is created for the first
  evaluation or changed model evaluation configuration after the required initialization; unchanged evaluation does
  not become a new parent merely because start is called again. Exact recovery APIs and copy mechanisms remain deferred.
- A configured instance has one active dataset and authoritative partition. `add_dataset()` prepares both, loading
  and validating an existing `split_path` without splitting; for a new partition, a splitter and its required
  configuration must be supplied.
  Accept only exact dataset associations with matching `sample_id`, corresponding `sample_label`, and valid
  memberships. Re-adding while configuration is mutable replaces the current association, not an old parent's
  fixed dataset/split. Other instances or later parents may reuse the artifact after the same validation.
- No general persistent-file clear API is required. Dataset removal changes configuration references; persistence
  artifacts remain files under the applicable ownership rules. A convenience clear API is optional. Explicit fresh
  initialization may replace the selected working directory with the user's overwrite opt-in; it is not recovery.

```text
constructor / initialize(root_dir, ...) -> initialized
initialized + configuration edits      -> initialized
initialized + start()                  -> persist configuration -> new parent -> consolidated
consolidated + unchanged evaluation    -> same parent / resume, with or without worker changes
consolidated + permitted evaluation edit -> unconsolidated
unconsolidated + start()               -> reject
unconsolidated + initialize(...)       -> initialized -> fresh start permitted
copy(instance)                        -> same attributes, root directory, and lifecycle state
```

Execution status, worker inactivity, and configuration-consolidation state are different scopes. Configuration changes
never redefine a previous parent. On stop/cancel/supersession, executing workers stop at the next cooperative boundary;
already-terminal or unnecessary pending handling ends immediately. Late results cannot enter a later parent.

**Model Scope & Task Flow:**

- Supported model inputs are scikit-learn-like regressors with `fit` and `predict`, scikit-learn-like classifiers with
  `fit`, `predict`, and `predict_proba`, and PyTorch reconstruction and training information with a user-selected
  weight initialization pattern.
- A parent model evaluation task is one fresh evaluation run created by `start()` from initialized state, containing
  exactly one dataset, one fixed task-wide evaluation configuration, and one or more models. Each model produces one
  unit task distinguished within its parent by its unique user-provided `model_label`. The parent dataset and
  configuration and each unit task's model input remain fixed.
- An execution attempt is one worker execution of a unit task, including all required folds, reports, and result
  delivery. Duplicate allocation creates another attempt of the same unit task. Folds and individual samples are not
  separate distributed tasks.
- Evaluation strategies include k-fold, grouped k-fold, Leave-One-Out (LOO), Leave-One-Group-Out (LOGO), and train-test
  splits. With no masks, validation results cover all samples for those cross-validation strategies and all test
  samples for a train-test split. With masks, results cover the realized permitted testing memberships; samples
  excluded from testing do not receive fabricated validation predictions.
- The root establishes the dataset's complete realized partition through `add_dataset()`; each fresh parent binds
  that current validated schedule before unit-task allocation. Every unit task and execution attempt uses exactly the
  same training and testing sample
  memberships. Models and workers must not independently regenerate, reshuffle, or alter them. Splitting randomness
  is resolved only during root-side split preparation; separate parent tasks may realize different partitions.
  Preparing or reusing a split must preserve its dataset association and the persistence contract below.
- Immediately after the complete realized partition set is generated, persist it once in one `dill` file on the
  root's hard drive, with its exact dataset identity, sample-ID/label mapping, and fold/split associations. Parents
  reference that artifact through their fixed execution state. The indices identify the saved
  training and testing `sample_id` memberships. Persistence must succeed before any unit-task allocation begins.
  This file is the single authoritative durable copy of that partition set; do not split the set across files, duplicate
  its authoritative storage, or repeatedly persist it during normal execution. Worker copies are execution inputs.
- Supply the applicable realized indices with online dataset synchronization, or local artifact preparation, as part
  of parent-context preparation. Workers select training/testing data using those indices; they never invoke the
  splitter. A retained ready dataset may be
  reused without retransmission, but the worker must receive the applicable parent partition context before allocation.
  Every unit task and attempt, including duplicates, uses the same saved memberships. After a root restart following
  successful persistence, load the existing `dill` file; do not rerun the splitter or create replacement partitions.
  Retaining only the splitting method or seed is insufficient.
- Worker outputs include fold data, validation results, performance and residual reports, visualization data, and an
  optional sample influence report.
- Within the active dataset, module-generated `sample_id` values identify samples internally, while root-assigned,
  user-provided unique `label` values identify them in user-facing reports. Preserve their one-to-one mapping.
  Neither identifier requires global or cross-dataset identity, and neither is a target/class label. Users supply
  sample labels, not Cocoonet's generated `sample_id` values; `sample_label` denotes the same user-facing label.
- The canonical dataset is sample-oriented and tabular. Support ordinary inline features (`sample_label`, `y`,
  feature values), inline tensor data (also a reconstructible shape), and file-backed data (`sample_label`, `y`,
  logical/dataset-relative `X_file` reference(s)). Tensor reconstruction must be deterministic and satisfy
  `product(shape) == number of flattened X values`; do not pad, truncate, repair, or infer inconsistent tensor data.
  Source references resolve against worker-local storage rather than root-machine absolute paths. Mixed inline/file
  rows and per-sample versus dataset-global shape encoding may be specified during dataset-schema design; support
  for either choice is not implied before that design.
- User tables may be grouped/ungrouped and masked/unmasked for each supported input form. Normalize the in-memory
  and stored dataset dataframe to the same grouped, masked structure, retaining generated `sample_id` and adding
  `group`, `train_mask`, and `test_mask` beside `sample_label`, target, and feature/source fields. Missing groups
  default to a distinct group ID for each sample; missing masks default to `1`.
- `group` supports grouped train-test splitting, grouped k-fold, and LOGO. `train_mask == 1` permits training use;
  `train_mask == 0` excludes the sample from every training population. `test_mask` independently applies the
  analogous testing permission/exclusion. Permission does not itself establish fold membership. The configured root
  split and masks determine saved memberships while preserving disjoint train/test rows and applicable group
  separation. Training-only augmented samples and test-only reserved samples are supported; split randomness is
  accepted. Group/mask metadata is part of the exact dataset representation and its split validation.
- Dataset registration includes root-side split preparation and inspection through `add_dataset()`. Start consumes
  the current validated dataset and partition; it does not invent a missing split or require a separate split call.
- Required public capabilities are initialization, dataset add/set with split preparation, inspect/remove, and
  model/worker add/list/inspect/summary/remove, with singular and bulk add/remove; start, break/cancel, recovery,
  and configuration modification/copy/reuse under the lifecycle rules. A general root-file clearing API is optional.
  Singular mutations are the primitive semantic
  actions; bulk operations are explicit conveniences. Define their commit/partial-failure behavior before
  implementation; do not silently leave undocumented partial configuration after validation failure. Exact names,
  signatures, return types, aliases, and containers remain deferred.
- Local-only, online-only, and mixed execution are supported. The root-local worker and online workers are
  independently optional and must be explicitly configured; root coordination alone is not an execution target.
- `start()` requires one active dataset, at least one model, and at least one usable execution target. Zero configured
  workers is allowed before start but must fail explicitly at start, before evaluation. Do not silently evaluate in
  the coordinator, enable a local worker, create a fake network worker, or wait for a never-configured worker.
  This is a configuration/startup failure, not baseline-training failure. Start does not wait for all configured
  workers. If none is currently runnable, fail startup explicitly; independent online connection retries continue
  so a later start may succeed. A disconnected worker stays configured but is unavailable for allocation.

**Fold Results:**

- Training and testing populations use the same logical sample structure: internal `sample_id`, user-provided sample
  label, group and training/testing masks, dataset-relative source-file reference(s), and target value or class label.
  Saved fold/split membership
  determines the population; do not define different sample-data schemas for training and testing.
- Internal fold records retain sample identities and their user-label mappings. User-facing fold reports identify
  samples by their user-provided labels. Predictions, predicted class labels, and class probabilities are computed
  results associated with those samples, not replacements for or changes to their underlying sample structure.
- Classification fold `y` results identify held-out samples and contain true class labels, predicted class labels,
  and probabilities with explicit class identities. Regression fold `y` results identify held-out samples and contain
  true and predicted target values.
- Fold `X` results reference source data rather than returning feature arrays. Every fold separately provides
  source-file references for its training and testing populations. References are relative to the dataset root and
  resolve through the parent's `dataset_id`. A sample may reference multiple files, and files may be shared by samples
  in different partitions. Overlapping file-reference sets do not imply overlapping sample membership.

**Classification Statistics:**

- Classwise overall accuracy is `(TP + TN) / (TP + TN + FP + FN)`, using each class's one-versus-rest confusion counts.
  `TN` means true negatives; the earlier `TF` notation was an error.
- Micro-average overall accuracy is `number of true predictions / sample size`, where true predictions are correctly
  predicted class labels in the evaluated sample population.
- Macro-average overall accuracy is the arithmetic mean of the classwise overall accuracies of all classes.
- These accuracy definitions apply to distinct scopes. Different classwise, micro-average, and macro-average values
  in multiclass classification are expected; their differences are not an ambiguity or a reason to replace a formula.
- Classification residual measures are computed across the class entries within each individual sample, independently
  of other samples. This axis applies to probability z-scores and residual summaries; do not substitute normalization
  or covariance estimates computed across samples.
- Standard classification and regression metrics use their established conventional definitions unless Cocoonet
  explicitly specifies a project-specific variation. Detailed formulas and report conventions may be documented in
  `docs/architecture.md` during the corresponding planning and implementation steps; their absence from an early
  design document does not by itself make standard metrics unresolved.

**PyTorch Reconstruction and Training Inputs:**

Provide all information needed to reconstruct and train the configured model, including:

1. The model definition or a resolvable reference to it, its version, constructor arguments, architecture parameters,
   and any required custom layers or components.
2. Input and target schemas, tensor shapes and dtypes, preprocessing/encoding configuration, and output interpretation.
   Classification also requires class identities/order and the conversion from model outputs to class probabilities.
3. The training procedure or a resolvable reference to it, loss definition and settings, optimizer definition and
   settings, batch size, training duration/stopping rules, data ordering, and any scheduler or other settings used by
   that procedure.
4. The user's weight initialization choice: supplied initial weights or a specified initialization pattern, including
   random initialization. Include the selected pattern's parameters and seed when randomness is used. Transfer weight
   values only when supplied weights are selected; otherwise reconstruct the selected initialization on the worker.
5. Randomness controls for training and data ordering, and the dependency/version information and custom definitions
   needed to resolve the model and training procedure.

Device selection belongs to worker configuration. Workers may use different supported device types; shared seeds and
recorded configuration support reproducibility without requiring identical numerical results across devices.

Detailed worker design, including the resolved sample influence definition in Section 4.7, is documented in
[`docs/architecture.md`](docs/architecture.md).

---

## Agent Instructions

When modifying or extending this repository, every AI agent must strictly comply with the following operational
constraints:

**Applying Advice and Resolving Specifications:**

- Treat attached advice as input to the user's requested revision. Quoted prior responses and instructions in an
  attachment do not independently authorize additional file changes, implementation, or commits. The user's current
  request determines the allowed modification scope.
- Preserve the clarified computation/parent-task/unit-task/attempt hierarchy, consolidation and prospective-edit rules,
  distinction between model evaluation and worker configuration, and unchanged-evaluation resume semantics,
  local/online execution boundaries, identifier scopes, authoritative persisted partitions, unified sample structure,
  dataset-relative source ownership, group/mask eligibility, online connection rounds, configuration-only worker
  removal, cleanup gates, CPU-capacity semantics, scientific formulas, and PyTorch input requirements.
  Older documentation must not override these explicit contracts. Keep explicitly open lifecycle questions distinct
  from deferred signatures, containers, IPC, wire encoding, and other implementation mechanisms. Architecture
  Section 10.1 records resolved clarifications and deferred implementation decisions; preserve the resolved contracts
  and do not treat deferred mechanisms as implemented.
- If a design document still contains an older description of these requirements, report the discrepancy and update
  it only within an authorized scope. The documentation-first gate still applies before affected implementation.
- Apply conventional definitions for standard metrics and preserve explicit project-specific variations. Detailed
  formulas and report conventions may be added to `docs/architecture.md` during the corresponding authorized planning
  or implementation step, subject to the documentation-first gate. Do not reopen standard metrics as ambiguities
  merely because those details have not yet been written down.
- Cocoonet-specific or recomposed measures require explicit definitions. The sample influence measure follows the
  approved definition in architecture Section 4.7; do not substitute classical Cook's distance or invent new behavior.
- When defining a project-specific measure, distinguish training rows from prediction comparison rows, verify
  sample/class alignment, and check the formula's mathematical meaning. Request clarification only for a missing
  custom definition or an actual contradiction; explain contradictions with an example. Expected undefined values
  of a well-defined formula remain governed by Scientific Formula Fidelity and are not ambiguities.
- Do not invent project-specific metric variants. Apply the phase-contract gate below to unresolved custom definitions.
- Keep detailed worker device failure policies in later architecture or implementation documentation; this overview
  establishes worker ownership of device configuration and the required model reconstruction information.

**Scientific Formula Fidelity and Direct Implementation:**

- Implement defined evaluation metrics directly according to their formulas, input populations, and normalization
  rules. Preserve their mathematical meaning and specified output behavior.
- Represent undefined metric values as `NaN`. Preserve small denominators and the resulting defined values.
  Do not introduce epsilon, clipping, imputation, invented probabilities, fallback values, or NaN-skipping
  aggregation to manufacture a defined or preferred result.
- Report formula-defined edge cases and risks clearly. Expected `NaN`, extreme defined values, and other consequences
  of a well-defined formula are not ambiguities and do not require requests for alternative handling.
  Request clarification only when the definition itself is missing or contradictory.
- Accept randomness effects from the configured computation. Do not manipulate results, select favorable runs,
  or introduce corrective procedures to suppress those effects.
- Raise clear, descriptive errors for failed operations or invalid inputs. Leave related undefined metrics as
  `NaN`; do not conceal failures through fallback calculations or replacement values.
- Keep all module code simple, clear, direct, and efficient. Avoid speculative abstractions, unnecessary computation,
  and special-case machinery. Improve performance while preserving the defined computation and output semantics.

**Failure Isolation and Reporting:**

- Configuration/startup failures occur before scientific evaluation and remain distinct from reconstruction or
  baseline-training failure, report/shared-intermediate failure, influence-removal failure, and worker/infrastructure
  failure. For example, missing execution targets at start must be reported as a startup prerequisite failure,
  never as a model fit failure. Adding local execution and reusable computations does not change the scientific
  failure boundaries below.
- The worker process is a long-running execution service expected to operate continuously, potentially for months.
  Contain normal application-level failures at the smallest applicable scope: measure, report, influence sample,
  unit-task evaluation, or bounded operation. None of these failures implies worker-process failure or termination.
- A measure failure leaves its affected values `NaN` or missing as defined, logs the error, and allows the remaining
  evaluation to continue subject to actual dependencies. A unit-task evaluation failure produces its defined failure
  outcome; the worker remains running and continues with subsequent work.
- A worker failure specifically means that the worker process itself has become unavailable, for example through
  an unrecoverable process crash, machine shutdown, external termination, or comparable infrastructure/runtime failure.
  Restarting the worker is not normal task or operation error recovery. If the process actually stops, external
  intervention may be needed to start the worker module again.
- Always identify the applicable scope of terms such as failure, retry exhaustion, and termination in instructions,
  architecture/lifecycle documentation, logs, and implementation. Keep operation status, model-evaluation outcome,
  logical-result acceptance, and worker-process availability distinct; apply each scope's existing lifecycle rules.
- Essential reconstruction or baseline-training failure, including any required baseline fold fit, stops the current
  unit-task evaluation and its influence analysis. Produce a failure-only result file without evaluation reports.
  Do not automatically retry the failed training or perform recovery fits. This failure does not automatically fail
  other unit tasks in the parent or terminate the worker process.
- Return that terminal unit-task evaluation failure as a failure-only result file containing its status and error
  information, without evaluation reports. This is a complete task outcome for delivery and arbitration, not termination
  of the worker process.
- After required baseline training succeeds, handle each report independently. A report failure must not invalidate successful reports or prevent other reports with available inputs from running. Ordinary report generation must not retrain the model to recover from an error.
- Respect actual computational dependencies. If shared predictions or intermediate results fail, mark only the dependent results unavailable and continue computations whose prerequisites remain valid. Independence does not justify fabricating missing inputs or repeating failed operations.
- Isolate influence analysis by training sample. If a sample-removal refit or influence calculation fails, leave the affected sample’s influence values `NaN`, record the failure, and continue with other samples. Preserve independently computable class-specific values where applicable. Failure of an influence refit is local to that removal evaluation; it does not constitute failure of the original baseline training or invalidate other reports.
- For report-level and sample-level failures, preserve sample identifiers, class identities, and successfully computed results. Represent unavailable result values as `NaN` within the defined output structure, accompanied by failure information. The essential baseline-training failure rule is the exception: it produces failure information without evaluation reports.
- Record detailed, traceable failure logs containing the evaluation/model identifier, computational stage, report name, fold and sample/class identifiers where applicable, exception type, descriptive message, and traceback. Identify the originating error when dependent computations are skipped. Capture errors at the appropriate isolation boundary so they remain visible without terminating the worker process.
- Keep failure handling simple, explicit, and proportional to the affected computation. Use clear boundaries and direct dependency checks; avoid speculative recovery mechanisms, cascading retries, and unnecessary recomputation.
- When any configured retry limit is reached, stop retrying and fail only the operation governed by that limit, such
  as transfer, persistence, or connection establishment. Log the operation, limit, attempts made, and final error with
  the failure-tracking context. Apply that scope's normal failure rules; do not terminate or restart the worker.
  Exhaustion alone requires no additional recovery, fallback, recomputation, or special user action. A retry limit does
  not authorize retries for operations whose failure policy prohibits them. An exhausted online connection round
  remains failed; the separately scheduled next round follows Online Connection Rounds and Allocation and does not
  reset any delivery retry budget.

**Roadmap-Driven Development:**

Before planning or implementing a phase, process project guidance in this order:

1. Read the complete root `AGENTS.md` and apply all relevant sections.
2. Read the relevant parts of `docs/architecture.md`.
3. Read `docs/roadmap.md`, the project-wide roadmap, and identify the current phase.
4. Read the corresponding `docs/roadmaps/roadmap_phase_<n>.md`, or create it under the phase-planning gate below.
5. Before each implementation unit, inspect the relevant code and tests.

- Before implementing a phase, its exact `docs/roadmaps/roadmap_phase_<n>.md` must exist, with `<n>` matching the
  project-wide phase number. Convert that phase into an implementation-level plan defining, where applicable:
  scope and completion criteria; data models, structures, types, and invariants; public APIs and signatures;
  computation formulas and numerical behavior; planned classes, functions, and methods with concise responsibilities
  and corresponding tests; dependency-safe implementation order; validation requirements; and decisions explicitly
  deferred to later phases. Keep detailed architecture and protocol design in the design documents, not `AGENTS.md`.
- Resolve every contract required to implement the current phase before implementation. Only decisions needed solely
  by later phases may remain explicitly deferred. If guidance sources conflict or a required current-phase contract
  remains unresolved, stop and report the issue instead of implementing; resolve it through authorized planning.
- After creating or materially revising a phase roadmap, stop and wait for explicit user approval before implementing
  the phase. Phase-roadmap approval and approval to implement a particular unit are separate gates: approval of the
  plan alone does not authorize production changes.
- Follow the approved phase roadmap. If implementation reveals that an approved API, data model, formula, function
  boundary, dependency, or other phase contract must change, stop implementation, report the discrepancy, propose the
  required phase-roadmap amendment, and wait for explicit user approval before continuing. Do not silently redesign
  the phase during implementation.

**Documentation-First Gatekeeping:**

- Functional improvements include new features and performance improvements. Fixes, scaffolding, and internal tooling
  are excluded from the requirement to update the design documents.
- For new features and major performance improvements that change or introduce architecture, update both
  `docs/architecture.md` and `docs/roadmap.md` before implementation. This update requirement does not apply when the
  feature or improvement neither changes existing architecture nor introduces new architecture.
- Assess and explain the design impact of the requested change. If its architectural impact or the significance of a
  performance improvement is unclear, identify the uncertainty before implementing the affected functionality.
- If a required design document is absent or lacks the relevant design, pause the affected implementation and request
  authorization to create or update the documents, unless the user has already authorized that documentation work.
  Complete the required documentation step before the corresponding implementation step.
- Exemptions from updating these two design documents do not waive the phase-roadmap or single-unit approval gates.
  Scaffolding and internal tooling functions or methods must still have docstrings describing their purpose and
  behavior, with detailed logic explained in comments above the relevant code line or block.

**Manual Commits:**

- Do not commit automatically. The user performs Git commits manually; an automatic commit is not a prerequisite for
  moving from completed design documentation to an authorized implementation step.
- At the end of each implementation, provide one or several suggested Git commit messages for the user's manual use.

**Single-Unit Production Changes:**

- For new development, by default add or change only one production function or one class method per implementation
  response, with only its directly corresponding tests, required typing/docstring updates, and minimal supporting
  edits. This default covers the entire turn, including all tool calls, subject to the helper-bundle exception below.
- A new class, schema, or other declaration may instead be its own approved change unit; a class declaration alone
  does not authorize bundling methods. Normally develop new classes method by method and explain, develop, and obtain
  approval for new helpers one unit at a time before composing their caller.
- When necessary or materially more coherent, propose an exact bounded set of tightly coupled helpers, or a caller
  together with those helpers, for one response. Identify every function/method and explain the grouping; do not
  implement the bundle until the user explicitly approves those exact units. A broad feature, phase, or roadmap
  request is not bundle approval. Do not use helper bundling to introduce unrelated behavior, reusable abstractions,
  or additional scope.
- Revisions to existing behavior may change related existing functions/methods together, including across files, when
  the requested revision is inherently cross-cutting or consistency requires coordinated edits. Use the smallest
  coherent bounded change and identify the affected units and why they belong together. Examples include parameter
  renames across callers/callees, shared type or signature changes, protocol-field updates across dependent handlers,
  and coordinated compatibility fixes. Do not force intermediate states that leave the repository inconsistent.
- Keep multi-unit revisions narrowly scoped to the requested revision. This allowance does not authorize unrelated
  new behavior, new abstractions, or new production functions/methods. New feature development does not qualify for
  this allowance merely because it edits existing code.
- If a revision also requires a new helper/function/method, identify that addition separately and apply the
  new-development default and helper-bundle approval rules above. Revision approval alone does not authorize a
  new-code bundle.
- An explicitly approved helper bundle or permitted bounded revision is one change unit for approval and targeted
  testing. Split large changes into bounded reviewable steps whenever they can be separated safely. The existing
  roadmap, documentation, validation, and approval gates apply to both new development and revisions.
- Even if the user requests an entire phase, feature, or roadmap item, complete only the next approved change unit,
  validate it, report the result, and stop. Obtain explicit user approval before the next change unit; do not advance
  automatically in the same response. Documentation-only edits remain limited to the user's requested document scope.

**Tests for the Targeted Unit:**

- The production change limits exclude testing code. A complete testing unit dedicated to the approved change unit
  may be added or updated in the same step, including the test functions needed to verify it.
- Keep tests focused on the approved production work; do not generate unrelated or project-wide test suites.
- If a production unit does not yet provide independently verifiable functionality, tests may be deferred. Explain
  what cannot yet be verified and why, then wait for user approval of that working unit before continuing.

**No Unsolicited Scope Expansion:**

- Do not introduce dependencies, utilities, abstractions, or unrelated edits outside the immediate approved scope.
- A required helper or coordinated revision does not expand the approved scope or waive the new-unit limit and
  approval gates above.

---

## Architecture Principles

**Local and Online Execution Boundaries:**

- Initialize the online-worker network subsystem only when online workers are used. Local-only evaluation must work
  without it. Network connection, authentication, heartbeat, reconnection, and transfer requirements below apply to
  online workers; applicable task, scientific, isolation, persistence, and acceptance requirements apply to all targets.
- The root-local worker is a separate locally managed process. Dispatch through a direct local execution/control
  boundary; do not create WSS loopback or fake network behavior. User model code must not execute in the root
  coordinator process. The local worker shares the evaluation implementation and technically enforced model
  restrictions with online workers; process separation alone does not replace those restrictions.
- Local execution requires no remote-only network authentication, heartbeat/reconnect, network dataset synchronization,
  or network result delivery. Framework-managed local data/model/result access must preserve dataset identity,
  authoritative saved partitions, parent/unit/attempt associations, failure scopes, and standardized result files.
  Local completion still requires root persistence and the same selected-result acceptance and acknowledgement
  semantics through the local control boundary. Exact IPC and artifact-sharing/copy mechanisms remain deferred.
- Workers are reusable root/computation resources that may serve successive parent tasks; a parent owns its unit
  tasks and attempts, not exclusive lifetime ownership of a worker. Reuse may retain online connections, the local
  worker process, and valid datasets under their applicable policies. Exact process/connection lifetimes and
  local-worker survival after root restart remain deferred. Configuration copying does not clone runtime resources.
  External public allocation and mandatory release cleanup still apply.

**Worker CPU Budget and Execution Ownership:**

- Each local or online worker has a user-configurable CPU budget `n_cpu`, defaulting to the machine's available CPU
  count `N`. Evaluation CPU capacity is `nCPU = max(1, min(n_cpu, N) - 1)`. `nCPU = 1` means serial evaluation;
  parallel evaluation occurs only when `nCPU > 1`. Budgets of one or two therefore intentionally produce serial
  evaluation. The minimum capacity of one takes precedence; do not claim that one physical CPU is always unused.
- Coordinator and worker-control responsiveness remain independent requirements. Cocoonet owns multiprocessing for
  scikit-learn and scikit-learn-like workflows; those models must not create their own multiprocessing pools.
  PyTorch retains its separate device/training contract. Process-pool mechanisms and thread-level behavior must not
  be conflated with the CPU-capacity formula; see architecture Section 8.

**Independent and Decoupled Execution:**

- Workers execute unit-task attempts independently using the parent's fixed dataset, evaluation configuration, and
  realized partitions, without inter-worker communication or dependence on shared mutable state with other workers.
  Internal evaluation state remains local to the worker executing the attempt.
- The coordinator may maintain state for scheduling, assignment, task lifecycle, failure tracking, and result
  acceptance. Coordinator-owned state does not make workers dependent on each other's execution state.

**Dataset Identity, Synchronization, and User Isolation:**

- The local module assigns a unique `dataset_id` to each exact dataset representation, including sample associations,
  features, targets, labels, and related metadata. A changed representation requires a new `dataset_id` regardless of
  its descriptive label. Task and split references must resolve against that same representation.
- Establish the parent's dataset readiness, fixed evaluation configuration, and realized partition schedule on each
  participating worker before unit-task execution. Unit-task transfers carry the model and required model-specific
  information, plus necessary lifecycle identifiers; they neither retransmit nor separately identify the dataset.
- Dataset retention follows its configured policy. A later parent task may reuse a retained, ready representation
  with the same `dataset_id`; cleared data must be made ready again through online synchronization or local artifact
  preparation as applicable. Mandatory release cleanup remains applicable.
- Each representation contains at least one source dataset file. Source references are dataset-root-relative paths
  interpreted with its `dataset_id`, independently of machine-specific storage paths.
- Root user-provided dataset source paths remain outside the Cocoonet-managed root working directory, under the user's
  filesystem control. Apply the bidirectional configuration validation in Project Overview. Cocoonet manages references
  and its own identity/configuration/partition/task metadata; it must not delete, move, rename, garbage-collect, or
  otherwise modify the underlying user source data. Dataset removal drops configuration references only. Root partition
  files are Cocoonet artifacts, governed by their persistence/recovery contract, not worker dataset cleanup.
- Cocoonet manages synchronized worker dataset copies. Successful cleanup invalidates that dataset's readiness on
  the worker; later use requires synchronization again. Private workers may retain copies under user control;
  public workers must remove user state before ownership release and before serving another user.
- Dataset cleanup targets one worker, a selected group, or all applicable workers. Normally request it after parent
  completion. Admission is all-or-nothing across the targeted set: every targeted worker must actually be inactive.
  If any is active, reject the entire request before deleting any copy. Do not partly clean or queue a rejected
  request for later; the user may request it again after inactivity. Interrupting the parent is not proof of worker
  inactivity. Cleanup is separate from configuration removal, scientific failure, delivery, and resource release.
- Root-local data access does not transfer ownership of root source files to the worker. Cleanup may remove
  Cocoonet-managed worker copies, never user-provided source files merely because a local worker references them.
- Public workers serve one root/user exclusively per allocation. Cocoonet must not treat connection loss itself as
  permission to serve another user. Public-pool allocation and release decisions belong to the external service.
  Apply the worker-side cleanup rules below before serving another user; do not allow prior work to recreate data.
- `dataset_id` encoding, synchronization verification, disconnect detection, and cleanup verification belong in the
  corresponding architecture/lifecycle specification; they must preserve these invariants.

**Worker Ownership and External Discovery:**

- Every online worker accepts only Cocoonet's internal protocol and serves at most one authenticated root/user at a
  time. Online workers never initiate connections to the root. No worker communicates directly with other workers.
  The root-local worker is controlled through its owning root's local boundary without network authentication.
- Private-worker registration generates connection information and a unique credential, returned directly to the user
  through the worker-side module, for example as a dictionary for root configuration. The root connects using this
  configured information. Private workers never appear in public discovery; their connection, disconnection, and
  lifecycle operations remain under explicit user control. Connection loss does not automatically release them.
- For a public worker, the root requests an idle worker from an external discovery service, receives the selected
  worker's connection information, and initiates the connection to that worker. Discovery does not cause a worker
  to connect back to the root.
- The public-worker service is outside Cocoonet's scope and is planned separately, for example as `CocoonService`.
  It owns worker availability, idle-worker selection, reservation/allocation state, and decisions about when a worker
  becomes available again. Pool state remains internal to that service; Cocoonet receives connection information,
  not responsibility for managing or reproducing the service's allocation state.
- Cocoonet may send lifecycle notifications required by the external interface, such as release notices. Reservation
  durations, timeout-based release, and other pool policies belong to the service. Do not implement a competing
  first-connection allocation policy, fixed pool-release timers, or an independent public credential-renewal workflow.
- Use the service endpoint, request/response format, authentication, connection-information format, and release
  notification protocol only when its interface is supplied for implementation. Do not invent these contracts in
  advance. This defers integration details without weakening authenticated, exclusive worker access.
- Private and public workers share the direct root-worker evaluation protocol. The discovery service does not proxy
  datasets, models, tasks, reports, or evaluation traffic. A worker available for another model task remains assigned
  to its owning root; task availability does not itself release the external service's allocation.

**Worker Configuration Removal and Explicit Management:**

- `remove_worker` only changes the root computation's worker configuration. It sends no stop, disconnect, cleanup,
  reset, or release instruction. If configuration cannot be modified, fail explicitly without compensating worker
  actions. Removal is permitted before execution, after it finishes, or after root interruption even while old work
  stops; it is locked during ordinary active execution. Worker changes do not change model evaluation configuration
  or create a new parent.
- The same rule applies to a worker reconnected after interruption or recovery, whether working, idle, or retaining
  data, whenever configuration modification is permitted. Any accompanying connection loss follows the existing
  disconnection policy; removal defines no additional timeout, stop, cleanup, or release behavior.
- `cleanup_dataset` removes selected retained worker dataset copies; `cleanup` removes retained worker user data.
  These are explicit worker-management operations, separate from root configuration edits. Dataset cleanup follows
  the all-target inactivity gate. Required public cleanup verification and execution-stop boundaries still apply.
- Public `release` cleans retained user data and performs the resource/state release required by the supplied external
  service interface. Private `release` has no resource-release function and warns that lifecycle is user controlled.
  Removing a public worker does not release it; removing a private worker does not stop its independent runtime.
  A disconnected public worker that was not explicitly released follows the existing service lifecycle rules.

**Public-Worker Cleanup and Service Lifecycle Boundary:**

- Maintain the root-worker heartbeat/status exchange for connection health. Connection loss is not a Cocoonet decision
  to release a public allocation; follow the external service's supplied lifecycle interface and policies.
- During release, stop prior user execution and remove Cocoonet-managed datasets, task state, models, reports, and other
  user state before the worker serves another user. No prior execution may continue or recreate state after cleanup.
- Require confirmed successful cleanup before accepting another user's work or reporting cleanup success. If cleanup
  fails or cannot be confirmed, keep the worker locally unavailable to another user and report the failure through
  the defined interface. The external service remains responsible for its pool's availability and allocation state.
- Specify cleanup verification and root-worker reconnection authentication before their implementation. Service-facing
  notifications and lifecycle coordination must follow the supplied external interface, without inventing pool policy.

**Network, Filesystem, and Model Execution Boundaries:**

- Evaluation networking is limited to authenticated dataset/model/configuration transfer, task control, heartbeat,
  status, and validated standardized result delivery between the root and worker. External discovery/lifecycle
  communication is limited to operations required by the supplied service interface. Do not expose general-purpose
  remote execution.
- Online workers accept serialized models only from the authenticated owning root through the designated model-transfer
  path. The root-local worker receives models through its owning root's designated local boundary.
  Restrict deserialization to the defined worker boundary; arbitrary received payloads must not be executed.
- User-provided model code has no direct filesystem read/write or network permission. Cocoonet manages dataset
  references, reads and worker copies without taking ownership of root source files, supplies in-memory data through
  the supported model API, validates model outputs, and constructs and
  persists standardized reports. Models return only outputs required by that interface.
- Specify technical enforcement of execution, filesystem, and network isolation before implementing the boundary.
  Cover executable deserialization, reconstruction, and model operations; authentication or supported-API validation
  alone does not make model code safe. Enforcement must not rely on Python conventions or cooperative model behavior.

**Task Identity, Duplicate Execution, and Result Acceptance:**

- In scheduling, duplicate allocation, and result acceptance, "logical task" means a unit task within its parent
  model evaluation task. Duplicate attempts use the same supplied model, parent dataset, evaluation configuration,
  and realized partitions. They create neither another unit task nor another parent task.
- Schedule local and online workers through the same logical task system. A local attempt and an online attempt of
  the same unit compete under the same receiving-order selection, durable acceptance, and duplicate-stop rules.
  Topology does not change result priority or permit a second accepted result. Local handoff/confirmation implements
  the applicable delivery semantics without introducing remote-only networking.

- Duplicate allocation is allowed only when no unallocated tasks remain, every remaining task is finished or already
  assigned, and a worker would otherwise be idle. Select an unfinished, currently running logical task, starting with
  the oldest by its first execution start time.
- A duplicate execution is another attempt at the same logical task, not a new task or logical result entry.
  For each logical task, the root serializes result reception: when duplicate attempts are ready to return results,
  select one according to receiving order and receive only that result file at a time. This restriction applies per
  logical task; receiving results for different logical tasks remains independent.
- Selection does not depend on whether the model evaluation succeeded or failed. An earlier received failure outcome
  takes precedence over a later success; expected `NaN` and recorded computational failures do not invalidate an
  outcome. Coordinator state must enforce at most one accepted result per logical task, including reconnects.
- Once the selected result file is completely received, successfully persisted, and accepted, the logical task is
  complete. Then send stop instructions to all remaining duplicate attempts without receiving their result files.
  Workers check stop instructions before each evaluation loop and stop at those cooperative boundaries; cancellation
  need not interrupt arbitrary model code immediately. Duplicate attempts ready to return results await the root's
  reception selection or stop instruction rather than sending unselected result files.
- Persistence waiting and resends apply only to the result file currently selected for reception. Unreceived results
  from the other duplicate attempts do not enter the resend process and are not delivery failures. A transfer or
  persistence failure for the selected file does not promote a later competing result; apply its bounded retry rules.
- Duplicate execution reduces completion latency; it is not failure recovery. Do not automatically retry or reallocate
  a terminal baseline-training failure, or start another duplicate while an existing completed result awaits delivery.
- Attempt identification, stop-message schemas, cancellation acknowledgements, and other wire details belong in the
  lifecycle/protocol specification; preserve receiving-order selection, serialized reception per logical task,
  acceptance before stopping the remaining attempts, and cooperative cancellation.

**Report Receipt, Persistence, and Acceptance:**

- Produce one evaluation-result file per execution attempt of a unit task containing all its reports, results,
  and failure information. Preserve successful computations alongside unavailable values and errors under the failure-isolation
  rules. A complete outcome may contain partial computational failures; do not create separate transfer or persistence
  states for individual reports. Baseline-training failure produces the failure-only file specified above.
- Result files are expected to be small. Write received serialized files to local disk promptly using asynchronous
  I/O. The initial design does not require bounded-memory streaming or a backpressure mechanism. Do not deserialize
  them in the active network event loop; root-side content processing occurs separately from the receive path.
- Accept the selected outcome and mark its logical task finished only after the complete evaluation file has been
  received and successfully written to root-side persistent storage. Confirm successful persistence to the worker.
  Bytes received, incomplete writes, and transient in-memory status do not establish completion. Never overwrite an
  already accepted result with a later duplicate.
- For the selected result file, network-transfer or root-persistence failure leaves the logical task unfinished.
  Log the failure and request resending that same existing worker file, subject to its configured retry limit.
  Do not recompute the evaluation. Report-computation failures recorded inside a complete file neither require
  retransmission nor prevent normal acceptance. This delivery policy does not apply to unselected duplicate results.
- After sending the selected file, its worker waits for the root's persistence confirmation or resend request. Until
  successful root persistence or retry-limit exhaustion, retain that file and do not assign a new task to its worker.
  At exhaustion, end the delivery operation and its wait as failed, log it under the normal failure rules, and return
  to ordinary task handling with the worker process still running. It may receive another task when its normal
  applicable local-control/online-connection, ownership, and readiness conditions hold. The delivery operation's
  failure is separate from logical
  result acceptance: it leaves the result unaccepted and does not finish the logical task or authorize recomputation.
  Continue to apply the configured file-retention and cleanup rules; these conditions do not determine worker lifetime.
- Configure worker-side retention as deletion after confirmed delivery, retention until all root tasks finish, or no
  automatic deletion. Mandatory public-worker release cleanup overrides local retention, including pending delivery:
  no prior user's file may survive reassignment. Outside that release cleanup, do not delete a pending-delivery file
  prematurely. Cleanup of an undelivered file does not mark its logical task finished.
- Associate each result file with its parent context, unit task, and producing execution attempt. Preserve sufficient
  internal identity for recovery and at-most-one acceptance per unit task. Dataset identity is resolved through the
  parent context rather than forming an independent part of unit-task identity. Recovery must distinguish a complete
  persisted outcome from an incomplete write; mere existence of a partial file is not completion evidence.

**Root Restart and Progress Recovery:**

- Reconstruct progress from the known task set and complete root-persisted evaluation files, then reconcile worker
  state through the applicable local control or root-initiated online reconnection boundary. Do not rely solely on
  pre-crash in-memory coordinator state.
- Before resuming dispatch, restore the parent dataset association and fixed evaluation configuration from committed
  parent execution state, and load its realized schedule from the referenced authoritative root-side `dill` file,
  written once during dataset/split preparation before any unit-task allocation. Continuing, resumed, and duplicate
  attempts must use those original training
  and testing `sample_id` memberships, including after unexpected root failure. Recovery loads the established
  schedule; it must not rerun the splitter, replace the partition set, or duplicate its authoritative storage.
- If a worker is still computing, no accepted result exists, and its parent has not been stopped/cancelled/superseded,
  restore its assignment and allow it to continue. Otherwise preserve its stop disposition. Apply duplicate-stop
  rules once another attempt's result has been accepted.
- If the task's complete accepted file is already persisted on the root, reconstruct the task as finished and confirm
  delivery as needed; no reevaluation or retransmission is required. The worker can become available to its owning
  root. Stop any remaining duplicate attempts without receiving their files. This also covers a crash after persistence
  but before the in-memory completion update or acknowledgement.
- If the root lacks the file selected for reception and its worker retains the completed result with retries remaining,
  resume that selected delivery within its remaining limit before assigning the worker another task. Preserve reception
  selection; do not receive an unselected duplicate's file in place of it.
- An already exhausted delivery operation remains failed through reconnection or restart; do not silently reset its
  limit or restart it merely because a retained file exists. Apply normal operation failure handling without inventing
  an additional recovery procedure for exhaustion.
- Recovery must preserve the receiving-order selection even if the root crashed before that file was fully persisted.
  Specify durable evidence for the pending reception selection before implementing recovery; selection alone does not
  constitute result acceptance or completion of the logical task.
- Exact message schemas, retry counts/timing, file integrity and publication mechanisms, identifier/filename formats,
  and acknowledgement encoding belong in later protocol documentation. They must implement these acceptance and
  recovery semantics rather than redefine them.

**Reproducibility and Auditability:**

- The framework does not guarantee deterministic model results. Accept randomness and nondeterminism from models,
  dependencies, execution environments, and hardware without corrective manipulation.
- Pass user-provided reproducibility settings, such as seeds, through where applicable. Do not treat them as a promise
  of deterministic execution or invent extra determinism controls. Determinism of user-defined models and their
  dependencies is the user's responsibility.
- Preserve actual results, including defined failure and `NaN` outcomes. Reliability depends on simple, direct
  execution and sufficient traceability to establish how an evaluation was performed.
- Trace parent tasks, unit tasks, execution attempts, internal model identities, dataset associations, and actual
  realized fold memberships, together with supplied reproducibility settings and relevant execution information.
  Shared partitions within a parent are mandatory; deterministic model outputs are not guaranteed. Operational
  provenance does not require Cocoonet to derive or catalogue user-level model metadata. Use structured logs for
  assignment, lifecycle transitions, acceptance, and failures. Concrete provenance fields and log schemas belong
  in the architecture/protocol documentation.

---

## Technical Stack

- Language support: Python 3.12 through the latest stable Python release, excluding prereleases.

**Asynchronous I/O and Worker Connections:**

- Use `asyncio` and `aiofiles`. For online workers, the root initiates a secure WebSocket (`WSS`) connection over port
  443
  and keeps it open for evaluation communication. Workers expose the designated Cocoonet endpoint reachable from the
  root using configured private-worker information or connection information supplied by the external public service.
- Online connection establishment, scheduling, and unit-task assignment are root-controlled. Online workers accept the
  root's
  connection; they do not initiate connections to the root or to other workers. The root tracks its connected workers
  as available or working. Assign a unit-task attempt through the existing connection only after compatibility checks
  and readiness of the parent dataset, evaluation configuration, and realized partition schedule. Reuse the established
  parent context without repeating dataset identity in individual unit-task transfers. External public-pool allocation
  remains the service's responsibility.
- An online worker marks itself working and evaluates independently. When selected for result reception by the root, it
  sends its evaluation-result file through the same connection. Sending alone does not make it available for another
  task: apply the persistence acknowledgement, pending-delivery, retry-limit, and restart-reconciliation rules in
  Architecture Principles. Availability to its owning root does not release a public allocation or make the worker
  publicly discoverable.
- Reuse the bidirectional connection for assignments, status, results, and subsequent tasks. Do not replace this model
  with worker polling or separate task-submission endpoints. On reconnection, the root again initiates the connection.
- Message schemas, authentication, acknowledgements, heartbeat/disconnect detection, reconnection, and message
  serialization details are defined incrementally before implementing the corresponding protocol units.

**Online Connection Rounds and Allocation:**

- Use one asynchronous per-worker connection/reconnection policy before start, during execution, and after disconnect.
  Configure a retry period and a sequence of delays within each round. For example, a 60-second period with delays
  `[1, 2, 4]` starts rounds periodically; after an initial failure, additional attempts follow those delays.
  Exhausting a round leaves the worker unavailable until the next scheduled round, without removing its configuration.
- A later scheduled connection round is a distinct policy operation, not a reset of an exhausted result-delivery
  budget. Connection failures do not fail unrelated tasks, block coordinator control, or authorize model reevaluation.
  Runtime availability changes do not edit configured worker membership or make the instance unconsolidated.
  Disconnect makes an assigned worker unavailable for new allocations; its existing attempt follows the established
  attempt, delivery, duplicate, and recovery rules. A usable reconnection restores eligibility after normal readiness.
- At allocation opportunities, asynchronously check configured idle online candidates. Skip unavailable workers;
  allocate first to the responsive eligible worker with the fastest response for that opportunity, then to remaining
  eligible idle workers while tasks remain. This observation is not permanent priority and does not change logical
  duplicate-task selection or result acceptance. Local execution does not require artificial network probes.
- Exact APIs, probe/response measurement, timer mechanics, and state representation remain implementation details.
  Existing authentication, exclusive ownership, and compatibility rejection requirements continue to apply.

**Serialization Ownership and Compatibility:**

- Use `dill` exclusively for Cocoonet serialized payloads; do not introduce a parallel `pickle` serialization path.
- The root prepares model inputs using the model-specific validation and reconstruction contracts below. Only the
  root creates serialized model payloads for workers, and only workers deserialize those payloads at the technically
  enforced model execution boundary. API validation and authenticated provenance are not safety guarantees.
- Workers construct and serialize standardized evaluation-result files; only the root deserializes them, outside
  the active receive event loop. Restrict model/result loading to their designated paths, not arbitrary serialized
  data. Separately, the root loads its authoritative partition `dill` file through the designated dataset/partition
  preparation or recovery path, validating the dataset association before reuse.
- Require exactly equal complete Cocoonet version identifiers for root and worker. For online workers, check during
  the connection handshake before normal protocol operation, dataset synchronization, task assignment, or model
  deserialization. Local compatibility checking does not require a network handshake. Matching major/minor versions
  is insufficient: root `1.2.3` accepts worker `1.2.3` and rejects `1.2.4`.
- Read compatibility metadata, including online handshake version metadata, without loading model or report payloads.
  Reject an incompatible worker immediately,
  identify both versions in the error, and suggest installing the same Cocoonet version. Do not attempt compatibility
  fallbacks or deserialize model payloads after rejection.
- Each Cocoonet release must operate across all Python versions it declares as supported. Root and worker Python
  versions may differ within that range; identical Python versions are not required. Dependency constraints and
  release validation must uphold this runtime and serialization contract. Do not add dependency or environment
  negotiation to the worker protocol.

**Model Preparation and Reconstruction:**

- The root assigns each supplied model an arbitrary module-generated unique `model_id`, associated with a user-provided
  `model_label` unique within the parent task. Use `model_id` internally for references, execution tracking, duplicate
  handling, and provenance; identify models by `model_label` in user-facing reports.
- The supplied model or construction information is the authoritative executable input and remains fixed across
  attempts of its unit task. Cocoonet does not derive or maintain a separate catalogue of user-level model descriptions,
  hyperparameter metadata, preprocessing provenance, or experiment lineage. Those remain the user's responsibility.
  Required reconstruction and training inputs, including parameters used by `get_params`/`set_params` and supplied
  PyTorch configuration, remain execution inputs.

- For standard scikit-learn models, identify the supported model type and obtain its parameters through `get_params`.
  Reconstruct/configure the corresponding worker-side model efficiently using `set_params` as appropriate.
- For custom scikit-learn-like models, validate only the required API: `fit` and `predict` for regression; `fit`,
  `predict`, `predict_proba`, and the required class information for classification. Do not perform deeper dependency,
  implementation, or execution-compatibility validation.
- For PyTorch, reconstruct from the complete construction and training information in Project Overview: definition
  and architecture, input/output configuration, criterion/loss, optimizer, applicable scheduler, training settings,
  initialization configuration, and optional supplied weights. Do not add third-party dependency negotiation.
- These paths establish the information needed to attempt evaluation; they do not guarantee that every supplied
  model will execute successfully. After the applicable validation and reconstruction, execute directly.
- Do not add compatibility prediction, dependency inspection/negotiation, environment repair, fallback reconstruction,
  model substitution, or corrective result manipulation. Report actual reconstruction, fitting, prediction, and other
  operation failures through the existing isolation and logging boundaries.
- Required reconstruction failure that prevents baseline training terminates the current unit-task evaluation with a
  failure-only result file. Baseline-fit failure has the same terminal scope; failures in later predictions, reports,
  or sample-removal refits affect only their dependent results as defined in Agent Instructions.

**Quality and Test Baseline:**

- Required tools are Ruff, mypy, Black, and pytest. Refer to `pyproject.toml` for their detailed configuration,
  including Ruff exclusions, pytest discovery, mypy settings, and documentation-build/version-switching checks.
- Ruff selects the `E`, `F`, `W`, `B`, `I`, `N`, and `C` rule families, subject to explicitly specified exclusions and
  the narrow suppression policy below. Use the exclusions in `pyproject.toml`; do not invent additional ones or rule
  families.
- mypy requires typed definitions and return checking, but not full `strict = true`. Ignore missing third-party
  import typing; project-owned code remains subject to the configured checks.
- Black uses a 120-character line length, with configured targets consistent with the supported Python versions.
- pytest runs tests under `tests/` using the discovery conventions in `pyproject.toml`.
- Applicable pytest, mypy, Black, and Ruff checks must all pass before accepting an implementation. When documentation
  is rebuilt, also validate local accessibility and version switching using the configured checks. Report unavailable
  checks accurately. Do not expand tool requirements or add dependencies outside the approved scope.

---

## Code Quality Standards

**NumPy-Style Source Docstrings and Public API Documentation:**

- Every class, function, and method, including private/internal components, must have a source docstring explaining
  its purpose and responsibility. Use NumPy-style sections only where applicable to the documented object.
- Use NumPy-style `Parameters`, rather than `Args`, for parameters requiring documentation, with types and
  descriptions; omit the section when none apply.
- Include `Returns` only for meaningful return values and `Yields` for yielded values. Classes and constructors do
  not require a return-value section merely because they construct an instance. Do not add empty placeholder sections.
- `Raises` documents exceptions intentionally exposed to callers as part of the interface contract. Do not list
  internally handled exceptions or exhaustively enumerate incidental third-party/user-component exceptions unless
  they are part of that contract.
- Include realistic usage examples where usage is non-trivial.
- User-facing API documentation includes public APIs only. Private/internal classes, functions, methods, helpers,
  and implementation details still require source docstrings, but documenting them does not make them public APIs.


**Clarity, Comments, and Decomposition:**

- Keep functions and methods focused, cohesive, simple, and directly understandable. Approximately 20-100 lines is a
  guideline, not a mechanical acceptance criterion. Do not pad short functions or split coherent logic just to meet it.
- Decompose around coherent operations when it improves readability, testability, reuse, or separation of
  responsibilities. Avoid trivial wrappers and unnecessary abstractions introduced only to shorten a function.
- Comment non-obvious reasoning, scientific assumptions, invariants, algorithmic decisions, performance-sensitive
  behavior, and constraints that cannot reliably be understood from the code alone. Explain why, not clear syntax.
- Prefer clear names and straightforward control flow to comments that restate the code. Preserve any specifically
  required documentation of scaffolding/internal tooling from Agent Instructions.
- Keep scientific computations as direct representations of their defined procedures. Do not fragment a coherent
  formula into helpers when doing so obscures its mathematical meaning or auditability.
- Keep source and test lines within 120 characters. Keep source files focused with clear domain boundaries, typically
  under 300-500 lines; file size is guidance, not permission for unrelated restructuring.


**Static Typing, Linting, and Narrow Suppressions:**

- Require typed definitions and return checking under the configured mypy baseline, not full strict mode. Missing
  third-party import typing is ignored as specified in Technical Stack. Project-owned code must pass its configured
  type-checking and linting rules.
- Permit narrow suppressions only for external dependency typing limitations: incomplete/inaccurate type information
  or external runtime behavior the checker cannot represent correctly. Keep each suppression local to the affected
  expression or line, use the most specific applicable error code, and briefly explain the external limitation.
- Do not use blanket suppressions or conceal errors in project-owned interfaces, implementation logic, or complexity.
  A suppression is not a substitute for correcting project code or seeking approval for necessary decomposition.
- Applicable pytest, mypy, Black, and Ruff checks must pass with only the explicitly permitted exclusions and
  suppressions. When documentation is rebuilt, also verify local accessibility and version switching.
