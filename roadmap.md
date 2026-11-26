# Cocoonet development roadmap

## General phase map

1. Shared identities, records, and component contracts
2. Filesystem ownership and durable persistence foundations
3. Reusable EvaluationTask configuration and lifecycle foundations
4. Dataset normalization and authoritative realized partitions
5. Technically enforced model execution boundary
6. Scikit-learn and custom-model preparation and execution
7. Baseline fold evaluation and validation results
8. Ordinary reports and visualization data
9. Optional sample influence
10. PyTorch reconstruction and evaluation integration
11. Bounded parallel evaluation and worker resource control
12. Complete attempt result bundles and worker persistence
13. Local coordinator, worker control, and result acceptance
14. Bounded operations, delivery waits, and result retention
15. Cooperative interruption, local reconciliation, and reuse
16. Private online-worker connection and control
17. Online synchronization, execution, and delivery
18. Mixed scheduling and duplicate-attempt arbitration
19. Explicit worker data management and cleanup
20. External public-worker service integration
21. Complete compatibility and lifecycle verification

## 1. Shared identities, records, and component contracts

### Intention

Establish a shared vocabulary for dataset, sample, model, parent, attempt, prediction, report, failure, and runtime observations before their producers and consumers diverge. Configuration, scientific outcomes, operational progress, and worker availability remain distinct. This foundation is recorded as implemented and verified in the existing development documents.

### Technical route

Small immutable records and pure validators represent identity scopes, exact sample/class alignment, fixed parent inputs, and failure provenance. Scientific value meaning follows the value and its failure associations, while evaluation outcome is derived from the complete result. Runtime records describe observations rather than performing scheduling or establishing acceptance. Compatibility is decided from plain release metadata. Internal record layouts and validation mechanics are documented in their audited source.

### Unit component map

| Unit | Component names | Intention | Technical tip |
| --- | --- | --- | --- |
| Identity vocabulary | `ReportName`, `validate_identifier`, `validate_label` | Distinguish scoped identities from user-facing labels and report categories. | Separate identity and label checks preserve their different domains. |
| Samples and partitions | `SampleRecord`, `PartitionReference`, `FoldRecord`, `PartitionSchedule`, `validate_source_reference`, `validate_sample_record`, `validate_partition_schedule` | Describe shared sample metadata and the complete ordered partition association. | Source syntax, sample validity, and cross-fold associations have separate validation owners. |
| Configuration references | `TaskKind`, `ModelFamily`, `ConsolidationState`, `ModelReference`, `EvaluationReference`, `ModelingSnapshot`, `validate_model_reference`, `validate_evaluation_reference`, `validate_modeling_snapshot` | Describe incomplete or committed root-side scientific configuration without executing it. | Container validation and cross-reference validation address different facts. |
| Fixed execution inputs | `ParentEvaluation`, `AttemptInputs`, `validate_parent_evaluation`, `validate_attempt_inputs` | Bind attempts to a fixed parent's model units. | Parent/model association supplies unit identity without another stored key. |
| Failures | `FailureScope`, `TraceContext`, `FailureRecord`, `validate_trace_context`, `validate_failure_record`, `validate_failure_links` | Preserve scoped errors and traceable prerequisite failures. | Shared origins avoid copying the same failure into dependent reports. |
| Predictions | `RegressionPredictions`, `ClassificationPredictions`, `validate_regression_predictions`, `validate_classification_predictions` | Associate model outputs and unavailable values with exact samples and classes. | Alignment and availability remain independently checkable. |
| Complete results | `ReportStatus`, `ReportOutcome`, `EvaluationOutcome`, `AttemptResult`, `validate_report_outcome`, `validate_attempt_result`, `derive_evaluation_outcome` | Represent complete scientific outcomes while preserving independent report failures. | Outcome derivation consumes validated evidence rather than maintaining another status. |
| Execution observations | `ExecutionStatus`, `AttemptStatus`, `StopStatus`, `StopReason`, `ParentProgress`, `StopDisposition`, `AttemptProgress`, `validate_parent_progress`, `validate_stop_disposition`, `validate_attempt_progress` | Describe parent execution, assigned attempts, and cooperative stop observations. | Stop disposition and attempt status describe different lifecycle facts. |
| Worker observations | `WorkerAvailability`, `WorkerProcessStatus`, `OwnershipStatus`, `CompatibilityStatus`, `ParentContextReadiness`, `WorkerObservation`, `validate_worker_observation` | Describe usability, parent readiness, and occupied attempt capacity separately. | Free capacity is derived from reservations rather than another maintained counter. |
| Operation observations | `OperationKind`, `OperationStatus`, `BoundedOperation`, `ReceiptStatus`, `PersistenceStatus`, `validate_bounded_operation` | Describe bounded operational work without selecting an accepted result. | A later connection round has its own operation identity. |
| Compatibility | `CompatibilityMetadata`, `validate_compatibility_metadata`, `require_compatible` | Establish release compatibility before executable payload access. | Metadata validation and cross-node comparison remain separate. |

## 2. Filesystem ownership and durable persistence foundations

### Intention

Establish visible managed storage, user-source ownership, safe fresh preparation, and reliable publication before configuration or execution can create persistent artifacts. Existing development records mark this foundation implemented and verified. State files support inspection; complete final result bundles remain the evidence of scientific completion.

### Technical route

Canonical physical paths provide the boundary for protected locations and bidirectional source ownership checks. Semantic path builders locate the contracted root and worker layouts. Focused file and directory publication operations distinguish complete artifacts from staging. Deterministic CSV codecs and serialized asynchronous writers preserve metadata and attempt-row identity; structured logs retain operational context. Existing-directory opening is separate from fresh preparation, and domain schemas remain with their producing phases.

### Unit component map

| Unit | Component names | Intention | Technical tip |
| --- | --- | --- | --- |
| Storage declarations | `StorageError`, `RootStorageLayout`, `WorkerStorageLayout`, `WorkerParentStorageLayout`, `BundlePublicationPaths` | Describe storage locations and scoped operational failure. | — |
| Physical paths | `resolve_physical_path`, `_normalized_path_parts`, `_paths_equal`, `_is_path_within` | Resolve aliases and compare physical filesystem boundaries. | Component comparisons avoid lexical-prefix ambiguity. |
| Protected roots | `discover_protected_locations`, `_validate_managed_work_dir_against`, `validate_managed_work_dir` | Identify locations unsuitable for managed root storage. | Discovery and pure comparison keep environmental evidence separate from decisions. |
| Source ownership | `_validate_canonical_dataset_source_ownership`, `validate_dataset_source_ownership`, `validate_work_dir_ownership` | Check proposed storage against all configured user sources before mutation. | Aggregate validation covers retained source paths as well as new ones. |
| Path identifiers | `_validate_dataset_id`, `_validate_parent_id`, `_validate_positive_decimal_identifier` | Check the semantic identifier forms used by managed paths. | Path-specific checks are narrower than general identity validation. |
| Root and worker layouts | `build_root_storage_layout`, `build_worker_storage_layout`, `build_worker_parent_storage_layout` | Locate contracted root, worker, and parent storage scopes. | Layout construction describes paths without preparing artifacts. |
| Dataset and model locations | `worker_dataset_directory`, `root_trusted_model_dill_path`, `worker_trusted_model_dill_path` | Locate dataset copies and the existing trusted-model path boundary. | Legacy dill-named helpers remain distinct from later family-specific serialization. |
| Bundle locations | `root_bundle_publication_paths`, `worker_bundle_publication_paths` | Locate sibling staging and final result directories. | Both node roles share publication naming while retaining separate ownership. |
| File publication | `staged_file_path`, `publish_staged_file`, `discard_staged_file` | Publish or discard one exact staged file. | Staging location and final authority remain separate facts. |
| Bundle publication | `publish_staged_bundle`, `discard_staged_bundle` | Publish or discard one complete staged result directory. | One directory boundary keeps bundle members in a common lifecycle. |
| Metadata | `NodeRole`, `CocoonetMetadata`, `current_compatibility_metadata`, `validate_cocoonet_metadata`, `encode_cocoonet_metadata_csv`, `decode_cocoonet_metadata_csv` | Capture and round-trip plain node metadata. | Encoding and decoding share the persistence layout in contracts. |
| Root preparation and opening | `prepare_work_directory`, `open_work_directory` | Separate fresh managed-directory preparation from non-destructive opening. | Existing metadata is evidence for opening rather than permission to initialize. |
| Attempt-state persistence | `ExecutionStateRecord`, `validate_execution_state_records`, `encode_execution_state_csv`, `decode_execution_state_csv`, `_execution_state_lock`, `_merge_execution_state_records`, `replace_execution_state`, `load_execution_state` | Preserve observed attempt rows through serialized complete-file updates. | Exact prior observations distinguish stale revisions from append-only additions. |
| Structured logs | `LogChannel`, `StructuredLogRecord`, `validate_structured_log_record`, `encode_structured_log_record`, `decode_structured_log_record`, `append_root_log` | Persist inspectable operational events and scoped failures. | The event codec and node-specific routing serve separate boundaries. |

## 3. Reusable EvaluationTask configuration and lifecycle foundations

### Intention

Provide reusable prospective root configuration without starting an evaluation. Incomplete configuration, ordered model and worker registries, safe fresh initialization, and read-only inspection form the public foundation. Model-evaluation edits and worker-only edits have different effects on parent reuse. Explicit configuration and progress cloning provide independently owned destinations, while ambiguous generic copying is rejected. The package root exposes only the primary task entry point and version inspection; optional output-record imports are isolated in `cocoonet.types` and are not required by ordinary task workflows.

### Technical route

Configuration records and pure lifecycle helpers precede the mutable configuration interface. Existing storage preparation supplies its filesystem boundary. Model inputs remain opaque candidates or canonical file references until isolated adapters are available. Validation and staging precede bounded registry commits, and inspection derives detached presentations. Worker registration describes configuration rather than runtime readiness. Generic copying fails explicitly. Configuration cloning prepares fresh storage, retains prospective configuration, and resets lifecycle progress. Progress cloning copies an inactive managed work directory and its lifecycle state into an independent destination, rebasing managed path references while retaining user-owned dataset references. Public release keeps the package root minimal. `EvaluationTask` owns record construction, while `cocoonet.types` provides an optional stable import path for output-record annotation and inspection.

### Unit component map

| Unit | Component names | Intention | Technical tip |
| --- | --- | --- | --- |
| Configuration records | `WorkerKind`, `ModelRegistration`, `WorkerConfiguration`, `validate_model_registration`, `validate_worker_configuration` | Describe prospective model and worker configuration. | Record validity and registry-wide agreement are different checks. |
| Registered model association | `_RegisteredModel`, `_RegisteredModel.model_id`, `_RegisteredModel.model_label` | Keep opaque model input associated with one immutable reference. | Derived properties avoid duplicated identity state. |
| Model staging | `_normalize_model_family`, `_normalize_task_kind`, `_stage_model_registration` | Prepare a validated prospective registration without committing it. | Shared staging preserves singular and bulk validation agreement. |
| Edit lifecycle | `_consolidation_after_evaluation_edit`, `_require_configuration_editable` | Separate configuration-edit admission from consolidation changes. | Worker-only edits bypass the scientific edit transition. |
| Parent preparation | `prepare_parent_evaluation` | Derive a fixed scientific parent description from complete prospective configuration. | — |
| Construction and initialization | `EvaluationTask`, `EvaluationTask.__init__`, `EvaluationTask.initialize` | Own reusable configuration and explicit fresh-directory preparation. | Successful preparation precedes replacing configuration ownership. |
| Inspection | `EvaluationTask.work_dir`, `EvaluationTask.state`, `EvaluationTask.parent_id`, `EvaluationTask.snapshot` | Expose root configuration without sharing mutable registries. | Snapshots derive their views from the owned records. |
| Model presentation | `_model_input_display`, `_filter_registered_models`, `_models_to_dataframe`, `EvaluationTask.list_models` | Present matching model metadata without invoking model behavior. | Presentation limits remain separate from registry mutation scope. |
| Model mutations | `EvaluationTask.add_model`, `EvaluationTask.remove_model`, `EvaluationTask.add_models`, `EvaluationTask.remove_models` | Apply singular or atomic bulk model configuration changes. | Singular semantics and shared staging support consistent batch commits. |
| Worker availability, network phase deferred | `EvaluationTask.list_available_workers` | Present worker-reported availability independently of task allocation membership. | Workers initiate root-directed connections. The future network implementation initializes the availability snapshot, replaces it after completed heartbeat and handshake rounds, and refreshes it before `run` allocates attempts. |
| Worker registry | `EvaluationTask.add_worker`, `EvaluationTask.add_workers`, `EvaluationTask.list_workers`, `EvaluationTask.remove_worker`, `EvaluationTask.remove_workers` | Maintain task-allocation membership independently of current worker availability. | Worker addition later retrieves worker-owned properties from the available snapshot. Runtime allocation requires a worker ID to occur in both the available snapshot and task registry. |
| Copy rejection | `EvaluationTask.__copy__` | Reject shallow copying that cannot express independent persistence and lifecycle ownership. | Direct users to the explicit configuration or progress cloning operation. |
| Configuration cloning | `EvaluationTask.clone_configuration` | Reuse prospective configuration in a freshly prepared work directory without execution progress. | Retain independently editable registry containers, reset parent and execution lifecycle state, and do not copy managed artifacts. |
| Progress cloning | `EvaluationTask.clone`, `_clone_work_directory` | Duplicate inactive persisted progress and corresponding in-memory state into an independent work directory. | Reject active execution; rebase references to managed paths while retaining external dataset references. Future persistence schemas should use owner-relative managed paths where possible and must add codec-owned rebasing for any absolute managed paths. |
| Public release surface, pending | `get_version`, `EvaluationTask` and the public declarations listed in contracts | Expose the primary task interface while keeping output-record imports optional and separate from ordinary workflows. | The package root exports only `EvaluationTask` and `get_version`. `cocoonet.types` exposes output records for optional annotation and inspection; users normally call task methods without importing or constructing these records. Lifecycle and selection enums and staging-oriented records remain internal. `get_version` is a public convenience only and must never be used internally. |

## 4. Dataset normalization and authoritative realized partitions

### Intention

Make dataset registration produce a validated dataset and one reusable realized partition schedule. Inline features, reconstructible tensors, and file-backed samples share sample identity, labels, grouping, and independent training/testing eligibility. User-owned sources remain outside managed root storage, and every model of a parent shares the same saved memberships.

### Technical route

Source adapters standardize supported tables one source at a time and preserve exact sample associations through reference generation. Root-side partition preparation either realizes configured splitting or validates an existing artifact. Group separation and mask eligibility apply to every saved fold. Transfer preparation produces reusable compressed or sharded representations from stored source sizes, with metadata-based cross-execution reuse. Mixed inline/file rows, tensor-shape encoding, empty-population admissibility, and artifact carry-forward remain phase-specific design decisions.

### Unit component map

| Unit | Component names (planned) | Intention | Technical tip |
| --- | --- | --- | --- |
| Source reading and standardization | `read_dataset_source`, `standardize_dataset_source`, `parse_tensor_shape` | Translate user tables into the common sample-oriented representation. | Initial label matching and later internal identity alignment serve different stages. |
| Dataset association | `prepare_dataset_reference`, `validate_dataset_association` | Bind standardized samples to their actual source files and dataset identity. | Logical references remain independent of machine storage paths. |
| Partition preparation | `realize_partition_schedule`, `load_partition_schedule`, `persist_partition_schedule` | Prepare or reuse the complete ordered schedule as one authoritative artifact. | Stored memberships preserve realized randomness more fully than a seed. |
| Transfer preparation | `prepare_tabular_transfer`, `load_transfer_preparation_state`, `persist_transfer_preparation_state` | Prepare reusable source-wise representations without changing original data. | Per-source sizing and dataset-wide shard naming address different scopes. |
| Dataset registration | `EvaluationTask.add_dataset` | Associate configuration with a prepared dataset and realized split. | — |
| Dataset inspection and batches | `inspect_dataset_configuration`, `configure_dataset_batch`, `remove_dataset_configuration`, `remove_dataset_configurations` | Support inspection and singular/bulk changes to dataset configuration. | These are provisional component names; public declarations and batch semantics remain deferred. |

## 5. Technically enforced model execution boundary

### Intention

Establish a worker-side boundary that contains executable loading, reconstruction, and model operations while keeping control responsive. Private and root-local execution retain trusted custom-model support, and public execution has a separate declarative input boundary. The coordinator remains outside model execution.

### Technical route

A technically enforced filesystem and network boundary surrounds model execution and permitted framework child processes. Framework-owned data access supplies in-memory inputs, and output validation returns aligned values or scoped failures. Responsive control and attempt execution have separate lifetimes. The enforcement mechanism remains an explicit design dependency rather than a claim established by authentication, process separation, or Python conventions alone.

### Unit component map

| Unit | Component names (planned) | Intention | Technical tip |
| --- | --- | --- | --- |
| Execution boundary | `ModelExecutionBoundary` | Own the selected enforcement mechanism and its execution lifetime. | — |
| Framework data access | `prepare_execution_inputs`, `validate_execution_outputs` | Mediate data entry and standardized model output. | Data access ownership and model permission remain distinct. |
| Isolated invocation | `load_model_in_boundary`, `invoke_model_in_boundary` | Cover both executable loading and later model operations. | Deserialization is an execution boundary as well as an input operation. |
| Responsive control | `WorkerControl` | Keep communication and cooperative lifecycle handling available during evaluation. | — |

## 6. Scikit-learn and custom-model preparation and execution

### Intention

Support serial model reconstruction, fitting, and prediction for standard scikit-learn and custom scikit-learn-like inputs. The same standardized prediction boundary serves later evaluation and reporting. Private/local trusted input and public declarative input retain their distinct scopes.

### Technical route

Isolated input loading dispatches from the complete user-file extension and validates the declared family and task. Standard estimators reconstruct from supported type and parameters; custom candidates receive the required interface validation. Managed serialization selects the family-specific representation independently of user input format. Pristine construction state supports fresh fold models, and declarative preparation supports controlled public components. Fixed-input capture precedes later parent commitment; deeper dependency prediction and environment repair are outside this route.

### Unit component map

| Unit | Component names (planned) | Intention | Technical tip |
| --- | --- | --- | --- |
| Input preparation | `load_user_model_input`, `validate_model_candidate`, `prepare_fixed_model_input` | Resolve an opaque registration into a fixed executable or declarative input. | Registration admission and isolated semantic validation occur at different boundaries. |
| Managed persistence | `serialize_managed_model`, `load_managed_model` | Preserve the selected trusted family representation for reuse. | The stored format identifies its loader independently of the original input suffix. |
| Initialization | `initialize_sklearn_model`, `initialize_sklearn_like_model`, `create_fold_model` | Produce pristine construction state and independent untrained fold models. | Fresh learned state and fixed construction inputs are complementary. |
| Execution adapters | `fit_model`, `predict_model`, `align_model_predictions` | Produce standardized regression or class-aligned predictions. | Explicit class identity prevents probability-column ambiguity. |
| Declarative input | `prepare_declarative_model`, `reconstruct_declarative_model` | Represent supported estimator compositions through controlled component identities. | Declarative structure remains separate from arbitrary Python objects. |

## 7. Baseline fold evaluation and validation results

### Intention

Evaluate one complete model attempt over the saved parent schedule and retain its primary held-out predictions. Reconstruction failure and primary-validation failure remain distinct scientific outcomes, and successful independent folds survive later fold failures. Fold computations remain within the attempt rather than becoming distributed tasks.

### Technical route

Initial reconstruction establishes pristine construction state. Each realized fold receives a fresh model, its saved training population, and its unchanged testing population. Standardized predictions and provenance form the validation payload and the later influence baseline. Independent folds can continue after scoped failures when construction remains usable. Initial reconstruction failure yields a failure-only outcome; incomplete primary validation blocks branches requiring a complete baseline.

### Unit component map

| Unit | Component names (planned) | Intention | Technical tip |
| --- | --- | --- | --- |
| Fold data access | `resolve_fold_populations`, `collect_fold_sources` | Resolve saved memberships and their sample/source associations. | Shared source files do not establish shared sample membership. |
| Primary fold | `evaluate_primary_fold` | Fit and predict one fold with independent learned state. | — |
| Attempt validation | `evaluate_primary_validation`, `build_validation_result` | Collect aligned fold results and their complete-stage outcome. | Stored baseline predictions provide a stable downstream comparison. |
| Scientific failure boundary | `build_reconstruction_failure_result`, `record_primary_fold_failure` | Preserve the distinct reconstruction and primary-fold failure scopes. | Failure provenance identifies unavailable values without replacing successful folds. |

## 8. Ordinary reports and visualization data

### Intention

Produce independent performance, residual, and visualization data from primary-validation results. Reports preserve explicit sample and class identity, actual scientific values, and scoped failures. Standard metrics retain conventional definitions, while agreed project-specific variations remain identifiable in their owning source documentation.

### Technical route

Report producers consume the shared validation evidence rather than retraining models. Classification follows the agreed classwise, micro-average, and macro-average measures and signed probability-residual analysis; regression follows its agreed performance and residual conventions. Shared intermediates support independent report branches. Scientific undefinedness, permitted infinities, computational unavailability, and non-applicability remain distinct. Visualization producers return rendering data, with workbook layouts located in contracts and detailed formulas, edge behavior, and approved missing-value choices documented in audited source.

### Unit component map

| Unit | Component names (planned) | Intention | Technical tip |
| --- | --- | --- | --- |
| Classification performance | `compute_classification_performance` | Produce classwise and aggregate classification measures. | — |
| Classification residuals | `compute_probability_residuals`, `compute_residual_z_scores`, `compute_residual_summaries`, `compute_residual_mahalanobis` | Describe signed class-probability errors and their sample-level structure. | Class order and shared residual populations connect the calculations. |
| Regression reports | `compute_regression_performance`, `compute_regression_residuals` | Produce regression measures and aligned residual analysis. | Sign and population conventions belong beside the owning calculation. |
| Visualization data | `build_roc_data`, `build_scatter_data`, `build_residual_plot_data` | Produce standardized rendering inputs associated with report evidence. | Reusing report evidence avoids a competing numeric summary. |
| Independent report orchestration | `evaluate_ordinary_reports` | Collect report outcomes without extending one failure into an independent branch. | — |

## 9. Optional sample influence

### Intention

Describe the normalized effect of removing a training sample on held-out predictions. Influence is optional within a complete attempt and depends on successful primary validation, while remaining independent of ordinary report computation afterward. Only folds in which the sample trains contribute to that sample's influence.

### Technical route

Fresh removal models reuse the parent's schedule and the stored primary predictions as a fixed baseline. Class-specific probability comparisons and single-target regression comparisons follow the approved generalized Cook-like measure. Baseline held-out error and actual predictor count determine normalization; aggregation follows exact held-out comparison counts, including repeated and zero-change comparisons. Removal failures remain local to sample, fold, and class, preserving independent contributions. Detailed formulas and numerical edge behavior belong in the audited calculation source.

### Unit component map

| Unit | Component names (planned) | Intention | Technical tip |
| --- | --- | --- | --- |
| Removal evaluation | `evaluate_sample_removal` | Refit without one eligible training sample and predict unchanged held-out rows. | — |
| Influence contributions | `compute_classification_influence`, `compute_regression_influence` | Quantify normalized prediction change against the stored baseline. | Baseline error and removal prediction change describe different comparisons. |
| Aggregation | `aggregate_sample_influence`, `count_prediction_comparisons` | Associate each sample's contributions with its exact comparison population. | Comparison count follows schedule membership rather than successful numeric contributions. |
| Influence report | `evaluate_sample_influence`, `build_influence_report` | Preserve requested influence values, coverage, and scoped failure provenance. | Ordinary reports and influence share a baseline without sharing failure ownership. |

## 10. PyTorch reconstruction and evaluation integration

### Intention

Extend the established evaluation and report path to supplied PyTorch reconstruction and training information. Worker-owned device selection and user-selected initialization remain visible execution inputs. Public workers use declarative components and non-executable supplied weights, while private/local workers retain trusted custom-model capabilities.

### Technical route

Reconstruction resolves model definition, architecture, data interpretation, training procedure, loss, optimizer, scheduler, initialization, and supplied randomness settings. Fresh fold and removal state feeds the existing validation and reporting components. Public supplied weights have a model-specific Parquet representation with exact tensor reconstruction metadata and reusable preparation. Framework loading/training settings determine accounted CPU demand for the next phase. Device-specific failures remain scoped outcomes rather than reasons for model substitution.

### Unit component map

| Unit | Component names (planned) | Intention | Technical tip |
| --- | --- | --- | --- |
| Reconstruction | `initialize_pytorch_model`, `initialize_pytorch_training_state` | Reconstruct the model and its supplied training state. | Pristine state covers optimizer and initialization as well as model structure. |
| Training and prediction | `fit_pytorch_model`, `predict_pytorch_model`, `align_pytorch_predictions` | Adapt PyTorch execution to the existing prediction boundary. | Output interpretation and class alignment connect framework outputs to shared reports. |
| Public weights | `prepare_public_weight_artifacts`, `load_public_weight_artifacts` | Preserve supplied tensor values in non-executable model-specific artifacts. | Weight identity remains separate from dataset identity. |
| Public components | `prepare_declarative_pytorch_model`, `reconstruct_declarative_pytorch_model` | Resolve public models through controlled supported components. | Public component restrictions do not redefine private/local reconstruction. |
| Evaluation and demand | `evaluate_pytorch_attempt`, `resolve_pytorch_cpu_demand` | Integrate scientific evaluation with framework-aware resource accounting. | Outer attempt concurrency and framework-managed parallelism have separate owners. |

## 11. Bounded parallel evaluation and worker resource control

### Intention

Allow concurrent complete CPU attempts while preserving responsive worker control and fixed model/framework settings. Requested concurrency, effective capacity, reservations, and actual process availability remain distinct. GPU execution retains a single Cocoonet attempt slot.

### Technical route

Worker control derives available CPU capacity after reserving control resources and accounts for the maximum eligible per-attempt framework demand. The agreed calculation either retains requested concurrency, reduces it once, or rejects incompatibility. Reservations cover whole attempt lifetimes, including pending delivery, and changed CPU availability triggers capacity revalidation. Cocoonet concurrency remains across attempts, with nested framework parallelism accounted for inside each attempt and isolated subprocess supervision outside the scientific calculations.

### Unit component map

| Unit | Component names (planned) | Intention | Technical tip |
| --- | --- | --- | --- |
| Capacity evidence | `detect_available_cpus`, `resolve_model_cpu_demand` | Observe worker capacity and fixed-input CPU demand. | Maximum eligible demand also covers duplicates of the highest-demand model. |
| Effective concurrency | `derive_effective_attempt_capacity` | Resolve the agreed one-step adaptation or incompatibility decision. | — |
| Reservations | `reserve_attempt_slot`, `release_attempt_slot` | Associate occupied capacity with complete attempt lifetimes. | Pending delivery continues to occupy its producing reservation. |
| Process ownership | `start_attempt_process`, `supervise_attempt_process`, `stop_attempt_process` | Own isolated attempt subprocess lifetimes independently of worker control. | Nested framework subprocesses remain part of the attempt's ownership scope. |
| Capacity inspection | `inspect_worker_capacity`, `revalidate_worker_capacity` | Expose capacity evidence and refresh admission after environmental changes. | A lower prospective limit affects admission rather than rewriting existing reservations. |

## 12. Complete attempt result bundles and worker persistence

### Intention

Turn every complete scientific attempt outcome into one immutable worker result bundle. Successful values, unavailable values, and failures remain associated with the same producing attempt. Reconstruction failure and validation failure can produce complete bundles without being mistaken for failed persistence.

### Technical route

Result assembly collects applicable reports, provenance, visualization data, references, and logs into a manifest-governed inventory. Workbook serialization preserves the contracted layouts and scientific cell distinctions. Complete staging content passes bundle validation before directory publication through the storage foundation. Bundle identity and completeness supply the later delivery boundary; individual members have no independent acceptance lifecycle. Model-keyed worker paths retain the producing attempt association inside their metadata.

### Unit component map

| Unit | Component names (planned) | Intention | Technical tip |
| --- | --- | --- | --- |
| Result tables | `encode_result_workbook`, `decode_result_workbook` | Preserve contracted result and metadata worksheets. | Cell representation distinguishes scientific undefinedness from unavailability. |
| Bundle assembly | `build_attempt_bundle`, `build_bundle_manifest` | Collect one attempt's complete artifact inventory and associations. | Conditional artifacts are explicit inventory members rather than missing-file guesses. |
| Bundle validation | `validate_attempt_bundle` | Establish complete schemas, identities, and inventory before publication. | — |
| Worker publication | `publish_worker_bundle`, `publish_staged_bundle` | Publish a validated complete attempt outcome in worker storage. | Scientific outcome and publication success describe separate facts. |

## 13. Local coordinator, worker control, and result acceptance

### Intention

Deliver the first complete normal evaluation through an explicitly configured root-local worker. The root commits a fixed parent, coordinates attempts, and accepts durable model results. Startup prerequisite failures remain distinct from scientific evaluation outcomes, and root coordination itself supplies no implicit execution target.

### Technical route

Local process control uses the established spawn and duplex-pipe boundary with explicit bounded message serialization. Parent commitment consumes fixed prepared model inputs and the saved partition, persisting execution configuration and initial attempt observations. Worker-parent readiness precedes assignment. A validated worker bundle travels through direct managed-storage handoff; successful root publication establishes acceptance and its acknowledgement. The root adds its own reference and inventory without recomputing scientific outputs or repeating worker-side result validation. Exact runtime declarations remain deferred to their conversation contracts.

### Unit component map

| Unit | Component names (planned except reused foundations) | Intention | Technical tip |
| --- | --- | --- | --- |
| Run entry | `EvaluationTask.run`, `validate_run_prerequisites`, `run_preflight` | Begin evaluation only with prepared scientific inputs and an explicit usable target. | A temporary preflight worker has a separate lifetime from normal target configuration. |
| Parent commitment | `commit_parent_evaluation`, `persist_execution_configuration`, `create_initial_attempts` | Bind one fixed parent and its initial attempt identities durably. | Commitment and attempt observations retain different persistence roles. |
| Local process control | `start_local_worker`, `send_local_control`, `receive_local_control`, `shutdown_local_worker` | Own the root-local worker's process and bounded control boundary. | Bulk artifacts stay in managed storage rather than control messages. |
| Readiness and assignment | `prepare_worker_parent_context`, `assign_attempt`, `replace_execution_state` | Associate ready worker capacity with the parent's attempt work. | Parent context is established before attempt allocation. |
| Root acceptance | `accept_local_bundle`, `publish_root_bundle`, `acknowledge_result_publication` | Make a durably published root model bundle the accepted attempt result. | Publication supplies completion evidence even across an interrupted acknowledgement. |

## 14. Bounded operations, delivery waits, and result retention

### Intention

Bound operational retries while keeping scientific completion, result acceptance, worker availability, and cleanup independently observable. A completed bundle remains reusable throughout its delivery operation. Retry exhaustion terminates the relevant operation or attempt without changing the already computed scientific result.

### Technical route

Operation-owned budgets and observations surround persistence or delivery of the same completed artifact. Terminal attempt failures remain terminal; any user-configured later model retry has a new attempt identity. Worker retention follows confirmed acceptance, applicable task completion, or explicit retention choices, with a separate parent-transition option. Pending delivery retains both its bundle and attempt reservation. Connection rounds introduced later have independent identities and do not reset these budgets.

### Unit component map

| Unit | Component names (planned) | Intention | Technical tip |
| --- | --- | --- | --- |
| Bounded operation | `run_bounded_operation` | Apply an operation-owned attempt budget and retain final failure evidence. | — |
| Result synchronization | `synchronize_completed_result`, `retry_result_synchronization` | Reuse one completed candidate until publication or terminal synchronization failure. | Artifact reuse separates delivery recovery from scientific recomputation. |
| New attempts | `create_retry_attempt` | Represent an explicitly permitted model retry with a new execution identity. | — |
| Retention | `apply_bundle_retention`, `cleanup_previous_parent_results` | Apply bundle-level retention and parent-transition cleanup. | Acceptance, terminal delivery, and deletion remain separate observations. |

## 15. Cooperative interruption, local reconciliation, and reuse

### Intention

Support parent interruption, prospective reconfiguration, and continuation of an existing execution without losing its scientific identity. Unchanged evaluation resumes the same parent, including after worker changes. Changed evaluation requires fresh preparation for a new parent, while late outcomes remain attached to their original execution.

### Technical route

Cooperative stop observations separate root configuration unlocking from actual worker inactivity. Recovery reconstructs fixed execution context and validates final root bundles before reconciling informational attempt rows. Terminal attempts remain historical, and unfinished model units receive new attempts under the configured policies. Root-process termination recreates the local worker rather than reconnecting it. Exact recovery declarations and coordination between copied configurations remain decisions for this phase, preserving publication authority across crash and acknowledgement windows.

### Unit component map

| Unit | Component names (planned) | Intention | Technical tip |
| --- | --- | --- | --- |
| Interruption | `EvaluationTask.interrupt`, `request_parent_stop`, `observe_attempt_stop` | Record parent interruption and cooperative worker stopping. | Permission for prospective edits does not establish inactivity. |
| Existing context | `load_fixed_parent_context`, `inspect_published_root_bundles` | Reconstruct execution inputs and authoritative model completion evidence. | A state claim and a validated final bundle provide different evidence. |
| Reconciliation | `reconcile_execution_state`, `resume_existing_parent` | Continue unfinished units while preserving accepted results and parent identity. | Publication-before-acknowledgement is reconciled from the final bundle. |
| Runtime reuse | `attach_existing_execution`, `recreate_local_worker` | Reestablish owned runtime resources for a retained execution. | Configuration copying and runtime ownership are distinct operations. |

## 16. Private online-worker connection and control

### Intention

Extend the stable local lifecycle to authenticated private online workers without changing root coordination authority. Connection direction, configured membership, runtime usability, parent readiness, and occupied slots remain independently observable. A disconnected worker stays configured while independent connection management continues.

### Technical route

One versioned TLS-protected bidirectional protocol carries control and later artifact traffic through asynchronous I/O. The selected direction policy governs both initial establishment and reconnection, with exact release compatibility checked before normal operation. Root-initiated reconciliation restores eligibility from observed ownership, context, and capacity. Per-worker periodic connection rounds have independent budgets and failure scopes. Framing, authentication, heartbeat detection, timer mechanics, and control acknowledgements remain explicit protocol decisions for this phase.

### Unit component map

| Unit | Component names (planned) | Intention | Technical tip |
| --- | --- | --- | --- |
| Connection establishment | `connect_private_worker`, `accept_private_worker_connection`, `authenticate_worker_connection` | Establish one authenticated compatible connection under the chosen direction policy. | Direction changes connection establishment rather than coordination ownership. |
| Connection rounds | `run_worker_connection_round`, `schedule_worker_connection_rounds` | Keep per-worker retries independent across periodic connection opportunities. | Distinct rounds do not replenish result-delivery budgets. |
| Control protocol | `encode_control_message`, `decode_control_message`, `exchange_worker_status` | Carry bounded task and lifecycle communication over the established connection. | Message framing and lifecycle evidence have separate responsibilities. |
| Health and reconnection | `monitor_worker_heartbeat`, `reconcile_connected_worker` | Observe lost contact and restore readiness after a usable connection returns. | Unknown process state after lost contact is not proof of process failure. |

## 17. Online synchronization, execution, and delivery

### Intention

Complete private online evaluation by preparing worker-local parent context, synchronizing fixed model inputs, and delivering immutable attempt results. Online transport retains the scientific and acceptance semantics established by the local path. Reusable data readiness remains separate from parent-specific partition readiness.

### Technical route

Synchronization consumes the fixed prepared representations and establishes worker-local references before assignment. Completed result bundles are packaged once and reused through bounded delivery. Package length, digest, and private authentication metadata precede safe extraction and schema validation outside the active network loop. Atomic root publication remains acceptance. Archive selection, tiling, compression settings, canonical authentication bytes, and acknowledgements remain phase-owned protocol decisions; public authentication remains tied to the supplied service interface.

### Unit component map

| Unit | Component names (planned except reused foundations) | Intention | Technical tip |
| --- | --- | --- | --- |
| Dataset readiness | `synchronize_dataset`, `prepare_online_parent_context`, `build_worker_data_reference` | Establish exact data and saved partition context on the worker. | Retained dataset readiness does not establish a new parent's context. |
| Model readiness | `synchronize_model`, `validate_synchronized_model_artifacts` | Make the unit's fixed reconstruction artifacts locally available. | Artifact readiness is separate from model semantic reconstruction. |
| Package production | `package_completed_bundle`, `authenticate_result_package` | Produce one reusable result package with integrity and private-origin evidence. | Authentication metadata describes the same immutable package on every retry. |
| Receipt and validation | `receive_result_package`, `verify_result_package`, `extract_result_package_safely`, `validate_attempt_bundle` | Establish a complete valid online candidate without blocking active communication. | Receipt and extraction remain transient stages rather than acceptance. |
| Online acceptance | `accept_online_bundle`, `publish_root_bundle`, `acknowledge_result_publication` | Reuse the root's durable publication boundary for online attempts. | Both topologies converge on the same model-level completion evidence. |

## 18. Mixed scheduling and duplicate-attempt arbitration

### Intention

Coordinate explicit local and online targets through one model-unit lifecycle. Spare eligible capacity may reduce completion latency through configured duplicate attempts, while only one bundle can become authoritative for a model. Connection-loss replacements and reconnected original attempts retain their shared parent identity.

### Technical route

Root scheduling combines current eligibility, parent readiness, effective capacity, and reservations. Online response speed is an allocation-time observation rather than permanent priority. Late duplicate eligibility follows the agreed idle-capacity conditions and oldest unfinished running unit; completed results awaiting delivery are a different situation. Per-model publication serialization arbitrates all candidates, and reconnection reconciles existing attempt identity against final bundle evidence. Scientific failures in a complete valid bundle do not remove its publication eligibility.

### Unit component map

| Unit | Component names (planned) | Intention | Technical tip |
| --- | --- | --- | --- |
| Allocation | `select_eligible_worker`, `allocate_ready_attempts` | Match ready work to currently available reserved capacity. | Availability, readiness, and free slots are separate eligibility inputs. |
| Late duplicates | `select_duplicate_candidate`, `create_duplicate_attempt` | Fill otherwise unused eligible capacity under the agreed duplicate conditions. | Execution age differs from time spent awaiting result delivery. |
| Publication competition | `serialize_model_publication`, `abort_competing_attempts` | Preserve first-successful-publication acceptance across concurrent candidates. | Arbitration follows durable publication rather than a preselected attempt. |
| Lost connections | `record_connection_loss`, `create_replacement_attempts`, `reconcile_reconnected_attempt` | Preserve disconnected identities while replacing unfinished work where permitted. | A returning original competes only while its model lacks an accepted bundle. |

## 19. Explicit worker data management and cleanup

### Intention

Provide explicit retained-dataset and user-data cleanup without conflating configuration removal, connection loss, scientific failure, or resource release. User-owned sources remain outside worker cleanup ownership. Cleanup eligibility reflects actual targeted worker activity rather than the root's interruption status.

### Technical route

Selection and activity evidence establish all-target admission before dataset deletion begins. Cleanup operates only on managed worker copies, verifies operational results, and invalidates readiness after successful deletion. A rejected active-target request has no partial or queued execution. Private runtime lifecycle remains user-controlled; stronger public release behavior is integrated through the next phase's supplied interface. Selection declarations, deletion mechanisms, and verification details remain phase-owned decisions.

### Unit component map

| Unit | Component names (planned) | Intention | Technical tip |
| --- | --- | --- | --- |
| Cleanup selection | `select_cleanup_targets`, `validate_cleanup_inactivity` | Establish the complete selected set and its admission evidence. | Admission failure differs from an operational failure after admitted deletion. |
| Dataset cleanup | `cleanup_dataset`, `verify_dataset_cleanup`, `invalidate_dataset_readiness` | Remove selected retained dataset copies and reflect the loss of readiness. | Successful deletion and readiness invalidation refer to the same worker copy. |
| User-data cleanup | `cleanup`, `verify_user_data_cleanup` | Remove retained managed user artifacts within explicit ownership boundaries. | Source references do not transfer ownership of original data. |
| Retained-data inspection | `inspect_retained_worker_data` | Expose cleanup-relevant managed data and activity observations. | — |

## 20. External public-worker service integration

### Intention

Integrate externally allocated public workers without reproducing the service's pool ownership or availability policy. Public execution uses controlled declarative models and non-executable model data. Confirmed cleanup prevents one user's work or artifacts from crossing into another allocation.

### Technical route

The separately supplied service interface provides allocation, connection information, authentication, and lifecycle notifications. Cocoonet's existing direct root-worker protocol carries evaluation traffic. Public admission resolves supported local components, while release coordinates stopping, verified managed-data cleanup, and service notification. Public parent transitions have mandatory earlier-result cleanup after applicable delivery settlement; allocation release has the stronger user-isolation boundary. Service endpoints, credentials, pool timers, and operator provisioning remain external dependencies rather than inferred Cocoonet contracts.

### Unit component map

| Unit | Component names (planned; service adapter names remain provisional) | Intention | Technical tip |
| --- | --- | --- | --- |
| Allocation integration | `request_public_worker`, `connect_allocated_public_worker` | Consume a service allocation and establish its root-owned connection. | Allocation ownership and task-slot availability remain separate. |
| Public model admission | `validate_public_model_specification`, `resolve_public_model_components` | Admit supported declarative reconstruction inputs and non-executable data. | Public restrictions retain their scope without narrowing trusted private/local models. |
| Parent-transition cleanup | `cleanup_public_previous_parent_results` | Apply mandatory earlier-parent result cleanup at public parent transitions. | — |
| Allocation release | `release`, `stop_public_user_execution`, `verify_public_release_cleanup`, `notify_public_worker_release` | Complete verified user isolation before external resource release. | Cleanup confirmation and the service's pool state have different owners. |
| Failed-cleanup containment | `mark_public_worker_unavailable` | Keep an unconfirmed worker unavailable to a new user under the supplied lifecycle. | — |

## 21. Complete compatibility and lifecycle verification

### Intention

Demonstrate the complete supported workflow across model families, declared Python versions, and local-only, online-only, and mixed execution. This phase closes integration coverage while preserving the unit-level verification already associated with each capability. Public examples describe actual implemented behavior.

### Technical route

Integration scenarios combine real scientific execution with persistence round trips, resource limits, interruption windows, delivery exhaustion, reconnection, duplicate arbitration, reuse, and cleanup. Cross-runtime coverage follows declared release support and exact Cocoonet version compatibility. Final bundle evidence and unchanged scientific associations provide the common audit trail. Documentation examples follow the completed public surface; any source fixes remain bounded development units rather than a redesign of established scientific or lifecycle semantics.

### Unit component map

| Unit | Component names (planned verification functions) | Intention | Technical tip |
| --- | --- | --- | --- |
| Runtime compatibility | `test_supported_runtime_compatibility`, `test_cross_runtime_artifact_roundtrip` | Exercise declared runtime combinations and persisted artifact interoperability. | Plain metadata compatibility and executable serialization interoperability need different evidence. |
| Topology workflows | `test_local_evaluation_workflow`, `test_online_evaluation_workflow`, `test_mixed_evaluation_workflow` | Demonstrate complete evaluations under every supported topology. | The same scientific inputs allow topology-independent association checks. |
| Lifecycle recovery | `test_publication_acknowledgement_recovery`, `test_interruption_and_resume`, `test_reconnection_and_duplicate_arbitration` | Exercise identity and acceptance across interruption and concurrent attempts. | Final bundles remain the common authority across crash windows. |
| Capacity and retention | `test_attempt_capacity_and_responsiveness`, `test_delivery_exhaustion_and_retention`, `test_dataset_reuse_and_cleanup` | Verify operational limits without altering scientific outcomes. | Capacity, acceptance, and cleanup observations remain independent. |
| Public service lifecycle | `test_public_worker_release_isolation` | Verify release and reuse through the supplied service integration. | — |
| Public examples | `test_public_workflow_examples` | Keep documented public workflows aligned with implemented behavior. | — |
