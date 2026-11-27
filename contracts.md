# Cocoonet public APIs and persistence layouts

FOR AGENTS: DO NOT MODIFY THIS FILE WITHOUT EXPLICIT USER APPROVAL. DO NOT AUTOMATICALLY APPROVE THE MODIFICATION OF THIS FILE.

## 1 Public APIs

### 1.1 Declaration set

```python
@dataclass(frozen=True, kw_only=True, slots=True)
class ModelReference:
    model_id: ModelId
    model_label: ScalarLabel
    family: ModelFamily
    task_kind: TaskKind


@dataclass(frozen=True, kw_only=True, slots=True)
class WorkerConfiguration:
    worker_id: WorkerId
    worker_label: ScalarLabel
    kind: WorkerKind
    device: str = "cpu"
    cpu_capacity: int | None = None
    n_attempt: int = 1
    n_cpu_per_attempt: int = 1
    memory_reserve_pct: float = 20.0
```

### 1.2 Root node evaluation task

```python
class EvaluationTask:
    def __init__(
        self,
        work_dir: PathInput,
        *,
        overwrite: bool = False,
    ) -> None: ...

    @property
    def work_dir(self) -> Path: ...

    @property
    def state(self) -> ConsolidationState: ...

    @property
    def parent_id(self) -> ParentId | None: ...

    def snapshot(self) -> ModelingSnapshot: ...

    def initialize(
        self,
        work_dir: Path | str,
        *,
        overwrite: bool = False,
    ) -> None: ...

    def add_model(
        self,
        model_input: object,
        model_label: str,
        family: Literal["sklearn", "sklearn-like", "pytorch", "cocoonet"],
        task_kind: Literal["regression", "classification"],
    ) -> ModelReference: ...

    def add_models(
        self,
        models: dict[str, object],
        family: Literal["sklearn", "sklearn-like", "pytorch", "cocoonet"] | dict[str, Literal["sklearn", "sklearn-like", "pytorch", "cocoonet"]],
        task_kind: Literal["regression", "classification"],
        *,
        suppress_return: bool = False,
        tabular_return: bool = True,
        limit: int = 1000,
    ) -> tuple[ModelReference, ...] | pd.DataFrame | None: ...

    def list_models(
        self,
        label: str | None = None,
        family: str | None = None,
        model_input: str | None = None,
        *,
        use_regex: bool = True,
        suppress_return: bool = False,
        tabular_return: bool = True,
        limit: int = 1000,
    ) -> tuple[ModelReference, ...] | pd.DataFrame | None: ...

    def remove_model(
        self,
        model_label: str,
    ) -> ModelReference | None: ...

    def remove_models(
        self,
        label: str | None = None,
        family: str | None = None,
        model_input: str | None = None,
        *,
        use_regex: bool = True,
        suppress_return: bool = False,
        tabular_return: bool = True,
        limit: int = 1000,
    ) -> tuple[ModelReference, ...] | pd.DataFrame | None: ...

    def add_worker(
        self,
        worker_id: str | None = None,
        kind: Literal['root-local', 'online'] = 'root-local',
        *,
        n_attempt: int | None = None,
        n_cpu_per_attempt: int = 1,
        memory_reserve_pct: float = 20.0,
        ) -> WorkerConfiguration: ...

    def add_workers(
        self,
        worker_id: str | Sequence[str] | None = None,
        worker_label: str | None = None,
        cpu_capacity: int | str | None = None,
        kind: str | None = None,
        device: str | None = None,
        *,
        n_attempt: int | dict[str, int] | None = None,
        n_cpu_per_attempt: int | dict[str, int] = 1,
        memory_reserve_pct: float | dict[str, float] = 20.0,
        use_regex: bool = True,
        suppress_return: bool = False,
        tabular_return: bool = True,
        limit: int = 1000,
        ) -> tuple[WorkerConfiguration, ...] | pd.DataFrame | None: ...

    def list_available_workers(
        self,
        worker_id: str | None = None,
        worker_label: str | None = None,
        cpu_capacity: int | str | None = None,
        kind: str | None = None,
        device: str | None = None,
        *,
        use_regex: bool = True,
        suppress_return: bool = False,
        tabular_return: bool = True,
        limit: int = 1000,
    ) -> tuple[WorkerConfiguration, ...] | pd.DataFrame | None: ...

    def list_workers(
        self,
        worker_id: str | None = None,
        worker_label: str | None = None,
        cpu_capacity: int | str | None = None,
        kind: str | None = None,
        device: str | None = None,
        *,
        use_regex: bool = True,
        suppress_return: bool = False,
        tabular_return: bool = True,
        limit: int = 1000,
    ) -> tuple[WorkerConfiguration, ...] | pd.DataFrame | None: ...

    def remove_worker(
        self,
        worker_id: str | None = None,
    ) -> WorkerConfiguration | None: ...

    def remove_workers(
        self,
        worker_id: str | None = None,
        worker_label: str | None = None,
        cpu_capacity: int | str | None = None,
        kind: str | None = None,
        device: str | None = None,
        *,
        use_regex: bool = True,
        suppress_return: bool = False,
        tabular_return: bool = True,
        limit: int = 1000,
    ) -> tuple[WorkerConfiguration, ...] | pd.DataFrame | None: ...

    def __copy__(self) -> EvaluationTask: ...

    def clone_configuration(
        self,
        new_work_dir: PathInput,
    ) -> EvaluationTask: ...

    def clone(
        self,
        new_work_dir: PathInput,
    ) -> EvaluationTask: ...
```

Public model-table return layout:

| Column order | Column | dtype |
| --- | --- | --- |
| 1 | `model_id` | pandas string |
| 2 | `model_label` | pandas string |
| 3 | `family` | pandas string |
| 4 | `task_kind` | pandas string |
| 5 | `model_input` | pandas string |

Public worker-table return layout:

| Column order | Column | dtype |
| --- | --- | --- |
| 1 | `worker_id` | pandas string |
| 2 | `worker_label` | pandas string |
| 3 | `kind` | pandas string |
| 4 | `device` | pandas string |
| 5 | `cpu_capacity` | pandas string |
| 6 | `n_attempt` | pandas string |
| 7 | `n_cpu_per_attempt` | pandas string |
| 8 | `memory_reserve_pct` | pandas string |

### 1.3 Public API declarations still deferred

| Public capability | Declaration status |
| --- | --- |
| Dataset registration, split preparation, inspection, removal, and bulk forms | Exact signatures deferred. |
| Evaluation settings and PyTorch reconstruction/training inputs | Exact signatures deferred. |
| Execution, optional preflight, interruption, and recovery | Exact signatures deferred. |
| Worker service startup, connection configuration, runtime inspection, cleanup, and release | Exact signatures deferred. |
| Delivery, retry, retention, compression, and runtime authentication settings | Exact signatures deferred. |
| Additional copy modes and independent-directory copying | Declarations unresolved in the current phase notes. |


## 2 Managed persistence layouts

### 2.1 Root working directory

```text
<work_dir>/
  cocoonet_metadata.csv
  root_data_reference.xlsx
  partition_schedules.xlsx
  execution_configuration.csv
  execution_state.csv
  models_for_transfer/
    model_<model_id>.<selected_model_extension>
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

### 2.2 Worker working directory

```text
<worker_work_dir>/
  datasets/
    dataset_<dataset_id>/
  models/
    parent_<parent_id>/
      model_<model_id>.<selected_model_extension>
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

Worker default location: `<worker_service_console_work_dir>/cocoonet/`.

### 2.3 Dataset transfer preparation beside source data

```text
<source_dataset_root_dir>/
  <original_source_files>
  dataset_for_transfer/
    <source_stem>_compressed.parquet
    cocoonet_dataset_shard_<zero-padded-one-based-number>.parquet
    dataset_<dataset_id>.<archive_extension>
    dataset_<dataset_id>_<tile_id>.<archive_extension>
```

### 2.4 Path and format variants

| Entry | Layout variants |
| --- | --- |
| Dataset directory identifier | `dataset_` followed by 12 lowercase hexadecimal characters |
| Worker parent directory identifier | `parent_` followed by 16 lowercase hexadecimal characters |
| Model and archive tile identifiers | Positive decimal text, beginning at `1`, without zero padding |
| Trusted PyTorch model extension | `.pt` |
| Trusted standard scikit-learn model extension | `.joblib` |
| Trusted scikit-learn-like model extension | One selected extension: `.joblib`, `.pkl`, or `.dill` |
| Dataset/model archive | Untiled archive or numbered tiled archives; archive extension deferred |
| Public model specification and supplied-weight files | Declarative specification and Parquet model-data artifacts; exact names/layout deferred |
| Result package for online delivery | `.tar.zst`; exact temporary package pathname deferred |
| Conditional bundle members | Task-specific performance workbook; optional influence, visualization data, rendered root plots, and error logs |
| File staging entry | `<filename>.partial`, beside its final file |
| Bundle staging entry | `model_<model_id>.staging`, beside `model_<model_id>/` |
| Transfer preparation reuse-state artifact | Exact filename and schema deferred |

### 2.5 Persisted file structures

#### 2.5.1 Metadata, configuration, execution CSV tables

| File | Physical columns |
| --- | --- |
| `cocoonet_metadata.csv` | `field,value` |
| `execution_configuration.csv` | `field,value`; exact field inventory deferred |
| `root_model_<model_id>_reference.csv` | `field,value`; includes `model_id`, `model_label`, `model_path` |
| `worker_model_<model_id>_reference.csv` | `field,value`; includes `model_id`, `model_label`, `model_path` |
| `execution_state.csv` | `parent_id,worker_id,model_label,model_id,attempt_id,attempt_status` |

CSV representation: UTF-8 without BOM, comma delimiter, CSV quoting, LF line endings;
undefined fields are empty. Metadata collections occupy hierarchical `field` keys
with explicit counts and zero-based indices. Model paths are owner-relative logical paths.

`cocoonet_metadata.csv` row layout:

```csv
field,value
storage_schema_version,1
node_role,<root|worker>
cocoonet_version,<complete version>
python_major,<nonnegative integer>
python_minor,<nonnegative integer>
python_micro,<nonnegative integer>
python_release_level,<alpha|beta|candidate|final>
```

`execution_state.csv` status values:

```text
not_allocated
allocation_failed
model_synchronizing
model_synchronized
model_synchronizing_failed
running
attempt_finished_without_error
attempt_finished_with_error
attempt_failed
result_synchronizing
result_synchronized
result_synchronization_failed
interrupted
connection_loss
aborted
```

#### 2.5.2 Dataset and partition workbooks

| File | Data layout |
| --- | --- |
| `root_data_reference.xlsx` | Sample IDs, labels, groups, train/test masks, targets, source associations, and applicable tensor metadata; exact sheet/column schema deferred. |
| `worker_data_reference.xlsx` | Corresponding sample metadata with worker-local synchronized paths; exact sheet/column schema deferred. |
| `partition_schedules.xlsx` | Ordered columns: `partition_id`, `dataset_id`, `fold_id`, `population`, `position`, `sample_id`, `sample_label`; exact worksheet name deferred. |

Partition populations: `training` and `testing`; population positions are one-based.
Fold order is the first-occurrence order of fold identities; training rows precede
testing rows within a fold.

#### 2.5.3 Result workbooks

Each result workbook has two worksheets: result data first and metadata second;
exact worksheet names remain deferred. Metadata has exactly `field,value` columns.
Common metadata fields include `creation_datetime`, `parent_id`, `attempt_id`,
`model_id`, `model_label`, and `worker_id`, with applicable dataset, partition, fold,
artifact, schema, integrity, and ordered class-to-column associations.

In the following column layouts, `<class>_*` denotes a column group in the exact
class order recorded in the workbook metadata.

| File / task | First-worksheet column layout |
| --- | --- |
| `validation_result.xlsx` / classification | `sample_id`, `sample_label`, `y_true`, `y_pred`, `<class>_probability` |
| `validation_result.xlsx` / regression | `sample_id`, `sample_label`, `fold_id`, `y_true`, `y_pred` |
| `classification_performance.xlsx` | `class`, `TP`, `TN`, `FP`, `FN`, `precision`, `recall`, `f1_score`, `accuracy`, `AUC` |
| `regression_performance.xlsx` | `mean_error`, `std_of_error`, `MAE`, `normalized_MAE`, `MSE`, `RMSE`, `normalized_RMSE`, `CV_RMSE`, `RPD`, `R2` |
| `residual_analysis.xlsx` / classification | `sample_id`, `sample_label`, `y_true`, `<class>_probability`, `<class>_residual`, `<class>_z_score`, `sum_of_absolute_residual`, `root_sum_of_squared_residual`, `Shannon_entropy_of_absolute_residual`, `Mahalanobis_distance_of_residual` |
| `residual_analysis.xlsx` / regression | `sample_id`, `sample_label`, `y_true`, `y_pred`, `residual`, `standardized_residual` |
| `sample_influence.xlsx` / classification | `sample_id`, `sample_label`, `prediction_comparison_count`, `<class>_influence`, `sum_of_influence`, `root_sum_of_squared_influence` |
| `sample_influence.xlsx` / regression | `sample_id`, `sample_label`, `influence`, `prediction_comparison_count` |
| `bundle_manifest.xlsx` | Complete bundle inventory; exact data schema deferred. |
| `attempt_metadata.xlsx` | Attempt and scientific outcome metadata; exact data schema deferred. |
| `fold_sources.xlsx` | Fold/sample associations and separate training/testing source references; exact data schema deferred. |
| `failures.xlsx` | Failure provenance and applicable value-level failure associations; exact data schema deferred. |

Classification performance row order: class rows in metadata order, then
`micro-average`, then `macro-average`. Aggregate confusion-count cells contain `NA`.
Regression performance contains one data row.

| Scientific cell representation | Stored XLSX value |
| --- | --- |
| Finite measure | Numeric cell |
| Permitted positive infinity | Text `Inf` |
| Permitted negative infinity | Text `-Inf` |
| Mathematically undefined | Text `NaN` |
| Computationally unavailable, with failure associations | Text `NA` |
| Schema-defined non-applicable value | Text `NA` |

#### 2.5.4 Logs and visualization files

| File family | Stored representation |
| --- | --- |
| Running and error `.log` files | UTF-8 JSON Lines; one structured event per line. |
| `roc_curve.dill` | Standardized ROC/AUC rendering data; exact payload schema deferred. |
| `scatter_plot.dill` | Standardized true/predicted-target rendering data; exact payload schema deferred. |
| `residual_plot.dill` | Standardized prediction/residual rendering data; exact payload schema deferred. |
| `rendered_plots/` | Requested root-rendered files; exact image names/formats deferred. |


## 3 Computational specifications

### 3.1 Mathematical formulas

#### 3.1.1 Classification accuracy

Let `N` be the evaluated sample count and `C` the number of classes. For each
class `c`, confusion counts are one-versus-rest counts:

```text
accuracy_c = (TP_c + TN_c) / (TP_c + TN_c + FP_c + FN_c)
micro_accuracy = number_of_correct_predicted_class_labels / N
macro_accuracy = (1/C) * sum_c accuracy_c
```

These are three distinct accuracy scopes.
Undefined results remain `NaN`; macro aggregation does not skip undefined class values.

#### 3.1.2 Classification probability residuals and z-scores

For evaluated sample `i` and class `c`, let `q_{i,c}` be its predicted probability
and `t_{i,c}` its one-hot true-class indicator. Class order is fixed and aligned.

```text
r_{i,c} = t_{i,c} - q_{i,c}
mu_c = (1/N) * sum_{i=1}^N r_{i,c}
sigma_c = sqrt((1/N) * sum_{i=1}^N (r_{i,c} - mu_c)^2)
z_{i,c} = (r_{i,c} - mu_c) / sigma_c
```

Z-scores are computed across evaluated samples separately for each class, with population standard deviation (`ddof=0`).
A zero standard deviation produces `NaN`.
Insufficient valid values or a resulting zero standard deviation produce `NaN`.

#### 3.1.3 Sample-level classification residual summaries

These summaries operate across classes for each sample:

```text
sum_of_absolute_residual_i = sum_{c=1}^C abs(r_{i,c})
root_sum_of_squared_residual_i = sqrt(sum_{c=1}^C r_{i,c}^2)
```

Both propagate `NaN`.

#### 3.1.4 Normalized Shannon entropy of absolute residuals

```text
p_{i,c} = abs(r_{i,c}) / sum_{k=1}^C abs(r_{i,k})
H_norm_i = -(1 / log_b(C)) * sum_{c=1}^C p_{i,c} * log_b(p_{i,c})
```

The zero term has the convention `0 * log_b(0) = 0`.
Consistent logarithm bases produce the same normalized result.
The project-specific all-zero-residual case is explicitly `H_norm_i = 1`.
Any required `NaN` residual produces `NaN` entropy.

#### 3.1.5 Mahalanobis distance of signed probability residuals

Let `r_i` be sample `i`'s signed residual vector over all classes, and let `N`
be the complete evaluated population in the residual report.

```text
r_bar = (1/N) * sum_{j=1}^N r_j
S = (1/(N-1)) * sum_{j=1}^N (r_j - r_bar) * (r_j - r_bar)^T
d_M(r_i) = sqrt((r_i - r_bar)^T * S^+ * (r_i - r_bar))
```

`S` is sample covariance (`ddof=1`), and `S^+` is the Moore–Penrose pseudoinverse.
Singular covariance alone is not a failure.
When `N < 2`, all distances are `NaN`.
A sample with any `NaN` residual receives `NaN` distance.
Fewer than two remaining complete vectors produce `NaN` for every distance.
Covariance or pseudoinverse computation failure makes all distances computationally unavailable, with failure provenance.

#### 3.1.6 Optional generalized Cook-like sample influence

Let `T_f` and `V_f` be the saved training and held-out populations of fold `f`.
For removed training sample `i`, `F_i` contains exactly the folds in which
`i` belongs to `T_f`. Baseline predictions are the stored successful
primary-validation predictions; removal predictions come from a fresh model
fitted on `T_f` without `i`, evaluated on the unchanged `V_f`.

`p` is the actual number of predictors in the input supplied for fitting,
not a model-parameter count. For classification, let `q_{f,j,c}` and
`q_{f,-i,j,c}` be baseline and removal probabilities for held-out sample `j`
and class `c`, with one-hot truth `t_{j,c}`:

```text
MSE_{f,c} = sum_{j in V_f} (t_{j,c} - q_{f,j,c})^2 / |V_f|
D_{f,i,c} = sum_{j in V_f} (q_{f,j,c} - q_{f,-i,j,c})^2 / (p * MSE_{f,c})
```

For single-target regression, with true held-out target `y_j`, baseline prediction
`y_hat_{f,j}`, and removal prediction `y_hat_{f,-i,j}`:

```text
MSE_f = sum_{j in V_f} (y_j - y_hat_{f,j})^2 / |V_f|
D_{f,i} = sum_{j in V_f} (y_hat_{f,j} - y_hat_{f,-i,j})^2 / (p * MSE_f)
```

The baseline MSE is against truth and remains fixed for the fold. Aggregation is:

```text
M_i = sum_{f in F_i} |V_f|
D_bar_{i,c} = sum_{f in F_i} D_{f,i,c} / M_i
D_bar_i = sum_{f in F_i} D_{f,i} / M_i
```

`M_i` counts every held-out prediction comparison, including zero changes and repeated comparisons. It is not the fold count or training size minus one; fold-specific MSE is not replaced by pooled MSE. A sample never used in training has no influence metric, rather than an observed zero. A held-out sample has no removal evaluation in that fold.

### 3.2 External dependency methods

| Measure or intermediate | Specified dependency method/expression |
| --- | --- |
| Signed residual matrix | NumPy array subtraction: `R = y_true_proba - y_pred_proba` |
| Per-class residual mean | `numpy.mean(R[:, c])` |
| Per-class population residual standard deviation | `numpy.std(R[:, c], ddof=0)` |
| Per-class residual z-score | `(R[:, c] - numpy.mean(R[:, c])) / numpy.std(R[:, c], ddof=0)` |
| Sum of absolute residuals | `numpy.sum(numpy.abs(R), axis=1)` |
| Root sum of squared residuals | `numpy.sqrt(numpy.sum(R**2, axis=1))` |
| Normalized entropy | `scipy.stats.entropy(numpy.abs(r_i), base=2) / numpy.log2(C)` |
| All-zero entropy override | `1.0` when `numpy.all(r_i == 0)` |
| Mean vector for Mahalanobis distance | `numpy.mean(R, axis=0)` |
| Sample residual covariance | `numpy.cov(R, rowvar=False, ddof=1)` |
| Moore–Penrose pseudoinverse | `numpy.linalg.pinv(covariance)` |
| Centered Mahalanobis distance | `scipy.spatial.distance.mahalanobis(r_i, mean_residual, inverse_covariance)` |

Here `R` has shape `(samples, classes)` and contains signed probability residuals; `r_i` is one row. These expressions describe its default propagation path.
