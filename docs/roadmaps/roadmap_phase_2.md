# Phase 2: Filesystem ownership and durable persistence foundations

Status: revised proposal; Phase 2 implementation is not authorized.

## 1. Authority and purpose

This roadmap plans only Phase 2 of [the project roadmap](../roadmap.md). It applies the complete root
[AGENTS.md](../../AGENTS.md), [architecture](../architecture.md), the implemented
[Phase 1 roadmap](roadmap_phase_1.md), and the current Phase 1 source and tests.

Phase 2 establishes the small private storage boundary needed by later phases:

- reject unsafe managed roots and root/dataset-source ownership conflicts before any persistent mutation;
- prepare or reopen visible root storage without touching user-owned dataset sources;
- construct the fixed semantic root and worker paths;
- persist compatibility metadata and lifecycle evidence in schema-controlled UTF-8 CSV;
- publish one file or one immutable result-bundle directory atomically; and
- append structured running and error logs.

It does not implement public computation APIs. Every declaration in this plan remains private under
`cocoonet._storage`.

Approval of this roadmap does not authorize source or test implementation. After the blocking governing-document
ambiguity is resolved and this roadmap is approved, each implementation unit still requires the separate review,
approval, implementation, verification, and stop cycle required by `AGENTS.md`. A change to any approved signature,
schema, invariant, or failure rule requires a roadmap amendment before implementation.

## 2. Phase boundary

### 2.1 In scope

Phase 2 implements:

1. canonical physical-path resolution;
2. protected-location discovery and managed-root validation;
3. dataset-source/root ownership validation in both configuration directions;
4. fresh root preparation and existing-root opening;
5. exact root and worker layout path builders;
6. the `cocoonet_metadata.csv` schema and codec;
7. a minimal snapshot and CSV codec for durable Phase 1 execution evidence;
8. one focused staged-file publication helper;
9. one focused staged-bundle publication helper; and
10. structured root running/error log records and append operations.

### 2.2 Out of scope

Phase 2 does not implement or decide:

- `Computation` construction, `initialize()`, `add_dataset()`, `run()`, copying, configuration mutation, or recovery
  entry points;
- dataset parsing, standardization, splitting, ID generation, source-state reuse, or transfer preparation;
- `root_data_reference.xlsx`, `worker_data_reference.xlsx`, `partition_schedules.xlsx`, or result-workbook schemas and
  writers;
- the XLSX engine dependency, which is selected by the first workbook-producing phase;
- `execution_configuration.csv` fields, which Phase 3 owns;
- model persistence, model-reference schemas, model transfer, evaluation, reports, influence, or rendering;
- result-bundle members, manifests, workbook metadata, package creation, `.tar.zst` settings, HMAC, or safe extraction;
- scheduling, attempt allocation, reception selection, result acceptance, retry transitions, recovery transitions,
  retention, cleanup, acknowledgements, IPC, or networking;
- worker-work-directory creation or clearing;
- a generic repository, transaction, serializer, artifact registry, content-addressed store, or filesystem abstraction;
- dill as a general persistence format; or
- a general root-file clearing API.

Later phases call the Phase 2 mechanisms but retain ownership of their domain schemas and lifecycle transitions.

## 3. Entry and completion criteria

### 3.1 Entry criteria

Implementation may begin only when:

- this revised roadmap is approved;
- Phase 1 contract tests pass unchanged; and
- the implementation unit under consideration receives its separate approval.

Phase 2 adds no dependency. It uses the standard library plus the existing `aiofiles` dependency. Its persistence
implementation must not import pandas, NumPy, torch, or dill.

### 3.2 Completion criteria

Phase 2 is complete only when:

- every destructive root operation first validates the canonical physical target and all configured dataset sources;
- missing, empty, nonempty, overwrite, non-directory, symlink/alias, and protected-root cases behave as specified;
- all constructed paths exactly match Architecture Section 6.2;
- no hidden `.cocoonet` store, hashed artifact filename, or opaque path identity is introduced;
- compatibility metadata and execution state round-trip through strict deterministic UTF-8 CSV;
- the execution-state snapshot preserves fixed parent context, known model units, attempts, worker observations,
  selected reception, acceptance, and bounded-operation retry evidence without performing transitions;
- immutable files and bundles become authoritative only through their designated atomic publication operations;
- mutable `execution_state.csv` uses staged atomic replacement;
- stale `.partial` files and `.staging` directories are never loaded or promoted automatically;
- structured logs preserve required context and tolerate an incomplete final line after a crash;
- all new tests use real temporary files and real Phase 1 records, with no mocks, monkeypatches, stubs, or fakes; and
- the full configured test and documentation checks pass.

## 4. Governing contracts and invariants

### 4.1 Canonical physical paths

`resolve_physical_path()` accepts a path-like value and returns an absolute, normalized, symlink-resolved `Path`.
Resolution must also work for a missing final path: resolve the nearest existing ancestor physically, then append the
missing lexical suffix without creating it.

Managed-root specifications may be absolute or relative. Resolve an absolute root directly. Resolve a relative root to
a location inside `<current console working directory>/cocoonet/<parent_id>/` and reject traversal outside that
parent-specific directory. An otherwise safe absolute path remains valid.

Containment comparisons use platform-normalized canonical components, not string prefixes. Equality is handled
explicitly. On case-insensitive filesystems, aliases that differ only by case compare as the same physical path.

Validation must not create, clear, rename, or remove any path.

### 4.2 Dataset-source ownership

For a candidate managed root `R` and every configured dataset source `S`:

- reject `S == R`;
- reject `S` contained in `R`;
- allow `R` contained in `S` for this ownership rule alone; and
- allow disjoint paths.

The permitted inverse containment is intentional: the contract protects user source data from being inside a directory
that Cocoonet may clear. Protected-location validation remains independent and may still reject `R`.

The same validator is used when adding or changing a source and when initializing, changing, or reassigning a root.
Validate the complete source collection before mutation. Report the conflicting source and canonical root.

### 4.3 Protected managed roots

Protected locations comprise:

- filesystem anchors;
- the running Python installation;
- Python package-installation directories;
- the installed Cocoonet package directory; and
- the complete Cocoonet source repository tree when running from a source checkout.

The candidate root is validated against canonical physical locations before any write. Reject equality with a
filesystem anchor. For every other listed risky location, reject equality, a candidate ancestor, or a candidate
descendant. Report the candidate root and protected location; do not reject a path solely because it is absolute.

### 4.4 Root preparation and opening

Fresh preparation has these outcomes:

| Candidate | `overwrite=False` | `overwrite=True` |
| --- | --- | --- |
| Missing path | create it | create it |
| Existing empty directory | use it | use it |
| Existing nonempty directory | reject without mutation | clear only its contents, then prepare |
| Existing non-directory | reject | reject |

Rules:

- Validate the root and every source before creating or clearing anything.
- Clearing is confined to the already validated root. Do not follow directory symlinks or junctions into their targets.
- Never modify, move, rename, or delete a dataset source.
- Fresh preparation creates only the Phase 2-owned entries: `cocoonet_metadata.csv`,
  `models_for_transfer/`, `attempt_results/`, and `logs/`.
- Later phases create dataset, configuration, state, model, and result artifacts when their owning operation succeeds.
- A preparation failure must not be reported as success. Phase 2 does not promise rollback of an arbitrary operating
  system failure during an explicitly authorized clear.
- Opening an existing root is non-destructive. It requires a directory and a valid final `cocoonet_metadata.csv`.
  A stale `cocoonet_metadata.csv.partial` is ignored when the final file is valid and is never promoted or deleted.
- Fresh preparation and opening/recovery remain distinct operations.

### 4.5 Visible semantic layouts

Path builders return paths only; they do not create directories or validate artifact contents.

Root layout:

~~~text
<root>/
  cocoonet_metadata.csv
  root_data_reference.xlsx
  partition_schedules.xlsx
  execution_configuration.csv
  execution_state.csv
  models_for_transfer/
  attempt_results/model_<model_id>/
  logs/root_running.log
  logs/root_error.log
~~~

Worker layout:

~~~text
<worker>/
  datasets/dataset_<dataset_id>/
  models/parent_<parent_id>/model_<model_id>.dill
  results/parent_<parent_id>/
    cocoonet_metadata.csv
    worker_data_reference.xlsx
    partition_schedules.xlsx
    execution_configuration.csv
    execution_state.csv
    attempt_results/model_<model_id>/
~~~

Identifier path validation is exact:

- `dataset_id` is 12 lowercase hexadecimal characters;
- `parent_id` is 16 lowercase hexadecimal characters;
- `model_id` and `tile_id` use canonical positive decimal text beginning at `1`, with no sign or zero padding;
- separators, `.`, `..`, absolute fragments, prefixes/suffixes, booleans as integers, and invalid lengths are rejected.

Phase 2 validates path representation only. Dataset/parent generation and collision handling remain in their owning
phases. Phase 1's general nonempty-string identity validator remains unchanged; a managed-path boundary applies the
narrower architecture format to the applicable identity. Phase 1 `ArtifactId` remains an opaque identity stored in
state; it is never converted into a filename.

### 4.6 Format ownership

- `cocoonet_metadata.csv` and `execution_state.csv` are schema-controlled UTF-8 CSV.
- Phase 2 supplies the same-directory staging/publication boundary for later authoritative XLSX writers, but does not
  create or parse workbooks.
- Result-bundle tables remain two-sheet XLSX workbooks owned by later scientific phases.
- Dill remains limited to trusted model payloads, trusted root-local IPC, and standardized visualization data.
- Logs append directly and are not staged.
- Persistence representation is independent from later transfer preparation and packaging.

### 4.7 Atomic publication

For an authoritative file `name.ext`, staging is exactly `name.ext.partial` in the same directory.

- The writer creates and closes the staging file.
- Validation occurs before publication.
- Publication flushes and synchronizes the staging file and then uses one atomic same-filesystem rename.
- Immutable publication rejects an existing final path.
- Mutable replacement is permitted only for `execution_state.csv` and uses one atomic replacement.
- A failure leaves the prior final file authoritative. Staging is never evidence of completion and is never promoted
  by a loader.
- Cleanup removes only the exact requested staging path.

For a result directory `model_<model_id>`, staging is the sibling `model_<model_id>.staging`.

- The caller creates and validates a complete staging tree.
- Publication synchronizes every regular file and atomically renames the directory.
- Publication rejects an existing final directory and never merges trees.
- A failure leaves staging non-authoritative.
- Cleanup removes only the exact requested staging directory and never an accepted final bundle.

The bundle helper establishes publication completeness only. It does not know result schemas, validate scientific
content, add root inventory, package data, accept a result, or change lifecycle state.

### 4.8 Deterministic CSV behavior

All Phase 2 CSV:

- is UTF-8 without a byte-order mark;
- uses the standard-library `csv` module, comma delimiter, RFC-compatible quoting, and `\n` line endings;
- includes one exact header row;
- encodes enums by `.value`, booleans as `0` or `1`, integers as canonical decimal, and finite floats with
  `repr()` round-trip semantics;
- never accepts `NaN` or infinity in lifecycle metadata;
- rejects unknown, duplicate, missing, or out-of-order columns and invalid row types; and
- emits deterministic row ordering.

Empty text and absent values must not be conflated. Nullable columns therefore have an explicit companion presence
column where empty text is otherwise valid.

## 5. Planned modules and records

### 5.1 Package surface

~~~text
src/cocoonet/_storage/
  __init__.py
  paths.py
  publication.py
  metadata.py
  execution_state.py
  logging.py
~~~

`_storage.__init__` exports only the private names explicitly needed by later internal phases. Nothing is re-exported
from the public `cocoonet` package in Phase 2.

### 5.2 Path and layout records

In `paths.py`:

~~~python
type PathInput = str | os.PathLike[str]

@dataclass(frozen=True, slots=True)
class RootStorageLayout:
    root: Path
    metadata_csv: Path
    root_data_reference_xlsx: Path
    partition_schedules_xlsx: Path
    execution_configuration_csv: Path
    execution_state_csv: Path
    models_for_transfer: Path
    attempt_results: Path
    logs: Path
    root_running_log: Path
    root_error_log: Path

@dataclass(frozen=True, slots=True)
class WorkerStorageLayout:
    root: Path
    datasets: Path
    models: Path
    results: Path

@dataclass(frozen=True, slots=True)
class WorkerParentStorageLayout:
    parent_root: Path
    metadata_csv: Path
    worker_data_reference_xlsx: Path
    partition_schedules_xlsx: Path
    execution_configuration_csv: Path
    execution_state_csv: Path
    attempt_results: Path

@dataclass(frozen=True, slots=True)
class BundlePublicationPaths:
    staging: Path
    final: Path
~~~

These records are immutable path collections. Construction helpers validate identifiers and containment; the records
perform no I/O.

### 5.3 Compatibility metadata

In `metadata.py`:

~~~python
class NodeRole(StrEnum):
    ROOT = "root"
    WORKER = "worker"

@dataclass(frozen=True, slots=True)
class CocoonetMetadata:
    storage_schema_version: int
    node_role: NodeRole
    compatibility: CompatibilityMetadata
~~~

`storage_schema_version` is exactly `1` in Phase 2. It versions the storage schema, not the Cocoonet release.

The exact `cocoonet_metadata.csv` schema is:

~~~text
field,value
storage_schema_version,1
node_role,root|worker
cocoonet_version,<complete version>
python_major,<nonnegative integer>
python_minor,<nonnegative integer>
python_micro,<nonnegative integer>
python_release_level,alpha|beta|candidate|final
~~~

Rows and order are exact. Unknown, missing, duplicate, or reordered fields are rejected.

### 5.4 Execution-state snapshot

In `execution_state.py`:

~~~python
@dataclass(frozen=True, slots=True)
class ExecutionStateSnapshot:
    storage_schema_version: int
    node_role: NodeRole
    parent_id: ParentId
    model_ids: tuple[ModelId, ...]
    progress: ParentProgress
    attempts: tuple[AttemptProgress, ...]
    workers: tuple[WorkerObservation, ...]
    receptions: tuple[ReceptionRecord, ...]
    operations: tuple[BoundedOperation, ...]
~~~

For a root snapshot, `model_ids` is the complete known ordered unit-task set; receptions may cover those units. For a
worker snapshot, `model_ids` contains only units represented by that worker's local assignments, attempts and workers
contain only that worker's observations, and `receptions` is empty because a worker must not persist root selection or
acceptance state. The snapshot references fixed context by identity; it does not duplicate the scientific configuration
owned by `execution_configuration.csv`.

The snapshot stores observations only. It does not allocate, select, accept, retry, recover, infer transitions, inspect
the filesystem for completion, or manufacture missing records.

Validation:

- validates every nested Phase 1 record with its existing validator;
- requires `storage_schema_version == 1`;
- requires `progress.parent_id == parent_id`, a valid managed-path `parent_id`, valid unique managed-path `model_ids`,
  and a nonempty root `model_ids` tuple;
- requires unique attempt IDs, worker IDs, reception parent/model pairs, and operation IDs;
- requires all attempts and receptions to reference `parent_id` and a listed model;
- requires worker active-attempt IDs to resolve to stored attempts assigned to that worker;
- requires root receptions only on `NodeRole.ROOT`;
- requires one worker identity on `NodeRole.WORKER` and every stored attempt assigned to it;
- validates failure-link integrity across operation failures; and
- does not impose future scheduler or transition rules beyond Phase 1 contracts.

### 5.5 Execution-state CSV schema

`execution_state.csv` uses one normalized table so it remains directly inspectable while preserving nested records.

The exact ordered columns are:

~~~python
EXECUTION_STATE_COLUMNS = (
    "schema_version", "record_type", "position", "subposition", "node_role",
    "parent_id", "model_id", "attempt_id", "worker_id",
    "execution_status", "attempt_status", "stop_status",
    "stop_reason_present", "stop_reason", "first_started_at_present", "first_started_at",
    "result_artifact_id_present", "result_artifact_id",
    "worker_availability", "worker_process_status", "ownership_status", "compatibility_status",
    "dataset_ready", "evaluation_ready", "partition_ready",
    "attempt_capacity_present", "attempt_capacity",
    "selected_attempt_id_present", "selected_attempt_id",
    "receipt_status", "persistence_status", "accepted",
    "operation_id", "operation_kind", "attempt_limit", "attempts_made", "operation_status",
    "stage", "fold_id_present", "fold_id", "sample_id_present", "sample_id",
    "class_label_present", "class_label", "report_name_present", "report_name",
    "failure_id_present", "failure_id", "failure_scope_present", "failure_scope",
    "exception_type_present", "exception_type", "message_present", "message",
    "traceback_present", "traceback", "origin_failure_id_present", "origin_failure_id",
)
~~~

The CSV writer emits these names as one physical header row.

Exact `record_type` values and ordering are:

1. one `snapshot` row carrying `node_role` and `parent_id`;
2. one `model_unit` row per `model_ids` order;
3. one `parent_progress` row;
4. `attempt` rows in tuple order;
5. `worker` rows in tuple order;
6. each worker's `worker_parent_readiness` rows in tuple order;
7. each worker's `worker_active_attempt` rows in tuple order;
8. `reception` rows in tuple order; and
9. `operation` rows in tuple order.

`position` is the zero-based outer tuple position; `subposition` is the zero-based nested position and is empty when
not applicable. Fields not owned by a row type are empty. Presence columns are `0`/`1` and are mandatory for nullable
fields whose present value may be empty text. An `operation` row includes its `TraceContext` and optional complete
`FailureRecord`; a missing last failure leaves all failure-presence columns `0`.

The decoder reconstructs the exact tuple order and Phase 1 nominal types, then validates the complete snapshot. It
rejects extra rows, missing required row types, gaps or duplicates in positions, values in inapplicable columns, or
noncanonical encodings. CSV row order is evidence; filesystem enumeration order is never used.

This Phase 2 schema persists the fixed parent association and known unit identities needed to interpret mutable state.
The complete scientific execution configuration remains solely owned by Phase 3's `execution_configuration.csv`.

### 5.6 Structured log record

In `logging.py`:

~~~python
class LogChannel(StrEnum):
    RUNNING = "running"
    ERROR = "error"

type LogScalar = None | bool | int | float | str

@dataclass(frozen=True, slots=True)
class StructuredLogRecord:
    timestamp_utc_ns: int
    node_role: NodeRole
    channel: LogChannel
    event: str
    message: str
    context: TraceContext | None
    failure: FailureRecord | None
    attributes: tuple[tuple[str, LogScalar], ...]
~~~

Each `.log` file is UTF-8 JSON Lines: one compact JSON object plus `\n` per append. JSON object keys are deterministic;
`ensure_ascii=False` and `allow_nan=False` are required. Attribute keys are nonempty and unique, and their tuple order is
preserved. A failure reuses its own context; when both `context` and `failure` are present they must agree rather than
record contradictory identity fields.

Phase 2 defines the storage format, not a catalogue of future events. Later phases define their event names and required
attributes.

## 6. Planned functions and responsibilities

### 6.1 `paths.py`

~~~python
def resolve_physical_path(path: PathInput, /) -> Path
~~~

Return the canonical physical path without mutation, including for a missing suffix. Managed-root callers preserve an
absolute input and resolve a relative input inside the current console working directory's
`cocoonet/<parent_id>/` location.

~~~python
def discover_protected_locations() -> tuple[Path, ...]
~~~

Return deterministic, deduplicated canonical protected paths for the current interpreter, package installation,
installed Cocoonet package, source checkout, and relevant filesystem anchors. It performs discovery only.

~~~python
def validate_managed_root(root: PathInput, /) -> Path
~~~

Canonicalize and validate one absolute or relative root against every discovered protected location using the approved
matrix. Reject relative traversal outside its parent-specific base. A private pure containment helper accepts explicit
canonical paths for direct tests; production validation cannot replace or omit discovered protections.

~~~python
def validate_dataset_source_ownership(
    root: PathInput,
    source: PathInput,
    /,
) -> tuple[Path, Path]
~~~

Validate one canonical root/source pair and return the canonical paths.

~~~python
def validate_root_ownership(
    root: PathInput,
    sources: Iterable[PathInput],
    /,
) -> tuple[Path, tuple[Path, ...]]
~~~

Validate the root and the entire materialized source collection before mutation. Reject duplicate canonical sources so
a caller cannot accidentally persist ambiguous duplicate associations.

~~~python
def build_root_storage_layout(root: PathInput, /) -> RootStorageLayout
def build_worker_storage_layout(root: PathInput, /) -> WorkerStorageLayout
def build_worker_parent_storage_layout(
    layout: WorkerStorageLayout,
    parent_id: ParentId,
    /,
) -> WorkerParentStorageLayout
def root_trusted_model_dill_path(
    layout: RootStorageLayout,
    model_id: ModelId,
    /,
) -> Path
def worker_dataset_directory(
    layout: WorkerStorageLayout,
    dataset_id: DatasetId,
    /,
) -> Path
def worker_trusted_model_dill_path(
    layout: WorkerStorageLayout,
    parent_id: ParentId,
    model_id: ModelId,
    /,
) -> Path
def root_bundle_publication_paths(
    layout: RootStorageLayout,
    model_id: ModelId,
    /,
) -> BundlePublicationPaths
def worker_bundle_publication_paths(
    parent_layout: WorkerParentStorageLayout,
    model_id: ModelId,
    /,
) -> BundlePublicationPaths
~~~

Build exact semantic paths and validate path-safe identifiers. The trusted-model helpers only locate architecture-owned
paths; they do not authorize dill use for public workers or perform serialization. None of these helpers creates an
entry.

### 6.2 `publication.py`

~~~python
def staged_file_path(final_path: PathInput, /) -> Path
~~~

Return the same-directory `<filename>.partial` path. Reject directory-like empty names and an input already ending in
`.partial`.

~~~python
def publish_staged_file(
    final_path: PathInput,
    /,
    *,
    replace: bool = False,
) -> None
~~~

Synchronize the exact staging file and atomically publish it. `replace=False` is immutable publication.
`replace=True` is accepted only when the final name is `execution_state.csv`.

~~~python
def discard_staged_file(final_path: PathInput, /) -> bool
~~~

Remove the exact staging file if present and return whether it was removed. Reject a directory at that path.

~~~python
def publish_staged_bundle(paths: BundlePublicationPaths, /) -> None
~~~

Require sibling `model_<id>.staging` and `model_<id>` paths, reject symlinks and an existing final path, synchronize
regular files without following links, and atomically rename the complete directory.

~~~python
def discard_staged_bundle(paths: BundlePublicationPaths, /) -> bool
~~~

Remove only the exact non-symlink staging tree if present and return whether it was removed. Never remove the final
bundle.

### 6.3 `metadata.py` and root operations

~~~python
def current_compatibility_metadata() -> CompatibilityMetadata
def validate_cocoonet_metadata(value: CocoonetMetadata, /) -> None
def encode_cocoonet_metadata_csv(value: CocoonetMetadata, /) -> bytes
def decode_cocoonet_metadata_csv(data: bytes, /) -> CocoonetMetadata
~~~

Capture current version data, validate the record, and encode/decode the exact schema. Decoding does not decide
cross-node compatibility; later worker connection code owns that comparison.

~~~python
def prepare_root_directory(
    root: PathInput,
    /,
    *,
    dataset_sources: Iterable[PathInput] = (),
    overwrite: bool = False,
) -> RootStorageLayout
~~~

Validate all ownership first, apply the fresh-preparation matrix, create only Phase 2-owned directories, write current
root metadata through immutable staged publication, and return the layout.

~~~python
def open_root_directory(
    root: PathInput,
    /,
    *,
    dataset_sources: Iterable[PathInput] = (),
) -> tuple[RootStorageLayout, CocoonetMetadata]
~~~

Validate ownership, require the existing Phase 2 layout and final metadata, decode it, and return without mutation.
It does not recover execution or create missing entries.

### 6.4 `execution_state.py`

~~~python
def validate_execution_state_snapshot(value: ExecutionStateSnapshot, /) -> None
def encode_execution_state_csv(value: ExecutionStateSnapshot, /) -> bytes
def decode_execution_state_csv(data: bytes, /) -> ExecutionStateSnapshot

async def replace_execution_state(
    path: PathInput,
    snapshot: ExecutionStateSnapshot,
    /,
) -> None

async def load_execution_state(
    path: PathInput,
    /,
) -> ExecutionStateSnapshot
~~~

The asynchronous writer validates and encodes before opening staging, writes with `aiofiles`, then calls the focused
mutable publication helper. The loader reads only the final file. Neither function applies transitions or treats state
as proof that a referenced bundle exists.

### 6.5 `logging.py`

~~~python
def validate_structured_log_record(value: StructuredLogRecord, /) -> None
def encode_structured_log_record(value: StructuredLogRecord, /) -> bytes
def decode_structured_log_record(data: bytes, /) -> StructuredLogRecord

async def append_root_log(
    layout: RootStorageLayout,
    record: StructuredLogRecord,
    /,
) -> None
~~~

`append_root_log()` requires `NodeRole.ROOT`, routes `RUNNING` and `ERROR` to the exact root log paths, and performs one
append through `aiofiles`. It does not rotate, truncate, stage, or rewrite the log. Decoding is a diagnostic/test
boundary for one complete line; an incomplete last line is ignored by future whole-log readers rather than repaired.

Worker bundle logs and general worker logging are integrated in their owning worker phases using the same record codec;
Phase 2 creates no worker runtime.

## 7. Validation and error behavior

### 7.1 Exception policy

- Incorrect Python argument types raise `TypeError` with the field name, expected type, and observed type.
- Invalid values, unsafe roots, ownership conflicts, malformed schemas, and invalid identifiers raise `ValueError` with
  the relevant path or field and the violated invariant.
- Operational filesystem, encoding, synchronization, and publication failures raise `StorageError` with the operation
  and exact target path, chaining the original exception.

`StorageError` is a small private `RuntimeError` subclass in `publication.py` and carries no retry policy. Later phases
record or retry failures according to their own lifecycle rules.

### 7.2 Numeric and text rules

- `bool` is rejected where an integer is required.
- Timestamps are exact nonnegative integers for nanoseconds or finite nonnegative floats where the Phase 1 record uses
  seconds.
- Empty identifiers and strings containing NUL are rejected.
- CSV decoders reject a UTF-8 BOM, invalid UTF-8, alternate newlines in unquoted fields, and trailing nonempty data
  outside the schema.
- NewType identities are reconstructed explicitly rather than left as untyped strings.

### 7.3 Publication safety

- Staging and final paths must share one parent.
- File helpers reject directories and symlinks.
- Bundle helpers reject symlinks anywhere in the staging tree.
- No helper resolves a user-controlled staging path by glob or deletes a parent directory.
- Existing immutable final content is never overwritten.
- Loaders never infer authority from path existence alone; they require the final path and valid schema.

## 8. Directly corresponding tests

All tests use `tmp_path`, real files/directories, real symlinks when the platform permits them, real Phase 1 records,
and real asynchronous I/O. Phase 2 has no network boundary, so its tests use no mock, monkeypatch, stub, or fake.
A symlink test may skip only when the test environment cannot create the required link. Tests do not monkeypatch
filesystem calls merely to force operating-system failures.

### 8.1 `tests/storage/test_paths.py`

Test:

- existing and missing-suffix canonical resolution;
- direct absolute-root resolution, relative-root resolution under `cocoonet/<parent_id>/`, and relative escape
  rejection;
- alias, symlink, case-normalization, equality, and component-aware containment;
- direct filesystem-anchor rejection, each risky protected-location containment direction, and acceptance of an
  otherwise safe absolute path;
- source equal to root, source below root, root below source, and disjoint paths;
- validation of all sources before any mutation;
- non-string path-like values and embedded NUL rejection;
- exact root/worker paths;
- exact worker dataset and trusted root/worker model paths;
- valid and invalid dataset, parent, model, and tile identifiers; and
- staging/final bundle path construction.

### 8.2 `tests/storage/test_publication.py`

Test:

- exact `.partial` and `.staging` naming;
- immutable file publication, existing-final rejection, and final bytes;
- mutable replacement only for `execution_state.csv`;
- stale staging remaining non-authoritative;
- exact staging discard behavior;
- complete directory publication without merge;
- existing bundle rejection;
- symlink rejection in bundle staging; and
- staging cleanup never deleting final content or siblings.

### 8.3 `tests/storage/test_metadata.py`

Test:

- current compatibility capture;
- exact metadata bytes, row order, newline, and UTF-8 behavior;
- round-trip equality;
- wrong schema version or role;
- missing, duplicate, reordered, or unknown fields;
- invalid version components and release level; and
- no executable or dill deserialization path.

### 8.4 `tests/storage/test_root_directory.py`

Test:

- missing, empty, nonempty, overwrite, and non-directory cases;
- unsafe/protected root rejection before mutation;
- source conflict rejection before mutation;
- outside user sources remaining byte-for-byte unchanged;
- explicit clearing confined to the validated root;
- root symlink/junction and nested-link safety where supported;
- exact initial directories and absence of later-phase artifacts;
- final metadata publication;
- non-destructive opening;
- missing/invalid final metadata rejection; and
- valid final metadata with stale `.partial` ignored and retained.

### 8.5 `tests/storage/test_execution_state.py`

Build complete root and worker snapshots from real Phase 1 contract records. Test:

- exact header and row ordering;
- round-trip equality for every row type and enum;
- empty text versus absent optional values;
- non-ASCII labels, embedded commas, quotes, and newlines;
- fixed parent association and ordered model-unit identities without duplicated scientific configuration;
- root selected-reception, acceptance, and remaining-operation-budget evidence;
- worker-local state excluding root-wide reception;
- invalid nested Phase 1 records and cross-record references;
- duplicate/gapped positions and unknown/inapplicable values;
- malformed encodings, nonfinite numbers, BOM, and schema-version mismatch;
- asynchronous initial write and repeated atomic replacement;
- loader ignoring stale `.partial`; and
- a state claim never causing automatic bundle promotion or lifecycle action.

### 8.6 `tests/storage/test_logging.py`

Test:

- running/error routing;
- deterministic one-object-per-line JSON;
- non-ASCII and structured failure round-trip;
- context/failure agreement and unique attributes;
- nonfinite attribute rejection;
- multiple asynchronous appends preserving complete lines;
- no staging file creation; and
- a deliberately incomplete final line remaining untouched and distinguishable from complete records.

## 9. Dependency-safe implementation order

Each numbered item is one approval unit unless the user explicitly approves a smaller or combined unit.

1. Add `_storage` package scaffolding and `StorageError`.
2. Add path/layout records with no I/O.
3. Implement `resolve_physical_path()` and containment primitives.
4. Implement protected-location discovery.
5. Implement managed-root validation.
6. Implement dataset-source and aggregate ownership validation.
7. Implement semantic identifier validators.
8. Implement root layout construction.
9. Implement worker and parent layout construction.
10. Implement worker dataset and trusted root/worker model path construction.
11. Implement root and worker bundle publication-path construction.
12. Implement staged-file path calculation.
13. Implement immutable/mutable staged-file publication.
14. Implement exact staged-file discard.
15. Implement staged-bundle publication.
16. Implement exact staged-bundle discard.
17. Add `NodeRole`, `CocoonetMetadata`, and their validators.
18. Implement compatibility metadata capture and CSV codec.
19. Implement fresh root preparation.
20. Implement non-destructive root opening.
21. Add `ExecutionStateSnapshot` and cross-record validation.
22. Implement deterministic execution-state CSV encoding.
23. Implement strict execution-state CSV decoding.
24. Implement asynchronous state replacement and loading.
25. Add structured log records, validation, and one-line codec.
26. Implement asynchronous root log routing and append.
27. Run Phase 2 targeted tests, the full configured suite, static checks, documentation checks, and a final
    scope/redundancy audit.

The order establishes validation before mutation, paths before publication, publication before writers, schemas before
I/O, and state encoding before durable replacement. No unit jumps into dataset, configuration, worker, result, or
network behavior.

## 10. Decisions explicitly deferred

| Decision | Owning phase |
| --- | --- |
| Public computation/configuration API and lifecycle integration | Phase 3 |
| `execution_configuration.csv` schema | Phase 3 |
| Dataset input schemas, dataset IDs, managed references, partition workbook, and XLSX engine | Phase 4 |
| Model adapters and trusted model artifacts | Phases 5–6 |
| Prediction/report/result workbook payloads | Phases 7–10 |
| Result bundle contents and validation | Phase 10 |
| Root-local process and IPC | Phase 13 |
| Scheduling, reception selection, acceptance, retry transitions | Phases 14–15 |
| Recovery orchestration | Phase 15 |
| Online protocol and reconnection | Phase 16 |
| `.tar.zst` creation, compression parameters, HMAC, transfer | Phase 17 |
| Retention and cleanup workflows | Phases 18–20 |
| Public service integration | Phase 20 |

The following fixed contracts are not reopened later: visible semantic paths, identifier path representations,
root/source ownership, strict CSV boundaries, `.partial` file staging, `.staging` bundle publication, and staging
non-authority.

## 11. Review against previous work

### 11.1 Phase 1 dependencies used without redesign

Phase 2 reuses Phase 1 identities, `CompatibilityMetadata`, `ParentProgress`, `AttemptProgress`,
`WorkerObservation`, `ReceptionRecord`, `BoundedOperation`, `TraceContext`, and `FailureRecord`. It validates and
serializes them but does not add transitions or scientific behavior.

### 11.2 Phase 1 terminology alignment

The Phase 1 roadmap uses the current `run()` and result-bundle terminology. Its implemented field shapes and invariants
remain usable: `ArtifactId` identifies the immutable result bundle, and Phase 2 does not add general dill persistence.

## 12. Consistency report

### 12.1 Resolved protected-root rule

Absolute and relative managed-root specifications are permitted. Absolute paths resolve directly. Relative paths
resolve inside `<current console working directory>/cocoonet/<parent_id>/` and cannot escape that parent-specific
directory. Reject a filesystem anchor itself. For Python installation/package-installation directories, the installed
Cocoonet package directory, and the complete source repository, reject equality, a candidate ancestor, or a candidate
descendant. This protects risky locations without rejecting otherwise safe absolute paths.

### 12.2 Other inconsistencies

No other blocking inconsistency was found among current `AGENTS.md`, `docs/architecture.md`, `docs/roadmap.md`, the
implemented Phase 1 contracts, and the Phase 2 scope.

## 13. Approval checkpoint

This revision replaces the obsolete hidden-store, hashed-filename, general-dill-container, JSON-metadata, and
single-result-file plan with the current visible CSV/XLSX and immutable-bundle architecture.

No source code, test, or dependency is modified by this roadmap revision. Phase 2 implementation must not begin until
this roadmap is approved.
