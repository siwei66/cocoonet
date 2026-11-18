# Phase 2: Filesystem ownership and durable persistence foundations

Status: implemented and verified.

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
- scheduling, attempt assignment, result synchronization, duplicate arbitration, retry transitions, recovery transitions,
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
- execution state uses the exact six-column attempt schema, with root state containing every created attempt and worker
  state containing only attempts assigned to that worker;
- empty fields preserve undefined values, existing rows preserve identity and order, and new attempts append in
  creation order;
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

Managed-root specifications may be absolute or relative. Resolve an absolute root directly and resolve a relative root
from the current console working directory. Use the user-provided directory directly. An otherwise safe absolute path
remains valid.

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
- uses the schema-controlled physical columns owned by its table;
- encodes enums by `.value`, booleans as `0` or `1`, integers as canonical decimal, and finite floats with `repr()`
  round-trip semantics;
- never accepts `NaN` or infinity in lifecycle metadata;
- rejects schema-invalid values; and
- emits deterministic row ordering.

Metadata-oriented CSV uses exactly two physical columns with the exact header `field,value`, places one canonical
metadata key in `field`, and places its scalar value or schema-defined text representation in `value`. Nested metadata
records and repeated values use schema-defined hierarchical keys, explicit collection counts, zero-based indices, and
explicit presence keys where required. Metadata decoders reject unknown, duplicate, missing, or out-of-order keys,
noncontiguous indices, and count mismatches.

`execution_state.csv` is the Phase 2 tabular exception. It has exactly this header:

~~~text
parent_id,worker_id,model_label,model_id,attempt_id,attempt_status
~~~

Undefined values use empty fields. The decoder rejects an incorrect header, malformed rows, unknown statuses, invalid
identifiers, duplicate attempt IDs within a parent, and inconsistent model identity/label associations. The replacement
writer rejects a nonempty worker ID that changes after assignment. Encoding preserves supplied row order. Decoding
preserves file order, but consumers must not assign semantic meaning to it.

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

### 5.4 Attempt execution-state record

In `execution_state.py`:

~~~python
@dataclass(frozen=True, slots=True)
class ExecutionStateRecord:
    parent_id: ParentId
    worker_id: WorkerId | None
    model_label: ScalarLabel
    model_id: ModelId
    attempt_id: AttemptId
    attempt_status: AttemptStatus
~~~

`worker_id=None` encodes as an empty CSV field. Other fields are required. `model_label` persists the exact fixed parent
model label through the existing `ScalarLabel` type.

`AttemptStatus` has exactly these values:

~~~text
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
~~~

A root state sequence contains every attempt created for one parent. A worker state sequence contains only attempts
assigned to that worker. The persistence layer validates record structure and cross-row identity consistency; it does
not perform lifecycle transitions, inspect result bundles, determine model completion, apply retry policy, or infer
worker cleanup.

### 5.5 Execution-state CSV schema

The file contains one header followed by one row per attempt:

~~~text
parent_id,worker_id,model_label,model_id,attempt_id,attempt_status
~~~

The root initially provides one `not_allocated` row per model. Later duplicate and retry attempts append rows.
Assignments fill the existing empty `worker_id`; state transitions update only `attempt_status`. Existing attempt rows
are not removed, duplicated, or reordered.

The root and each worker serialize updates to their own file through one writer. The writer stages and atomically
replaces the complete CSV. Atomic replacement prevents partial publication. Updating an existing row requires the exact
previously observed record as an optimistic concurrency precondition; a mismatching current record rejects the stale
revision. Newly created attempts require no precondition and remain mergeable as append-only rows.

The loader returns records in file order. Row order is preserved for stable inspection but has no lifecycle meaning.
The file is informational and never establishes result-bundle existence or model completion. The complete scientific
execution configuration remains solely owned by Phase 3's `execution_configuration.csv`.

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

Each `.log` file is UTF-8 JSON Lines: one compact JSON object plus `\n` per append. Encoding uses deterministic JSON
object key order, `ensure_ascii=False`, and `allow_nan=False`. Decoding requires the exact schema-controlled key sets
but does not depend on their physical JSON object order. Attribute keys are nonempty and unique, and their tuple order
is preserved. A failure reuses its own context; when both `context` and `failure` are present they must agree rather
than record contradictory identity fields.

Phase 2 defines the storage format, not a catalogue of future events. Later phases define their event names and required
attributes.

## 6. Planned functions and responsibilities

### 6.1 `paths.py`

~~~python
def resolve_physical_path(path: PathInput, /) -> Path
~~~

Return the canonical physical path without mutation, including for a missing suffix. Managed-root callers preserve an
absolute input and resolve a relative input from the current console working directory.

~~~python
def discover_protected_locations() -> tuple[Path, ...]
~~~

Return deterministic, deduplicated canonical protected paths for the current interpreter, package installation,
installed Cocoonet package, source checkout, and relevant filesystem anchors. It performs discovery only.

~~~python
def validate_managed_root(root: PathInput, /) -> Path
~~~

Canonicalize and validate one absolute or relative root against every discovered protected location using the approved
matrix. A private pure containment helper accepts explicit canonical paths for direct tests; production validation
cannot replace or omit discovered protections.

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
def validate_execution_state_records(
    value: tuple[ExecutionStateRecord, ...],
    /,
) -> None

def encode_execution_state_csv(
    value: tuple[ExecutionStateRecord, ...],
    /,
) -> bytes

def decode_execution_state_csv(
    data: bytes,
    /,
) -> tuple[ExecutionStateRecord, ...]

async def replace_execution_state(
    path: PathInput,
    records: tuple[ExecutionStateRecord, ...],
    /,
    *,
    expected_records: tuple[ExecutionStateRecord, ...] = (),
) -> None

async def load_execution_state(
    path: PathInput,
    /,
) -> tuple[ExecutionStateRecord, ...]
~~~

The asynchronous writer validates and encodes before opening staging, preserves existing identity order, and permits
new attempts only as appended rows. An existing row may change only when its exact currently published record matches
the supplied optimistic concurrency precondition. The writer rejects stale revisions, writes with `aiofiles`, and then
calls the focused mutable publication helper. The loader reads only the final file. Neither function applies lifecycle
transitions or treats state as proof that a result bundle exists.

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
- Empty identifiers are rejected. Cocoonet must not produce NUL characters in lifecycle metadata strings; Phase 2
  does not require redundant persistence-boundary NUL validation for internally produced record values.
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
- direct absolute-root resolution and direct relative-root resolution from the current console working directory;
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

Build root and worker attempt-state records from real Phase 1 identities. Test:

- exact six-column header and deterministic UTF-8 CSV encoding;
- root initial records with empty `worker_id`;
- worker records limited to their assigned attempts;
- every exact status value;
- non-ASCII model labels and RFC-compatible commas, quotes, and newlines;
- round-trip preservation of row order;
- assignment and status updates with exact-current-record preconditions;
- immutable worker assignment after it is established;
- rejection of an existing-row change without its current-record precondition;
- duplicate and retry attempts appended in creation order without preconditions;
- duplicate attempt-ID and inconsistent model-ID/model-label rejection;
- malformed rows, incorrect headers, BOM, unknown statuses, and invalid identifiers;
- serialized concurrent appends without lost rows;
- competing updates based on the same observation permitting one revision and rejecting the stale competitor;
- stale observations unable to regress a newer published status;
- initial write and repeated atomic replacement;
- loader ignoring stale `.partial`; and
- state claims never establishing result-bundle existence or model completion.

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
21. Add `ExecutionStateRecord` and record-sequence validation using the established Phase 1 `AttemptStatus`.
22. Implement deterministic six-column execution-state CSV encoding.
23. Implement strict six-column execution-state CSV decoding.
24. Implement serialized asynchronous state replacement and loading.
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
| Result bundle contents and validation | Phase 12 |
| Root-local process and IPC | Phase 13 |
| Scheduling, result synchronization, duplicate arbitration, retry transitions | Phases 14–18 |
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

Phase 2 reuses Phase 1 identities, `ScalarLabel`, `CompatibilityMetadata`, and the established `AttemptStatus`
vocabulary without redesign. It persists minimal attempt-state observations without performing transitions. Phase 2
adds only the storage-owned `ExecutionStateRecord` representation and no scientific behavior.

### 11.2 Phase 1 terminology alignment

The Phase 1 roadmap uses the current `run()` and result-bundle terminology. Its implemented field shapes and invariants
remain usable: `ArtifactId` identifies the immutable result bundle, and Phase 2 does not add general dill persistence.

## 12. Consistency report

### 12.1 Resolved protected-root rule

Absolute and relative managed-root specifications are permitted. Absolute paths resolve directly. Relative paths
resolve from the current console working directory, and the user-provided directory is used directly. Reject a
filesystem anchor itself. For Python installation/package-installation directories, the installed
Cocoonet package directory, and the complete source repository, reject equality, a candidate ancestor, or a candidate
descendant. This protects risky locations without rejecting otherwise safe absolute paths.

### 12.2 CSV persistence formats

Persisted metadata-oriented CSV tables and result-workbook metadata worksheets use exactly two columns named `field`
and `value`.

`execution_state.csv` is intentionally not metadata-oriented. It is an attempt-state domain table and uses the exact
six-column schema defined in Section 5.5. This exception avoids hierarchical snapshot serialization and persists only
the minimal user-inspection state required for attempts. It does not change the approved `field,value` metadata schema
for each XLSX result workbook.

### 12.3 Other inconsistencies

No other blocking inconsistency was found among current `AGENTS.md`, `docs/architecture.md`, `docs/roadmap.md`, the
implemented Phase 1 contracts, and the Phase 2 scope.

## 13. Approval checkpoint

This revision replaces the obsolete hidden-store, hashed-filename, general-dill-container, JSON-metadata, and
single-result-file plan with the current visible CSV/XLSX and immutable-bundle architecture.

No source code, test, or dependency is modified by this roadmap revision. Phase 2 implementation must not begin until
this roadmap is approved.
