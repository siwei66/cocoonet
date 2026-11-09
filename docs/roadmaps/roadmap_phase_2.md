# Phase 2: Filesystem ownership and durable persistence foundations

Status: proposed; no Phase 2 implementation is authorized.

## 1. Authority, purpose, and approval gates

This plan covers only Phase 2, **Filesystem ownership and durable persistence foundations**, in
[the project roadmap](../roadmap.md). It follows the complete root [AGENTS.md](../../AGENTS.md), the complete
[architecture](../architecture.md), and the implemented [Phase 1 roadmap](roadmap_phase_1.md). The governing documents
and the Phase 1 source and directly corresponding contract tests were inspected before this plan was drafted.

The goal is to establish a small private storage boundary that later phases can use without redefining physical-path
ownership, managed-root preparation, artifact publication, dill-container metadata, execution-checkpoint durability,
or structured diagnostic logs. Phase 2 makes storage operations reliable and inspectable; it does not implement the
configuration, dataset, evaluation, scheduling, delivery, or recovery behavior that will call them.

The architectural impact is a concrete implementation of storage mechanisms already assigned to Phase 2. It does not
change the computation/parent/unit/attempt hierarchy, scientific contracts, lifecycle rules, or network architecture.
The file layout, publication protocol, and internal signatures below are Phase 2 implementation decisions submitted
for approval.

Approval of this roadmap and authorization to implement an individual unit remain separate gates. After this roadmap
is approved, each implementation response must:

1. inspect the relevant source and tests and review the proposed unit for necessity and over-engineering;
2. present the exact declaration, function, method, or explicitly bounded helper bundle and obtain approval;
3. implement only that unit and its directly corresponding tests and minimal supporting edits;
4. run the applicable targeted and configured checks, report results, suggest a manual commit message, and stop.

If implementation requires changing a signature, schema, durability rule, file format, or invariant in this roadmap,
stop and obtain approval for a roadmap amendment first. Do not commit automatically.

This planning task creates only this file. It does not authorize production code, tests, dependencies, or edits to
`AGENTS.md`, `docs/architecture.md`, or `docs/roadmap.md`.

### 1.1 In scope

- Canonical physical-path resolution for ownership comparison, without changing user source files.
- Bidirectional enforcement that a configured dataset source path is neither equal to nor contained within the
  Cocoonet-managed root directory.
- Fresh managed-root preparation for missing, empty, and explicitly overwritten nonempty directories.
- A separate read-only opener for an existing managed root; recovery never calls fresh preparation.
- A stable private directory layout for partition artifacts, committed execution configuration, mutable execution
  checkpoints, result files, and structured logs.
- A versioned one-file Cocoonet dill-container format with non-executable metadata readable before payload loading.
- Same-directory incomplete files, atomic final publication, flush/synchronization requirements, and explicit
  inspection of absent, incomplete, and complete publication evidence.
- Immutable artifact publication, atomic replacement of the one mutable execution checkpoint, and designated loading
  that validates identity, kind, schema, compatibility, payload type, and payload invariants.
- Persistence of Phase 1's parent progress, attempts, receptions, and bounded-operation observations in a validated
  execution checkpoint, including the known logical-task set, fixed-parent association, selected result, durable
  acceptance evidence, and unchanged retry counts.
- Asynchronous persistence of an already packaged result-file byte stream without deserializing it in the active
  receive path.
- Structured root/worker running and error logs using a fixed, non-executable JSON Lines diagnostic schema.
- Explicit storage failures, direct tests, and cross-platform path/content behavior.

### 1.2 Out of scope

- The public computation class, constructor, `initialize()`, configuration registries, mutation/state-transition
  algorithms, copying, bulk operations, or start prerequisites: Phase 3 and later lifecycle phases.
- Dataset table reading, normalization, `data_reference.csv`, dataset/sample identity generation, split generation,
  split strategy validation, explicit/default `split_path` selection, transfer preparation, and registration:
  Phase 4. Phase 2 only persists an already validated `PartitionSchedule` in the managed default artifact store.
- Technical model isolation, executable model serialization/loading, reconstruction, fitting, prediction, reports,
  influence, PyTorch integration, and worker resource control: Phases 5–11.
- Construction of concrete attempt-result payloads or worker-side result packaging: Phase 12. Phase 2 provides the
  reusable dill container and raw-byte persistence boundary only.
- Parent commitment, unit/attempt allocation, result selection, acceptance decisions, acknowledgements, delivery
  retries, retention, interruption, reconciliation, or recovery algorithms: Phases 13–18. Phase 2 stores validated
  observations but does not create their transitions.
- Network transfer, authentication, heartbeat/reconnection, synchronization, cleanup, public-service integration, or
  protocol/wire schemas: Phases 16–20.
- Cross-release storage migration, content-addressed artifacts, payload deduplication, encryption, compression,
  transfer checksums, file locking across independent computation controllers, log rotation, a database, or a generic
  persistence framework.
- A general API for clearing managed-root files. Fresh initialization with `overwrite=True` is the only Phase 2
  root-content clearing operation.

## 2. Entry and completion criteria

### 2.1 Entry criteria

Before the first Phase 2 implementation unit:

- This roadmap is approved and the exact next unit is separately authorized.
- The implemented Phase 1 contracts remain the dependency baseline. In particular, Phase 2 consumes
  `ArtifactId`, `PartitionSchedule`, `ParentEvaluation`, `ParentProgress`, `AttemptProgress`, `ReceptionRecord`,
  `BoundedOperation`, `CompatibilityMetadata`, and their validators rather than replacing them.
- `aiofiles` and `dill` remain the already declared dependencies; Phase 2 introduces no dependency.
- Existing unrelated worktree content is preserved. The untracked development prompt is not a Phase 2 output.
- No blocking ambiguity in Section 11 has been introduced by intervening project changes.

### 2.2 Completion criteria

Phase 2 is complete only when every approved declaration and function in Sections 4–8 exists with its directly
corresponding tests and all of the following hold:

- Every ownership conflict is rejected before a directory or file is created, removed, truncated, or replaced.
- Missing and empty roots are prepared successfully; nonempty roots require exact `overwrite=True`; a filesystem
  anchor is never an admissible managed root.
- Clearing a managed root does not follow child symlinks or junction-like entries into external targets, and configured
  source files outside the root remain byte-for-byte and metadata unchanged by preparation.
- Fresh preparation and opening an existing root are separate functions. The existing-root opener performs no clearing
  or repair.
- The private layout and root metadata are stable, versioned, and validated before artifact use.
- Every object payload is serialized with dill protocol 5 in the one-file container. Plain JSON is used only for the
  bounded non-executable container header, managed-root metadata, and diagnostic log lines; it is not an alternative
  object-payload serialization path.
- Compatibility/container metadata can be read and rejected without deserializing the dill payload.
- Successful publication means validation completed, the complete file was written and flushed, the file descriptor
  was synchronized, and a same-directory atomic replace exposed the final path. A failure never reports success.
- A published immutable artifact is never overwritten. An execution-checkpoint replacement leaves the previous
  complete checkpoint usable until the replacement is atomically published.
- Incomplete evidence is never accepted as complete, and explicit inspection distinguishes final and incomplete paths,
  including the valid case where an old complete checkpoint and a failed replacement coexist.
- Partition artifacts load only as valid `PartitionSchedule` values and preserve exact dataset/partition/sample/fold
  associations. No split is generated or rewritten.
- Execution checkpoints preserve the exact Phase 1 observations and counts, contain exactly the fixed parent's known
  logical units, and cannot claim complete result persistence or acceptance without a complete published result file.
- The asynchronous result-byte writer never calls `dill.load`, `dill.loads`, a payload validator, or result processing.
  It publishes only after normal end-of-stream and durable completion.
- Structured log encoding is deterministic UTF-8 JSON Lines, preserves trace/failure context, and rejects malformed,
  nonfinite, or contradictory records without logging scientific payloads.
- Storage/load errors identify the operation and bounded path or artifact identity and preserve the original exception
  as their cause. Invalid caller values continue to use descriptive built-in `TypeError`/`ValueError` diagnostics.
- Applicable pytest, mypy, Black, and Ruff checks pass under `pyproject.toml`. No new suppression is added without
  explicit user approval.

Completion establishes storage foundations, not a runnable computation, dataset API, packaged result producer,
scheduler, delivery loop, or recovery engine.

## 3. Resolved filesystem ownership and root-preparation contract

### 3.1 Accepted path inputs and canonical comparison

Internal Phase 2 path functions accept `str | os.PathLike[str]`. They reject empty strings, embedded NUL characters,
non-path objects, and paths that cannot be resolved because of malformed syntax or a symlink loop. A relative input is
resolved against the process current working directory at the time the configuration operation is invoked. The future
configuration owner stores the returned absolute path so later behavior does not depend on a changed working directory.

For ownership comparison, resolve existing symlinks/junctions and existing ancestors with
`Path.resolve(strict=False)`, then compare platform-normalized absolute path keys (`os.path.normcase` on platforms
where case normalization applies). Missing final components remain allowed at this layer. Dataset existence, file type,
supported suffix, and readability are Phase 4 validation, not ownership validation.

Path comparison has exactly the architecture's meaning:

```text
resolved_source == resolved_root                  -> ownership conflict
resolved_source is contained within resolved_root -> ownership conflict
resolved_source is outside resolved_root           -> ownership check passes
```

The inverse containment relationship is not added as a new restriction. A root located below a source's parent
directory passes when the actual configured source path remains outside the root. This preserves the explicit
architecture rule rather than replacing it with a broader “no overlapping ancestors” policy.

Validation is bidirectional by caller:

- dataset addition/path change calls `validate_dataset_source_ownership()` against the current root;
- construction, initialization, root reassignment, or copied-configuration root change calls
  `validate_root_ownership()` against every retained source path.

Both functions are read-only and return `None`. The preparation function resolves and validates the complete source
tuple before its first filesystem mutation. It never modifies, moves, opens for writing, hashes, or deletes a source.

### 3.2 Root safety and overwrite behavior

A resolved filesystem anchor such as `/` or `C:\` is rejected as a managed root even with `overwrite=True`. This is a
minimal destructive-operation safety condition, not a broader restriction on user-selected directories.

After all source ownership checks succeed:

| Resolved root state | Required behavior |
| --- | --- |
| Missing | Create it and the Phase 2 layout; return prepared storage. |
| Existing empty directory | Create the layout; return prepared storage. |
| Existing nonempty directory, `overwrite=False` | Raise `FileExistsError`; change nothing. |
| Existing nonempty directory, `overwrite=True` | Remove its contents without following child links, create the layout, and return prepared storage. |
| Existing non-directory | Raise `NotADirectoryError`; change nothing. |

`overwrite` must be an actual `bool`; integers are rejected. “Empty” means no directory entry of any kind, including a
hidden entry. Clearing removes every entry within the selected root. It does not selectively preserve managed files,
because configured source paths have already been proved outside the root. A symlink/reparse-point child is unlinked as
an entry; its external target is not traversed. The selected root directory itself remains present.

The preparation function may leave a partially prepared managed layout if an operating-system failure occurs after an
authorized clear. It must raise `StorageError` and must not claim initialization success. A later fresh preparation of
that nonempty directory again requires explicit `overwrite=True`. Phase 2 does not hide destructive progress or invent
a rollback that could itself lose evidence.

### 3.3 Fresh preparation versus existing-root opening

`prepare_root_directory()` is exclusively a fresh-initialization primitive. `open_root_directory()` is exclusively a
non-mutating existing-storage primitive. The latter requires the root metadata and every required Phase 2 directory,
validates them, and returns the layout. It neither creates missing entries nor accepts `overwrite`.

Later recovery must call `open_root_directory()` and then load the committed configuration/checkpoint. It must never
call `prepare_root_directory()`. Phase 2 implements no recovery transition or automatic repair.

## 4. Storage layout, records, and invariants

All new modules remain private under `src/cocoonet/_storage/`; nothing is exported from `cocoonet.__init__` in Phase 2.
The package contains cohesive modules rather than one module per declaration:

| Module | Responsibility |
| --- | --- |
| `paths.py` | Physical path resolution, ownership checks, and `StorageLayout`. |
| `root.py` | Shared storage error, current compatibility metadata, fresh root preparation, safe content clearing, root metadata, and read-only opening. |
| `artifacts.py` | Artifact references, container metadata/header, paths, publication, inspection, loading, and raw result-byte persistence. |
| `checkpoint.py` | Execution-checkpoint record, validation, durable replacement, and loading. |
| `logging.py` | Structured log records, validation, JSON encoding/decoding, and append. |

### 4.1 Private managed layout

Fresh preparation creates exactly this Phase 2 layout:

```text
<root_dir>/
  .cocoonet/
    root.json
    partitions/
    execution-configuration/
    execution-state/
    results/
    logs/
```

Later phases may add their documented managed files, including `data_reference.csv` and dataset preparation
directories, without changing these Phase 2 paths. User source files never become children of this layout.

`StorageLayout` is a frozen, keyword-only, slotted dataclass containing:

```python
root_dir: Path
control_dir: Path
root_metadata_path: Path
partition_dir: Path
execution_configuration_dir: Path
execution_state_dir: Path
result_dir: Path
log_dir: Path
```

All fields are absolute paths derived from `root_dir`; callers cannot supply individual child paths. The record has no
custom methods and performs no I/O during construction.

`ManagedRootMetadata` is stored as canonical UTF-8 JSON at `.cocoonet/root.json` and contains only:

```python
storage_format_version: int
compatibility: CompatibilityMetadata
```

`STORAGE_FORMAT_VERSION` is `1`. JSON is appropriate here because this metadata must be non-executable and readable
before any dill payload is loaded. Metadata encoding uses sorted keys, compact separators, `ensure_ascii=False`,
`allow_nan=False`, and a final `\n`. A successful root preparation writes `.cocoonet/root.json.partial`, flushes and
synchronizes it, and atomically publishes `.cocoonet/root.json`.

Opening an existing root requires format version 1 and exact Cocoonet-version compatibility through Phase 1's
`require_compatible()`. The Python versions may differ within `SUPPORTED_PYTHON_SERIES`, as Phase 1 already permits.
No cross-Cocoonet-release storage migration or fallback is inferred.

### 4.2 Artifact reference and kinds

`ArtifactKind` is a `StrEnum` with exactly:

```text
partition
execution_configuration
execution_state
attempt_result
```

`ArtifactReference` is a frozen, keyword-only, slotted dataclass:

```python
artifact_id: ArtifactId
kind: ArtifactKind
```

It is a storage handle, not another dataset, partition, parent, unit, or attempt identity. A later committed
configuration may bind a `PartitionReference` to its partition `ArtifactReference`; the parent continues to use the
Phase 1 partition reference as its scientific identity. No schedule is duplicated by this handle.

`new_artifact_id() -> ArtifactId` returns a new UUID4 hexadecimal string wrapped as `ArtifactId`. Consumers continue to
treat the value as opaque; UUID spelling is the generator's implementation choice, not a validation or filename
contract. `validate_artifact_reference()` accepts any nonempty opaque `ArtifactId` and the declared enum kind.

The execution-configuration and execution-state files use module-private reserved artifact references scoped to one
managed root. This makes their designated paths discoverable during opening/recovery without inventing a computation
ID. Partition and attempt-result references are generated per artifact.

The reserved references are constants, not generated identities:

```python
EXECUTION_CONFIGURATION_REFERENCE = ArtifactReference(
    artifact_id=ArtifactId("execution-configuration"),
    kind=ArtifactKind.EXECUTION_CONFIGURATION,
)
EXECUTION_STATE_REFERENCE = ArtifactReference(
    artifact_id=ArtifactId("execution-state"),
    kind=ArtifactKind.EXECUTION_STATE,
)
```

### 4.3 Artifact metadata and one-file dill container

`ArtifactMetadata` is a frozen, keyword-only, slotted dataclass:

```python
format_version: int
reference: ArtifactReference
payload_schema: str
compatibility: CompatibilityMetadata
```

`format_version` is `1`; `payload_schema` is a nonempty exact producer-owned schema name such as
`cocoonet.execution-state.v1`. Artifact kind and schema are separate: kind selects storage/lifecycle behavior, while
schema selects the concrete payload contract. Later payload producers define their own schema names before publishing
them; Phase 2 does not create placeholder dataset, configuration, or result schemas.

Every object artifact is one file ending in `.dill` with this exact container framing:

```text
magic bytes: b"COCOONET-DILL\x00"
header length: unsigned 32-bit big-endian integer
header: canonical UTF-8 JSON representation of ArtifactMetadata
payload: one dill protocol-5 object stream
```

The header is limited to 65,536 bytes and contains no executable data. The reader rejects a wrong magic value, zero or
oversized header, truncated header, invalid UTF-8/JSON, unknown field, invalid metadata, kind/identity/schema mismatch,
or incompatible version before calling `dill.load()`.

Protocol 5 is fixed so declared Python versions use the same supported protocol rather than whichever interpreter has
the newest default. This setting applies to Phase 2 data/configuration/state/result containers, not to later executable
model-payload choices.

### 4.4 Artifact path and publication evidence

Artifact kinds map to layout directories as follows:

| Artifact kind | Directory |
| --- | --- |
| `partition` | `partition_dir` (`.cocoonet/partitions/`) |
| `execution_configuration` | `execution_configuration_dir` (`.cocoonet/execution-configuration/`) |
| `execution_state` | `execution_state_dir` (`.cocoonet/execution-state/`) |
| `attempt_result` | `result_dir` (`.cocoonet/results/`) |

An artifact final filename is derived without treating the opaque ID as path syntax:

```text
token = sha256((kind.value + "\x00" + artifact_id).encode("utf-8")).hexdigest()
final = <kind-directory>/<token>.dill
incomplete = <kind-directory>/<token>.partial
```

The hash is a safe deterministic filename mapping, not a content digest, identity-equivalence claim, transfer integrity
check, or automatic-reuse digest. The container header retains the exact reference and detects a hypothetical filename
collision or wrong-file substitution.

`PublicationInspection` is a frozen record:

```python
reference: ArtifactReference
complete_exists: bool
incomplete_exists: bool
```

The booleans are independent. An immutable artifact normally progresses from neither, to incomplete, to complete. A
failed mutable-checkpoint replacement may legitimately leave `complete_exists=True` for the previous checkpoint and
`incomplete_exists=True` for the unsuccessful replacement. File existence alone never validates metadata or payload.

An incomplete path is not silently deleted or reused. `discard_incomplete_artifact()` removes only that exact derived
`.partial` file and refuses to touch a final file. Later retry/recovery owners call it explicitly before restarting a
whole-file write. Phase 2 defines no resume-from-offset behavior.

### 4.5 Execution checkpoint

`ExecutionCheckpoint` is a frozen, keyword-only, slotted dataclass containing only mutable execution observations:

```python
progress: ParentProgress
attempts: tuple[AttemptProgress, ...]
receptions: tuple[ReceptionRecord, ...]
operations: tuple[BoundedOperation, ...]
```

The fixed `ParentEvaluation` is stored in the separately published committed execution-configuration payload. It is
passed to checkpoint validation/loading rather than duplicated in the checkpoint. The checkpoint schema name is
`cocoonet.execution-state.v1`.

`validate_execution_checkpoint(value, *, parent)` enforces:

- both the checkpoint and supplied parent pass their Phase 1 validators;
- `progress.parent_id == parent.parent_id`;
- attempt IDs are unique, and every attempt validates against that parent;
- the reception collection passes `validate_reception_set()` against the checkpoint attempt inputs;
- receptions contain exactly one `(parent_id, model_id)` record for every model in `parent.models`, with no missing or
  extra logical unit; attempts may be empty before allocation;
- bounded-operation IDs are unique and every operation passes `validate_bounded_operation()`;
- any operation context that supplies a parent/model/attempt coordinate resolves to this parent and the exact attempt
  routing, including worker identity; context omitted where Phase 1 permits it is not invented;
- all tuples and nested records remain unchanged.

The validator performs no I/O. `replace_execution_checkpoint()` additionally checks publication evidence before it
writes: every reception whose `persistence` is `complete` must reference a complete `attempt_result` artifact whose
header identity/kind are valid. This also covers accepted results. Incomplete persistence does not require a partial
file to remain, because a failed operation or explicit discard may have removed it.

The checkpoint uses the reserved execution-state reference. Replacement writes a complete `.partial` file and then
atomically replaces the final checkpoint. Until replacement succeeds, an older final checkpoint remains the only
complete state. Phase 2 does not determine when a transition is permitted; later owners construct the next validated
snapshot and must durably replace it before relying on new selection, acceptance, or retry-count evidence.

The checkpoint is sufficient storage evidence for later lifecycle code:

- the known logical-task set is the fixed parent's ordered models plus the checkpoint's exact reception record per unit;
- attempts preserve routing, progress, stop disposition, and result artifact references;
- selected attempts, receipt/persistence observations, and acceptance remain distinct;
- bounded operations preserve `attempt_limit`, `attempts_made`, status, and last failure exactly, so restart cannot
  manufacture a reset budget.

It is not a recovery algorithm, event journal, scheduler, or source of new transitions.

### 4.6 Structured diagnostic logs

`NodeRole` has `root` and `worker`; `LogChannel` has `running` and `error`. `LogScalar` is
`str | int | float | bool | None`.

`StructuredLogRecord` is a frozen, keyword-only, slotted dataclass:

```python
timestamp: float
node_role: NodeRole
channel: LogChannel
event: str
message: str
context: TraceContext | None
failure: FailureRecord | None
attributes: tuple[tuple[str, LogScalar], ...] = ()
```

Invariants are:

- `timestamp` is a finite, nonnegative UTC Unix timestamp in seconds;
- `event` and `message` are nonempty strings and are preserved without normalization;
- attribute names are nonempty and unique, values use exact scalar types, and floating attributes are finite;
- supplied context and failure records pass Phase 1 validators;
- if `failure` is present, `context` is present and exactly equals `failure.context`;
- the running channel forbids a failure record; the error channel permits a failure record but does not require one,
  because configuration/startup errors are direct exceptions outside `FailureScope`;
- scientific arrays, tables, tensors, model objects, and report payloads are not log attributes. Logs reference their
  identities/artifacts instead.

Each record is one canonical UTF-8 JSON object followed by `\n`, with enum/opaque IDs encoded as their strings and
nested context/failure fields explicitly named. Encoding uses sorted keys, compact separators, `ensure_ascii=False`,
and `allow_nan=False`. Files are:

```text
logs/root-running.jsonl
logs/root-error.jsonl
logs/worker-running.jsonl
logs/worker-error.jsonl
```

The role remains in each record even though it is also reflected in the filename, so an independently copied log line
retains provenance. Logs are diagnostic, append-only evidence and are not authoritative execution state. Phase 2 does
not define event-specific required attributes, log rotation, retention, indexing, or history presentation; each later
producer defines its event name and scalar attributes with its implementation contract.

## 5. Internal APIs and exact signatures

Phase 2 adds no user-facing top-level API. These private signatures are the stable boundaries for later phases.

### 5.1 Path and root functions

```python
type PathInput = str | os.PathLike[str]

def resolve_physical_path(value: PathInput, *, field: str) -> Path: ...

def validate_dataset_source_ownership(
    source_path: PathInput,
    *,
    root_dir: PathInput,
) -> None: ...

def validate_root_ownership(
    root_dir: PathInput,
    *,
    source_paths: tuple[PathInput, ...],
) -> None: ...

def build_storage_layout(root_dir: Path) -> StorageLayout: ...

def current_compatibility_metadata() -> CompatibilityMetadata: ...

def validate_managed_root_metadata(value: ManagedRootMetadata) -> None: ...

def encode_managed_root_metadata(value: ManagedRootMetadata) -> str: ...

def decode_managed_root_metadata(value: str) -> ManagedRootMetadata: ...

def prepare_root_directory(
    root_dir: PathInput,
    *,
    source_paths: tuple[PathInput, ...] = (),
    overwrite: bool = False,
) -> StorageLayout: ...

def open_root_directory(root_dir: PathInput) -> StorageLayout: ...
```

Responsibilities:

- `resolve_physical_path` performs the one canonical resolution described in Section 3.1 and no mutation.
- `validate_dataset_source_ownership` is the singular primitive for a source edit.
- `validate_root_ownership` validates a complete immutable tuple before a root edit and reports the source index.
- `build_storage_layout` derives child paths from an already resolved absolute root and performs no I/O.
- `current_compatibility_metadata` reads only installed Cocoonet metadata and `sys.version_info`, constructs the Phase 1
  record, and validates it;
- the root-metadata validator and codec enforce the exact keys, format version, compatibility record, canonical JSON,
  and non-executable metadata boundary;
- `prepare_root_directory` validates everything first, applies the table in Section 3.2, writes root metadata, and
  returns only after the layout is complete.
- `open_root_directory` validates an existing layout and metadata without changing it.

The focused internal helper `_clear_directory_contents(root_dir: Path) -> None` may be proposed in the same exact
approved bundle as `prepare_root_directory`, because it has no independent caller and exists solely to make the
non-following deletion rule testable. It must verify the resolved target is the expected non-anchor root immediately
before traversal.

### 5.2 Artifact metadata and path functions

```python
def new_artifact_id() -> ArtifactId: ...

def validate_artifact_reference(value: ArtifactReference) -> None: ...

def validate_artifact_metadata(value: ArtifactMetadata) -> None: ...

def validate_publication_inspection(value: PublicationInspection) -> None: ...

def artifact_paths(
    layout: StorageLayout,
    *,
    reference: ArtifactReference,
) -> tuple[Path, Path]: ...

def inspect_artifact_publication(
    layout: StorageLayout,
    *,
    reference: ArtifactReference,
) -> PublicationInspection: ...

def discard_incomplete_artifact(
    layout: StorageLayout,
    *,
    reference: ArtifactReference,
) -> bool: ...

def read_artifact_metadata(
    layout: StorageLayout,
    *,
    reference: ArtifactReference,
    expected_schema: str,
) -> ArtifactMetadata: ...

def _encode_artifact_header(value: ArtifactMetadata) -> bytes: ...

def _decode_artifact_header(value: bytes) -> ArtifactMetadata: ...
```

`artifact_paths()` never uses an ID directly as a path component. `read_artifact_metadata()` reads only the bounded
header and verifies the expected path/reference/kind/schema and compatibility.

The two private header helpers are the only JSON-to-record conversion boundary for artifact metadata. They are proposed
with `read_artifact_metadata()` as one exact helper/caller bundle because neither helper has another independent
responsibility.

### 5.3 Dill publication and loading

```python
def publish_dill_artifact[T](
    layout: StorageLayout,
    *,
    reference: ArtifactReference,
    payload_schema: str,
    payload: T,
    validator: Callable[[T], None],
) -> ArtifactMetadata: ...

def replace_dill_artifact[T](
    layout: StorageLayout,
    *,
    reference: ArtifactReference,
    payload_schema: str,
    payload: T,
    validator: Callable[[T], None],
) -> ArtifactMetadata: ...

def load_dill_artifact[T](
    layout: StorageLayout,
    *,
    reference: ArtifactReference,
    expected_schema: str,
    payload_type: type[T],
    validator: Callable[[T], None],
) -> T: ...
```

`publish_dill_artifact()` accepts partition, execution-configuration, and attempt-result kinds and refuses an existing
final or incomplete path. `replace_dill_artifact()` accepts only the reserved execution-state reference and preserves
the previous final until atomic replacement. Both validate metadata and payload before opening the partial file.

The only planned low-level synchronous publication helper is:

```python
def _write_dill_container[T](
    partial_path: Path,
    final_path: Path,
    *,
    metadata: ArtifactMetadata,
    payload: T,
    replace_final: bool,
) -> None: ...
```

It owns framing, dill protocol 5, flush/fsync, same-directory replacement, and failure cleanup rules; it performs no
payload validation or path selection. It is proposed only in the exact approved bundle with
`publish_dill_artifact()`. `replace_dill_artifact()` reuses it with `replace_final=True`; immutable publication uses
`False` and has already established final absence under the single-writer precondition.

`load_dill_artifact()` accepts only a derived complete final path. It reads and validates metadata before one
`dill.load()` call, checks the exact outer payload type, runs the supplied concrete validator, rejects trailing bytes,
and returns the unchanged validated object. It never imports or loads from a caller-selected arbitrary file path.
Phase 4 will define the separate explicit-`split_path` designation boundary before loading a partition outside this
managed layout; it will reuse the same private container reader rather than add pickle or bypass metadata validation.

The `Callable` validator is not persisted. It lets Phase 2 support concrete later payload types without open dictionaries,
placeholder records, or a second storage implementation. Purpose-specific wrappers below prevent routine callers from
repeating checkpoint rules.

### 5.4 Partition and execution-state wrappers

```python
def publish_partition_schedule(
    layout: StorageLayout,
    *,
    reference: ArtifactReference,
    schedule: PartitionSchedule,
) -> ArtifactMetadata: ...

def load_partition_schedule(
    layout: StorageLayout,
    *,
    reference: ArtifactReference,
) -> PartitionSchedule: ...

def validate_execution_checkpoint(
    value: ExecutionCheckpoint,
    *,
    parent: ParentEvaluation,
) -> None: ...

def replace_execution_checkpoint(
    layout: StorageLayout,
    *,
    value: ExecutionCheckpoint,
    parent: ParentEvaluation,
) -> ArtifactMetadata: ...

def load_execution_checkpoint(
    layout: StorageLayout,
    *,
    parent: ParentEvaluation,
) -> ExecutionCheckpoint: ...
```

`publish_partition_schedule()` requires a caller-created reference of kind `partition`, validates the schedule first,
publishes schema `cocoonet.partition-schedule.v1`, and returns its metadata. It performs no split generation. The
caller retains the reference even if publication fails, so incomplete evidence can be inspected or explicitly discarded.
Phase 4 owns the one-time call and binds that reference to the schedule's `PartitionReference`; immutable publication
ensures the same storage handle cannot overwrite or silently replace the authoritative file.

The execution-checkpoint wrappers use the reserved state reference and fixed schema, validate all Phase 1 associations,
and perform the complete-result publication checks in Section 4.5. They do not update a reception, increment an
operation count, choose an attempt, or infer a transition.

Committed execution-configuration publication uses `publish_dill_artifact()` with the reserved configuration reference.
Its concrete payload/schema and validator belong to Phase 3, because Phase 1's `ComputationSnapshot` intentionally does
not contain full dataset source, model input, evaluation-settings, or worker configuration payloads. Phase 2 must not
invent that later schema.

### 5.5 Asynchronous result-byte persistence

```python
async def persist_result_stream(
    layout: StorageLayout,
    *,
    reference: ArtifactReference,
    chunks: AsyncIterable[bytes],
) -> ArtifactMetadata: ...
```

The reference must be `attempt_result`. The function refuses existing final/incomplete paths, opens the derived partial
path through `aiofiles`, requires each yielded chunk to be nonempty `bytes`, writes in order, and requires at least one
byte overall. After normal end-of-stream, it flushes/synchronizes and validates only the bounded container header from
the partial file. It never loads the dill payload. Only then does it atomically publish the final path and return the
validated metadata. Cancellation, iterator error, header error, write failure, or sync/publication failure leaves no
final file and raises; the incomplete path remains evidence.

Phase 13 and later decide when a selected stream may be received and when the resulting publication permits a
`ReceptionRecord` update.

### 5.6 Structured logging functions

```python
def validate_structured_log_record(value: StructuredLogRecord) -> None: ...

def encode_structured_log_record(value: StructuredLogRecord) -> str: ...

def decode_structured_log_record(value: str) -> StructuredLogRecord: ...

async def append_structured_log(
    layout: StorageLayout,
    *,
    value: StructuredLogRecord,
) -> None: ...
```

Encoding/decoding is explicit rather than `dataclasses.asdict()` so enum values, optional coordinates, failure fields,
unknown keys, booleans versus integers, and bounded diagnostics retain the approved schema. Decoding rejects missing or
extra keys and reconstructs the exact Phase 1 record types before validation.

`append_structured_log()` derives the path from record role/channel, writes one encoded line with `aiofiles` in append
mode, flushes and synchronizes before returning, and raises `StorageError` on operational failure. It does not catch an
application exception or create a `FailureRecord`; the owning boundary supplies the already constructed log record.

### 5.7 Error behavior

Phase 2 introduces one private exception:

```python
class StorageError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        operation: str,
        path: Path,
        reference: ArtifactReference | None = None,
    ) -> None: ...
```

It exposes `operation`, `path`, and optional `reference` attributes and reports an operating-system,
serialization, format, synchronization, or publication failure after caller values have passed validation. Its message
identifies the operation, a bounded path, and the artifact reference when applicable; the original exception is
chained. It carries no retry, failure-scope, or worker-termination policy.

Use built-in exceptions for caller/precondition errors:

- `TypeError` for wrong runtime types;
- `ValueError` for invalid values, ownership conflicts, root-anchor selection, schema mismatch supplied by a caller,
  or invalid record associations;
- `FileExistsError` for nonempty fresh root without overwrite or an already present publication path;
- `NotADirectoryError` for a root/layout path of the wrong filesystem kind;
- `FileNotFoundError` for a missing existing root or complete artifact.

Do not convert a payload validator's `TypeError`/`ValueError` during pre-write validation. On load, invalid persisted
metadata/payload is wrapped as `StorageError` because the stored artifact, rather than the caller candidate, is invalid.

## 6. Publication, durability, and trust rules

### 6.1 Immutable artifact publication

The direct algorithm is:

1. validate layout, reference, schema, compatibility metadata, payload, and final/incomplete absence;
2. create the incomplete file exclusively in the final file's directory;
3. write magic, header length, canonical header, and exactly one `dill.dump(..., protocol=5)` payload;
4. flush the Python file object and call `os.fsync()` on its descriptor;
5. recheck the expected final-path precondition under the storage component's single-writer ownership;
6. atomically move the incomplete path to the final path with `os.replace()`;
7. synchronize the final file and, where the platform exposes directory synchronization, its parent directory;
8. return metadata only after all required calls succeed.

Phase 2 assumes one owning storage component for a managed root. It does not claim that these functions coordinate two
independent processes controlling the same copied computation. Architecture already forbids copying from authorizing
uncoordinated controllers; the concrete cross-controller exclusion mechanism remains a Phase 3/15 decision.

If a required flush, `fsync`, replace, or supported directory-sync call fails, publication raises and does not report a
complete artifact. Hardware and operating-system guarantees beyond successful synchronization calls are not invented.

### 6.2 Mutable checkpoint replacement

Checkpoint replacement follows the same write/sync steps, but an existing final state is expected. The incomplete file
contains the entire next checkpoint; no in-place mutation or append occurs. Only the final `os.replace()` changes the
visible checkpoint. A failed pre-replace write leaves the prior final intact. A failure reported after replace is
treated conservatively as uncertain publication: the caller reloads/inspects before relying on it; Phase 2 does not
fabricate success or roll back the new file.

### 6.3 Loader and deserialization boundary

Header inspection is non-executable and may occur wherever metadata is needed. Dill payload loading occurs only in
`load_dill_artifact()` or a purpose-specific wrapper, only from a path derived by `StorageLayout` and an approved
reference, and only after metadata checks. Partition loading is root-side; standardized result loading is root-side and
outside the active receive event loop. Later private/local model loading uses its separately enforced execution boundary
and is not implemented by this general data/result loader.

Phase 2 does not claim that dill is safe for untrusted arbitrary data. Public workers still reject root-supplied Python
model objects; this storage foundation does not weaken Phase 5's isolation or Phase 20's declarative public-model rules.

### 6.4 Reception and acceptance sequencing supplied to later phases

Phase 2 provides, but does not invoke, the following required ordering:

```text
persist checkpoint containing selected_attempt_id
    -> receive/persist only that selected result stream
    -> verify complete result publication
    -> persist checkpoint with persistence=complete, accepted=False
    -> later coordinator accepts
    -> persist checkpoint with accepted=True
    -> acknowledge/stop duplicates in their owning phases
```

The complete-but-unaccepted checkpoint is intentionally representable for the crash window. A complete result artifact
does not itself update or imply acceptance. No Phase 2 function promotes another duplicate after a failed selected
write, resets a retry count, or recomputes a result.

## 7. Directly corresponding tests

Tests use `tmp_path` and standard-library facilities. Async functions are exercised with `asyncio.run()`; Phase 2 does
not add `pytest-asyncio`. Tests do not launch workers, generate splits, execute models, or simulate a scheduler.

### 7.1 Path and root tests

Planned module: `tests/storage/test_paths.py` and `tests/storage/test_root.py`.

| Unit | Direct tests |
| --- | --- |
| `resolve_physical_path` | String/path-like acceptance; absolute result; relative and `..` resolution; existing symlink resolution where supported; empty/NUL/wrong type/symlink-loop rejection; no mutation. |
| `validate_dataset_source_ownership` | Equal source; nested source; outside source; sibling-prefix trap (`root` versus `root2`); different-drive outside case where available; case behavior follows platform; source remains untouched. |
| `validate_root_ownership` | Empty tuple; all outside; first/middle/last conflict reports index; copied-style retained path; entire tuple validated without mutation. |
| `build_storage_layout` | Exact absolute child paths; no filesystem creation; fields immutable. |
| `prepare_root_directory` | Missing, empty, nonempty/no overwrite, nonempty/overwrite, non-directory, exact bool, anchor rejection; hidden entries count; conflict precedes create/clear; exact layout/metadata; child symlink unlinked without target deletion; source contents/size/mtime unchanged. |
| `open_root_directory` | Valid root; missing root; missing/wrong child; malformed metadata; storage-version mismatch; exact Cocoonet mismatch; supported differing Python series; proof of no writes or repair. |

Symlink/junction tests skip with an explicit reason only when the current platform/test account cannot create them.
Ordinary ownership coverage must still run.

### 7.2 Artifact tests

Planned module: `tests/storage/test_artifacts.py`.

| Unit | Direct tests |
| --- | --- |
| records/validators | Exact kind vocabulary; frozen records; empty ID/schema, wrong enum/version/compatibility, and malformed inspection fields rejected. |
| `current_compatibility_metadata` | Matches installed version and current final supported Python metadata; no payload access. |
| `new_artifact_id` | Nonempty valid opaque IDs and uniqueness over a practical sample; tests do not make UUID syntax a consumer contract. |
| `artifact_paths` | Stable 64-hex filename; kind separation; hostile ID text cannot escape its directory; no file creation; exact SHA-256 mapping. |
| `inspect_artifact_publication` | Absent, incomplete only, complete only, and complete plus incomplete; no payload loading. |
| `discard_incomplete_artifact` | Removes only exact partial; absent returns false; final survives; wrong type/kind rejected. |
| `read_artifact_metadata` | Reads header without calling dill; accepts exact identity/kind/schema; rejects bad magic, length, UTF-8/JSON, extra/missing keys, truncation, collision/mismatch, version mismatch, and oversized header. |
| `publish_dill_artifact` | Round-trip of typed Phase 1 fixture; protocol/header checks; pre-validation before file creation; final/partial collision refusal; no overwrite; write/dump/flush/fsync/replace failure leaves no claimed success; source object unchanged. |
| `replace_dill_artifact` | First publication; successful replacement; failed next write preserves old final and leaves partial; wrong kind rejected; no in-place state mutation. |
| `load_dill_artifact` | Exact type and validator; missing/incomplete file; wrong schema/kind/ID; invalid payload; trailing bytes; dill failure chained; metadata rejection occurs before a spy `dill.load`; designated-path-only behavior. |
| partition wrappers | Valid `PartitionSchedule` round-trip with exact associations/order; wrong-kind reference and invalid schedule create no file; repeated publication to the same reference is refused; loading cannot resplit or alter records. |

Failure-injection tests monkeypatch the narrow file/dill/synchronization boundary, not global unrelated APIs. They assert
the final/partial evidence and exception chaining rather than brittle full OS error wording.

### 7.3 Execution-checkpoint tests

Planned module: `tests/storage/test_checkpoint.py`.

| Unit | Direct tests |
| --- | --- |
| declaration | Frozen tuple-owned observations; construction has no I/O or validation side effect. |
| `validate_execution_checkpoint` | Valid no-attempt initial checkpoint; unique attempts; exact parent; exactly one reception per model; selected attempt routing; accepted and complete-but-unaccepted states; unique operations; exact retry counts; mismatched parent/model/attempt/worker contexts; no mutation. |
| `replace_execution_checkpoint` | Selection checkpoint becomes durable before a caller proceeds; complete/accepted reception requires a complete matching result header; partial/missing/wrong-kind result rejected; failed replacement preserves prior checkpoint; exact counts survive. |
| `load_execution_checkpoint` | Valid round-trip against supplied parent; invalid stored associations and missing accepted result rejected; complete-but-unaccepted crash window preserved; no transition/retry/reconciliation side effects. |

Tests explicitly demonstrate that an exhausted operation remains exhausted after round-trip and that replacing worker or
scientific state is not inferred from loading a checkpoint.

### 7.4 Asynchronous result persistence tests

Planned module: `tests/storage/test_result_persistence.py`.

| Unit | Direct tests |
| --- | --- |
| `persist_result_stream` | Ordered multi-chunk exact bytes; metadata-only validation; empty stream/chunk and wrong chunk type; existing final/partial refusal; iterator exception, cancellation, write/flush/fsync/replace failure; incomplete retained; final absent on pre-publication failure; no overwrite. |

Spies assert that no dill deserializer or payload validator is called. A complete worker-produced Phase 2 container is
stored byte-for-byte, and returned metadata comes only from its header.

### 7.5 Structured logging tests

Planned module: `tests/storage/test_logging.py`.

| Unit | Direct tests |
| --- | --- |
| declaration/validator | Both roles/channels; finite timestamp; nonempty event/message; unique scalar attributes; bool/int distinction; context/failure equality; running failure rejection; error without `FailureRecord` accepted for configuration errors; no mutation. |
| encoder/decoder | Exact round-trip of minimal, complete attempt-context, operational, and origin-linked failure examples; deterministic key order/newline; Unicode; escaped traceback newlines; unknown/missing keys, invalid enums/IDs, nonfinite values, and malformed JSON rejected. |
| `append_structured_log` | Exact role/channel file; multiple complete lines in order; flush/sync; write failure raises `StorageError`; no execution-state mutation; no dill/payload access. |

### 7.6 Quality and documentation checks

Every class/function/method, including private helpers, receives a NumPy-style source docstring. Non-helper objects end
their `Notes` section with the applicable node-usage statement. Tests check behavior and required diagnostic context,
not exact docstring wording.

For each approved unit, run its targeted pytest module/cases plus applicable mypy, Black `--check`, and Ruff commands.
At phase close, run the complete existing and Phase 2 test suite and configured checks. This documentation-only planning
step does not add or run production tests merely to validate the plan.

## 8. Dependency-safe implementation order

The order below keeps dependencies moving from pure records and path checks to mutation, serialization, state, and
async consumers. Each named function or declaration is a separate default approval unit unless an exact helper bundle
is identified here or separately approved.

| Order | Declaration/function unit | Direct test module |
| ---: | --- | --- |
| 1 | Create private `_storage` package; declare `PathInput` and `StorageLayout`. No I/O. | `test_paths.py` |
| 2 | `resolve_physical_path` | `test_paths.py` |
| 3 | `validate_dataset_source_ownership` | `test_paths.py` |
| 4 | `validate_root_ownership` | `test_paths.py` |
| 5 | `build_storage_layout` | `test_paths.py` |
| 6 | Declare `StorageError`. | `test_root.py`, `test_artifacts.py` |
| 7 | `current_compatibility_metadata` | `test_root.py`, `test_artifacts.py` |
| 8 | Declare `ManagedRootMetadata`, `STORAGE_FORMAT_VERSION`, and its pure validator/JSON codec as separately approved units. | `test_root.py` |
| 9 | `prepare_root_directory` with the exact `_clear_directory_contents` helper bundle described in Section 5.1 | `test_root.py` |
| 10 | `open_root_directory` | `test_root.py` |
| 11 | Declare `ArtifactKind`, `ArtifactReference`, `ArtifactMetadata`, and `PublicationInspection` in dependency order. | `test_artifacts.py` |
| 12 | `new_artifact_id`; then the artifact validators. | `test_artifacts.py` |
| 13 | `artifact_paths` | `test_artifacts.py` |
| 14 | `inspect_artifact_publication`; then `discard_incomplete_artifact`. | `test_artifacts.py` |
| 15 | Artifact header encoder/decoder and `read_artifact_metadata`, as an exact bounded helper/caller bundle if approved. | `test_artifacts.py` |
| 16 | `publish_dill_artifact` with only its low-level write/sync helper bundle explicitly approved for this caller. | `test_artifacts.py` |
| 17 | `replace_dill_artifact`, reusing the approved container writer. | `test_artifacts.py` |
| 18 | `load_dill_artifact` | `test_artifacts.py` |
| 19 | `publish_partition_schedule`; then `load_partition_schedule`. | `test_artifacts.py` |
| 20 | Declare `ExecutionCheckpoint`; then `validate_execution_checkpoint`. | `test_checkpoint.py` |
| 21 | `replace_execution_checkpoint` | `test_checkpoint.py` |
| 22 | `load_execution_checkpoint` | `test_checkpoint.py` |
| 23 | `persist_result_stream` with the exact async write/sync helper bundle required by `aiofiles`. | `test_result_persistence.py` |
| 24 | Declare `NodeRole`, `LogChannel`, `LogScalar`, and `StructuredLogRecord`. | `test_logging.py` |
| 25 | `validate_structured_log_record` | `test_logging.py` |
| 26 | `encode_structured_log_record`; then `decode_structured_log_record`. | `test_logging.py` |
| 27 | `append_structured_log` | `test_logging.py` |
| 28 | Phase-wide contract/layout/security review and complete configured checks; no new production unit. | All targeted modules |

Before every unit, confirm that its caller/consumer needs it now. Do not add repositories, service objects, registries,
lock managers, async abstractions, generic serializers, or recovery journals beyond this plan.

## 9. Validation, numerical, and cross-platform behavior

Phase 2 implements no scientific metric, split randomness, probability calculation, influence formula, CPU-capacity
formula, or model execution. It must preserve Phase 1 scientific values byte-for-byte through dill; it must not inspect,
repair, normalize, clip, impute, or aggregate them.

The only Phase 2 computations are operational and exact:

- artifact filename token: SHA-256 of UTF-8 `kind + NUL + artifact_id`;
- header length: unsigned 32-bit big-endian byte count, at most 65,536;
- timestamps: finite nonnegative floating-point Unix seconds;
- storage format version: integer exactly 1, excluding `bool`;
- file byte order: magic, length, header, one dill stream, no trailing bytes;
- operation attempt counts: preserved and validated by Phase 1 without increment/reset;
- JSON numbers: finite only; `allow_nan=False`.

Filesystem behavior and logical content must be invariant across supported operating systems:

- derive paths with `pathlib`, not string separator concatenation;
- compare canonical platform path keys while preserving returned canonical spelling;
- use UTF-8 explicitly for JSON/log text and `\n` as the on-disk line separator;
- use same-directory partial/final paths so replacement never crosses filesystems;
- do not depend on filesystem iteration order; clearing behavior is set-based, and diagnostics sort only bounded path
  names when deterministic reporting needs them;
- do not follow child links while clearing;
- reject rather than silently downgrade when a required available synchronization operation fails.

Validators follow the existing diagnostic rules: wrong types identify field, expected type, observed type, and bounded
value; invalid values/associations identify context, expected condition, and bounded observed condition. Filesystem
errors report bounded absolute paths and never dump payloads, source data, model objects, or complete large collections.

## 10. Contract resolutions made by this plan

| Required Phase 2 question | Resolution |
| --- | --- |
| How are filesystem aliases/symlinks compared? | Resolve existing links/ancestors with `Path.resolve(strict=False)` and compare platform-normalized absolute keys. |
| Is inverse ancestor overlap rejected? | No. Reject only a source equal to or within the root, exactly as architecture specifies. |
| Can a filesystem root be selected with overwrite? | No; anchors are always rejected as a minimal destructive safety rule. |
| Does overwrite preserve selected files? | No. After ownership validation it clears every root child; source files are outside. |
| How is recovery kept separate from initialization? | Separate non-mutating `open_root_directory`; recovery must never call fresh preparation. |
| Where are files stored? | Fixed `.cocoonet` layout in Section 4.1; later managed dataset files may coexist outside it. |
| How are opaque IDs made filename-safe? | SHA-256 mapping of kind and exact ID; exact ID remains in the header. |
| What is a complete publication? | Validated full file, flush/fsync, same-directory atomic replace, and successful required post-publication synchronization. |
| What happens to incomplete writes? | They remain `.partial`, never count as complete, and require explicit discard before a whole-file retry. |
| Are immutable artifacts overwritten? | Never. Execution state is the only Phase 2 replaceable dill artifact. |
| How is compatibility read without loading a payload? | Bounded canonical JSON header precedes the dill stream in the same file. |
| Is JSON another payload serializer? | No. It is limited to non-executable headers, root metadata, and diagnostic logs; object payloads use dill only. |
| Which pickle protocol is used? | Dill protocol 5 for stable support across declared Python series. |
| How is the known task set persisted? | Fixed parent from committed configuration plus exactly one checkpoint reception per parent model. |
| How is acceptance backed by storage? | A complete/accepted reception cannot be checkpointed unless its exact final result artifact and header are complete. |
| Does the async receive writer deserialize? | No; it writes exact bytes and validates only bounded header metadata before final publication. |
| Are logs authoritative recovery state? | No. The execution checkpoint is authoritative; logs are structured diagnostics. |
| What error scope applies to checkpoint persistence? | The primitive raises `StorageError`. Phase 2 does not mislabel it as dataset or result persistence or invent a retry policy. |

## 11. Inconsistencies, blocking ambiguities, and later deferrals

### 11.1 Remaining blocking ambiguity

No blocking Phase 2 ambiguity remains after the decisions in this roadmap. All contracts required to implement the
storage foundation are specified. Approval is still required before any implementation unit begins.

### 11.2 Inconsistencies and stale documentation found

1. **Roadmap location prose and links:** the actual project roadmap is `docs/roadmap.md`. Its opening relative links
   assume a repository-root location, while its final paragraph says it is intentionally at the repository root and
   calls `docs/roadmap.md` references a mismatch. This is the same nonblocking stale-location issue recorded by the
   Phase 1 roadmap. This task is not authorized to edit the governing roadmap.
2. **Architecture scaffold status:** architecture Section 1 says the repository is at the scaffold stage, while Phase 1
   contracts and tests are implemented and its roadmap status is “implemented and verified.” This is stale status
   prose, not a behavioral conflict or Phase 2 blocker. The implemented Phase 1 contracts are the actual dependency.
3. **Checkpoint persistence failure scope:** Phase 1's `BoundedOperation(PERSISTENCE)` permits
   `DATA_PREPARATION` or `RESULT_PERSISTENCE` failures, but Phase 2 must also atomically persist execution checkpoints.
   Neither existing scope accurately describes a checkpoint write. This plan therefore exposes a direct `StorageError`
   and structured error log and implements no checkpoint retry operation. If a later lifecycle phase requires bounded
   retries and a `FailureRecord` for checkpoint persistence, it must first obtain approval for a focused Phase 1
   contract/roadmap amendment rather than misclassify the error. This does not block the Phase 2 storage primitive.

No conflict was found between the root ownership/overwrite rules, architecture, project-wide Phase 2 scope, or the
implemented Phase 1 schemas that prevents this plan.

### 11.3 Decisions explicitly deferred to later phases

| Deferred decision | Owning phase/boundary |
| --- | --- |
| Public `Computation` constructor/`initialize()` signatures, storage ownership lifetime, configuration collections, source-path retention, root reassignment, and bulk commit behavior | Phase 3 |
| Full committed execution-configuration payload/schema, including authoritative dataset/model/evaluation/worker inputs | Phase 3, completed with Phase 4 dataset records and later model inputs |
| Dataset input APIs, explicit/default `split_path`, partition ID and sample ID generation, external partition designation/loading, schedule strategy admissibility, and `data_reference.csv` | Phase 4 |
| Dataset transfer preparation/reuse metadata files and 16 MB/1 GB policies | Phase 4; Phase 17 consumes them online |
| Model payload format and executable dill settings for trusted local/private models | Phases 5–6, under enforced isolation |
| Concrete validation/report/influence/PyTorch payload schema names and validators | Phases 7–10 |
| Attempt-result payload schema, worker packaging call, failure-only packaging, and result-retention ownership | Phase 12 |
| Local IPC/artifact handoff, parent commitment/start sequencing, attempt allocation, selected-result state transitions, acceptance, and acknowledgement | Phase 13 |
| Retry limits/timing, partial-result discard timing, resend policy, retention modes, and any checkpoint-persistence retry/failure-scope amendment | Phase 14, with recovery consequences in Phase 15 |
| Restart reconciliation, historical checkpoint progression, incomplete-file cleanup policy, copy/controller exclusion or locking, and runtime process lifetime | Phase 15 |
| Wire framing, transfer checksum/integrity, authentication, heartbeat, connection rounds, and online result transport | Phases 16–17 |
| Duplicate selection/ties and mixed local/online arbitration | Phase 18 |
| Worker dataset/user-data deletion selection and verification | Phase 19 |
| Public-service lifecycle, mandatory release cleanup, and externally supplied API | Phase 20 |
| Cross-supported-Python integration matrix and any future storage-format or cross-release migration process | Phase 21 or a separately authorized future design |
| Log rotation, retention, querying, rendered history, database/index adoption, artifact compression/encryption/deduplication | Not required by the initial roadmap; future design only if explicitly authorized |

These deferrals do not weaken source ownership, one authoritative partition artifact, exact parent identity, durable
selection/acceptance evidence, unchanged retry counts, one result file per attempt, or the prohibition on loading a
result in the active receive event loop.

## 12. Review checkpoint

Implementation order, in summary:

```text
path records and canonical resolution
    -> source/root ownership validation
    -> managed layout and fresh/open root boundaries
    -> artifact records, header, paths, and publication inspection
    -> immutable dill publication and designated loading
    -> partition wrappers
    -> execution checkpoint and atomic replacement
    -> asynchronous raw result persistence
    -> structured diagnostic logs
    -> phase-wide validation and quality checks
```

The plan deliberately separates pure validation, destructive root preparation, immutable artifact publication, mutable
checkpoint replacement, raw result receipt, and diagnostic logging. Later development can consume these foundations
without adding scheduling or scientific behavior to the storage layer.

Stop after this roadmap is reviewed. Plan approval does not authorize Phase 2 source or test implementation; the next
production unit still requires its own explicit approval.
