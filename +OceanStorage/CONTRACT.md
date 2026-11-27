# OceanStorage milestone 1 — frozen contract

Status: implementation in progress, 2026-10-16.

## Scope and boundaries

A from-scratch, single-node Rust B+ tree byte-key/byte-value engine, separate
from OceanData. No existing database engine dependency. No changes to existing
stores or data. This milestone is not a DDBMS, SQL engine, LSM implementation,
columnar implementation, or replacement of MetadataRegistry yet.

Rust source: `+OceanStorage/rust/` (crate `ocean-storage`, library
`ocean_storage`, executable `ocean-storage`). Language clients are thin process
adapters. A long-lived child process owns the directory. Stdio is the first
versioned transport; HTTP, C ABI and multiple concurrent clients are later work.

## Engine API (owned by engine builder)

`pub type Result<T> = std::result::Result<T, Error>`; Error implements
Display + std::error::Error.

```
Engine::open(path: impl AsRef<Path>) -> Result<Engine>
Engine::put(&mut self, key: &[u8], value: &[u8]) -> Result<()>
Engine::get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>>
Engine::delete(&mut self, key: &[u8]) -> Result<()>
Engine::scan(&mut self, start: &[u8], end: Option<&[u8]>)
    -> Result<Vec<(Vec<u8>, Vec<u8>)>>
Engine::inspect(&mut self) -> Result<String>
```

Bytewise lexicographic ordering; empty keys and values supported; put replaces;
delete absent is a no-op; scan is start-inclusive/end-exclusive. Reversed/equal
bounds return empty. Keys <=128 bytes, values <=256 bytes; oversize rejected
before any mutation. Public constants MAX_KEY_SIZE, MAX_VALUE_SIZE. Reads after
an engine storage I/O failure must fail until reopen (fail-stop).

Actual B+ tree with sorted leaf records, internal separators, linked leaves,
recursive splitting and root growth, stored in fixed 4096-byte pages. Maximum
seven keys per node is an intentional teaching parameter. Deletion may use
tombstones without merging/reclaiming in this milestone, explicitly documented.
No map-as-tree substitute. Clean read cache is bounded; uncommitted dirty pages
may be buffered separately. Inspection emits actual page IDs, node structure,
leaf links and measured I/O/cache counters as readable text (no plots).

OS-backed exclusive lock, automatically released on process exit; a second
owner fails without modifying the database. Use std File::try_lock on Rust >=1.89
or a small locking dependency. Persistent data file and WAL, versioned headers,
checksums; invalid formats and corruption fail explicitly.

Each mutation is one transaction. Full-page redo WAL is acceptable: collect
dirty page images including metadata, write a checksummed committed frame,
sync WAL, install data pages, sync data, then reset WAL. Recovery replays only
complete validated committed frames idempotently. An incomplete trailing frame
is discarded; complete checksum corruption is an error. Never acknowledge
before sync. Initial files must be synced; document any parent-directory/power
loss portability limits. WAL recovery must precede validation of possibly torn
data pages. Fail-stop on ambiguous commit I/O errors.

## Protocol v1 (owned by client builder)

Invocation: `ocean-storage serve <database-directory>`.
ASCII, one request per line, tab-separated fields, lowercase or uppercase hex
input bytes (output lowercase), no spaces inside hex. Empty hex fields retained.
First response is `READY\t1`. Engine-open failure: stderr + nonzero exit.

Requests/responses:

| Request | Response |
| --- | --- |
| `PUT\t<keyhex>\t<valuehex>` | `OK` |
| `GET\t<keyhex>` | `VALUE\t<valuehex>` or `MISSING` |
| `DELETE\t<keyhex>` | `OK` |
| `SCAN\t<starthex>\t<endhex or *>` | zero or more `ROW\t<keyhex>\t<valuehex>` followed by `END` |
| `INSPECT` | `TEXT\t<utf8 text encoded as hex>` |
| `CLOSE` | `OK` then exit |

Errors: `ERR\t<utf8 error encoded as hex>`, always flushed. Invalid requests do
not mutate storage. Bound request size; a broken output pipe terminates cleanly.
No user-generated logging on stdout. A process crash may leave an unacknowledged
write committed; acknowledged writes survive restart, assuming filesystem sync
guarantees. Clients must not blindly retry ambiguous writes.

Python client `clients/python/ocean_storage.py`: stdlib subprocess, bytes API,
context manager. Java `clients/java/OceanStorageClient.java`: stdlib ProcessBuilder,
byte[] API, AutoCloseable. MATLAB `+OceanStorage/Client.m`: Java ProcessBuilder
transport, uint8 API, close method (reserve delete for MATLAB destructor; use
remove for record deletion). Executable path supplied by caller. MATLAB tests
in `+OceanTest/+OceanStorage/ClientTest.m`, using OceanTest.BaseTestCase.

## Acceptance and critic gate

Engine: deterministic randomized operations against std BTreeMap, multi-level
splits, binary/empty bytes, overwrite, tombstones, range edges, bounded cache,
reopen persistence, lock exclusion, checksums/format rejection and crash injection
at WAL/data boundaries in real child processes. Independent validation feeds
repository text records to our engine and Python SQLite, comparing ordered results.
Clients: execute Python/Java/MATLAB roundtrips against actual Rust executable,
including persisted shared records, errors and missing/empty distinction.

Critic is fresh, review-only. Score /10: contract 2, API/style 2, algorithmic
correctness 2, durability/robustness 2, reproducible tests/docs 2. Pass >=8 with
no blockers. Maximum three critique rounds. Findings identify owner and concrete
failure. Execution evidence and remaining limitations go in BUILD_LOG.md.

## Later milestones

Catalog/MetadataRegistry adapter; LSM under the same byte-store semantics;
columnar/tensor layouts (separate from tree indexing); streaming scans; page
reclamation; transaction batches; query engine; service transport; distribution.
