# OceanStorage

From-scratch database internals for study, implemented in Rust and accessed from
other languages. The first milestone is a single-owner, single-node, durable
B+ tree byte store. It is separate from `OceanData`; existing stores and datasets
are not migrated or modified.

## Start here

- [Frozen contract](CONTRACT.md): exact API, limits, protocol and durability promises.
- [Build log](BUILD_LOG.md): executed verification and outstanding limitations.
- [Recipes](../docs/recipes/ocean_storage.md): Rust, Python, Java and MATLAB examples.

```text
MATLAB / Python / Java clients
             |
     versioned stdio protocol
             |
        Rust Engine
             |
    B+ tree and bounded page cache
             |
    full-page redo WAL + page file
```

## Build

Requires Rust 1.89 or newer, Cargo and the platform linker. No database or other
third-party Rust crate is required.

```powershell
cargo build --manifest-path '+OceanStorage/rust/Cargo.toml'
cargo test --manifest-path '+OceanStorage/rust/Cargo.toml'
cargo clippy --manifest-path '+OceanStorage/rust/Cargo.toml' --all-targets -- -D warnings
```

This build's workspace-local toolchain is ignored by Git. In this checkout it
can be selected without changing global PATH:

```powershell
$env:CARGO_HOME = Join-Path $PWD '+OceanStorage/.toolchain/cargo'
$env:RUSTUP_HOME = Join-Path $PWD '+OceanStorage/.toolchain/rustup'
$env:PATH = "$env:CARGO_HOME/bin;$env:PATH"
```

## What is being studied

The tree stores payloads in sorted leaves. Internal separators route searches;
leaf links support ordered scans. Insertion splits full nodes and grows the root.
Deletion currently marks a tombstone: it does not merge nodes or reclaim pages.
The seven-key node limit makes splitting visible in small experiments. Pages are
4096 bytes; keys are at most 128 bytes, values at most 256 bytes. These are explicit
first-milestone limits, not a large-dataset storage format.

A mutation gathers changed page images into a redo transaction, syncs the WAL,
installs the pages, syncs the data file and clears the WAL. Opening first recovers
a committed WAL, then validates data pages. Inspection reports actual node/page
state and measured counters. No existing database engine is hidden underneath.

The protocol exchanges bytes as hex; bytes carry no implicit text, table, matrix
layout or serialization. Callers choose how to encode values. An empty value is
different from a missing key. A single child process owns the directory and holds
an OS lock, which the OS releases if it exits. A second owner must fail. The
current child-process clients are intended for sequential ownership; they are not
multiple simultaneous network clients.

## Durability limits

Each successful mutation is synced before acknowledgement. If a process dies
before acknowledgement, that mutation may or may not have committed; clients
must not infer rollback or retry automatically. Filesystem and device sync
guarantees still apply. The prototype does not provide a portable guarantee that
newly created directory entries survive sudden power loss on every platform.
Tests simulate process failures and torn bytes, not physical power cuts.

Checksums detect accidental corruption; they are not authentication. An I/O
failure during operation makes the engine fail-stop until reopened. Do not edit
database files while the engine owns them. Open performs validation proportional
to the number of allocated pages. Scans currently materialize their results;
streaming and page reclamation are later work.

## Roadmap

1. Durable B+ tree and interoperable clients (this milestone).
2. Dataset/experiment catalog and an explicit `OceanData` adapter.
3. From-scratch LSM backend under the same byte-store semantics.
4. Columnar table storage and tensor layout metadata. Column-major array layout
   is separate from B+ tree/LSM indexing and from table columnar encoding.
5. Query operators, planning and SQL.
6. Replication, partitioning and distributed recovery.

The frozen first-milestone contract does not claim completion of later phases.
