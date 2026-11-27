# P0: Foundations and Contracts (phase plan)

**Status:** written 2026-10-12, awaiting approval. No code until approved.
**Master plan:** `Planning/Game Theory/Game_Theory_Practice_Web_App_Plan.md` (v3)
**Goal:** By the end of P0 a skeleton runs end to end: a Rust contract crate defines `sl/1`, the schema and TypeScript types are *generated* from it, and a tiny WASM engine call runs in a browser Web Worker. Every later phase then fills in this skeleton without changing its shape.

---

## 1. Prerequisites (machine state checked 2026-10-12)

> **Superseded for installation:** implementation moved to a **Fedora Linux** computer (handoff `AGENTS.md` D14). Use `AGENTS.md` §4 for the Linux checks and install commands. The table below describes the original Windows machine and is kept only for reference.

| Tool | State | Action (needs your OK: it downloads and installs software) |
|---|---|---|
| Node.js | v25.6.1 present | none |
| npm | 11.9.0 present | none |
| Docker | 29.2.1 present | none (used from P8) |
| **MSVC C++ build tools** | **missing** (Visual Studio installed, but no C++ workload) | Visual Studio Installer → add "Desktop development with C++". Rust on Windows uses the MSVC linker. The same install also covers Production SI. |
| **Rust** (`rustup`, stable) | **missing** | install from rustup.rs; then `rustup target add wasm32-unknown-unknown` |
| **wasm-pack** or `wasm-bindgen-cli` | missing | `cargo install` (decided at start: wasm-pack is simplest; check that it is still maintained, else wasm-bindgen-cli + a build script) |
| **pnpm** | missing (Node 25 no longer ships corepack) | `npm install -g pnpm` |
| gcloud, terraform, firebase CLI | missing | **not needed until P8**; no action now |

Versions are verified at install time; none are pinned here.

## 2. Folder skeleton

```
StrategyLab/
├─ AGENTS.md                ← entry point for agents: rules (master plan §12) + links
├─ README.md                ← how to build/test, for humans
├─ .gitignore               ← node_modules/, target/, dist/, pkg/, .env, *.tfstate*
├─ package.json             ← root scripts (pnpm workspace): build, test, check, gen
├─ pnpm-workspace.yaml
├─ Cargo.toml               ← Rust workspace
├─ docs/
│  ├─ phases/               ← this file and later phase plans
│  └─ adr/                  ← architecture decision records (first: 0001-rust-wasm-engine, 0002-gcp-stack)
├─ contracts/
│  ├─ schema/               ← GENERATED JSON Schemas (committed, drift-checked)
│  └─ grammar/expr.ebnf     ← expression language grammar (hand-written spec)
├─ engine/
│  ├─ sl-contracts/         ← Rust types for sl/1 (source of truth) + schema generator binary
│  ├─ sl-engine/            ← engine crate (P0: version + model validation only)
│  └─ sl-wasm/              ← thin wasm-bindgen layer over sl-engine
├─ web/                     ← Vite + React + TS app (P0: one page + Web Worker)
│  └─ src/contracts/        ← GENERATED TS types (never hand-edited)
├─ scenarios/               ← sl/1 model files (inputs) used by tests and the library
├─ services/                ← empty until P9 (api/, sweep-worker/)
└─ infra/                   ← empty until P8 (terraform/, compose/)
```

## 3. Contracts to define

### 3.1 Numbers: `Num`
Every numeric output says whether it is exact:
`{ "exact": "3/4" }` | `{ "approx": 0.75, "err": 1e-9 }` (`err` optional, an absolute error bound when known).
Inputs accept plain numbers or rational strings.

### 3.2 `Model` (`schema: "sl/1"`): the input
| Field | Content |
|---|---|
| `schema`, `meta` | version tag; id, title, created, author (user id later) |
| `parameters[]` | name, value, unit, optional range and sweep spec (1-D/2-D) |
| `players[]` | id, display name |
| `primitives` | `demand`: a `Source` (below); per-player `cost`; `capacity`; `distributions` (named); `policy` (tax, subsidy, cap, transfers) |
| `Source` (tagged union) | `formula {expr, var, inverse?}` · `data {points | csv_ref, interpolation}` · `derived {population: distribution of reservation values or costs}` · `approximated {…}` (reserved for P7, rejected by validation until then) |
| `rule` | tagged union: `matrix` (finite normal form, payoffs as `Num` tables) · `cournot` · `bertrand` · `bertrand_edgeworth {rationing}` · `stackelberg {leader}` · later kinds reserved by name |
| `finance` (optional) | horizon, discount rate, investment schedule (used from P5/P6) |
| `request` | which analyses to run (equilibria, best_responses, regimes, statics, welfare, finance, diagnostics) |

Units: each parameter and expression variable has a dimension (`price`, `quantity`, `money`, `rate`, `dimensionless`, `time`); validation rejects mismatches (e.g. adding a price to a quantity).

### 3.3 `Analysis`: the output
`schema`, `engine_version`, `model_hash`, and one optional section per requested analysis: `equilibria[]` (kind `pure | mixed | set`, profile, **certificate**, `exact`), `best_responses`, `regimes`, `statics`, `welfare`, `finance`, `diagnostics`, plus `warnings[]`. Every section records `provenance` (which `Source` kinds its inputs came from). In P0 these sections are *declared* in the schema but the engine returns only `diagnostics` (validation messages).

### 3.4 Model hash (used for cloud result caching from P9)
`model_hash = SHA-256( canonical_json(model) )`, canonical JSON per **RFC 8785 (JCS)**. The cache key is `(model_hash, engine_version)`. [addition: standard, not from the textbook]

### 3.5 Expression grammar (spec only; the parser is P1)
EBNF in `contracts/grammar/expr.ebnf`: numbers/rationals, parameters, variables, `+ − × ÷ ^`, `min`, `max`, `abs`, `if(cond, a, b)`, comparisons, a fixed whitelist of functions (`exp`, `ln`, `sqrt`, distribution `cdf`/`pdf`). No user-defined functions, no loops, so every expression always terminates.

### 3.6 Generation pipeline (no hand-copied types)
`sl-contracts` (Rust + `serde` + `schemars`) → `cargo run -p sl-contracts --bin gen-schema` → `contracts/schema/*.json` → `json-schema-to-typescript` → `web/src/contracts/*.ts`.
The REST API (OpenAPI) is added to this pipeline in P9, not now.

## 4. End-to-end smoke path

`sl-engine` exposes `engine_version()` and `validate(model_json) → Analysis{diagnostics}`. `sl-wasm` wraps both. The web page loads the WASM in a **Web Worker**, sends a scenario file, and shows the diagnostics. This proves the Rust → WASM → Worker → React chain before any maths is written.

## 5. Tests ("done when")

1. `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test` all pass.
2. **Schema round-trip:** every file in `scenarios/` deserialises in Rust → re-serialises identically (after canonicalisation) → validates against the generated JSON Schema in TS (`ajv`) → its parsed TS object re-serialises to the same canonical JSON.
3. **Rejection tests:** invalid models (unit mismatch, unknown rule, `approximated` source, missing player) are rejected with a clear message. Test inputs are written deliberately as *invalid inputs*, not as fake results.
4. **Drift check:** `pnpm gen && git diff --exit-code contracts/ web/src/contracts/` reports no changes (this checks generated files are up to date; it does not commit anything).
5. **Hash stability:** the same model with keys reordered or whitespace changed → same `model_hash`.
6. **WASM smoke:** Vitest runs the WASM `validate` and gets the same result as native `cargo test`. Playwright loads the page and sees "valid" for a scenario file.
7. `pnpm check` runs all of the above with one command.

## 6. Out of scope for P0
Any maths (solvers, expression evaluation), UI design beyond one page, cloud resources, Docker compose, API service.

## 7. Questions for approval
1. OK to install the prerequisites in §1 (MSVC C++ workload, rustup + wasm target, wasm-pack, pnpm)? I'll explain each command when we run it. The Visual Studio workload is installed by you through the Visual Studio Installer window.
2. Starter scenarios for the round-trip tests: a 2×2 matrix game, a linear-demand Cournot duopoly, and a Bertrand–Edgeworth duopoly with capacity. These are model *inputs*; no results are written by hand. OK?
