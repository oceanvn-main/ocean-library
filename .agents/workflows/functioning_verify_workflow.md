---
description: Manual-Implementation Verification Workflow
---

# AGENTS.md — Manual-Implementation Verification Workflow

**Scope.** Governs any task where an agent manually re-implements a component that already exists in a mature production library (e.g. `linear_regression`, `kmeans`, `fft`, a UI widget). Goal: match a trusted reference on correctness, and pass explicit bars on speed, API/UI, and robustness that a reference doesn't automatically guarantee.

**Assumption.** A comparable reference implementation exists and is accessible. If none exists, verification degrades to spec-only testing (§6.1) — label this explicitly, don't silently substitute it for oracle-verification.

---

## 1. Roles

| Role | Job | Must NOT do |
|---|---|---|
| **Spec Author** | Writes the formal contract (§3) | Write implementation code |
| **Builder** | Implements the contract manually | Call the reference library inside the implementation; see Critic's rubric before submitting |
| **Oracle** | Runs reference lib on same inputs, diffs vs Builder | Modify Builder's code; use judgment — purely tolerance-based |
| **Critic** | Reviews axes the Oracle can't check (§7) | Patch code — critique only |
| **Integrator** | Applies gate policy (§9) | Overrule a failed gate without a logged reason |

If run as one agent/session, still produce each role's artifact (§10) as a distinct step — don't fold Oracle/Critic checks into a mental pass during the build step.

---

## 2. Pipeline

```
Request
 → [1] SPEC            → contract.md
 → [2] ORACLE SELECT    → oracle.md (which reference, which version, why trusted)
 → [3] TEST HARNESS     → test_matrix.md (written BEFORE implementation)
 → [4] BUILD             → impl + builder_notes.md
 → [5] VERIFY (Oracle)   → verification_report.md
 → [6] CRITIQUE          → critic_report.md
 → gates pass? NO → back to [4] with both reports as input
              YES → [7] ACCEPT → added to regression suite
```

Test harness is written from the contract, before the implementation exists — so the Builder can't shape tests around its own shortcuts.

---

## 3. Contract (before any code)

```
Component: <name>
Definition: <precise mathematical/behavioral definition>
Assumptions: <e.g. IID data, real-valued input, n > 1>
Domain of validity: <where definition holds / what's out of scope>
Signature: <inputs, types, shapes, outputs, error contract>
Complexity target: <e.g. O(n log n)>
Special cases: <degenerate inputs with distinct handling — don't conflate with general case>
Non-goals: <what this deliberately doesn't do>
```

If ambiguous mid-build, Builder escalates back to Spec Author — no silent reinterpretation.

---

## 4. Oracle Selection

```
Oracle: <component>
Reference: scipy.stats.<fn> v<ver> / sklearn.<mod> / MATLAB Stats Toolbox <ver>
Why trusted: <peer-reviewed algorithm, widely used>
Reference caveats: <e.g. uses a different-but-valid algorithm variant>
```

Two correct implementations of the same *concept* can use different *algorithms* and legitimately diverge at the edges (tie-breaking, convergence criteria). Recording the reference's own assumptions stops the Oracle from failing a correct-but-differently-specified implementation.

---

## 5. Test Harness (from contract, before build)

1. **Golden values** — hand-computed/textbook cases, exact expected output.
2. **Randomized differential tests** — N random in-domain inputs; compare Builder vs Oracle.
   - Deterministic: `|builder - oracle| <= atol + rtol*|oracle|`, tolerance set per-component (no blanket global tolerance).
   - Stochastic (samplers, randomized algorithms): compare *distributions* (KS test, multi-seed summary stats), never a single-seed point value.
3. **Boundary cases derived from contract's "domain of validity"/"special cases"** — empty input, single element, NaN/Inf if in-domain, ill-conditioned input, ties, max-size for perf. Each case traces to a named contract clause.

`test_matrix.md` records per test: generator, tolerance/criterion, which contract clause it checks.

---

## 6. Build (Builder)

- **Before writing the implementation**, write a short `usage_demo.py` (or `.m`) that calls the real production library people would actually use in practice (e.g. `scipy.stats`, `sklearn`, `statsmodels`) end-to-end on representative inputs. Purpose: learn the practical workflow around the component — typical call signature, default parameters, shape/type of inputs and outputs, common surrounding steps (e.g. fit → predict → score) — not to copy internals. This script's outputs double as extra golden values for §5.
- Implement from `contract.md` only. No importing the reference lib inside the final implementation (makes verification tautological) — `usage_demo.py` is a learning/reference artifact, kept separate from `impl.<ext>`.
- May read the reference's docs/paper for the algorithm; implement the method, don't transliterate its source.
- Produces `builder_notes.md`: algorithm chosen + why, expected complexity, any contract deviations + why, plus what `usage_demo.py` revealed about real-world usage conventions.

### 6.1 No reference available
Fall back to property-based tests from contract invariants + hand-worked examples. Report must say "spec-verified, not oracle-verified" — a real, weaker guarantee, not a formality.

---

## 7. Verify (Oracle) — mechanical, not subjective

```
Verification: <component>
| Test | Result | Detail |
| golden_1 | PASS | |
| diff_random(n=500) | PASS | max err 3.1e-9, tol 1e-6 |
| edge_empty_input | FAIL | raises IndexError, contract wants ValueError |
| dist_compare(seed×50) | PASS | KS p=0.41 |
Verdict: FAIL — 1 blocking
```

Any FAIL is blocking by default — goes back to Builder before Critic reviews (no point critiquing UI/speed on numerically wrong output).

---

## 8. Critique (Critic) — fixed rubric, comparable across revisions

| Axis | Checks | Fail example |
|---|---|---|
| Correctness beyond harness | Hand-re-derive 2–3 cases the generator didn't sample | Passes generated tests, fails a valid untested input |
| Performance | Benchmark ratio vs Oracle on matched sizes | 50× slower, no justification |
| Complexity vs target | Empirical scaling at n, 2n, 4n vs contract | Contract says O(n log n), measured O(n²) |
| API/UI (if user-facing) | Naming/signature consistency, error messages, defaults; for UI: layout, responsiveness, contrast, styling consistency | Silent failure instead of raised error |
| Robustness | Malformed types, extreme scale, repeated/concurrent calls | Crashes instead of documented error |
| Style/maintainability | Conventions, inline assumption docs, no unexplained magic numbers | Undocumented tolerance constant |

```
Critique: <component> rev N
Correctness(beyond harness): PASS
Performance: WARN — 4.2x slower at n=1e5; contract silent on target → flag to Spec Author
API/UI: FAIL — bare Exception, inconsistent with module
Robustness: PASS
Style: PASS
Verdict: FAIL — 1 blocking, 1 needs-spec-clarification
```

If the contract never set a bar (e.g. no speed target), Critic flags the gap rather than inventing one and failing against it — keeps the Critic from becoming a source of new, unagreed requirements.

---

## 9. Revision Loop

- Builder gets both reports, not a rewritten contract — fixes its own code.
- **Cap: 3 rounds.** Same failure class recurring past 3 rounds → escalate to Spec Author (usually means the contract is under-specified, not that the Builder needs another guess).
- Each round versioned (`rev1`, `rev2`...); reports kept, not overwritten, so regressions across rounds stay visible.

---

## 10. Acceptance Gate

- **Blocking:** golden values, differential/statistical tests, any Critic FAIL.
- **Non-blocking, recorded:** Critic WARN, performance below reference with no stated target — accepted with a logged caveat, never silently dropped.
- On accept: implementation + its exact `test_matrix.md` join the permanent regression suite.

---

## 11. Layout

```
/components/<name>/
    contract.md
    oracle.md
    test_matrix.md
    test_cases/
    usage_demo.py
    impl.<ext>
    builder_notes.md
    verification_report_rev1.md
    critic_report_rev1.md
    ACCEPTED.md
```

---

## 12. Anti-patterns

- **Self-graded review** — the session that built it shouldn't also run Oracle/Critic without re-reading only `contract.md` first.
- **Tautological verification** — Builder importing the reference lib internally invalidates the differential test.
- **One tolerance for everything** — set per-component from the contract's stated precision.
- **Point-value comparison for stochastic algorithms** — needs distributional comparison, not single-seed diffs.
- **Critic inventing requirements** — flag missing bars to Spec Author, don't fail against an unagreed target.
- **Tests written after seeing the implementation** — tends to test what the code does, not what the contract requires.