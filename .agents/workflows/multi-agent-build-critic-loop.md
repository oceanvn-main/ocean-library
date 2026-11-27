# Workflow: multi-agent build → independent critic → bounded iteration loop

> [!IMPORTANT]
> **Living workflow document.** Like `.agents/pipeline-architecture-spec.md`, this is not
> fixed. It generalizes a process that was run several times, live, during the OceanNLP
> build (`Planning/NLP/NLP_Library_Master_Plan.md`, `Planning/NLP/Dual_Representation_Addendum.md`)
> — every claim about "what works" below is drawn from something that actually happened in
> that build, not theorized in advance. Update this file when the process teaches you
> something new; don't let it calcify into a checklist nobody re-examines.

## 1. When to use this workflow

Not every task needs it. Use the full workflow when a task has **two or more** of:

- Multiple genuinely independent pieces of work (different files/packages, no shared state) —
  the parallelism is free, not forced.
- A real correctness risk: a published algorithm/formula being implemented, numeric code,
  anything where "looks right" and "is right" can diverge.
- Enough surface area that one agent reviewing its own work is a meaningfully weaker check
  than a second, fully independent agent reviewing it.

For a small, well-scoped change (one file, a docstring fix, a single new method with clear
precedent) — just do it directly, verify it yourself, done. Spinning up a builder-swarm and a
critic for a one-line fix is the kind of premature process the rest of this repo's own
engineering guidance already warns against. (Concretely, in the session this workflow was
extracted from: the `CosineSimilarity` LaTeX-formula bug, the `WeightingFactory` addition, and
the `IIdfScheme` Strategy-pattern refactor were all done directly, with direct MATLAB
verification, specifically *because* they didn't meet the bar above — worth reading as the
negative-space definition of when this workflow is overkill.)

## 2. Roles

**Orchestrator** (you, coordinating) — never writes the deliverable's code directly once
building has started. Freezes the contracts (§4), dispatches builders, keeps a live execution
log, routes critic findings back to the correct owner, and decides when to stop the loop.

**Builder agent** — owns a scoped, file-disjoint piece of work. Must:
- Read the frozen contract before writing anything.
- Actually *execute* what it writes (run the code, run the tests) — a deliverable that has
  only been read, not run, is not done. Paste real command + real output in its report, not a
  summary of what it expects would happen.
- Report deviations from the contract and the reasoning, not just the diff.
- Self-flag anything it's unsure about, rather than silently picking an answer and moving on —
  these flags become the critic's starting checklist, not a substitute for the critic's own
  judgment.

**Critic agent** — always a *fresh* agent invocation, never a builder reviewing its own work,
never an agent that shares conversation context with any builder. Must:
- Re-derive claims independently rather than trusting builder reports — re-run the test suite
  itself, re-derive at least one numeric/logical example itself rather than re-reading a
  builder's worked example and assuming it's right.
- Score against an explicit, numbered rubric (§6), not a vibe.
- Tag every finding with severity and owning builder (§6.2) — the orchestrator routes fixes
  by that tag, so vague findings cost a full extra round to clarify.
- Never fixes code. Review only. (If the critic and the builder are ever the same agent
  identity, the review is worthless — it's grading its own homework.)

**(Domain-dependent) External validation agent/step** — for anything implementing a published
algorithm or formula, an independent cross-check against a production reference on real data,
*separate from* the unit test suite (§5). Unit tests can be self-consistently wrong — they
verify the implementation matches itself. External validation checks whether the *chosen*
algorithm is the one actually intended. (This is what caught nothing new in the OceanNLP
build — all four external checks passed exact — but it's what would have caught it if the
BM25 formula chosen had been the wrong variant; the value is in the check existing, not in it
always finding something.)

## 3. Phase shape

```text
Phase 0   Freeze contracts (orchestrator only, not delegated)
             |
             +---------+---------+---------+
             v         v         v         v
Phase 1   Builder A  Builder B  Builder C  ...     (parallel — file-disjoint, no
             |         |         |                  coordination needed mid-phase)
             +---------+---------+
                       v
Phase 2+  Dependent builders     (sequential — read prior phases' files, compose
             |                    them; never redefine their contracts)
             v
Phase N   Integration verification   (prove components work TOGETHER, not just
             |                        individually — see §5.2)
             v
Phase N+1 Test authoring   (formalize every builder's ad hoc verification into
             |              real, executed test files — see §5.1)
             v
Phase N+2 (domain-dependent) External validation against a production reference,
             |               on real data (see §5.3)
             v
Phase N+3 Critic review (§6)
             |
      score >= threshold? --no--> route tagged findings to owning builder(s) only,
             | yes                re-run critic. Max ROUNDS total (§7).
             v                    If still below threshold at the cap: STOP, report
          Done                    to the human, do not loop indefinitely or ship
                                   silently.
```

Phase 0 is never delegated, for the same reason it wasn't in the OceanNLP build: every
downstream builder needs an *identical, unambiguous* contract to avoid integration conflicts
that only surface after everyone's already written code against slightly different
assumptions. If any part of Phase 0 is already partially designed, include a worked
reference implementation for the highest-risk piece directly in the frozen contract, not just
a method signature — builders adapt/harden/verify a real starting point faster and more
consistently than they design one from a bare interface. (`Planning/NLP/NLP_Library_Master_Plan.md`
Appendix A did this for `CosineSimilarity`/`TfidfWeighting`/`BM25Weighting`; every builder that
got a worked reference shipped closer to first-try-correct than the ones designing from
scratch.)

## 4. What goes in a frozen contract (Phase 0)

1. **Method-level API contract.** Concrete signatures, naming conventions, and where state
   lives. If the codebase already has a convention (this repo's is PyTorch-style: a pure
   functional core wrapped by a thin stateful class, `fit()`/`forward()` naming, learned state
   held as buffers on `self` and mutated in place rather than threaded through as a separate
   return value — see `feedback_nlp_pytorch_style.md` in memory), state it explicitly per
   class, don't assume builders will infer it consistently on their own. Where the contract
   is silent on some smaller detail (a type choice, an empty-value convention), builders
   should read already-completed sibling files and match, not reinvent — this repo's build
   converged independently on `strings(1,0)` for "no formula" and `cellstr` for
   `ComposesFrom` across agents with no coordination, purely by "read the sibling file first."
   That convergence is a signal the contract was clear enough; divergence is a signal to
   tighten Phase 0 next time.
2. **The domain-specific "no faking it" rule.** This repo's is R7 from
   `.agents/pipeline-architecture-spec.md`: an operator without a valid closed-form
   representation is explicitly tagged non-symbolic, never given a fabricated formula. Every
   domain has an equivalent — name it in Phase 0, don't leave it implicit.
3. **The 3-pillar contract, when it applies** (`.agents/pipeline-architecture-spec.md` §1.1):
   every operator needs a numeric map (`f`), a symbolic representation (`φ`) or an honest
   `IsSymbolic=false`, and a diagram/port spec (`𝒟`). State per-class, in Phase 0, which
   pillars are required and which formula/port-domain values are expected — don't leave
   pillar 1/3 as a follow-up retrofit if it can be specified up front instead (it was a
   retrofit in this build's history; specifying it in Phase 0 next time is strictly cheaper
   than a whole second addendum pass).
4. **Software design pattern the contract expects**, when relevant — e.g. "this varies along
   exactly one axis, so it should be a Strategy behind an interface + factory, matching
   `IActivation`/`ISimilarity`/`ITokenizer`/`IIdfScheme`," stated in Phase 0 rather than
   discovered as a refactor afterward. Recognizing the axis of variation up front (what
   actually differs between the classic/smooth/BM25 idf formulas — only `compute(df,N)` —
   versus what stays fixed — the owning class's `fit`/`forward` shape) is the actual design
   skill; naming the pattern is just the label for having done that analysis correctly.
5. **Sparse-safety / the domain's equivalent robustness checklist**, if the domain has one —
   this repo's: never assume an operation preserves sparsity, verify empirically with
   `issparse()` on a real sparse input, and know the specific anti-pattern (`X + scalar`
   densifies; `X .* vector` or `X ./ vector` usually doesn't, because zero stays zero under
   multiplication/division but not under addition — this is why the BM25 single-document bug
   was a `find()`-orientation bug and not a "forgot sparsity" bug: the sparse-safe pattern was
   already correctly applied, just with the wrong vector orientation on one edge case).
6. **Package/file layout**, matching existing repo conventions rather than inventing new ones
   per task.

## 5. Between "builders finish" and "critic runs"

### 5.1 Test authoring
Every ad hoc verification a builder ran interactively becomes a real, persisted, executed
test — not re-derived from scratch, but formalized. If something turns out to be
untestable (this build hit exactly one case: a constructor's error-path guard that never
actually fires because the underlying platform is more permissive than expected), **do not
write a test that asserts a false pass.** Document the limitation in the test file and move
on. A known gap is honest; a green checkmark on an untested path is a liability with a
delay timer on it.

### 5.2 Integration verification
Unit-correct components can still fail when composed — this is where this build's one real
bug surfaced (a single-document corpus crashed `BM25Weighting`, caught only when the
`Retrieval` layer's builder exercised it end-to-end, not by either component's own unit
tests). Budget for this as its own step, not an assumed side effect of unit testing.

### 5.3 External validation (when the domain calls for it)
For anything implementing a published algorithm: reproduce it independently — ideally in a
different language/library, on real public data, not synthetic data engineered to make the
implementation look right. Isolate what's being checked: feed *identical* upstream inputs
(e.g. the same tokenized count matrix) to both sides so a mismatch means the *formula*
differs, not the preprocessing. When two legitimate variants of the "same" published
algorithm exist (this build's case: Lucene/Elasticsearch's non-negative BM25 idf vs.
`rank_bm25`'s classic Robertson formula), validate against the variant the implementation
actually documents itself as, and report the other as context, not as a failure.

## 6. Critic rubric template

Dimensions are domain-specific; the shape is not. Five dimensions, weighted evenly, summing
to 10, is what this build used and is a reasonable default:

| # | Dimension | 2 pts | 1 pt | 0 pts |
|---|---|---|---|---|
| A | Architecture/contract conformance | Matches Phase 0 exactly; no scope leaking between components | Minor placement/naming drift | A real boundary violated |
| B | API/style correctness | Fully matches the stated convention (e.g. PyTorch-style, §4.1) everywhere | One class missing one convention | No consistent convention followed |
| C | Correctness (numeric/logical) | Independently re-derived by the critic, own examples, matches | Correct, but an edge case unverified | Wrong, or an untested claim of correctness |
| D | Domain robustness (e.g. sparse-safety, thread-safety, whatever applies) | Empirically confirmed by the critic on the critic's own inputs | Safe in the common case, one edge-path leak | A known anti-pattern present |
| E | Tests & docs | Full coverage, reproduced by the critic from a cold run | Coverage exists but incomplete, or docs thin | Missing |

### 6.1 What the critic must actually do, not just read
- Re-run the full test suite itself; don't trust a builder's "N passed" claim.
- Re-derive at least one non-trivial example per correctness-bearing component itself, with
  its own input, computed before looking at the implementation's output.
- Empirically verify the domain's robustness property on its own constructed input, not by
  reading the code and reasoning it should hold.
- Read every file end to end at current on-disk state — not a summary, not the Phase 0 draft
  (implementations legitimately evolve past the initial contract; review what's real).

### 6.2 Findings format
Every finding: **severity** (`blocker` / `should-fix` / `nit`) + **owning builder** + what's
actually wrong (not "consider improving X" — state the concrete failure scenario). A finding
without an owner or a concrete failure scenario is not actionable and shouldn't gate a round.

## 7. The loop

1. Critic scores. `>= threshold` (this build used 8/10): done, no more rounds needed even if
   individual nits remain open — a nit is explicitly *not* blocking by definition; don't let
   the loop become "keep going until zero findings of any severity," that's a different,
   unbounded process.
2. `< threshold`: route *only* the tagged findings back to *only* the owning builder(s) — not
   a full re-build, not other builders whose work wasn't flagged. Each routed finding should
   be fixed and every one of that component's *original* acceptance criteria re-verified
   (not just the new fix in isolation), since a fix that breaks something the original build
   already validated is a net loss.
3. Re-run the critic — a fresh invocation again, not the same critic continuing its own
   thread, for the same self-grading reason builders and critics are never the same agent.
4. Cap at a fixed number of rounds (this build used 3). If still below threshold at the cap:
   **stop and escalate to the human** with the outstanding findings. Do not attempt a 4th
   round without explicit sign-off, and do not silently ship a sub-threshold result either —
   both are worse than surfacing the gap and asking.

## 8. Definition of done

- Every Phase 0 contract item is satisfied and verified, not assumed.
- Critic score `>= threshold` within the round cap, or an explicit human decision to accept
  less / extend rounds.
- A short, *actually executed* end-to-end usage example recorded in the plan's living log —
  proof the pieces compose, not just that each piece passed in isolation.
- The plan document itself stays live during the build (status per phase, updated as it
  actually happens) rather than being written once upfront and left to drift from reality.

## 9. Anti-patterns observed (and avoided) in the session this was extracted from

- **Grading your own homework**: never have a builder review its own package, or a critic
  reuse a builder's conversation context.
- **Fabricating a pass**: a test that asserts something false to turn red into green, or a
  formula documented as closed-form when it isn't (§4.2's "no faking it" rule) — both are
  worse than an honest gap, because they cost more to discover later.
- **Vague findings**: a critic finding with no severity, no owner, or no concrete failure
  scenario just generates a clarification round instead of a fix round.
- **Unbounded iteration**: looping past the round cap without escalating turns a
  quality gate into an unmonitored background process. Stop and ask.
- **Process for its own sake**: running the full swarm-plus-critic machinery on a change
  small enough to verify directly (§1) — the workflow has real overhead (agent-launch cost,
  round-trip latency, a plan document to maintain); reserve it for work that actually needs
  independent parallel verification.

## 10. Next application of this workflow (context, not the plan itself)

The task that prompted extracting this document: inverted-index-based retrieval, IR
evaluation metrics (precision/recall@k, MAP, NDCG, MRR — filling a gap explicitly identified
as missing from the current `OceanNLP` build), and dense-vector retrieval across multiple
BERT-family embedding models. That work gets its **own** Phase-0-through-critic plan document
(same shape as `Planning/NLP/NLP_Library_Master_Plan.md`), written *using* this workflow, not
folded into this file — this file is the reusable process; the next plan is one more
application of it, complete with its own frozen contracts (continuing the 3-pillar +
PyTorch-style + Strategy-pattern conventions already established), its own rubric, and its
own Python-based external validation phase (e.g. against `pytrec_eval`/`ir_measures` for the
evaluation metrics, and `sentence-transformers`/`transformers` for the dense embedding
baselines).
