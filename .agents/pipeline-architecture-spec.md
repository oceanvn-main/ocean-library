# Architecture spec: hierarchical, dual-representation signal pipeline

> [!IMPORTANT]
> **Living Specification & Evolutionary Architecture**: This architecture specification is a living, non-rigid reference document. It is not fixed or perfect and will be modified, expanded, or refined as necessary during library implementation and experiment development. Any AI agent working on this codebase should understand that architectural patterns and contracts here may evolve dynamically to meet practical requirements.

## 1. Purpose & Core Contract

A system where computational pipelines (across Signal Processing, Time Series, Computer Vision, NLP, and ML) can be built by dragging components onto a canvas and wiring them, where every component carries a numeric implementation, a mathematical (LaTeX) representation, and a self-describing port diagram specification.

### 1.1 The Mandatory 3-Pillar Component Contract: $\mathcal{O} = (\phi, f, \mathcal{D})$

Every component added to the library MUST implement three unified representations:

1. **Pillar 1: Symbolic Representation ($\phi$) — "Why & What"**:
   - Closed-form LaTeX formulas (`SymbolicLatex`), textbook definitions (Hamilton, Oppenheim, Cover & Thomas, Hastie).
   - Symbol Interpretation Dictionary (`Symbols` struct): maps every LaTeX symbol token in `SymbolicLatex` (e.g., `V_n`, `\alpha`, `\mu_n`, `W_n`) to its plain-language description for UI/extractor rendering.
   - Formal mathematical assumptions ($\mathcal{A}$: linear, stationary, causal, positive-definite).
   - Exposed via `getMetadata().SymbolicLatex` and `getMetadata().Symbols`.

2. **Pillar 2: Computational Representation ($f$) — "How & Execute"**:
   - High-performance numeric forward execution pass (`compute()`, `fit()`, `simulate()`, `forward`).
   - Vector/matrix numerical evaluation, residual calculation, and error diagnostics.

3. **Pillar 3: Diagram & Architectural Specification ($\mathcal{D}$) — "Wiring & Canvas"**:
   - Enumerated `PortSpec` input/output signatures (`Inputs`, `Outputs`, domains, data types, tensor shapes).
   - Structural grouping kind (`VIRTUAL` composite vs. `NONVIRTUAL` stateful composite).
   - Flexible diagram visual metadata (`DiagramSymbol`): specifies which symbol/icon to render on canvas (supports custom loaded/imported SVG/image assets, mathematical glyphs, or defaults to a clean Simulink-style rectangular block displaying the function/operator name in the center).
   - Exposed via `getPipelineSpec()` or prototype registration to enable programmatic Simulink block generation and web diagram canvas rendering.

---

## 2. Formal requirements

| # | Requirement | Formal statement |
|---|---|---|
| R1 | 3-Pillar Representation | Every operator MUST provide a numeric map `f: X → Y`, a symbolic LaTeX/closed-form map `φ`, and a diagram port spec `𝒟` |
| R2 | Composability | `φ` must satisfy `φ_(T2∘T1) = φ_T2 ∘ φ_T1` wherever both sides are defined |
| R3 | Hierarchical abstraction | Any subgraph with a finite, enumerated input/output boundary can be packaged into a named composite |
| R4 | Substitution transparency | `f_composite(x) = expand(composite).f(x)` for all valid `x` — folding is lossless |
| R5 | Bounded rendering | Rendering cost at any moment is `O(k)`, `k` = siblings at the current frontier, independent of total node count |
| R6 | Interactive authoring | Components are dragged from a palette onto a canvas; connections are drawn port-to-port, validated at draw time |
| R7 | Honest opacity | Operators without a valid `φ` are explicitly tagged non-symbolic, never silently faked |


Everything below exists to satisfy these seven; nothing else is in scope for v1.

## 3. Core data model

### 3.1 Prototype vs. instance

A **prototype** is a registered operator *type* — it declares its port signature before anything else exists (mirrors `mdlInitializeSizes`):

```python
@dataclass(frozen=True)
class PortSpec:
    name: str
    domain: str          # 'discrete-time', 'freq-domain', 'continuous-time', ...
    dtype: str            # 'real', 'complex', 'int'
    shape: tuple | None   # None = unconstrained/inferred

@dataclass(frozen=True)
class OperatorPrototype:
    id: str                                   # unique registry key
    inputs: tuple[PortSpec, ...]
    outputs: tuple[PortSpec, ...]
    forward: Callable                         # f: numeric
    expr_fn: Callable | None                  # φ: sympy expr(s) -> expr(s), or None
    assumptions: dict                         # 𝒜: linear, causal, stationary-input-required, ...
    is_symbolic: bool                         # False forces opaque rendering (R7)
```

An **instance** is a prototype dropped onto the canvas, with concrete parameter values and a canvas position:

```python
@dataclass
class OperatorInstance:
    instance_id: str
    prototype_id: str
    params: dict
    position: tuple[float, float]
```

### 3.2 Signal (edge)

```python
@dataclass
class Signal:
    id: str
    source: tuple[str, str]   # (instance_id, output_port_name)
    target: tuple[str, str]   # (instance_id, input_port_name)
    domain: str                # resolved during compile, see §6.1
    symbol: object | None      # sympy expression, resolved during compile
```

### 3.3 Composite node (the abstraction mechanism)

```python
class CompositeKind(Enum):
    VIRTUAL = "virtual"        # pure grouping — flattens for free, no separate semantics
    NONVIRTUAL = "nonvirtual"  # has real execution semantics of its own (state, own rate)

@dataclass
class Composite:
    id: str
    name: str                             # the "named block" — e.g. Φ
    kind: CompositeKind
    boundary_inputs: tuple[PortSpec, ...]
    boundary_outputs: tuple[PortSpec, ...]
    subgraph: "PipelineGraph"             # always retained — R4 requires this never be discarded
    icon_label: str | None = None          # mask-equivalent display override
```

The `kind` field is the explicit choice the Simulink review surfaced: a `VIRTUAL` composite is guaranteed substitution-transparent by construction (it's just a grouping, flattened mechanically at compile time — no extra test needed beyond the round-trip check in §8). A `NONVIRTUAL` composite is a genuinely new operator with its own state or timing and **must** supply its own `φ` and its own correctness tests — it does not get R4 for free.

### 3.4 Pipeline graph

```python
@dataclass
class PipelineGraph:
    nodes: dict[str, OperatorInstance | Composite]
    edges: dict[str, Signal]
```

A `Composite.subgraph` is itself a `PipelineGraph` — this recursive definition is what makes the structure a *tree of graphs* rather than a flat graph, and is the single data-structure decision that makes R5 achievable.

## 4. System layers

```
┌─────────────────────────────────────────────┐
│ UI layer          palette · canvas · inspector · zoom navigator │
├─────────────────────────────────────────────┤
│ Rendering layer    LatexRenderer · GraphRenderer · ValuePreview  │
├─────────────────────────────────────────────┤
│ Compiler layer     attribute propagation · flatten · sort · validate │
├─────────────────────────────────────────────┤
│ Execution layer    numeric evaluation, per compiled schedule     │
├─────────────────────────────────────────────┤
│ Domain layer       PipelineGraph · Composite · OperatorInstance · Signal │
├─────────────────────────────────────────────┤
│ Persistence layer  serialization, prototype registry             │
└─────────────────────────────────────────────┘
```

Strict rule: each layer only talks to the layer directly below it. The UI layer never touches the execution layer directly — it goes through the compiler's output. This is what keeps rule "renderer never depends on more than one level of the tree" enforceable rather than aspirational.

## 5. Component inventory — what you actually need to build

| Component | Responsibility | Depends on |
|---|---|---|
| `PrototypeRegistry` | Register/lookup operator prototypes by id; enforce unique namespacing (rule 5 from earlier) | — |
| `GraphStore` | In-memory + persisted `PipelineGraph`, including nested composites | Persistence layer |
| `PortValidator` | Check type/domain/shape compatibility when a wire is drawn (§6.1, incremental) | `PrototypeRegistry` |
| `Compiler` | Full compile pass: attribute propagation, virtual-subsystem flattening, topological sort, validation (§6.1–6.2) | `GraphStore` |
| `Executor` | Run the compiled schedule, produce numeric values per signal | `Compiler` output |
| `LatexRenderer` | Visitor producing named-intermediate LaTeX per node/subgraph (§6.4) | `GraphStore`, SymPy |
| `SubstitutionChecker` | Test harness verifying R4 for every composite (§8) | `Compiler`, `Executor` |
| `LayoutCache` | Per-node cached layout, invalidated only on that node's own change (§6.3) | `GraphStore` |
| `ZoomNavigator` (frontend) | Drill-down/breadcrumb state machine; camera animation between levels | `LayoutCache`, `GraphStore` |
| `Palette` (frontend) | Lists registered prototypes + user-defined composites available for drag | `PrototypeRegistry` |
| `Canvas` (frontend) | Renders current frontier only; handles drag-drop, wire-drawing | `ZoomNavigator`, `PortValidator` |
| `Inspector` (frontend) | Three-panel signal view (symbol / value / provenance) from the previous design | `LatexRenderer`, `Executor` |

## 6. Key algorithms

### 6.1 Attribute propagation and port validation

Directly modeled on Simulink's compile phase. Two triggers:

- **Incremental (draw-time)**: when a wire is proposed, check `source.dtype/domain/shape` against `target`'s declared `PortSpec`; reject or flag on mismatch before the edge is committed.
- **Full (compile-time)**: forward topological walk over the flattened graph; each node's output attributes are computed from its declared `PortSpec` plus its inputs' resolved attributes (exactly "propagates the attributes of a source signal to the inputs of the blocks it drives," generalized to also carry the `Signal.symbol`).

### 6.2 Flatten + sort (compile)

1. Recursively replace every `VIRTUAL` composite with its `subgraph`'s contents (safe by construction — see §3.3). `NONVIRTUAL` composites are *not* expanded; they remain atomic execution units with their own scheduling slot.
2. Topologically sort the resulting flat graph to get execution order. Cycles are only valid through an explicit feedback/delay construct (never an implicit loop) — reject otherwise.
3. Emit a compiled schedule: an ordered list of `(node, resolved_inputs)` the executor runs in sequence.

### 6.3 Fold/unfold with the layout cache

- Folding a subgraph into its parent composite's box never discards the subgraph (R4/rule 3).
- `LayoutCache[node_id]` stores the computed 2D layout for that node's children, computed once on first expansion.
- Invalidation is local: editing inside composite `C` invalidates only `LayoutCache[C]`, never ancestors or siblings — this is what bounds rendering cost (R5) as the system grows.

### 6.4 LaTeX rendering — named intermediates by default

Per node: `y_i = φ_node(inputs)`, rendered as its own line, not substituted into neighbors. Full substitution across a user-selected span is available on demand only (§ prior discussion — expression swell is the failure mode this avoids). For `NONVIRTUAL` or non-symbolic nodes, render the declared name as an opaque function, `T(\cdot)`, with a plain-language note — never a fabricated formula (R7).

### 6.5 Zoom mechanics — the honest version

**Important scoping decision, stated explicitly rather than left implicit**: "Google Maps style" here means *level-based drill-down with animated camera transitions*, not literal continuous rendering of infinitely nested SVG at every zoom scale. Practically:

1. The canvas renders exactly one level of the tree at a time (current frontier), per R5.
2. Clicking a composite: camera animates toward that node's bounding box (CSS/canvas transform, cheap), then the renderer swaps in the child `PipelineGraph` from `LayoutCache`, and the camera settles — the animation is what creates the "zoom" feeling; the actual work done is a level swap, not a continuous re-render of the whole tree.
3. Breadcrumb trail (`system > filter bank > filter 1`) gives direct navigation back up, matching rule 5's scoped-naming requirement.

This is the same technique Figma/Miro/draw.io use for "zoom into frame" — it is the practical realization of the Google Maps UX, not a literal reimplementation of tile-based mapping.

## 7. Implementation stack

**Backend** (Python — matches your existing stack):
- `NetworkX` for the flattened-graph topological sort
- `SymPy` for `φ` and `sympy.latex()`
- `NumPy`/`SciPy` for numeric execution
- `Pydantic` for schema validation of prototypes/instances (also gives you JSON schema for free, useful for §9)
- `FastAPI` (or similar) exposing: register-prototype, get-graph, compile, run, get-node-latex, get-signal-inspector

**Frontend**:
- React + TypeScript
- Canvas/graph rendering: a node-based editor library (e.g. React Flow) for the palette-drag-and-wire interaction *within one level*; the drill-down/breadcrumb navigation across levels is custom state on top of it (§6.5) — do not rely on the library's built-in continuous zoom to deliver R5, it doesn't by itself
- LaTeX display: KaTeX or MathJax for rendering the strings `LatexRenderer` returns

**Alternative for a MATLAB-only path**: MATLAB App Designer + Symbolic Math Toolbox can implement the same layered model, but the graph/zoom/UI ecosystem is considerably thinner than the Python/web stack — recommend Python/web for the tool itself, and treat MATLAB/Simulink as reference architecture rather than implementation target, unless there's a hard requirement to stay inside MATLAB.

## 8. Testing strategy (enforces §2 mechanically)

| Test | Checks | Required for |
|---|---|---|
| Numeric correctness | `f` matches expected output on reference inputs | every prototype |
| Compositionality | `φ` composes correctly with at least one real neighbor | every prototype with `is_symbolic=True` |
| Round-trip (substitution transparency) | `f_composite(x) == expand(composite).f(x)` | every composite, both kinds |
| Port validation | Rejects mismatched domain/dtype/shape at draw time | `PortValidator` |
| Layout invalidation scope | Editing node `C` does not invalidate cache entries outside `C` | `LayoutCache` |
| Opacity honesty | Non-symbolic nodes never emit a `φ`-derived expression | `LatexRenderer` |

## 9. Serialization format (sketch)

A model is a JSON document mirroring §3 directly — this is your `.slx`-equivalent:

```json
{
  "prototypes_used": ["lti_filter", "dft"],
  "graph": {
    "nodes": {
      "n1": {"type": "instance", "prototype_id": "lti_filter", "params": {...}, "position": [x, y]},
      "c1": {"type": "composite", "kind": "virtual", "name": "filter_bank",
             "boundary_inputs": [...], "boundary_outputs": [...],
             "subgraph": { "...": "recursive PipelineGraph" }}
    },
    "edges": {
      "e1": {"source": ["n1", "out"], "target": ["c1", "in0"]}
    }
  }
}
```

## 10. Rules for anyone (or any AI) coding in this repo

Consolidated from the earlier discussion, made concrete against this data model:

1. No `OperatorPrototype` may be registered without a fully enumerated `inputs`/`outputs` port signature.
2. No composite skips the substitution-transparency test in §8, regardless of `kind`.
3. `Composite.subgraph` is never deleted or omitted when folding — if a code path can't produce it, that code path is wrong.
4. `is_symbolic=False` is a first-class, explicit state — never a fallback triggered by an exception in `expr_fn`.
5. Prototype and composite names are namespaced to their defining scope; the registry rejects collisions rather than silently shadowing.
6. UI/render code imports only from the rendering layer's public interface — never reaches into `Compiler` internals or a composite's grandchildren.
7. `LayoutCache` invalidation touches only the node that changed; a PR that invalidates a parent or sibling cache on unrelated edits should be rejected in review.
8. A `VIRTUAL` composite's icon/label/summary is computed from its subgraph at render time, never hand-authored and stored separately.
9. Changing a composite's `boundary_inputs`/`boundary_outputs` is a breaking change requiring a version bump; internal edits to `subgraph` that preserve the boundary are not.
10. No PR merges a new prototype or composite without all six tests in §8 passing.

## 11. Suggested build order

1. Domain layer (§3) + `PrototypeRegistry` + in-memory `GraphStore` — no UI yet.
2. `Compiler` (§6.1–6.2) against a hand-built test graph (3–4 signal-processing prototypes: gain, FIR filter, DFT).
3. `Executor` + numeric correctness tests.
4. `LatexRenderer` for the same 3–4 prototypes; verify named-intermediate output by hand.
5. Serialization (§9) — round-trip save/load.
6. Minimal canvas: palette + drag + wire-draw + `PortValidator`, single level, no zoom yet.
7. Composite creation (select subgraph → package as `VIRTUAL` composite) + substitution-transparency test harness.
8. `LayoutCache` + `ZoomNavigator` drill-down/breadcrumb (§6.5).
9. Signal inspector panel (three-view, from the earlier design).

Steps 1–5 are pure backend and fully testable without any UI — worth resisting the pull to build the canvas first, since the compositionality guarantees are what the whole system depends on, and they're cheapest to get right before there's a UI built on top of them.

## 12a. Extensibility hooks: parameters and representation levels

Not implemented in v1, but the data model below must exist from the start so neither capability requires a redesign later.

**Hyperparameter controls.** A prototype's tunable parameters are declared, not smuggled into an untyped `params` dict:

```python
@dataclass(frozen=True)
class ParamSpec:
    name: str
    symbol: str                 # LaTeX symbol, e.g. r'\beta', r'\epsilon'
    default: float
    range: tuple[float, float] | None
    scale: str = "linear"       # or "log"
    live: bool = False          # eligible for a UI slider

OperatorPrototype.params: tuple[ParamSpec, ...]
```

Because the spec lives on the prototype, the UI can render a control for any node generically — no per-operator UI code. `LatexRenderer` must support two render modes for any node: **general form** (parameters left as free symbols — the form for understanding the math) and **instantiated form** (parameters substituted with current values — the form for a specific run). Both are legitimate views of the same `φ`; neither replaces the other.

Extend the invalidation rule from §10 rule 7 to computed values, not just layout: changing a `live` parameter on node `N` invalidates `N` and its topological descendants only — never upstream nodes, siblings, or unrelated subgraphs. This is what keeps a slider drag cheap regardless of total pipeline size (R5, applied to recompute cost).

**Representation level.** Every port distinguishes the *operand representation* (what `f` actually consumes/produces — governs `PortValidator` compatibility) from the *display representation* (what a human sees in the Inspector). These coincide for most DSP/vision signals (a pixel or sample tensor is already legible) and diverge for NLP signals (token ids/embeddings/logits require a real, sometimes lossy, sometimes stochastic transformation to become text).

```python
@dataclass(frozen=True)
class PortSpec:
    name: str
    domain: str
    dtype: str
    shape: tuple | None
    decode_fn: Callable | None = None   # operand -> display representation; identity where they coincide
```

`decode_fn` must always be declared explicitly, even when it's `identity` — never omitted or defaulted. Decoding never happens implicitly inside the rendering layer (rule 6 already forbids this; this just names the specific violation to watch for): a lossy or stochastic transformation (argmax + detokenize) is real computation and must be a registered, inspectable function like any other operator, not logic hidden inside a UI component.

New rules for §10:

11. Every prototype's tunable parameters are declared via `ParamSpec`; nothing UI-adjustable is read from an untyped dict.
12. Value-cache invalidation (like layout-cache invalidation) touches only the changed node and its downstream descendants.
13. `PortSpec.decode_fn` is always explicit. A port with no real decode step still declares `identity` — it is never left unset.

## 12. Assumptions and domain of validity

This architecture assumes: operators are predominantly closed-form or template-parameterized (true for classical signal processing — filters, transforms, modulation); pipelines are primarily feed-forward with explicit, bounded feedback constructs where needed; and the primary cost driver for scaling is component *count*, not edge density (a graph with very high fan-out per node will stress §6.5's `O(k)` claim — worth flagging if that becomes a real use case, since it would need an additional fan-out-bucketing step not covered here).
