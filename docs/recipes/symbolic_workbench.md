# MATLAB Symbolic Workbench

From the Ocean Library root in MATLAB, run:

```matlab
app = OceanTools.SymbolicWorkbench.launch();
```

Requires MATLAB and Symbolic Math Toolbox. Python is supported as an imported source language; no Python runtime is invoked. No browser server or API key is needed for local extraction and symbolic mathematics.

## Extract symbols from MATLAB or Python

Select **Import file...** and choose a `.m` or `.py` file. The workbench reads source text but never executes it. Each discovered function is classified as:

- **transparent** when a straight-line output formula can be traced safely;
- **opaque** when the callable interface is useful but control flow or implementation details cannot be reduced safely; or
- **not extractable** when no mathematical output contract is available.

Review the function signature, exact source lines, confidence, rationale, and formula candidates. An exact-code formula and a documentation-derived formula are separate evidence. Accept a formula only when it represents the function you need; otherwise accept the function as an abstract operator such as `optimize(loss, theta0)`. Opaque calls can be composed and contract-checked, but they cannot be expanded or computed through without an explicit mathematical rule.

Accepted names are collision-safe (`f`, `f_2`, and so on), and every rename is announced. Declare input/output domains and shapes in the contract editor before composition. Unknown compatibility remains **unknown**; it is never reported as passed. Only a transparent expression whose contracts explicitly say every input and output is scalar can be handed to the MATLAB symbolic calculator.

The browser extractor can export a `.symbolic.json` handoff containing source text and candidates. The native workbench recognizes this packet and safely rescans its embedded source; it does not trust the packet's precomputed claims. The workbench's own JSON format remains the trusted format for saved workspace symbols and definitions.

## First calculation

Select **Gaussian example** to explicitly create a mathematical example, not data extracted from a file. Its standard deviation is positive. The compact call retains access to the complete expression and contract checks. Select **differentiate**, choose `x`, and calculate. Use **substitute** with `mu=0, sigma=1` to specialize the expression.

Alternatively declare symbols, enter an expression with explicit multiplication (`2*x`, not `2x`), then calculate. Supported scalar operations are inspection, simplification, expansion, factorization, differentiation, indefinite integration, polynomial solving (degrees 1–4), substitution and numerical evaluation. Integration results omit the arbitrary additive constant. Factor returns MATLAB's vector of factors. Matrix algebra, units, arbitrary code execution, and complete algorithm-to-formula translation are not supported.

## Reusable blocks from scripts

```matlab
w = OceanMath.Symbolic.Workspace();
w.addSymbol("x", "real");
w.addSymbol("mu", "real");
w.addSymbol("sigma", "positive");
w.define("G", ["t","m","s"], ...
    "exp(-((t-m)/s)^2/2)/(s*sqrt(2*pi))", ...
    ["real","real","positive"], "positive");
r = w.inspect("G(x,mu,sigma)");
disp(r.compact)
disp(r.expanded)
d = w.calculate("G(x,mu,sigma)", "differentiate", "x");
disp(d.result)
```

Input arity and parameter/output domains are checked. Definite violations fail; undecidable claims are marked **unknown**, not passed. These scalar contracts are not a proof of a source algorithm's correctness. Symbol assumptions are isolated from your interactive MATLAB symbols.

Duplicate names receive a suffix such as `G_2`. Notices identify the change. Explicit renaming updates dependent block calls; local parameter names are protected against accidental capture. Keep the returned actual name when scripting:

```matlab
actualName = w.rename("G", "density");
disp(w.Notices)
bundle = w.exportBundle();
other = OceanMath.Symbolic.Workspace();
renames = other.importBundle(bundle);
```

## Import and provenance

Imported source functions and their formulas retain file and line provenance. Control-flow-heavy functions remain callable abstract nodes rather than being falsely translated. Use the original source to verify assumptions and meaning before accepting an interface or formula.

Save/import the workbench's own JSON bundles to retain symbols, definitions, contracts and provenance. An HTML source-inspector manifest is not itself a symbolic bundle.

## Optional AI

The AI action sends the imported source text, source path and question to OpenAI only after the user requests it. API charges may apply. Configure `OPENAI_API_KEY` in MATLAB's environment securely; never put it into a source file or shared bundle. No key is needed for other actions.

AI drafts are proposals, not source facts or verified mathematics. Review their cited lines, assumptions, limitations, parameters and domains before creating a block. A safe mathematical parser then checks accepted expressions; neither imported code nor generated code is evaluated. Live API behavior requires a configured key and network connection.

## Verification

Validated with MATLAB R2026a and Symbolic Math Toolbox. Tests cover symbolic operations, rejected code, contracts, hygienic rename and import, MATLAB/Python static extraction, transparent and opaque interfaces, the repository Entropy implementation, typed composition, formal opaque derivatives, provenance, AI draft validation, and native UI bindings/cleanup. Live OpenAI requests were not exercised without credentials.

To rerun the tests and regenerate both native window previews:

```matlab
addpath('Experiments/Scripts/NLP');
results = verify_symbolic_workbench();
```
