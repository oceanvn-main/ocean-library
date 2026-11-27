# Method Metadata Standard for Ocean Library

## When to Apply This Rule

**Every concrete implementation of `IScatteredInterpolant`, `IModelStrategy`,
`ISplineStrategy`, `IMetricStrategy`, or any learnable/estimable algorithm class
MUST include a `static getMetadata()` method AND a companion `.meta.json` sidecar file.**

The metadata drives the GUI learner application (`OceanViz.App`). It must be accurate,
pedagogically clear, and include LaTeX equations for mathematical display.

---

## Required Metadata Fields

All fields are **required** unless marked optional.

```
id                 — MATLAB class short name (e.g. "ScatteredRBF")
display_name       — Full human-readable name
short_description  — One sentence
category           — Broad method family (e.g. "Kernel / RBF", "Tree-based", "Bayesian")
textbook           — Source reference
  .reference       — Full citation string
  .section         — Section identifier (e.g. "§4.1.1")
  .equations       — Array of equation labels (e.g. ["4.3", "4.8", "4.11"])

pedagogy           — All pedagogical content for the learner GUI
  .motivation      — Why this method exists. What problem prompted it?
                     Write for a learner who has never seen the method.
  .assumptions     — Array of bullet-point assumptions the method makes
  .data_model      — How the method mathematically represents the data.
                     Describe f(x), what is known, what is estimated.
  .what_is_missing — What information is unavailable that must be inferred?
                     This is the core estimation problem.
  .how_it_solves   — Plain-language description of the algorithmic idea.
                     Step-by-step if possible.

equations          — Key math equations as LaTeX strings.
                     Keys are short descriptive labels.
                     Values are objects with:
                       .label       — Short name shown in GUI
                       .latex       — LaTeX string (full equation, NOT display mode wrapper)
                       .description — What this equation means / when it is used

Symbols            — Symbol Interpretation Dictionary mapping every LaTeX math symbol token in SymbolicLatex / equations
                     (e.g., "V_{n-1}", "\alpha", "\mu_n") to a plain-language description for UI/extractor display.
                     Formatted as an array of structs:
                       [struct('Symbol', 'V_n', 'Description', 'Estimated value/mean at step n'), ...]

parameters         — Array of tunable parameter objects:
  .name            — Property name as it appears in MATLAB code
  .latex_symbol    — LaTeX for the symbol (e.g. "\\lambda", "c")
  .default         — Default value as a string
  .description     — What the parameter controls
  .effect          — How increasing/decreasing it changes the model behaviour
  .optional        — (optional) true if the parameter is optional

complexity         — Computational complexity
  .training        — Big-O string for fitting (e.g. "O(N^3)")
  .prediction      — Big-O string for predict (e.g. "O(MN)")
  .memory          — Big-O string for memory usage

strengths          — Array of bullet strings: when the method works well
weaknesses         — Array of bullet strings: when the method fails or is inappropriate
when_to_use        — One paragraph of guidance

references         — Array of citation strings
```

---

## File Locations

For a class `+PackageA/+PackageB/MyMethod.m`:
- **JSON sidecar**: `+PackageA/+PackageB/meta/MyMethod.meta.json`
- **Static MATLAB method**: `static getMetadata()` inside `MyMethod.m`

### Example static method (copy-paste template):

```matlab
methods (Static)
    function meta = getMetadata()
        % GETMETADATA  Return pedagogical metadata for GUI learner display.
        %   Loads the companion .meta.json sidecar file.
        here = fileparts(mfilename('fullpath'));
        raw  = fileread(fullfile(here, 'meta', 'MyMethod.meta.json'));
        meta = jsondecode(raw);
    end
end
```

---

## LaTeX Conventions

- Use `\\` for `\` inside JSON strings (JSON requires double-escape).
- Do NOT wrap equations in `$$` or `\[...\]` — the GUI renderer adds the display wrapper.
- Use `\\mathbf{}` for vectors and matrices.
- Use `\\|` for norm bars (`\\|\\mathbf{x}\\|`).
- Use `\\sum_k`, `\\phi`, `\\lambda`, `\\epsilon` for common symbols.
- For cases / piecewise: use `\\begin{cases}...\\end{cases}`.

### Example JSON equation block:
```json
"equations": {
  "interpolant": {
    "label": "Interpolant",
    "latex": "f(\\mathbf{x}) = \\sum_k w_k \\phi(\\|\\mathbf{x} - \\mathbf{x}_k\\|)",
    "description": "The predicted value at any query point x is a weighted sum of basis functions centered at each training point."
  }
}
```

---

## Quality Checklist

Before committing any algorithm class, verify:

- [ ] `static getMetadata()` method added
- [ ] `.meta.json` sidecar file created in `meta/` subfolder
- [ ] All 8 top-level fields present in JSON
- [ ] All `pedagogy` sub-fields are non-empty and written for a learner audience
- [ ] At least the primary equation is in `equations` with valid LaTeX
- [ ] `Symbols` array of struct objects present in `getMetadata()` mapping LaTeX symbols to plain-language descriptions
- [ ] Every tunable property appears in `parameters`
- [ ] `complexity.training` and `complexity.prediction` filled in
- [ ] JSON validated (no trailing commas, balanced braces)

---

## Rule Enforcement

This rule applies to:
- All classes in `+OceanMath/+Interpolation/`
- All classes in `+OceanStats/+Supervised/`
- All classes in `+OceanStats/+NonParametric/`
- All classes in `+OceanMath/+Information/+Strategies/`
- Any class implementing an algorithm that a learner would study

It does NOT apply to:
- Abstract interfaces (`IScatteredInterpolant`, `IModelStrategy`, etc.)
- Factory classes (`InterpolationFactory`, `ModelFactory`)
- Utility/helper classes (`InterpolationError`, `ScatterFitter`)
- Infrastructure classes (adapters, pipeline stages)
