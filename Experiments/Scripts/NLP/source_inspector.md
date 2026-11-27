# Source inspector

For the native MATLAB symbolic environment, run `app = OceanTools.SymbolicWorkbench.launch();` from the library root in MATLAB. See [the symbolic workbench recipe](../../../docs/recipes/symbolic_workbench.md) for named blocks, domain checks, and optional AI drafts. The browser inspector below remains source-only.

Open `component_extractor.html` in a browser and choose **Open a source file**.
Keep `component_extractor.css`, `component_extractor.js`, and `source_inspector_core.js` beside the HTML. No installation or internet connection is required for local files or pasted source. Files are read as UTF-8 text, with a 2 MB limit.

To use library-relative server paths, serve the Ocean Library root and open `/Experiments/Scripts/NLP/component_extractor.html`. Paths are resolved from that server root. A stopped server, missing path, timeout, or HTML fallback produces an explicit error; local file inspection remains available.

## Coverage

| Languages | Extracted evidence |
| --- | --- |
| MATLAB | Classes, functions, declared return names, returned struct fields, argument/property blocks, metadata, documentation, recognized dependency references |
| Python | Classes, functions, multiline signatures, annotations, explicit returns and returned dictionary fields, imports, comments and docstrings |
| JavaScript / TypeScript | Classes, functions, common methods and arrows, explicit returns and returned object fields, imports/re-exports/requires, annotations and comments |
| Java, C, C++, C#, Go, Rust, R | Common declaration/import/return/comment outlines; advanced syntax and language-specific contracts may be missed |
| Other text files | Literal text, with a manual language override |

Filename extensions select a language. The Language selector can override the choice, including ambiguous extensions such as `.m` and `.h`.

This is a bounded lexical inspector, not a compiler or full semantic analyzer. It separates comments and string literals before recognizing supported syntax. It does not execute source, import its dependencies, establish runtime shapes, prove absence of side effects, or establish mathematical purity. Known coverage limits and inventory caps appear in the page and exported manifest.

## How to read the result

- **File interpretation** gives a cautious assessment, supported by source lines: numerical operations, symbolic operations, interfaces, plotting calls, or insufficient evidence.
- **What could be plotted?** traces possible numeric outputs and recognizes plotting call syntax. Every suggestion includes its evidence and the data, dimensions, dependencies, or labels still needed. No sample measurements, covariance values, charts, or regularity results are generated.
- **Structure & contracts** organizes declarations, output candidates, validation/metadata, and imports. An output expression is source text, not its evaluated result.
- **Documentation & mathematical statements** preserves author-written comments and formula metadata as text. These describe intent, not verified behavior.
- **Read source & verify evidence** opens the selected line. Large files are paged; all loaded source text remains available here.
- **Export manifest** saves the current inspection as JSON. Export is disabled while loading or after a failed read, so a previous file cannot be mistaken for the current inspection.

The export schema is version 2.0: source identity and language coverage, declarations, outputs, dependencies, documentation, formulas, contracts, assessment, plot candidates, and warnings. Inferences live in `assessment` and `plotCandidates`; each includes source evidence. The full original source file is not embedded in the export.

## Verification

Run the portable parser checks from the library root:

```powershell
node --test --test-isolation=none Experiments/Scripts/NLP/source_inspector_core.test.js
```

`source_inspector.browser.test.cjs` additionally verifies offline local-file and paste workflows, output exports, source citations, language overrides, desktop and mobile layout, inert rendering of source text, HTTP errors, and competing file requests. It uses the bundled Playwright package and an installed Edge browser on this development machine; adjust the executable/runtime paths for another machine. Review screenshots are generated in `.inspector-review/`.
