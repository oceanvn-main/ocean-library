# RAG Pipeline Demo — Implementation Plan

## Architecture

**UIHTML app with a MATLAB backend.** A programmatic MATLAB `.m` entry point hosts the web frontend in a `uihtml` component. MATLAB retains ownership of corpus indexing, retrieval, prompt construction, and generation through `OceanNLP.RAG.RAGPipeline`; HTML/CSS/JavaScript provide the interactive interface and pipeline visualization.

**Serialization:** Standalone programmatic MATLAB `.m` entry point; companion local HTML, CSS, and JavaScript files.

**Layout:** Explorer — users configure and run a live RAG pipeline in a stable left-side setup panel, then inspect the pipeline flow and outputs in the main workspace.

## Structure

- **Setup sidebar:** Corpus text entry (documents separated by `---`), chunker selection and relevant parameters, retriever selection (BM25 lexical or TF-IDF/cosine sparse retrieval) with Top-K, generator mode (transparent no-LLM demo or local LLM endpoint/model), and explicit **Index corpus** / **Run query** actions.
- **Workspace header:** App identity, indexing state, active configuration summary, and status/error feedback.
- **Pipeline view:** A stable stage flow for chunking, indexing, retrieval, prompt assembly, and generation. Stage emphasis and status update during a run; the view distinguishes indexing stages from query-time stages.
- **Results area:** Answer, ranked retrieved chunks with scores and chunk IDs, and the final prompt, organized for inspection without changing the main layout.
- **MATLAB bridge:** Request/response events validate inputs and call the existing package APIs; MATLAB errors are returned to the UI as visible, actionable messages.

## Key behaviors

- Enter a small corpus and index it; show indexed state and chunk count.
- Choose a supported chunker (sliding, recursive, or sentence-window) and retrieval strategy (BM25 or TF-IDF/cosine); validate settings before constructing the pipeline.
- Submit a question only after indexing; display retrieved contexts, scores, assembled prompt, and generated answer from `OceanNLP.RAG.RAGResult`.
- Show pipeline progress and completion/error state. Avoid presenting estimated stage timings as measured telemetry.
- Configure the local OpenAI-compatible generation endpoint and model using `OceanNLP.Generation.LocalLLMGenerator`.
- Select a no-LLM demo generator implemented with `OceanNLP.Generation.FunctionGenerator` to exercise real indexing, retrieval, and prompt assembly without contacting an inference server; identify the returned response as a mock.
- Surface unsupported/missing backend dependencies as explicit errors rather than silently substituting a fake live answer.

## Existing backend integration

- Construct `OceanNLP.RAG.RAGPipeline` using `OceanNLP.Chunking.ChunkerFactory`, `OceanNLP.Retrieval.BM25Ranker` or `OceanNLP.Retrieval.SemanticRanker`, and the selected generation adapter.
- Use `RAGPipeline.index` and `RAGPipeline.query` for real execution; use `RAGResult` fields (`Answer`, `ChunkIds`, `Scores`, `Chunks`, `Prompt`) to populate the UI.
- Keep implementation within `+OceanNLP/+RAG/GUI`; do not change shared package behavior for presentation purposes.

## Internal references that will be used

| Reference | Role |
|---|---|
| `references/archetypes/explorer.md` | Explorer spatial skeleton and control/display responsibilities. [src: `references/archetypes/explorer.md` §Explorer / Control Panel Layout] |
| `references/uihtml/setup.md` | Full-figure `uihtml` setup, local source paths, and `setup()` contract. [src: `references/uihtml/setup.md` §MATLAB: Creating the uihtml Component; §JavaScript: The setup() Function] |
| `references/uihtml/bridge-guide.md` | Bridge invariants and implementation checklist. [src: `references/uihtml/bridge-guide.md` §Critical Rules; §Core Architecture] |
| `references/uihtml/communication-patterns.md` | Request-response events for UI actions and backend results. [src: `references/uihtml/communication-patterns.md` §Pattern 4] |
| `references/uihtml/data-types.md` | Safe MATLAB/JavaScript payload types and struct conversion. [src: `references/uihtml/data-types.md` §Type Conversion Table; §Serialization Rules] |
| `references/uihtml/error-handling.md` | Validate bridge requests and show errors on both sides. [src: `references/uihtml/error-handling.md` §MATLAB Side; §Input Validation at the Bridge Boundary] |
| `references/uihtml/styling-guide.md` | CSS Grid/Flexbox, design tokens, responsive and interactive styling. [src: `references/uihtml/styling-guide.md` §Critical Rules; §Design Tokens; §Layout Patterns] |

## External skill guidance

- `web-artifacts-builder` (user-provided GitHub skill link; not installed in this environment): apply its frontend craft guidance to the local UI—clear information hierarchy, deliberate visual restraint, and cohesive React/web-style interaction principles. Its Claude artifact initialization and single-file bundling workflow is not applicable to a MATLAB `uihtml` bridge, so no separate artifact bundle will be produced.
- No separate UX GUI design skill was found among the installed skills. The Explorer archetype and UIHTML design references above guide UX/layout.

## File organization

```text
+OceanNLP/+RAG/GUI/
├── rag-pipeline-demo-plan.md
├── RAGPipelineDemoApp.m     % MATLAB uifigure/uihtml host and RAG event handlers
├── index.html              % UI shell and semantic structure
├── styles.css              % Local design tokens, layout, responsive states
├── app.js                  % setup(htmlComponent), UI state, events, rendering
└── README.md               % Launch instructions and local backend requirements
```

## Implementation sequence

1. Confirm current constructor and method contracts for the selected chunkers, BM25 ranker, local generator, and RAG result against package source.
2. Establish the MATLAB host and `uihtml` source path using `setup.md`; create the stable Explorer shell and pipeline/results regions.
3. Implement UI request-response events using `communication-patterns.md`; validate payload fields/types per `error-handling.md` and send only bridge-safe structs.
4. Wire corpus indexing and query execution to the existing RAG classes; return index status and `RAGResult` details.
5. Style and make the frontend responsive using local CSS design tokens and Grid/Flexbox; provide loading, disabled, success, and error states.
6. Validate source structure and MATLAB syntax where the installed MATLAB runtime permits; exercise or document runtime/backend prerequisites.

## Scope and assumptions

- Initial retrievers are BM25 lexical ranking and TF-IDF/cosine sparse vector ranking; dense/hybrid strategies require additional encoder and fusion configuration and are not exposed in this first UI.
- Initial live generator is the existing local OpenAI-compatible endpoint adapter; alternatively, the demo mode uses a transparent no-LLM mock generator to validate the preceding pipeline stages.
- Corpus entry uses pasted text with a documented `---` document delimiter; file import can be added later if desired.
- RAGPipeline currently returns final query artifacts, not per-stage timing/trace events. The visualization will reflect actual execution states and returned artifacts, not fabricate fine-grained telemetry.
