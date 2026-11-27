# OceanNLP RAG Pipeline Demo

An interactive MATLAB UIHTML app for indexing a small text corpus, asking questions against it, and inspecting the real retrieval contexts, generated answer, and assembled prompt.

## Launch

```matlab
addpath('C:\Ocean Library\Matlab code practice\+OceanNLP\+RAG\GUI')
RAGPipelineDemoApp
```

The app adds the parent folder of `+OceanNLP` to the MATLAB path when it starts. Any separate dependencies used by your OceanNLP installation must also be available on the MATLAB path.

## Use

1. Edit the example corpus or paste your own documents. Separate documents with a line containing exactly `---`.
2. Choose a chunking strategy, retrieval strategy (**BM25 · lexical** or **TF-IDF · cosine**), and Top-K. Select **Demo (no LLM)** to test indexing, retrieval, and prompt assembly without a local inference server, or choose **Local LLM** and configure its endpoint/model.
3. You can type a question at any time. Select **Index corpus** before running it; **Run query** stays disabled until indexing succeeds.
4. Select **Run query** once the corpus is indexed.
5. Inspect the answer, retrieved chunks with BM25 scores, or final prompt in the result tabs.

Editing the corpus or pipeline settings invalidates the current index; re-index before running another query. Queries use `OceanNLP.RAG.RAGPipeline.query`. The pipeline returns final result artifacts but currently does not expose per-stage timings.

The TF-IDF/cosine choice is sparse lexical vector-space retrieval, not a transformer embedding model. Dense and hybrid rankers are not currently configurable in this demo because they require extra encoder and fusion setup.

## Backend requirements

- MATLAB with `uifigure` and `uihtml` support.
- OceanNLP package and its runtime dependencies available on the MATLAB path.
- **Demo (no LLM):** No inference server required. This runs real corpus indexing, BM25 retrieval, and prompt assembly, then uses `OceanNLP.Generation.FunctionGenerator` to return a clearly labeled demonstration message instead of invoking a model. Inspect the retrieved chunks and final prompt to verify the pipeline inputs.
- **Local LLM:** An available OpenAI-compatible local chat-completions server is required. The default endpoint is `http://localhost:1234/v1/chat/completions`; configure a model identifier recognized by that server.

The frontend assets are local files next to `index.html`; they do not use CDNs. If local inference is selected and the server is unavailable, the app displays the MATLAB error rather than silently switching to demo mode.
