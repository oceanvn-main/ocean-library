# OceanNLP RAG Pipeline

The RAG implementation is composed from replaceable objects:

```matlab
chunker = OceanNLP.Chunking.ChunkerFactory.create("sliding", ...
    'WindowSize', 128, 'Overlap', 32);
ranker = OceanNLP.Retrieval.BM25Ranker();
generator = OceanNLP.Generation.FunctionGenerator(@(p) "answer from injected LLM");
pipe = OceanNLP.RAG.RAGPipeline('Chunker', chunker, 'Ranker', ranker, ...
    'Generator', generator);
pipe.index(documents);
result = pipe.query("question");
```

The default query workflow is a `ComposablePipeline` with retrieval, prompt assembly, and
generation stages. Replace it with `setWorkflow()` when a project needs query rewriting,
reranking, validation, or a different order. Each stage receives and returns a state struct.
The default stages declare `Requires`/`Provides` fields (`retrieve`: Query → ChunkIds, Scores, Chunks;
`prompt`: Query, Chunks → Prompt; `generate`: Prompt → Answer), and every query validates the
workflow before running it. Call `pipe.Workflow.analyze('InitialFields', "Query")` to inspect it.

For vector search, use `OceanAlgo.Search.HNSWIndex`, `IVFPQIndex`, or another
`OceanAlgo.Search.IVectorIndex`; domain code only maps data to numeric vectors and maps row
IDs back to records. Product Quantization is shared through
`OceanAlgo.Compression.ProductQuantizer`.

Inject `ParentChildCorpusStore` when retrieval should operate on small child chunks but
generation should receive the corresponding parent documents. `RAGPipeline` detects the
parent-expansion capability and records the parent IDs in the query state.
