# NLP Process Explorer

The Process Explorer is a separate, text-first MATLAB app. It displays the actual input and output of each enabled stage in an `OceanNLP.Pipeline.ComposablePipeline`.

```matlab
pipeline = OceanNLP.Pipeline.ComposablePipeline();
tokenizer = OceanNLP.Tokenize.RegexTokenizer();
pipeline.add(OceanNLP.Pipeline.Stage.fromFunction("tokenize", ...
    @(text) tokenizer.forward(text)));
[tokens, reader] = OceanNLP.Inspect.PipelineAdapter.run(pipeline, inputText);
app = OceanViz.App.NLP.ProcessExplorer(reader);
```

Use the left map to select a step. The center shows input, decision summary, and output, with text, table, structure, and raw views. The context panel shows recorded IDs and status. Search filters the step map; arrow keys and the bottom buttons navigate.

`PipelineAdapter` captures stage values in memory up to `MaxArtifactBytes` (default 8 MB per artifact) and `MaxTotalBytes` (default 64 MB total). Omitted values keep their metadata and lineage. This first version does not stream or persist artifacts. A stage's private internal operations appear only when that component emits its own inspection events.

For an actual multilayer HNSW trace:

```matlab
% X: actual N-by-D document embedding matrix from your NLP pipeline.
% q: actual 1-by-D query embedding produced by the same encoder.
index = OceanAlgo.Search.HNSWIndex();
inspector = OceanAlgo.Search.Inspect.HNSWInspector('MaxEvents', 500);
inspector.attach(index);
index.fit(X);
[ids, scores] = index.search(q, 2);
reader = OceanAlgo.Search.Inspect.HNSWReaderAdapter.fromInspector( ...
    inspector, index, 'CenterNode', ids(1), 'NodeLimit', 50);
app = OceanViz.App.NLP.ProcessExplorer(reader);
```

The explorer lists real build/search events and a bounded MATLAB graph neighborhood. `DroppedCount` on the inspector reports events omitted when the cap is reached. Graph layout positions are visual arrangement, not embedding coordinates.

When a pipeline stage fails, `PipelineAdapter.run` rethrows the original error. Use `PipelineAdapter.capture` to inspect the completed and failed steps alongside the returned exception.
