# RAG recipes

## Hybrid retrieval

Construct multiple `IRanker` implementations and combine their ranked IDs:

```matlab
fusion = OceanNLP.Fusion.FusionFactory.create("rrf");
hybrid = OceanNLP.Retrieval.HybridRanker('Rankers', {bm25, dense}, 'Fusion', fusion);
hybrid.fit(chunks);
[ids, scores] = hybrid.rank(question);
```

## Editable workflow

Declare what each stage reads (`Requires`) and writes (`Provides`). The pipeline checks the
contracts before every run, so a reordered or missing stage fails before any model call.

```matlab
stage = @OceanNLP.Pipeline.Stage.fromFunction;
workflow = OceanNLP.Pipeline.ComposablePipeline();
workflow.add(stage("retrieve", @(s) retrieveStage(s), 'Requires', "Query", 'Provides', ["ChunkIds" "Scores" "Chunks"]));
workflow.add(stage("rerank",   @(s) rerankStage(s),   'Requires', ["Query" "Chunks"], 'Provides', ["ChunkIds" "Chunks"]));
workflow.add(stage("prompt",   @(s) promptStage(s),   'Requires', ["Query" "Chunks"], 'Provides', "Prompt"));
workflow.add(stage("generate", @(s) generateStage(s), 'Requires', "Prompt", 'Provides', "Answer"));
workflow.analyze('InitialFields', "Query")   % per-stage reads/writes and any problems
pipe.setWorkflow(workflow);
```

Edits never warn by default, so you can pass through a broken order on purpose. To have
every edit checked, turn it on:

```matlab
workflow.InitialFields = "Query";
workflow.AutoValidate = true;   % warns OceanNLP:Pipeline:ChainBroken after a breaking edit
```

Stages without `Requires`/`Provides` still run; `validate` reports them as unchecked.
Struct stages (`struct('Name',...,'Step',...)`) are deprecated.

## Generic ANN

`HNSWIndex`, `IVFPQIndex`, and `ProductQuantizer` accept numeric matrices and do not
depend on NLP. Use exact dense retrieval as the reference when measuring recall@K.
