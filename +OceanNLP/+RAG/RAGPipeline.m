classdef RAGPipeline < handle
    % RAGPIPELINE  Editable end-to-end Retrieval-Augmented Generation (RAG) pipeline facade.
    %
    % PURPOSE:
    %   Orchestrates document indexing (chunking, storing, ranker fitting) and
    %   query processing (retrieval, optional reranking, prompt assembly, LLM generation)
    %   by composing internal components into a ComposablePipeline workflow.
    %
    % COMPONENTS:
    %   Chunker   - IChunker strategy (e.g., SlidingWindowChunker, RecursiveChunker)
    %   Ranker    - IRanker strategy (e.g., BM25Ranker, DenseRanker, HybridRanker)
    %   Store     - ICorpusStore storage strategy (e.g., InMemoryCorpusStore)
    %   Builder   - PromptBuilder context budgeting & formatting component
    %   Generator - IGenerator LLM generation strategy (e.g., LocalLLMGenerator, OpenAIGenerator)
    %   Reranker  - Optional IReranker strategy (e.g., CrossEncoderReranker)
    %
    % EXAMPLES:
    %   pipeline = OceanNLP.RAG.RAGPipeline(...
    %       'Chunker', OceanNLP.Chunking.SlidingWindowChunker('WindowSize', 100), ...
    %       'Ranker', OceanNLP.Retrieval.BM25Ranker(), ...
    %       'Generator', OceanNLP.Generation.LocalLLMGenerator('Model', "gemma-4-12b"));
    %   pipeline.index(["Doc 1 text...", "Doc 2 text..."]);
    %   result = pipeline.query("What is in Doc 1?");
    %
    % See also: OceanNLP.RAG.RAGResult, OceanNLP.RAG.RAGEvaluator, OceanNLP.Pipeline.ComposablePipeline

    properties (SetAccess = private)
        Chunker
        Ranker
        Store
        Builder
        Generator
        Reranker = []
        TopK (1,1) double = 5
        IsIndexed logical = false
        Workflow
    end

    methods
        function obj = RAGPipeline(options)
            arguments
                options.Chunker (1,1) OceanNLP.Chunking.IChunker
                options.Ranker (1,1) OceanNLP.Retrieval.IRanker
                options.Store (1,1) OceanNLP.Store.ICorpusStore = OceanNLP.Store.InMemoryCorpusStore()
                options.Builder (1,1) OceanNLP.Generation.PromptBuilder = OceanNLP.Generation.PromptBuilder()
                options.Generator (1,1) OceanNLP.Generation.IGenerator
                options.Reranker = []
                options.TopK (1,1) double {mustBeInteger, mustBePositive} = 5
            end
            obj.Chunker = options.Chunker;
            obj.Ranker = options.Ranker;
            obj.Store = options.Store;
            obj.Builder = options.Builder;
            obj.Generator = options.Generator;
            obj.Reranker = options.Reranker;
            obj.TopK = options.TopK;

            obj.Workflow = OceanNLP.Pipeline.ComposablePipeline('InitialFields', "Query");
            stage = @OceanNLP.Pipeline.Stage.fromFunction;
            obj.Workflow.add(stage("retrieve", @(s)obj.stageRetrieve_(s), 'Requires', "Query", 'Provides', ["ChunkIds", "Scores", "Chunks"]));
            obj.Workflow.add(stage("prompt", @(s)obj.stagePrompt_(s), 'Requires', ["Query", "Chunks"], 'Provides', "Prompt"));
            obj.Workflow.add(stage("generate", @(s)obj.stageGenerate_(s), 'Requires', "Prompt", 'Provides', "Answer"));
        end
        function index(obj,documents)
            arguments
                obj
                documents string
            end
            all=strings(1,0); sources=zeros(1,0);
            for d=1:numel(documents), c=obj.Chunker.chunk(documents(d)); all=[all,c]; sources=[sources,repmat(d,1,numel(c))]; end %#ok<AGROW>
            obj.Store.fit(all,sources); obj.Ranker.fit(all); obj.IsIndexed=true;
        end
        function result=query(obj,question)
            arguments
                obj
                question string
            end
            if ~obj.IsIndexed, error('OceanNLP:RAGPipeline:NotIndexed','Call index first.'); end
            state=struct('Query',question); state=obj.Workflow.run(state);
            if ~isfield(state,'ParentIds'), state.ParentIds=[]; end
            result=OceanNLP.RAG.RAGResult(question,state.Answer,state.ChunkIds,state.Scores,state.Chunks,state.Prompt,state.ParentIds);
        end
        function replaceComponent(obj,name,component)
            arguments
                obj
                name string
                component
            end
            switch name, case "Chunker", obj.Chunker=component; case "Ranker", obj.Ranker=component; case "Store", obj.Store=component; case "Builder", obj.Builder=component; case "Generator", obj.Generator=component; case "Reranker", obj.Reranker=component; otherwise, error('OceanNLP:RAGPipeline:UnknownComponent','Unknown component.'); end
        end
        function setWorkflow(obj,workflow)
            arguments
                obj
                workflow (1,1) OceanNLP.Pipeline.ComposablePipeline
            end
            obj.Workflow=workflow;
        end
        function state=runWorkflow(obj,state)
            arguments
                obj
                state struct
            end
            state=obj.Workflow.run(state);
        end
        function meta=getMetadata(obj)
            meta=struct('Name','RAGPipeline','Indexed',obj.IsIndexed,'TopK',obj.TopK,'Components',{{class(obj.Chunker),class(obj.Ranker),class(obj.Store),class(obj.Builder),class(obj.Generator),class(obj.Reranker)}});
        end
        function spec=getPipelineSpec(~)
            spec=struct('Inputs',"documents/query",'Outputs',"RAGResult",'Stages',{{"chunk","store","fit-ranker","retrieve","assemble-prompt","generate"}},'EditableComponents',true);
        end
    end
    methods (Access=private)
        function state=stageRetrieve_(obj,state)
            [ids,scores]=obj.Ranker.rank(state.Query); take=min(obj.TopK,numel(ids)); state.ChunkIds=ids(1:take); state.Scores=scores(1:take); state.Chunks=obj.Store.getChunk(state.ChunkIds);
            if ~isempty(obj.Reranker), if ~obj.Reranker.isFitted(), obj.Reranker.fit(state.Chunks); end, [state.ChunkIds,state.Scores]=obj.Reranker.rerank(state.Query,state.ChunkIds,state.Chunks); state.Chunks=obj.Store.getChunk(state.ChunkIds); end
            if ismethod(obj.Store,'getParents'), [state.Chunks,state.ParentIds]=obj.Store.getParents(state.ChunkIds); end
        end
        function state=stagePrompt_(obj,state), state.Prompt=obj.Builder.build(state.Chunks,state.Query); end
        function state=stageGenerate_(obj,state), state.Answer=obj.Generator.generate(state.Prompt); end
    end
end
