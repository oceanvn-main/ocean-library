classdef RAGResult
    % RAGRESULT  Immutable value record encapsulating the output of a RAG query run.
    %
    % PURPOSE:
    %   Stores complete query execution state including input query, LLM answer,
    %   retrieved chunk IDs, scores, context chunks, constructed prompt, and parent document IDs.
    %
    % PROPERTIES:
    %   Query     - Original user question string
    %   Answer    - Generated natural language answer string
    %   ChunkIds  - Vector of retrieved context chunk IDs
    %   Scores    - Vector of retrieval/reranking scores for retrieved chunks
    %   Chunks    - String array of retrieved context text chunks
    %   Prompt    - Final assembled prompt string sent to the LLM generator
    %   ParentIds - Vector of parent document IDs (if parent-child store used)
    %
    % See also: OceanNLP.RAG.RAGPipeline, OceanNLP.RAG.RAGEvaluator

    properties (SetAccess = private)
        Query string
        Answer string
        ChunkIds double
        Scores double
        Chunks string
        Prompt string
        ParentIds double
    end

    methods
        function obj = RAGResult(query, answer, ids, scores, chunks, prompt, parentIds)
            arguments
                query (1,1) string
                answer (1,1) string
                ids double
                scores double
                chunks string
                prompt (1,1) string
                parentIds double = []
            end
            obj.Query = query;
            obj.Answer = answer;
            obj.ChunkIds = ids;
            obj.Scores = scores;
            obj.Chunks = chunks;
            obj.Prompt = prompt;
            obj.ParentIds = parentIds;
        end
    end
end
