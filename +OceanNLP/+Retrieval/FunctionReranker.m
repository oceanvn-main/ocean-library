classdef FunctionReranker < OceanNLP.Retrieval.IReranker
    % FUNCTIONRERANKER  Function Handle Adapter for Candidate Reranking.
    %
    % PURPOSE:
    %   Serves as an adapter / wrapper pattern that turns ANY custom function
    %   handle fn(query, documentText) into a pluggable IReranker object.
    %
    % ROLES:
    %   - ScoreFunction : Function handle @(query, text) returning a scalar score
    %   - IsReady       : Logical flag tracking fitted/ready status
    %
    % See also: OceanNLP.Retrieval.IReranker,
    %           OceanNLP.Retrieval.CrossEncoderReranker

    properties (SetAccess = private)
        ScoreFunction function_handle  % Custom scoring handle @(q, text) -> double
        IsReady logical = false        % State readiness flag
    end

    methods
        function obj = FunctionReranker(fn)
            % FUNCTIONRERANKER  Construct wrapper around a custom function handle.
            arguments
                fn (1,1) function_handle
            end
            obj.ScoreFunction = fn;
        end

        function fit(obj, candidates)
            % FIT  Mark adapter ready (function handles require no corpus training).
            arguments
                obj
                candidates string
            end
            obj.IsReady = true;
        end

        function [ids, scores] = rerank(obj, query, candidateIds, candidateTexts)
            % RERANK  Re-score candidate document texts using the function handle.
            arguments
                obj
                query string
                candidateIds double
                candidateTexts string
            end
            if ~obj.IsReady
                error('OceanNLP:FunctionReranker:NotFitted', 'Call fit(candidates) first.');
            end

            % 1. Evaluate score function for each candidate text
            scores = zeros(1, numel(candidateIds));
            for i = 1:numel(candidateIds)
                scores(i) = obj.ScoreFunction(query, candidateTexts(i));
            end

            % 2. Re-sort candidates by descending score
            [scores, o] = sort(scores, 'descend');
            ids = candidateIds(o);
        end

        function tf = isFitted(obj)
            tf = obj.IsReady;
        end
    end
end

