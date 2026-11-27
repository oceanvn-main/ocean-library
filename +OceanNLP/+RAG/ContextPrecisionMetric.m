classdef ContextPrecisionMetric < handle
    % CONTEXTPRECISIONMETRIC  Calculates the fraction of retrieved chunks that are relevant.
    %
    % PURPOSE:
    %   Evaluates context retrieval quality by computing the ratio of retrieved chunk IDs
    %   that match ground-truth relevant chunk IDs: score = count(retrieved & relevant) / total_retrieved.
    %
    % See also: OceanNLP.RAG.FaithfulnessMetric, OceanNLP.RAG.RAGEvaluator

    methods (Static)
        function value = score(result, relevantIds)
            arguments
                result
                relevantIds double
            end
            if isempty(result.ChunkIds)
                value = 0;
                return;
            end
            value = mean(ismember(result.ChunkIds, relevantIds));
        end
    end
end
