classdef RAGEvaluator < handle
    % RAGEVALUATOR  Evaluates RAG system performance metrics across query runs.
    %
    % PURPOSE:
    %   Computes summary statistics (mean retrieved chunks count, faithfulness scores)
    %   across an array of RAGResult objects.
    %
    % PROPERTIES:
    %   FaithfulnessFunction - Optional handle `@(ragResult) -> double` for factual groundedness
    %
    % EXAMPLES:
    %   evaluator = OceanNLP.RAG.RAGEvaluator('FaithfulnessFunction', @(r) double(contains(r.Answer, "MATLAB")));
    %   report = evaluator.evaluate(results);
    %
    % See also: OceanNLP.RAG.RAGResult, OceanNLP.RAG.FaithfulnessMetric, OceanNLP.RAG.ContextPrecisionMetric

    properties (SetAccess = private)
        FaithfulnessFunction
    end

    methods
        function obj = RAGEvaluator(options)
            arguments
                options.FaithfulnessFunction = []
            end
            obj.FaithfulnessFunction = options.FaithfulnessFunction;
        end

        function report = evaluate(obj, results)
            arguments
                obj
                results
            end
            n = numel(results);
            report = struct('NumResults', n, 'MeanRetrieved', 0, 'Faithfulness', NaN);
            if n == 0
                return;
            end
            counts = arrayfun(@(r) numel(r.ChunkIds), results);
            report.MeanRetrieved = mean(counts);
            if ~isempty(obj.FaithfulnessFunction)
                vals = zeros(1, n);
                for i = 1:n
                    vals(i) = obj.FaithfulnessFunction(results(i));
                end
                report.Faithfulness = mean(vals, 'omitnan');
            end
        end
    end
end
