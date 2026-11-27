classdef FaithfulnessMetric < handle
    % FAITHFULNESSMETRIC  Evaluates whether the LLM answer is faithful to retrieved context.
    %
    % PURPOSE:
    %   Delegates factual groundedness checking to an evidence verification function
    %   `@ (answer, chunks, query) -> double [0.0 to 1.0]`.
    %
    % See also: OceanNLP.RAG.ContextPrecisionMetric, OceanNLP.RAG.RAGEvaluator

    properties (SetAccess = private)
        Checker function_handle
    end

    methods
        function obj = FaithfulnessMetric(checker)
            arguments
                checker (1,1) function_handle
            end
            obj.Checker = checker;
        end

        function value = score(obj, result)
            arguments
                obj
                result
            end
            value = double(obj.Checker(result.Answer, result.Chunks, result.Query));
        end
    end
end
