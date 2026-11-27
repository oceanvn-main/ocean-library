classdef RAGTestRanker < OceanNLP.Retrieval.IRanker
    properties, Fitted logical=false, end
    methods
        function fit(obj,corpus), obj.Fitted=true; end
        function [ids,scores]=rank(obj,query), if ~obj.Fitted, error('not fitted'); end, ids=1; scores=1; end
        function tf=isFitted(obj), tf=obj.Fitted; end
        function s=name(~), s="test-ranker"; end
    end
end
