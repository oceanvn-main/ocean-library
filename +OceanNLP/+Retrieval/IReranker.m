classdef (Abstract) IReranker < handle
    %IRERANKER  Candidate reranking contract, independent of first-stage retrieval.
    methods (Abstract)
        fit(obj,candidates)
        [ids,scores]=rerank(obj,query,candidateIds,candidateTexts)
        tf=isFitted(obj)
    end
end
