classdef (Abstract) IVectorIndex < handle
    %IVECTORINDEX Domain-neutral numeric vector index.
    methods (Abstract)
        fit(obj, X)
        [ids, scores] = search(obj, Q, k, options)
        meta = getMetadata(obj)
    end
end
