classdef VectorSearchAdapter < handle
    %VECTORSEARCHADAPTER  Domain-neutral encoder-to-index adapter.
    properties (SetAccess=private)
        Encoder
        Index
        IsFitted (1,1) logical=false
    end
    methods
        function obj=VectorSearchAdapter(encoder,index)
            arguments
                encoder (1,1) function_handle
                index (1,1) OceanAlgo.Search.IVectorIndex
            end
            obj.Encoder=encoder; obj.Index=index;
        end
        function fit(obj,data)
            arguments
                obj
                data
            end
            obj.Index.fit(obj.Encoder(data)); obj.IsFitted=true;
        end
        function [ids,scores]=search(obj,queries,k,options)
            arguments
                obj
                queries
                k (1,1) double {mustBeInteger,mustBePositive}
                options double=[]
            end
            if isempty(options), [ids,scores]=obj.Index.search(obj.Encoder(queries),k); else, [ids,scores]=obj.Index.search(obj.Encoder(queries),k,'EfSearch',options); end
        end
        function meta=getMetadata(obj), meta=struct('Name','VectorSearchAdapter','Domain','none','Index',obj.Index.getMetadata()); end
    end
end
