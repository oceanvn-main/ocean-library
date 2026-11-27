classdef QueryExpander < handle
    properties (SetAccess=private), ExpansionFunction function_handle, NumQueries (1,1) double=3, end
    methods
        function obj=QueryExpander(fn,options)
            arguments
                fn (1,1) function_handle
                options.NumQueries (1,1) double {mustBeInteger,mustBePositive}=3
            end
            obj.ExpansionFunction=fn; obj.NumQueries=options.NumQueries;
        end
        function queries=expand(obj,query)
            arguments
                obj
                query string
            end
            queries=string(obj.ExpansionFunction(query,obj.NumQueries));
            queries=queries(1:min(obj.NumQueries,numel(queries)));
        end
    end
end
