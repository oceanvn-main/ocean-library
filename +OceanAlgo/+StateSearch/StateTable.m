classdef StateTable < handle
    % STATETABLE  Map from a state key (ISearchModel.key) to a positive integer id. The key type (double or
    % string) is fixed by the first key stored; mixing types is an error.
    %
    %   t = OceanAlgo.StateSearch.StateTable();
    %   t.set(k, id);  id = t.get(k)  (0 when absent);  tf = t.has(k);  n = t.count();  t.clear()
    properties (Access = private)
        Map_ = []
        IsNumeric_ logical = logical([])
    end
    methods
        function set(obj, k, id)
            obj.ensure_(k);
            obj.Map_(obj.norm_(k)) = id;
        end
        function id = get(obj, k)
            id = 0;
            if isempty(obj.Map_), return; end
            nk = obj.norm_(k);
            if isKey(obj.Map_, nk), id = obj.Map_(nk); end
        end
        function tf = has(obj, k), tf = obj.get(k) > 0; end
        function n = count(obj)
            if isempty(obj.Map_), n = 0; else, n = double(obj.Map_.Count); end
        end
        function clear(obj), obj.Map_ = []; obj.IsNumeric_ = logical([]); end
    end
    methods (Access = private)
        function ensure_(obj, k)
            if ~isempty(obj.Map_), return; end
            obj.IsNumeric_ = isnumeric(k);
            if obj.IsNumeric_
                obj.Map_ = containers.Map('KeyType', 'double', 'ValueType', 'double');
            else
                obj.Map_ = containers.Map('KeyType', 'char', 'ValueType', 'double');
            end
        end
        function nk = norm_(obj, k)
            if isnumeric(k) && isscalar(k) && isreal(k) && isfinite(k)
                if ~isempty(obj.IsNumeric_) && ~obj.IsNumeric_
                    error('OceanAlgo:StateSearch:BadKey', 'State keys must all be numeric or all be strings.');
                end
                nk = double(k);
            elseif (isstring(k) && isscalar(k)) || (ischar(k) && isrow(k))
                if ~isempty(obj.IsNumeric_) && obj.IsNumeric_
                    error('OceanAlgo:StateSearch:BadKey', 'State keys must all be numeric or all be strings.');
                end
                nk = char(k);
            else
                error('OceanAlgo:StateSearch:BadKey', 'A state key must be a finite real scalar or a string scalar.');
            end
        end
    end
end
