function assertValueData(x, name)
% ASSERTVALUEDATA  The single "value data only" rule for OceanAgents records and artifacts.
% Throws OceanAgents:Core:InvalidRecord if x contains, at ANY depth (struct fields, struct arrays, cells), a handle
% object, a function handle, a Java object, or any object other than string/datetime/duration/categorical.
% A record or artifact that held a live reference would silently link every copy of it (A0 section 1).
%
%   OceanAgents.Core.assertValueData(provenance, 'Provenance')
arguments
    x
    name (1,1) string = "value"
end
check_(x, name, 0);
end

function check_(v, path, depth)
if depth > 50
    error('OceanAgents:Core:InvalidRecord', '%s is nested more than 50 levels deep.', path);
end
if isa(v, 'handle') || isa(v, 'function_handle') || isjava(v)
    error('OceanAgents:Core:InvalidRecord', '%s holds a %s; records carry value data only.', path, class(v));
end
if isstruct(v)
    f = fieldnames(v);
    for k = 1:numel(v)
        for j = 1:numel(f)
            sub = sprintf('%s.%s', path, f{j});
            if numel(v) > 1, sub = sprintf('%s(%d).%s', path, k, f{j}); end
            check_(v(k).(f{j}), sub, depth + 1);
        end
    end
elseif iscell(v)
    for k = 1:numel(v)
        check_(v{k}, sprintf('%s{%d}', path, k), depth + 1);
    end
elseif isobject(v) && ~(isstring(v) || isa(v, 'datetime') || isa(v, 'duration') || isa(v, 'categorical'))
    % user value classes, dictionary, table, ...: they may hold handles internally, and walking every container
    % type reliably is not worth the risk, so records accept only plain built-in value data
    error('OceanAgents:Core:InvalidRecord', '%s holds an object of class %s; records carry plain value data only.', ...
        path, class(v));
end
end
