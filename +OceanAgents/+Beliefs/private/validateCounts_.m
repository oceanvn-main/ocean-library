function n = validateCounts_(n)
% VALIDATECOUNTS_  An imported counts struct must have exactly the disposition fields, each a non-negative integer.
want = fieldnames(dispositionCounts_());
if ~(isstruct(n) && isscalar(n)) || ~isempty(setxor(fieldnames(n), want))
    error('OceanAgents:Beliefs:BadImport', 'Counts must be a struct with fields %s.', strjoin(want, ', '));
end
for k = 1:numel(want)
    x = n.(want{k});
    if ~(isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0 && x == fix(x))
        error('OceanAgents:Beliefs:BadImport', 'Counts.%s must be a non-negative integer.', want{k});
    end
    n.(want{k}) = double(x);
end
n = orderfields(n, want);
end
