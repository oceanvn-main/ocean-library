function tf = asRowFlags_(tf, n, name)
% ASROWFLAGS_  1 x n logical row (n = [] accepts any length).
if isnumeric(tf) && all(tf(:) == 0 | tf(:) == 1), tf = logical(tf); end
if ~(islogical(tf) && (isrow(tf) || isempty(tf)))
    fail_('InvalidRecord', '%s must be a logical row vector.', name);
end
tf = reshape(tf, 1, []);
if ~isempty(n) && numel(tf) ~= n
    fail_('InvalidRecord', '%s must have %d elements, got %d.', name, n, numel(tf));
end
end
