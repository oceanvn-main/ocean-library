function s = asString_(s, name, allowEmpty)
% ASSTRING_  Scalar string (char row accepted). Empty text allowed only when allowEmpty.
if nargin < 3, allowEmpty = false; end
if ischar(s) && (isrow(s) || isempty(s)), s = string(s); end
if ~(isstring(s) && isscalar(s)) || ismissing(s)
    fail_('InvalidRecord', '%s must be a string scalar.', name);
end
if ~allowEmpty && strlength(s) == 0
    fail_('InvalidRecord', '%s must be non-empty.', name);
end
end
