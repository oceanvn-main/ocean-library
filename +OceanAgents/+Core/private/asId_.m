function x = asId_(x, name)
% ASID_  Episode/frame/tick identifiers: real double, non-negative integer scalar.
if ~(isnumeric(x) && isscalar(x) && isreal(x))
    fail_('InvalidRecord', '%s must be a numeric scalar.', name);
end
x = double(x);
if ~isfinite(x), fail_('NonFinite', '%s must be finite.', name); end
if x < 0 || x ~= fix(x), fail_('InvalidRecord', '%s must be a non-negative integer.', name); end
end
