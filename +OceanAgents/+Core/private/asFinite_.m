function x = asFinite_(x, name)
% ASFINITE_  Real finite numeric array, returned as double. NaN/Inf -> NonFinite.
if ~(isnumeric(x) && isreal(x)) && ~islogical(x)
    fail_('InvalidRecord', '%s must be real numeric.', name);
end
x = double(x);
if ~all(isfinite(x(:))), fail_('NonFinite', '%s contains NaN or Inf.', name); end
end
