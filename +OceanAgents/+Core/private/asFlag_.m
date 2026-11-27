function tf = asFlag_(tf, name)
% ASFLAG_  Logical scalar (0/1 numerics accepted).
if isnumeric(tf) && isscalar(tf) && (tf == 0 || tf == 1), tf = logical(tf); end
if ~(islogical(tf) && isscalar(tf)), fail_('InvalidRecord', '%s must be a logical scalar.', name); end
end
