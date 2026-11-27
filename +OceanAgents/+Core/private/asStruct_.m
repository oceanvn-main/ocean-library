function s = asStruct_(s, name)
% ASSTRUCT_  Scalar struct holding plain value data only, at any depth (see OceanAgents.Core.assertValueData).
if ~(isstruct(s) && isscalar(s)), fail_('InvalidRecord', '%s must be a scalar struct.', name); end
OceanAgents.Core.assertValueData(s, name);
end
