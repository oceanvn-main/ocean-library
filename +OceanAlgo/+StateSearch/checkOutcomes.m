function o = checkOutcomes(o, requireDeterministic)
% CHECKOUTCOMES  Validate an outcome list returned by IDistributionModel.outcomes.
%
%   o = OceanAlgo.StateSearch.checkOutcomes(o)          fields, finite rewards, probabilities summing to 1
%   o = OceanAlgo.StateSearch.checkOutcomes(o, true)    ... and exactly one outcome (deterministic model)
arguments
    o
    requireDeterministic (1,1) logical = false
end
if ~isstruct(o) || isempty(o) || ~all(isfield(o, {'Next', 'Reward', 'Terminated', 'Probability'}))
    error('OceanAlgo:StateSearch:BadOutcomes', 'Outcomes must be a nonempty struct array with Next, Reward, Terminated, Probability.');
end
p = [o.Probability]; r = [o.Reward]; t = [o.Terminated];
if numel(p) ~= numel(o) || numel(r) ~= numel(o) || numel(t) ~= numel(o) || ~isreal(r) || ~all(isfinite(r)) || ...
        ~all(isfinite(p)) || any(p < 0) || abs(sum(p) - 1) > 1e-9 || ~(islogical(t) || all(t == 0 | t == 1))
    error('OceanAlgo:StateSearch:BadOutcomes', ...
        'Each outcome needs a finite scalar Reward, a logical Terminated and a Probability in [0,1]; probabilities must sum to 1.');
end
o = o(p > 0);              % zero-probability rows (allowed, as in ExactModelAdapter) never happen: drop them
if requireDeterministic && numel(o) ~= 1
    error('OceanAlgo:StateSearch:NotDeterministic', ...
        'This engine needs a deterministic model (one outcome per action); got %d outcomes with p > 0.', numel(o));
end
end
