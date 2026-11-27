function index = categoricalSample(probabilities, u)
%CATEGORICALSAMPLE Sample a categorical distribution from an explicit draw.
%   The first cumulative probability strictly greater than u is selected,
%   with u in [0,1). No random stream or global RNG is accessed.
if ~(isnumeric(probabilities) && isreal(probabilities) && isvector(probabilities) ...
        && ~isempty(probabilities) && all(isfinite(probabilities), 'all') ...
        && all(probabilities >= 0, 'all'))
    fail_("probabilities must be a nonempty finite nonnegative vector.");
end
if abs(sum(double(probabilities)) - 1) > 1e-12
    fail_("probabilities must sum to one within tolerance.");
end
if ~(isnumeric(u) && isreal(u) && isscalar(u) && isfinite(u) && u >= 0 && u < 1)
    fail_("u must be a finite scalar in [0, 1).");
end
cumulative = cumsum(double(probabilities(:)));
index = find(double(u) < cumulative, 1, 'first');
if isempty(index)
    index = numel(probabilities);
end
end
function fail_(message), error('OceanStats:RL:Dyna:Functional:InvalidInput', '%s', message); end
