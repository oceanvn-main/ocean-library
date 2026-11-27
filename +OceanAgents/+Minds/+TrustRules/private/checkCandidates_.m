function checkCandidates_(c)
% CHECKCANDIDATES_  Shape check of the candidates struct handed to a trust rule (A2 contract section 4.1).
need = {'Labels', 'Subjects', 'IsOutside', 'Estimate', 'Variance', 'Alpha', 'Beta', 'Payoff'};
if ~(isstruct(c) && isscalar(c) && all(isfield(c, need)))
    error('OceanAgents:Minds:Incompatible', 'Candidates must be a struct with fields %s.', strjoin(need, ', '));
end
n = numel(c.Labels);
if n == 0
    error('OceanAgents:Minds:Incompatible', 'A trust rule needs at least one legal candidate.');
end
for f = need(2:7)
    if numel(c.(f{1})) ~= n
        error('OceanAgents:Minds:Incompatible', 'Candidates.%s must have one entry per label (%d).', f{1}, n);
    end
end
if ~(isnumeric(c.Payoff) && numel(c.Payoff) == 3 && all(isfinite(c.Payoff)))
    error('OceanAgents:Minds:Incompatible', 'Candidates.Payoff must be [Success Failure Outside], finite.');
end
end
