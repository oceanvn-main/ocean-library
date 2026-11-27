function target = qLearningTarget(reward, nextQ, terminated, gamma, validMask)
%QLEARNINGTARGET One-step off-policy target with explicit terminal semantics.
%   y = r when terminated, otherwise y = r + gamma*max_a Q(s',a)
%   over actions selected by validMask.
validateReward_(reward);
validateFlag_(terminated);
validateGamma_(gamma);
if terminated
    target = double(reward);
    return
end
validateActionValues_(nextQ, validMask);
target = double(reward) + double(gamma) * max(double(nextQ(validMask)));
end

function validateReward_(x)
if ~(isnumeric(x) && isreal(x) && isscalar(x) && isfinite(x)), fail_("reward must be a finite real scalar."); end
end
function validateFlag_(x)
if ~(islogical(x) && isscalar(x)), fail_("terminated must be a scalar logical."); end
end
function validateGamma_(x)
if ~(isnumeric(x) && isreal(x) && isscalar(x) && isfinite(x) && x >= 0 && x <= 1), fail_("gamma must be in [0, 1]."); end
end
function validateActionValues_(q, mask)
if ~(isnumeric(q) && isreal(q) && isvector(q) && all(isfinite(q), 'all')), fail_("nextQ must be a finite real vector."); end
if ~(islogical(mask) && isvector(mask) && isequal(size(mask), size(q))), fail_("validMask must be a logical vector matching nextQ."); end
if ~any(mask), fail_("A nonterminal successor must have a valid action."); end
end
function fail_(message), error('OceanStats:RL:Dyna:Functional:InvalidInput', '%s', message); end
