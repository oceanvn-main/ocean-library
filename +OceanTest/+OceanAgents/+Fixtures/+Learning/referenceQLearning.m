function [Q, N] = referenceQLearning(Q, steps, alpha)
% REFERENCEQLEARNING  Independent plain-arithmetic one-step Q-learning (Sutton & Barto 2018 eq. 6.8) for tests.
% Deliberately shares NO code with OceanStats.RL.TD (no TabularQ, targets or step sizes), so agreement with
% OceanAgents.Learners.OneStepQLearner is evidence rather than self-consistency.
%
%   [Q, N] = OceanTest.OceanAgents.Fixtures.Learning.referenceQLearning(Q0, steps, alpha)
%
%   Q0     S x A initial table
%   steps  struct array, in update order, with fields (all in TABLE indices):
%            s, a         row and column of the updated entry
%            r            reward
%            s2           successor row (ignored when term)
%            term         true for a terminal transition (no bootstrap)
%            discount     gamma^duration
%            valid2       1 x A logical successor legality (ignored when term)
%   alpha  constant step size
%   Q, N   final table and per-entry update counts
N = zeros(size(Q));
for k = 1:numel(steps)
    t = steps(k);
    if t.term
        target = t.r;
    else
        best = -Inf;
        for j = 1:size(Q, 2)
            if t.valid2(j) && Q(t.s2, j) > best
                best = Q(t.s2, j);
            end
        end
        target = t.r + t.discount * best;
    end
    Q(t.s, t.a) = Q(t.s, t.a) + alpha * (target - Q(t.s, t.a));
    N(t.s, t.a) = N(t.s, t.a) + 1;
end
end
