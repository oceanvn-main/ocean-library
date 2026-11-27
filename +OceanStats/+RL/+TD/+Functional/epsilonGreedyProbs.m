function p = epsilonGreedyProbs(q, valid, eps)
% EPSILONGREEDYPROBS  Action probabilities of an epsilon-greedy policy over the valid actions.
% Ties for the maximum share the greedy mass equally, so p sums to 1 and is symmetric in tied actions.
%   p(a) = eps / |A(s)| + (1-eps) * [a is greedy] / (number of greedy actions),  0 for invalid a.
p = zeros(size(q));
nv = nnz(valid);
p(valid) = eps / nv;
qv = q(valid);
best = valid & (q >= max(qv));
p(best) = p(best) + (1 - eps) / nnz(best);
end
