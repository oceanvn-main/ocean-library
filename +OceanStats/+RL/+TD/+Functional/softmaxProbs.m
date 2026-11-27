function p = softmaxProbs(q, valid, tau)
% SOFTMAXPROBS  Boltzmann distribution exp(q/tau)/sum over the valid actions (max-shifted, numerically stable).
p = zeros(size(q));
z = q(valid) / tau;
z = exp(z - max(z));
p(valid) = z / sum(z);
end
