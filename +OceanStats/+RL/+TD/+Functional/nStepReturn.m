function G = nStepReturn(rewards, bootstrap, gamma)
% NSTEPRETURN  G = R_1 + gamma R_2 + ... + gamma^(k-1) R_k + gamma^k * bootstrap,  k = numel(rewards).
%   rewards    the k rewards actually collected (k <= n; fewer than n near the end of an episode)
%   bootstrap  the value estimate after those k steps; pass 0 when the episode TERMINATED within them
%              (Sutton & Barto eq. 7.1, 7.4). If the episode was merely truncated, pass the estimate.
rewards = rewards(:);
k = numel(rewards);
G = sum((gamma .^ (0:k-1)') .* rewards) + gamma^k * bootstrap;
end
