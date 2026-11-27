function G = lambdaReturns(rewards, nextValues, terminated, gamma, lambda)
% LAMBDARETURNS  Forward-view lambda-returns G^lambda_t for every step t of ONE episode.
%   rewards     (T x 1) R_{t+1}
%   nextValues  (T x 1) current estimates v(S_{t+1}); the entry for a terminal S_T is ignored
%   terminated  logical; TRUE if the last step ended in a terminal state (no bootstrap after T)
% Backward recursion (Sutton & Barto eq. 12.17, truncated at the episode end):
%   G_T   = R_T + gamma * (1-terminated) * v(S_T)
%   G_t   = R_{t+1} + gamma * [ (1-lambda) v(S_{t+1}) + lambda G_{t+1} ]
% lambda=0 gives the one-step TD target, lambda=1 the Monte Carlo return (terminated episodes).
T = numel(rewards);
G = zeros(T, 1);
last = rewards(T) + gamma * (~terminated) * nextValues(T);
G(T) = last;
for t = T-1:-1:1
    G(t) = rewards(t) + gamma * ((1 - lambda) * nextValues(t) + lambda * G(t+1));
end
end
