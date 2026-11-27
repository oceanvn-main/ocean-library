function y = oneStepTarget(r, bootstrap, terminated, gamma)
% ONESTEPTARGET  The TD target  y = r + gamma * (1 - terminated) * bootstrap.
%   r           immediate reward R_{t+1}
%   bootstrap   the estimate that stands in for the rest of the return (V(S'), Q(S',A'), max_a Q(S',a), ...)
%   terminated  TRUE only for a true MDP terminal state (bootstrap is dropped). A time-limit
%               truncation is NOT terminal: the caller passes terminated=false and the bootstrap stays.
%   gamma       discount in [0,1]
% Every one-step TD algorithm (TD(0), Sarsa, Expected Sarsa, Q-learning) is this function with a
% different `bootstrap`. Pure and vectorised over equally sized inputs.
y = r + gamma .* (~terminated) .* bootstrap;
end
