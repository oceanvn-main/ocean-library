function [G, steps, reason] = rolloutReturn(model, s, firstAction, policy, stream, discount, maxSteps, leafValue)
% ROLLOUTRETURN  One simulated discounted return from a sample model (Sutton & Barto 2nd ed., Sections 8.10-8.11).
%
%   [G, steps, reason] = OceanAlgo.StateSearch.rolloutReturn(model, s, firstAction, policy, stream, ...
%                            discount, maxSteps, leafValue)
%
%   S_0 = s, A_0 = firstAction (NaN: chosen by the rollout policy), then A_k = policy.act(...), using only
%   model.sample, for at most maxSteps transitions:
%
%     G = R_1 + γ R_2 + ... + γ^(T-1) R_T                      the episode ended at T <= maxSteps
%     G = R_1 + γ R_2 + ... + γ^(H-1) R_H + γ^H h(S_H)        truncated at H = maxSteps
%
%   Truncated returns are corrected by the stored evaluation function h = leafValue (function handle), as the
%   text suggests for rollouts cut short of complete episodes. A state with no legal action ends the
%   simulation with the same correction γ^k h(S_k). The episode ends without continuation value on a
%   terminated transition or on reaching a goal state (model.isGoal). maxSteps = 0 returns h(s) (0 at a goal).
%
%   reason: "terminated" | "goal" | "horizon" | "no_action"
%
% See also: OceanAlgo.StateSearch.RolloutSearch, OceanAlgo.StateSearch.MonteCarloTreeSearch
arguments
    model (1,1) OceanAlgo.StateSearch.ISampleModel
    s
    firstAction (1,1) double
    policy (1,1) OceanAlgo.StateSearch.IRolloutPolicy
    stream (1,1) RandStream
    discount (1,1) double {mustBeInRange(discount, 0, 1)}
    maxSteps (1,1) double {mustBeInteger, mustBeNonnegative}
    leafValue (1,1) function_handle
end
G = 0; scale = 1; steps = 0;
if model.isGoal(s)
    reason = "goal"; return
end
if maxSteps == 0
    G = leaf_(leafValue, s); reason = "horizon"; return
end
a = firstAction;
if isnan(a)
    if isempty(model.actions(s))
        G = leaf_(leafValue, s); reason = "no_action"; return
    end
    a = policy.act(model, s, stream);
end
while true
    o = model.sample(s, a, stream);
    steps = steps + 1;
    G = G + scale * o.Reward;
    scale = scale * discount;
    if o.Terminated
        reason = "terminated"; return
    end
    s = o.Next;
    if model.isGoal(s)
        reason = "goal"; return
    end
    if steps >= maxSteps
        G = G + scale * leaf_(leafValue, s); reason = "horizon"; return
    end
    if isempty(model.actions(s))
        G = G + scale * leaf_(leafValue, s); reason = "no_action"; return
    end
    a = policy.act(model, s, stream);
end
end

function v = leaf_(f, s)
v = f(s);
if ~(isscalar(v) && isreal(v) && isfinite(v))
    error('OceanAlgo:StateSearch:BadLeafValue', 'LeafValue must return a finite real scalar.');
end
end
