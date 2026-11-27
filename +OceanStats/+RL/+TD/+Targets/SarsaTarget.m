classdef SarsaTarget < OceanStats.RL.TD.Targets.ITarget
    % SARSATARGET  On-policy action-value target r + gamma * Q(S', A') with A' the action actually taken.
    properties (Constant)
        Latex = "y = R_{t+1} + \gamma\, Q(S_{t+1}, A_{t+1})"
        Reference = "Sutton & Barto (2018) eq. 6.7, Sarsa"
        IsOffPolicy = false
    end
    methods
        function y = compute(~, r, qNext, aNext, terminated, gamma, ~, ~)
            y = OceanStats.RL.TD.Functional.oneStepTarget(r, qNext(aNext), terminated, gamma);
        end
        function s = name(~), s = "sarsa"; end
    end
end
