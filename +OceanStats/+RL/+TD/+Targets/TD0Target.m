classdef TD0Target < OceanStats.RL.TD.Targets.ITarget
    % TD0TARGET  State-value target r + gamma * V(S').  (prediction; Sutton & Barto eq. 6.2)
    properties (Constant)
        Latex = "y = R_{t+1} + \gamma\, V(S_{t+1})"
        Reference = "Sutton & Barto (2018) eq. 6.2, TD(0)"
        IsOffPolicy = false
    end
    methods
        function y = compute(~, r, qNext, ~, terminated, gamma, ~, ~)
            y = OceanStats.RL.TD.Functional.oneStepTarget(r, qNext(1), terminated, gamma);
        end
        function s = name(~), s = "td0"; end
        function k = valueKind(~), k = "state"; end
    end
end
