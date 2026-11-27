classdef AfterstateTarget < OceanStats.RL.TD.Targets.ITarget
    % AFTERSTATETARGET  Target for afterstate state-value evaluation V(S'_a).
    %   Sutton & Barto (2018) Sec 6.8: Afterstates.
    properties (Constant)
        Latex = "y = R_{t+1} + \gamma\, V(S'_{a})"
        Reference = "Sutton & Barto (2018) Sec 6.8, Afterstates"
        IsOffPolicy = false
    end
    methods
        function y = compute(~, r, qNext, ~, terminated, gamma, ~, ~)
            if terminated
                b = 0;
            else
                b = qNext(1);
            end
            y = OceanStats.RL.TD.Functional.oneStepTarget(r, b, terminated, gamma);
        end
        function s = name(~), s = "afterstate"; end
        function k = valueKind(~), k = "state"; end
    end
end
