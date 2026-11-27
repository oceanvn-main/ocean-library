classdef DoubleSarsaTarget < OceanStats.RL.TD.Targets.ITarget
    % DOUBLESARSATARGET  Target r + gamma * Q_B(S_{t+1}, A_{t+1}) for Double Sarsa.
    %   Sutton & Barto (2018) Sec 6.7: Double Learning.
    properties (Constant)
        Latex = "y = R_{t+1} + \gamma\, Q_B(S_{t+1}, A_{t+1})"
        Reference = "Sutton & Barto (2018) Sec 6.7, Double Sarsa"
        IsOffPolicy = false
    end
    methods
        function y = compute(~, r, qNext, aNext, terminated, gamma, ~, ~)
            if terminated
                b = 0;
            else
                b = qNext(aNext);
            end
            y = OceanStats.RL.TD.Functional.oneStepTarget(r, b, terminated, gamma);
        end
        function s = name(~), s = "doublesarsa"; end
        function tf = supportsDouble(~), tf = true; end
    end
end
