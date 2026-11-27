classdef DoubleExpectedSarsaTarget < OceanStats.RL.TD.Targets.ITarget
    % DOUBLEEXPECTEDSARSATARGET  Target r + gamma * sum_a pi(a|S_{t+1}) Q_B(S_{t+1}, a) for Double Expected Sarsa.
    %   Sutton & Barto (2018) Sec 6.7: Double Learning.
    properties (Constant)
        Latex = "y = R_{t+1} + \gamma \sum_a \pi(a \mid S_{t+1})\, Q_B(S_{t+1}, a)"
        Reference = "Sutton & Barto (2018) Sec 6.7, Double Expected Sarsa"
        IsOffPolicy = false
    end
    methods
        function y = compute(~, r, qNext, ~, terminated, gamma, probsNext, valid)
            if terminated
                b = 0;
            else
                b = OceanStats.RL.TD.Functional.expectedValue(qNext, probsNext, valid);
            end
            y = OceanStats.RL.TD.Functional.oneStepTarget(r, b, terminated, gamma);
        end
        function s = name(~), s = "doubleexpectedsarsa"; end
        function tf = supportsDouble(~), tf = true; end
    end
end
