classdef ExpectedSarsaTarget < OceanStats.RL.TD.Targets.ITarget
    % EXPECTEDSARSATARGET  r + gamma * sum_a pi(a|S') Q(S', a) with pi the behaviour policy (on-policy form).
    % Same expected update as Sarsa but without the sampling variance of choosing A'.
    properties (Constant)
        Latex = "y = R_{t+1} + \gamma \sum_a \pi(a \mid S_{t+1})\, Q(S_{t+1}, a)"
        Reference = "Sutton & Barto (2018) eq. 6.9, Expected Sarsa"
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
        function s = name(~), s = "expectedsarsa"; end
    end
end
