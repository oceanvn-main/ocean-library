classdef QLearningTarget < OceanStats.RL.TD.Targets.ITarget
    % QLEARNINGTARGET  Off-policy target r + gamma * max_a Q(S', a) over the valid actions.
    properties (Constant)
        Latex = "y = R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a)"
        Reference = "Watkins (1989); Sutton & Barto (2018) eq. 6.8, Q-learning"
        IsOffPolicy = true
    end
    methods
        function y = compute(~, r, qNext, ~, terminated, gamma, ~, valid)
            if terminated
                b = 0;
            else
                b = OceanStats.RL.TD.Functional.greedyValue(qNext, valid);
            end
            y = OceanStats.RL.TD.Functional.oneStepTarget(r, b, terminated, gamma);
        end
        function s = name(~), s = "qlearning"; end
        function tf = supportsDouble(~), tf = true; end
    end
end
