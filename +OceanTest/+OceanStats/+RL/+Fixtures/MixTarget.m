classdef MixTarget < OceanStats.RL.TD.Targets.ITarget
    % MIXTARGET  TEST FIXTURE for the extension rule: a brand-new target written as ONE file, in a foreign package,
    % registered at run time. y = r + gamma * ( Q(S',A') + max_a Q(S',a) ) / 2 (an average of Sarsa and Q-learning).
    % It overrides nothing but compute(): every mode the learner offers must work for it through the capability
    % methods of ITarget (valueKind / bootstrap), with no edit to the learner.
    properties (Constant)
        Latex = "y = R_{t+1} + \gamma\,\tfrac12\left(Q(S_{t+1},A_{t+1}) + \max_a Q(S_{t+1},a)\right)"
        Reference = "test fixture (average of Sarsa and Q-learning targets)"
        IsOffPolicy = false
    end
    methods
        function y = compute(~, r, qNext, aNext, terminated, gamma, ~, valid)
            if terminated
                b = 0;
            else
                b = 0.5 * qNext(aNext) + 0.5 * OceanStats.RL.TD.Functional.greedyValue(qNext, valid);
            end
            y = OceanStats.RL.TD.Functional.oneStepTarget(r, b, terminated, gamma);
        end
        function s = name(~), s = "mix"; end
    end
end
