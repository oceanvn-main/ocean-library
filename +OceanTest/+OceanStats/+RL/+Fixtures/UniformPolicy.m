classdef UniformPolicy < OceanStats.RL.TD.Policies.IBehaviourPolicy
    % UNIFORMPOLICY  TEST FIXTURE for the extension rule: a brand-new behaviour policy written as ONE file, supplied through
    % the 'Policy' hook: uniform over the admissible actions, ignoring the value estimates.
    properties (Constant)
        Latex = "b(a \mid s) = 1 / |A(s)|"
        Reference = "test fixture (uniform random behaviour)"
    end
    methods
        function s = name(~), s = "uniform"; end
        function p = probabilities(~, ~, ~, valid, ~), p = double(valid) / sum(valid); end
    end
end
