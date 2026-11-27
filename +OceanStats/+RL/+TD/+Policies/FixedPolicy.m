classdef FixedPolicy < OceanStats.RL.TD.Policies.IBehaviourPolicy
    % FIXEDPOLICY  A given stochastic policy P (S x A) that is followed, and evaluated, without exploration. Used for
    % prediction. P is validated against the admissible actions: right size, finite and non-negative, every non-terminal
    % row sums to 1 (no zero rows), no probability on an inadmissible action.
    %   policy = OceanStats.RL.TD.Policies.FixedPolicy(P, validMask)    P = [] means uniform over the admissible actions
    properties (Constant)
        Latex = "b(a \mid s) = \pi(a \mid s)"
        Reference = "the fixed policy being evaluated"
    end
    properties (SetAccess = private)
        P double = []
    end
    methods
        function obj = FixedPolicy(P, validMask)
            arguments
                P double
                validMask logical
            end
            if isempty(P)
                P = double(validMask) ./ max(sum(validMask, 2), 1);
            else
                OceanStats.RL.TD.Policies.FixedPolicy.validate(P, validMask);
            end
            obj.P = P;
        end
        function s = name(~), s = "fixed"; end
        function p = probabilities(obj, ~, si, ~, ~), p = obj.P(si, :); end
    end
    methods (Static)
        function validate(P, validMask)
            id = 'OceanStats:RL:TD:TDLearner:BadPolicy';
            if ~isequal(size(P), size(validMask))
                error(id, 'PolicyProbs must be %d x %d (states x actions), got %s.', size(validMask, 1), size(validMask, 2), mat2str(size(P)));
            end
            if any(P(:) < 0) || any(~isfinite(P(:)))
                error(id, 'PolicyProbs must be finite and non-negative.');
            end
            live = any(validMask, 2);
            if any(abs(sum(P(live, :), 2) - 1) > 1e-8)
                error(id, 'Every non-terminal row of PolicyProbs must sum to 1 (zero rows are not allowed).');
            end
            if any(any(P(live, :) > 0 & ~validMask(live, :)))
                error(id, 'PolicyProbs puts probability on an inadmissible action.');
            end
        end
    end
end
