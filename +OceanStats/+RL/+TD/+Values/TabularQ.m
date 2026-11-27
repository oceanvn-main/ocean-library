classdef TabularQ < OceanStats.RL.TD.Values.TabularActionValues
    % TABULARQ  Dense action-value table Q(s,a) (S x A) with per-entry visit counts. Invalid actions stay at 0.
    %   vf = OceanStats.RL.TD.Values.TabularQ(validMask, initialValue)
    properties (Constant)
        Latex = "Q(s,a) \leftarrow Q(s,a) + \alpha_n \left[ y - Q(s,a) \right]"
        Reference = "Sutton & Barto (2018) sec. 6.4-6.5 (tabular action values)"
    end
    methods
        function obj = TabularQ(validMask, initialValue)
            arguments
                validMask logical
                initialValue (1,1) double = 0
            end
            obj@OceanStats.RL.TD.Values.TabularActionValues(validMask, initialValue);
        end
        function s = name(~), s = "tabularq"; end
        function coin = oneStep(obj, target, stepSize, si, ai, r, s2i, a2, term, gamma, p2, valid2, ~)
            y = target.compute(r, obj.Q(s2i, :), a2, term, gamma, p2, valid2);
            obj.Counts(si, ai) = obj.Counts(si, ai) + 1;
            alpha = stepSize.alpha(obj.Counts(si, ai));
            obj.Q(si, ai) = obj.Q(si, ai) + alpha * (y - obj.Q(si, ai));
            coin = [];
        end
    end
end
