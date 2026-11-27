classdef TabularV < OceanStats.RL.TD.Values.IValueFunction
    % TABULARV  Dense state-value table V(s) (S x 1) with per-state visit counts (prediction).
    %   vf = OceanStats.RL.TD.Values.TabularV(numStates, initialValue)
    properties (Constant)
        Latex = "V(s) \leftarrow V(s) + \alpha_n \left[ y - V(s) \right]"
        Reference = "Sutton & Barto (2018) sec. 6.1 (tabular state values)"
    end
    properties (SetAccess = protected)
        V double = []
        Counts double = []
    end
    methods
        function obj = TabularV(numStates, initialValue)
            arguments
                numStates (1,1) double {mustBeInteger, mustBePositive}
                initialValue (1,1) double = 0
            end
            obj.V = initialValue * ones(numStates, 1);
            obj.Counts = zeros(numStates, 1);
        end
        function s = name(~), s = "tabularv"; end
        function k = kind(~), k = "state"; end
        function x = estimates(obj, si), x = obj.V(si); end
        function x = behaviourEstimates(obj, si), x = obj.V(si); end
        function x = get(obj, si, ~), x = obj.V(si); end
        function n = touch(obj, si, ~)
            obj.Counts(si) = obj.Counts(si) + 1;
            n = obj.Counts(si);
        end
        function update(obj, si, ~, alpha, y)
            obj.V(si) = obj.V(si) + alpha * (y - obj.V(si));
        end
        function T = zerosLike(obj), T = zeros(size(obj.V)); end
        function applyIncrement(obj, dT), obj.V = obj.V + dT; end
        function coin = oneStep(obj, target, stepSize, si, ~, r, s2i, ~, term, gamma, ~, ~, ~)
            y = target.compute(r, obj.V(s2i), 1, term, gamma, [], true);
            obj.Counts(si) = obj.Counts(si) + 1;
            alpha = stepSize.alpha(obj.Counts(si));
            obj.V(si) = obj.V(si) + alpha * (y - obj.V(si));
            coin = [];
        end
        function v = stateValues(obj, idx), v = obj.V(idx); end
        function V = finalV(obj), V = obj.V; end
    end
end
