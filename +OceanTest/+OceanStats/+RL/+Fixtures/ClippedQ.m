classdef ClippedQ < OceanStats.RL.TD.Values.TabularActionValues
    % CLIPPEDQ  TEST FIXTURE for the extension rule: a brand-new value function written as ONE file, in a foreign package,
    % supplied through the 'ValueFunction' hook. Tabular action values that are clipped to [-Limit, Limit] after every update.
    properties (Constant)
        Latex = "Q(s,a) \leftarrow \mathrm{clip}\left( Q(s,a) + \alpha [y - Q(s,a)],\, -L,\, L \right)"
        Reference = "test fixture (clipped tabular action values)"
    end
    properties (SetAccess = private)
        Limit (1,1) double = Inf
    end
    methods
        function obj = ClippedQ(validMask, initialValue, limit)
            obj@OceanStats.RL.TD.Values.TabularActionValues(validMask, initialValue);
            obj.Limit = limit;
        end
        function s = name(~), s = "clippedq"; end
        function update(obj, si, ai, alpha, y)
            obj.Q(si, ai) = max(-obj.Limit, min(obj.Limit, obj.Q(si, ai) + alpha * (y - obj.Q(si, ai))));
        end
        function coin = oneStep(obj, target, stepSize, si, ai, r, s2i, a2, term, gamma, p2, valid2, ~)
            y = target.compute(r, obj.Q(s2i, :), a2, term, gamma, p2, valid2);
            cnt = obj.touch(si, ai);
            obj.update(si, ai, stepSize.alpha(cnt), y);
            coin = [];
        end
    end
end
