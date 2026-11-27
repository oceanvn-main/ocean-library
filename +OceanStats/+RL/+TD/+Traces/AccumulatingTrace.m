classdef AccumulatingTrace < OceanStats.RL.TD.Traces.ITrace
    % ACCUMULATINGTRACE  e(s,a) <- e(s,a) + 1 on a visit.
    properties (Constant)
        Latex = "e_t(s,a) = \gamma\lambda\, e_{t-1}(s,a) + \mathbb{1}[(S_t,A_t) = (s,a)]"
        Reference = "Sutton & Barto (2018) eq. 12.5 (accumulating traces)"
    end
    methods
        function E = visit(~, E, idx), E(idx) = E(idx) + 1; end
        function s = name(~), s = "accumulating"; end
    end
end
