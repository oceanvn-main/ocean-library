classdef ReplacingTrace < OceanStats.RL.TD.Traces.ITrace
    % REPLACINGTRACE  e(s,a) <- 1 on a visit (Singh & Sutton 1996); avoids traces above 1 on revisits.
    properties (Constant)
        Latex = "e_t(s,a) = \begin{cases} 1 & (S_t,A_t)=(s,a) \\ \gamma\lambda\, e_{t-1}(s,a) & \text{otherwise} \end{cases}"
        Reference = "Singh & Sutton (1996); Sutton & Barto (2018) sec. 12.7 (replacing traces)"
    end
    methods
        function E = visit(~, E, idx), E(idx) = 1; end
        function s = name(~), s = "replacing"; end
    end
end
