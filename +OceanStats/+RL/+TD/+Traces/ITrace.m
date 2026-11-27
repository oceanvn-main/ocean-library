classdef (Abstract) ITrace < OceanStats.RL.TD.ITDComponent
    % ITRACE  Strategy: how the eligibility trace of the visited entry is bumped (Sutton & Barto ch. 12).
    % The learner decays ALL traces by gamma*lambda each step; the strategy only defines the visit.
    %   E = trace.visit(E, idx)     E: trace table, idx: linear index of the visited entry
    % Traces that need the step size or the decay (Dutch traces) override visitWith; the learner calls it.
    methods (Abstract)
        E = visit(obj, E, idx)
    end
    methods
        function E = visitWith(obj, E, idx, ~, ~)
            % E = trace.visitWith(E, idx, alpha, gammaLambda)
            E = obj.visit(E, idx);
        end
    end
end
