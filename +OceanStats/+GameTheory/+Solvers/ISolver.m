classdef (Abstract) ISolver < handle
    % ISOLVER  Abstract interface for normal-form game solvers.
    %
    %   All solvers take a NormalFormGame and return a solver-specific
    %   result (e.g. IteratedDominanceSolver returns the reduced game).
    %   Mirrors OceanMath.Image.Variational.Solvers.ISolver's
    %   solve()/getHistory() convention so later solvers (Nash equilibrium,
    %   etc.) slot into the same pattern without modifying existing code
    %   (Open/Closed Principle).
    %
    %   Author: Ocean Library System
    %   Date: 2026-09-23

    methods (Abstract)
        % SOLVE  Run the solver against a NormalFormGame.
        result = solve(obj, game)
    end

    methods (Access = public)
        function history = getHistory(obj)
            % GETHISTORY  Return the record of the last solve() call.
            history = obj.History_;
        end
    end

    properties (Access = protected)
        History_ = struct()
    end
end
