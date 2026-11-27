classdef GridPosition < OceanAgents.Components.IComponent
    % GRIDPOSITION  Where a body is on a grid: the body's own data (design D9). Anyone may ask where it is; it MOVES
    % only on a request from its trusted mover (the terrain device), never by an overwrite (D10; DV1 contract §3).
    %
    %   p = OceanAgents.Components.GridPosition(48);                     % a 4 x 12 grid, mover "terrain"
    %   query   struct('Op', "where")                 -> reply Cell
    %   command struct('Op', "moveTo", 'Cell', c)     -> reply From, To   (only from the trusted mover)

    properties (SetAccess = private)
        Cell double = 0                         % 0 = not placed yet
        NumCells double
        TrustedMover string
        Moves double = 0
        WorldID double = 0                      % the router of the one world it is wired into (DV1 critic F1, F2)
    end

    methods
        function obj = GridPosition(numCells, options)
            arguments
                numCells (1,1) double {mustBeInteger, mustBePositive}
                options.TrustedMover (1,1) string = "terrain"
                options.ID (1,1) string = "position"
            end
            R = @(role) OceanAgents.Minds.Signals.SignalType('ValueKind', "record", 'Role', role);
            obj.setDeclarations_(options.ID, [R("command") R("query")], OceanAgents.Minds.Signals.SignalType.reply(), "calendar");
            obj.NumCells = numCells;
            obj.TrustedMover = options.TrustedMover;
        end

        function s = exportState(obj)
            s = struct('Cell', obj.Cell, 'Moves', obj.Moves);
        end
    end

    methods (Access = {?OceanAgents.Minds.Signals.Router})
        function joinWorld_(obj, routerID)
            % JOINWORLD_  Called by the Router when the position is attached: a body stands in one world at a time.
            if obj.WorldID ~= 0 && obj.WorldID ~= routerID
                error('OceanAgents:Devices:BadConfig', 'Position "%s" is already wired into another world.', obj.ID);
            end
            obj.WorldID = routerID;
        end
    end

    methods (Access = protected)
        function [ok, reason] = verify(obj, signal)
            v = signal.Value;
            ok = isfield(v, 'Op');
            reason = "a request needs Op";
            if ~ok, return; end
            if signal.Type.Role == "query"
                ok = string(v.Op) == "where";
                reason = "a position answers only ""where""";
                return
            end
            if signal.Sender ~= obj.TrustedMover || obj.WorldID == 0 || signal.Via ~= obj.WorldID
                ok = false;
                reason = sprintf('only "%s" of the world this body stands in may move it; "%s" tried', ...
                    obj.TrustedMover, signal.Sender);
                return
            end
            ok = string(v.Op) == "moveTo" && isfield(v, 'Cell') && isnumeric(v.Cell) && isscalar(v.Cell) && ...
                v.Cell >= 1 && v.Cell <= obj.NumCells && v.Cell == fix(v.Cell);
            reason = "moveTo needs a cell of the grid";
        end

        function out = act(obj, signal)
            if signal.Type.Role == "query"
                out = OceanAgents.Minds.Signals.Signal.ok(struct('Cell', obj.Cell));
                return
            end
            from = obj.Cell;
            obj.Cell = signal.Value.Cell;
            obj.Moves = obj.Moves + 1;
            out = OceanAgents.Minds.Signals.Signal.ok(struct('From', from, 'To', obj.Cell));
        end
    end
end
