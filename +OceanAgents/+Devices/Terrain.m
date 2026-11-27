classdef Terrain < OceanAgents.Devices.IDevice
    % TERRAIN  The grid's ground (DV1 contract §3; S1 step S1.1): its size, the moves, the column wind, the cliff, the
    % start, and the cells the Laws forbid entering. Given a body's move it asks the body's position where it is, works
    % out where the move ends (an edge keeps you in place; wind pushes up; the cliff sends you back to the start) and
    % asks the position to move there. It also answers which moves the Laws permit from a cell.
    %
    %   t = OceanAgents.Devices.Terrain(4, 12, 'Start', [4 1], 'Cliff', 38:47);
    %   t = OceanAgents.Devices.Terrain(7, 10, 'Start', [4 1], 'Wind', [0 0 0 1 1 1 2 2 1 0], 'StochasticWind', true);
    %   [to, fell] = t.resolve(cell, a, u);           % pure: the rule itself; u in (0,1) picks a random outcome
    %   [cells, probs, fell] = t.outcomes(cell, a);   % pure: every outcome in the native order (the exact model)
    %   m = t.permitted(cell);                        % pure: the moves the Laws permit from cell
    %   t.useStream(stream);                          % the world's random stream (exactly one draw per move)
    %
    % Moves: default 1..4 = up, right, down, left (as the native grid environments); 'Moves' gives any A x 2 list of
    % (row, col) offsets in label order (king moves, stay). Cells are numbered row by row.
    % Wind (one value per column, pushes UP by that many rows) acts on the column the body moves FROM; with
    % StochasticWind and a wind w > 0 the push is w-1, w or w+1 with probability 1/3 each, in that order (the native
    % WindyGridworldEnv). Randomness: with a stream, EXACTLY ONE uniform per move, even when the move is deterministic,
    % then the outcome k = find(u * c(end) <= c, 1), c = cumsum(probs) - the native draw rule, so worlds built from
    % this device are bit-identical to the native environments. Without a stream a deterministic move draws nothing.
    % A wall (Forbidden) cannot be combined with wind yet: a push into a wall is not defined.

    properties (SetAccess = private)
        Rows double
        Cols double
        StartCell double
        Cliff logical
        Forbidden logical
        Moves double = [-1 0; 0 1; 1 0; 0 -1]
        Wind double
        StochasticWind logical = false
        Labels string = string(1:4)
    end

    properties (Access = private)
        Here_ = 0                               % the cell asked for in verify, reused by act (one question per move)
        Stream_ = []                            % the world's random stream (set by the world; none = no draws)
    end

    properties (Constant)
        Deltas = [-1 0; 0 1; 1 0; 0 -1]         % the default four moves
    end

    methods
        function obj = Terrain(rows, cols, options)
            arguments
                rows (1,1) double {mustBeInteger, mustBePositive}
                cols (1,1) double {mustBeInteger, mustBePositive}
                options.Start (1,2) double = [rows 1]
                options.Cliff double = zeros(1, 0)            % cell indices
                options.Forbidden double = zeros(1, 0)        % cells the Laws forbid entering (a wall)
                options.Moves double = [-1 0; 0 1; 1 0; 0 -1] % A x 2 (row, col) offsets in label order
                options.Wind double = []                      % 1 x cols upward push per column (default none)
                options.StochasticWind (1,1) logical = false
                options.ID (1,1) string = "terrain"
            end
            S = rows * cols;
            M = options.Moves;
            if ~(isnumeric(M) && ismatrix(M) && size(M, 2) == 2 && size(M, 1) >= 1 && all(M(:) == fix(M(:))))
                error('OceanAgents:Devices:BadConfig', 'Terrain: Moves must be an A x 2 list of integer (row, col) offsets.');
            end
            wind = options.Wind;
            if isempty(wind), wind = zeros(1, cols); end
            if ~(isnumeric(wind) && isequal(size(wind), [1 cols]) && all(isfinite(wind)))
                error('OceanAgents:Devices:BadConfig', 'Terrain: Wind must be 1 x %d.', cols);
            end
            if any(wind ~= 0) && ~isempty(options.Forbidden)
                error('OceanAgents:Devices:BadConfig', 'Terrain: a wall cannot be combined with wind (a push into a wall is not defined).');
            end
            labels = string(1:size(M, 1));
            label = OceanAgents.Minds.Signals.SignalType('ValueKind', "label", 'Labels', labels, 'Role', "command");
            rec = OceanAgents.Minds.Signals.SignalType('ValueKind', "record", 'Role', "command");
            obj.setDeclarations_(options.ID, [label rec], OceanAgents.Minds.Signals.SignalType.reply(), "calendar");
            obj.Rows = rows;  obj.Cols = cols;
            obj.Moves = double(M);  obj.Labels = labels;
            obj.Wind = double(wind);  obj.StochasticWind = options.StochasticWind;
            obj.StartCell = (options.Start(1) - 1) * cols + options.Start(2);
            obj.Cliff = false(S, 1);  obj.Cliff(options.Cliff) = true;
            obj.Forbidden = false(S, 1);  obj.Forbidden(options.Forbidden) = true;
            if obj.Cliff(obj.StartCell) || obj.Forbidden(obj.StartCell)
                error('OceanAgents:Devices:BadConfig', 'Terrain: the start cell cannot be cliff or forbidden.');
            end
        end

        function useStream(obj, stream)
            % USESTREAM  The world hands the terrain its random stream; from then on every move draws exactly once.
            obj.Stream_ = stream;
        end

        function [cells, probs, fell] = outcomes(obj, cell, a)
            % OUTCOMES  Every way move a from cell can end, before the cliff sends a fall back to the start: cells
            % (K x 1, native outcome order), probabilities (K x 1) and which are cliff cells (K x 1).
            r = floor((cell - 1) / obj.Cols) + 1;  c = mod(cell - 1, obj.Cols) + 1;
            w0 = obj.Wind(c);
            if obj.StochasticWind && w0 > 0
                winds = w0 + [-1 0 1];  probs = [1; 1; 1] / 3;
            else
                winds = w0;  probs = 1;
            end
            r2 = min(max(r + obj.Moves(a, 1) - winds(:), 1), obj.Rows);
            c2 = min(max(c + obj.Moves(a, 2), 1), obj.Cols);
            cells = (r2 - 1) * obj.Cols + c2;
            fell = obj.Cliff(cells);
        end

        function [to, fell] = resolve(obj, cell, a, u)
            % RESOLVE  Where move a from cell ends, and whether it fell off the cliff (back to the start). u in (0, 1]
            % picks among random outcomes by the native rule; it may be omitted when the move is deterministic.
            [cells, probs, falls] = obj.outcomes(cell, a);
            k = 1;
            if numel(cells) > 1
                if nargin < 4
                    error('OceanAgents:Devices:BadConfig', 'Terrain: a random move needs a uniform draw u.');
                end
                cp = cumsum(probs);
                k = find(u * cp(end) <= cp, 1);
            end
            to = cells(k);  fell = falls(k);
            if fell, to = obj.StartCell; end
        end

        function m = permitted(obj, cell)
            % PERMITTED  The moves the Laws permit from cell: those whose own step does not end in a forbidden cell.
            A = size(obj.Moves, 1);
            m = true(1, A);
            if ~any(obj.Forbidden), return; end
            r = floor((cell - 1) / obj.Cols) + 1;  c = mod(cell - 1, obj.Cols) + 1;
            for a = 1:A
                r2 = min(max(r + obj.Moves(a, 1), 1), obj.Rows);
                c2 = min(max(c + obj.Moves(a, 2), 1), obj.Cols);
                m(a) = ~obj.Forbidden((r2 - 1) * obj.Cols + c2);
            end
        end

        function n = numCells(obj)
            n = obj.Rows * obj.Cols;
        end

        function n = numMoves(obj)
            n = size(obj.Moves, 1);
        end
    end

    methods (Access = protected)
        function [ok, reason] = verify(obj, signal)
            ok = true;  reason = "";
            if signal.Type.ValueKind == "label"
                here = obj.ask_("where", struct('Op', "where")).Cell;
                obj.Here_ = here;
                ok = here >= 1;
                if ok
                    m = obj.permitted(here);
                    ok = m(str2double(signal.Value));
                end
                reason = sprintf('the Laws forbid move %s from cell %d', signal.Value, here);
            elseif ~(isfield(signal.Value, 'Op') && string(signal.Value.Op) == "start")
                ok = false;  reason = "the terrain takes moves or a start request";
            end
        end

        function out = act(obj, signal)
            if signal.Type.ValueKind == "record"                     % place the body at the start
                obj.ask_("moveTo", struct('Op', "moveTo", 'Cell', obj.StartCell));
                out = OceanAgents.Minds.Signals.Signal.ok(struct('To', obj.StartCell));
                return
            end
            here = obj.Here_;                                    % asked in verify, a moment ago
            a = str2double(signal.Value);
            if isempty(obj.Stream_)
                [to, fell] = obj.resolve(here, a);
            else
                [to, fell] = obj.resolve(here, a, rand(obj.Stream_));   % exactly one draw per move
            end
            obj.ask_("moveTo", struct('Op', "moveTo", 'Cell', to));
            out = OceanAgents.Minds.Signals.Signal.ok(struct('From', here, 'To', to, 'Fell', fell));
        end
    end
end
