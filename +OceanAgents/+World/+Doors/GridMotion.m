classdef GridMotion
    % GRIDMOTION  The grid-motion door: an entity's cell on a Rows x Cols grid, moved by "Move" events with optional
    % column wind (deterministic or stochastic), a cliff that sends the mover back to Start, and terminal cells that
    % block movement. Built to reproduce the native grid environments (OceanStats.RL.Env.Classic.*) bit for bit.
    %
    %   cfg  = struct('Rows', 4, 'Cols', 12, 'Moves', [-1 0; 0 1; 1 0; 0 -1], 'Wind', zeros(1, 12), ...
    %                 'StochasticWind', false, 'Cliff', cliffMask, 'Start', [4 1], 'Terminal', goalOrCliffMask, ...
    %                 'AgentID', "agent1");
    %   door = OceanAgents.World.Doors.GridMotion.spec(cfg);          % DoorSpec "GridMotion"
    %   [next, prob] = OceanAgents.World.Doors.GridMotion.transitions(cfg, 37, 2);   % exact model row (cell indices)
    %
    % Cells are (row, col) with row 1 at the top; the native state index is (row - 1) * Cols + col.
    % Owns     Position: value, double, width 2, dimensionless, public.
    % Stream   "legacy-mt19937ar" (the World gives this door RandStream('mt19937ar', 'Seed', seed) at reset(seed)).
    % Init     draws EXACTLY ONE u = rand(stream) and picks the start by the native rule
    %          c = cumsum(StartProbs); k = find(u * c(end) <= c, 1)  (StartProbs one-hot at Start, row-major cells),
    %          then spawns the agent with Position, AgentID (Agency) and Steps = 0 (RewardTermination).
    % Rule Move (phase "act", consumes "Move" with payload Label; one event per decision), per event in delivery order:
    %          1. a move from a Terminal cell is blocked: no draw, no event, no write (the host never sends one);
    %          2. EXACTLY ONE u = rand(stream), even when the move is deterministic (native TabularEnvironment.step);
    %          3. outcomes in native order: StochasticWind and wind w0 > 0 -> winds w0 + [-1 0 1], probs [1 1 1]/3;
    %             otherwise wind w0, prob 1; each outcome r2 = min(max(r + dr - w, 1), Rows), c2 = min(max(c + dc, 1), Cols);
    %          4. k = find(u * c(end) <= c, 1) with c = cumsum(probs(:));
    %          5. a Cliff cell sends the entity to Start and emits "CliffFall"; otherwise it moves and emits "Moved".
    %          Both events: payload From, To (native state indices; To = the cell actually occupied afterwards, i.e.
    %          Start after a fall), and for CliffFall also Fell (the cliff cell); SourceEntity = the entity.
    % Projection Cell: View.Index = (row - 1) * Cols + col for every row having Position (one per agent entity; the grid
    %          worlds are single-agent, so the viewer reads its own cell as a scalar).
    % Contract: W2_Grid_Doors_Contract.md section 3.1.

    properties (Constant)
        Name = "GridMotion"
    end

    methods (Static)
        function door = spec(cfg)
            % SPEC  The DoorSpec for a validated configuration (see the class help for cfg fields).
            cfg = OceanAgents.World.Doors.GridMotion.checkConfig(cfg);
            T = @(varargin) OceanAgents.World.Guard.ComponentType(varargin{:});
            res = struct('Rows', cfg.Rows, 'Cols', cfg.Cols, 'Moves', cfg.Moves, 'Wind', cfg.Wind, ...
                'StochasticWind', cfg.StochasticWind, 'Cliff', cfg.Cliff, 'Start', cfg.Start, 'Terminal', cfg.Terminal);
            grid = ["Rows" "Cols" "Moves" "Wind" "StochasticWind" "Cliff" "Start" "Terminal"];
            move = OceanAgents.World.RuleSpec('Name', "Move", 'Phase', "act", 'Requires', "Position", ...
                'Reads', ["Position" grid], 'Writes', "Position", 'Consumes', "Move", ...
                'Emits', ["Moved" "CliffFall"], 'Fn', @OceanAgents.World.Doors.GridMotion.moveRule);
            cellView = OceanAgents.World.RuleSpec('Name', "Cell", 'Phase', "project", 'Kind', "projection", ...
                'Requires', "Position", 'Reads', ["Position" "Cols"], 'Fn', @OceanAgents.World.Doors.GridMotion.cellProjection);
            agentID = cfg.AgentID;
            door = OceanAgents.World.DoorSpec('Name', OceanAgents.World.Doors.GridMotion.Name, 'Version', "1", ...
                'Owns', struct('Name', "Position", 'Type', T('Width', 2)), 'Resources', res, ...
                'Membership', "Position", 'Rules', [move cellView], 'Drive', "on-event", ...
                'Stream', "legacy-mt19937ar", ...
                'Init', @(stream) OceanAgents.World.Doors.GridMotion.initSpawn(cfg, agentID, stream));
        end

        function cfg = checkConfig(cfg)
            % CHECKCONFIG  Validate and normalise a GridMotion configuration (OceanAgents:World:BadValue).
            bad = @(msg) error('OceanAgents:World:BadValue', 'GridMotion: %s', msg);
            need = {'Rows', 'Cols', 'Moves', 'Wind', 'StochasticWind', 'Cliff', 'Start', 'Terminal', 'AgentID'};
            if ~isstruct(cfg) || ~isscalar(cfg) || ~all(isfield(cfg, need))
                bad(sprintf('cfg needs fields %s.', strjoin(need, ', ')));
            end
            R = cfg.Rows;  C = cfg.Cols;
            isPosInt = @(x) isnumeric(x) && isscalar(x) && x >= 1 && x == fix(x);
            if ~isPosInt(R) || ~isPosInt(C), bad('Rows and Cols must be positive integers.'); end
            if ~(isnumeric(cfg.Moves) && size(cfg.Moves, 2) == 2 && size(cfg.Moves, 1) >= 1 && all(isfinite(cfg.Moves(:))) ...
                    && all(cfg.Moves(:) == fix(cfg.Moves(:))))
                bad('Moves must be an A x 2 integer array of (row, col) offsets in action-label order.');
            end
            if ~(isnumeric(cfg.Wind) && isequal(size(cfg.Wind), [1 C]) && all(isfinite(cfg.Wind)))
                bad('Wind must be 1 x Cols.');
            end
            if ~(islogical(cfg.StochasticWind) && isscalar(cfg.StochasticWind)), bad('StochasticWind must be a logical scalar.'); end
            if ~(islogical(cfg.Cliff) && isequal(size(cfg.Cliff), [R C])), bad('Cliff must be a Rows x Cols logical.'); end
            if ~(islogical(cfg.Terminal) && isequal(size(cfg.Terminal), [R C])), bad('Terminal must be a Rows x Cols logical.'); end
            s = cfg.Start;
            if ~(isnumeric(s) && isequal(size(s), [1 2]) && all(s >= 1) && all(s == fix(s)) && s(1) <= R && s(2) <= C)
                bad('Start must be a (row, col) cell of the grid.');
            end
            if cfg.Terminal(s(1), s(2)) || cfg.Cliff(s(1), s(2)), bad('Start cannot be a terminal or cliff cell.'); end
            id = string(cfg.AgentID);
            if ~isscalar(id) || strlength(id) == 0, bad('AgentID must be a non-empty string.'); end
            cfg.Rows = double(R);  cfg.Cols = double(C);
            cfg.Moves = double(cfg.Moves);  cfg.Wind = double(cfg.Wind);  cfg.Start = double(s);  cfg.AgentID = id;
        end

        function cmds = initSpawn(cfg, agentID, stream)
            % INITSPAWN  Reset: one draw selects the start cell by the native rule; spawn the agent there.
            S = cfg.Rows * cfg.Cols;
            p = zeros(S, 1);
            p((cfg.Start(1) - 1) * cfg.Cols + cfg.Start(2)) = 1;          % StartProbs, one-hot, row-major
            c = cumsum(p);
            k = find(rand(stream) * c(end) <= c, 1);
            pos = [floor((k - 1) / cfg.Cols) + 1, mod(k - 1, cfg.Cols) + 1];
            T = @(varargin) OceanAgents.World.Guard.ComponentType(varargin{:});
            offer = struct( ...
                'Position', struct('Value', pos, 'Type', T('Width', 2)), ...
                'AgentID', struct('Value', agentID, 'Type', OceanAgents.World.Doors.Agency.idType()), ...
                'Steps', struct('Value', 0, 'Type', OceanAgents.World.Doors.RewardTermination.stepsType()));
            cmds = OceanAgents.World.Command.spawn(offer);
        end

        function out = moveRule(ctx)
            % MOVERULE  Rule Fn of Move (see the class help). Pure except for the draws from ctx.Stream.
            g = ctx.Read;
            pos = ctx.Read.Position;                        % n x 2, current cells of ctx.Rows (updated per event)
            moved = false(numel(ctx.Rows), 1);
            events = OceanAgents.World.Event.empty(1, 0);
            A = size(g.Moves, 1);
            for e = reshape(ctx.Events, 1, [])
                i = find(ctx.IDs == e.SourceEntity, 1);
                if isempty(i)
                    error('OceanAgents:World:BadValue', 'GridMotion.Move: entity %d has no Position.', e.SourceEntity);
                end
                label = e.Payload.Label;
                if ~(isnumeric(label) && isscalar(label) && label >= 1 && label <= A && label == fix(label))
                    error('OceanAgents:World:BadValue', 'GridMotion.Move: label %s is not in 1..%d.', mat2str(label), A);
                end
                r = pos(i, 1);  c = pos(i, 2);
                if g.Terminal(r, c), continue; end          % blocked: no draw, nothing happens
                u = rand(ctx.Stream);                        % exactly one draw per move
                [cells, probs] = outcomes_(g, r, c, label);
                cp = cumsum(probs(:));
                k = find(u * cp(end) <= cp, 1);
                from = (r - 1) * g.Cols + c;
                to = cells(k, :);
                if g.Cliff(to(1), to(2))
                    fell = (to(1) - 1) * g.Cols + to(2);
                    pos(i, :) = g.Start;
                    payload = struct('From', from, 'To', (g.Start(1) - 1) * g.Cols + g.Start(2), 'Fell', fell);
                    events(end + 1) = OceanAgents.World.Event.make("CliffFall", payload, 'SourceEntity', e.SourceEntity); %#ok<AGROW>
                else
                    pos(i, :) = to;
                    payload = struct('From', from, 'To', (to(1) - 1) * g.Cols + to(2));
                    events(end + 1) = OceanAgents.World.Event.make("Moved", payload, 'SourceEntity', e.SourceEntity); %#ok<AGROW>
                end
                moved(i) = true;
            end
            out = struct('Writes', struct('Component', "Position", 'Rows', ctx.Rows(moved), 'Values', pos(moved, :), ...
                'Rates', []), 'Events', events);
        end

        function out = cellProjection(ctx)
            % CELLPROJECTION  View.Index = (row - 1) * Cols + col (the native state index) per row.
            p = ctx.Read.Position;
            out = struct('View', struct('Index', (p(:, 1) - 1) * ctx.Read.Cols + p(:, 2)));
        end

        function [next, prob] = transitions(cfg, stateIndex, label)
            % TRANSITIONS  Exact model row: next cell indices (native outcome order; a cliff outcome is replaced by
            % Start) and their probabilities, as column vectors. A terminal state has none (empty).
            cfg = OceanAgents.World.Doors.GridMotion.checkConfig(cfg);
            S = cfg.Rows * cfg.Cols;
            if ~(isnumeric(stateIndex) && isscalar(stateIndex) && stateIndex >= 1 && stateIndex <= S && stateIndex == fix(stateIndex))
                error('OceanAgents:World:BadValue', 'GridMotion.transitions: state %s is not in 1..%d.', mat2str(stateIndex), S);
            end
            A = size(cfg.Moves, 1);
            if ~(isnumeric(label) && isscalar(label) && label >= 1 && label <= A && label == fix(label))
                error('OceanAgents:World:BadValue', 'GridMotion.transitions: label %s is not in 1..%d.', mat2str(label), A);
            end
            r = floor((stateIndex - 1) / cfg.Cols) + 1;  c = mod(stateIndex - 1, cfg.Cols) + 1;
            if cfg.Terminal(r, c)
                next = zeros(0, 1);  prob = zeros(0, 1);
                return
            end
            [cells, probs] = outcomes_(cfg, r, c, label);
            fall = cfg.Cliff(sub2ind([cfg.Rows cfg.Cols], cells(:, 1), cells(:, 2)));
            cells(fall, :) = repmat(cfg.Start, nnz(fall), 1);
            next = (cells(:, 1) - 1) * cfg.Cols + cells(:, 2);
            prob = probs(:);
        end

        function [cells, probs, fell] = outcomesOf(cfg, stateIndex, label)
            % OUTCOMESOF  The raw outcome cells (before the cliff teleport), their probabilities and which are cliff
            % cells - what the model's reward function needs.
            r = floor((stateIndex - 1) / cfg.Cols) + 1;  c = mod(stateIndex - 1, cfg.Cols) + 1;
            [cells, probs] = outcomes_(cfg, r, c, label);
            fell = cfg.Cliff(sub2ind([cfg.Rows cfg.Cols], cells(:, 1), cells(:, 2)));
        end
    end
end

function [cells, probs] = outcomes_(g, r, c, label)
% Outcome cells (K x 2) and probabilities (1 x K) of one move, in the native order.
w0 = g.Wind(c);
if g.StochasticWind && w0 > 0
    winds = w0 + [-1 0 1];  probs = [1 1 1] / 3;
else
    winds = w0;  probs = 1;
end
dr = g.Moves(label, 1);  dc = g.Moves(label, 2);
cells = zeros(numel(winds), 2);
for k = 1:numel(winds)
    cells(k, :) = [min(max(r + dr - winds(k), 1), g.Rows), min(max(c + dc, 1), g.Cols)];
end
end
