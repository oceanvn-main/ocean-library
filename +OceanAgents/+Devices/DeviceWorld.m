classdef DeviceWorld < OceanStats.RL.Env.IEnvironment & OceanStats.RL.Env.IModelEnvironment
    % DEVICEWORLD  A grid world made of devices (DV1 contract): no environment class and no door holds its rules. The
    % terrain device moves the body's position component; the reward device pays and ends episodes; this facade only
    % wires them on one router, delivers the action, reads the outcome back, and presents the result as a native
    % IEnvironment + IModelEnvironment so the existing golden tests, solvers and runners use it unchanged.
    %
    %   env = OceanAgents.Devices.DeviceWorld.cliffWalking();                         % 4 x 12, start 37, goal 48
    %   env = OceanAgents.Devices.DeviceWorld.cliffWalking('Rows', 3, 'Cols', 5, 'TimeLimit', 20);
    %   env = OceanAgents.Devices.DeviceWorld.cliffWalking('Position', p);   % p = the body's own GridPosition
    %   s = env.reset('Seed', 7);  [s2, r, term, trunc, info] = env.step(2);
    %
    % One step = one instant of the world's router: move -> terrain (asks the position where it is, asks it to move)
    % -> reward (pays, ends). The facade wires and relays: it passes the terrain's outcome on to the reward device and
    % pre-checks the action against ValidMask (built from the terrain's Laws and the reward's goal, so it cannot
    % drift); the terrain also refuses a forbidden move itself when driven directly. Randomness: one uniform at reset and one per step from an mt19937ar stream, as the
    % native grid environments draw (cliff walking does not use them; they keep seeds aligned for stochastic worlds).

    properties (SetAccess = private)
        Name string = "DeviceWorld"
        NumStates double = 0
        NumActions double = 4
        TerminalMask (:,1) logical
        ValidMask logical
        Terrain
        Reward
        Position
    end

    properties (Access = private)
        MoveType_
        EventType_
        CommandType_
        Router_
        Port_
        Stream_ = []
        Live_ (1,1) logical = false
        Started_ (1,1) logical = false
    end

    properties (Dependent)
        Accepts
    end

    methods
        function obj = DeviceWorld(terrain, reward, options)
            arguments
                terrain (1,1) OceanAgents.Devices.Terrain
                reward (1,1) OceanAgents.Devices.Reward
                options.Position = []
                options.Name (1,1) string = "DeviceWorld"
            end
            S = terrain.numCells();
            position = options.Position;
            if isempty(position), position = OceanAgents.Components.GridPosition(S); end
            if ~(isa(position, 'OceanAgents.Components.GridPosition') && position.NumCells == S && ...
                    position.TrustedMover == terrain.ID)
                error('OceanAgents:Devices:BadConfig', ['DeviceWorld: the position must be a GridPosition of %d cells ' ...
                    'moved by "%s".'], S, terrain.ID);
            end
            obj.Name = options.Name;
            obj.Terrain = terrain;  obj.Reward = reward;  obj.Position = position;
            obj.NumStates = S;
            obj.ObservationSpace = OceanStats.RL.Env.Spaces.Discrete(S);
            A = terrain.numMoves();
            obj.NumActions = A;
            obj.ActionSpace = OceanStats.RL.Env.Spaces.Discrete(A);
            terminal = terrain.Cliff;  terminal(reward.Goal) = true;      % cliff cells are never occupied (as natively)
            obj.TerminalMask = terminal;
            valid = false(S, A);
            for s = find(~terminal)', valid(s, :) = terrain.permitted(s); end
            obj.ValidMask = valid;
            % wiring: one router for the world, every route negotiated once and sealed
            T = @(kind, role) OceanAgents.Minds.Signals.SignalType('ValueKind', kind, 'Role', role);
            r = OceanAgents.Minds.Signals.Router();
            obj.Port_ = r.attach(obj, "world");
            terrainPort = r.attach(terrain, terrain.ID);
            rewardPort = r.attach(reward, reward.ID);
            r.attach(position, position.ID);
            routes = {"move", terrain.ID, "command", OceanAgents.Minds.Signals.SignalType('ValueKind', "label", ...
                          'Labels', terrain.Labels, 'Role', "command"); ...
                      "place", terrain.ID, "command", T("record", "command"); ...
                      "event", reward.ID, "sense", T("record", "sense"); ...
                      "resetReward", reward.ID, "command", T("record", "command"); ...
                      "look", position.ID, "query", T("record", "query"); ...
                      "where", position.ID, "query", T("record", "query"); ...
                      "moveTo", position.ID, "command", T("record", "command")};
            for k = 1:size(routes, 1)
                r.addRoute(routes{k, 1}, routes{k, 2}, routes{k, 3});
                r.negotiate(routes{k, 1}, routes{k, 4});
            end
            r.seal();
            obj.MoveType_ = routes{1, 4};  obj.EventType_ = routes{3, 4};  obj.CommandType_ = routes{2, 4};
            terrain.connect(terrainPort);
            reward.connect(rewardPort);
            obj.Router_ = r;
        end

        function a = get.Accepts(~)
            a = OceanAgents.Minds.Signals.SignalType.empty(1, 0);
        end

        function out = receive(~, ~)
            out = OceanAgents.Minds.Signals.Signal.empty(1, 0);
        end

        function setSeed(obj, seed)
            obj.Stream_ = RandStream('mt19937ar', 'Seed', seed);
            obj.Terrain.useStream(obj.Stream_);
        end

        function [obs, reward, terminated, truncated, info] = step(obj, action)
            if ~obj.Started_
                error('OceanStats:RL:Env:TabularEnvironment:NotReset', 'step called before reset.');
            end
            here = obj.Position.Cell;
            if ~obj.Live_ || ~(isnumeric(action) && isscalar(action) && any(action == 1:obj.NumActions) && obj.ValidMask(here, action))
                error('OceanStats:RL:Env:TabularEnvironment:InvalidAction', ...
                    'Action %s is not admissible in state %d (or the episode has ended).', mat2str(action), here);
            end
            obj.Router_.advanceTo(obj.Router_.Now + 1);
            moved = obj.send_("move", obj.MoveType_, string(action));
            outcome = obj.send_("event", obj.EventType_, struct('From', moved.From, 'To', moved.To, 'Fell', moved.Fell));
            obs = moved.To;
            reward = outcome.Reward;
            terminated = logical(outcome.Terminated);
            truncated = logical(outcome.Truncated);
            obj.Live_ = ~(terminated || truncated);
            info = struct('action_mask', obj.ValidMask(obs, :) & ~terminated);
            if truncated, info.successor_action_mask = obj.ValidMask(obs, :); end
        end

        function [next_states, probabilities, immediate_rewards] = get_transitions(obj, state, action)
            s = state(1);
            if obj.TerminalMask(s) || ~obj.ValidMask(s, action)
                error('OceanStats:RL:Env:TabularEnvironment:InvalidAction', ...
                    'Action %d is not admissible in state %d.', action, s);
            end
            [cells, probs, falls] = obj.Terrain.outcomes(s, action);     % native outcome order
            next_states = cells;
            next_states(falls) = obj.Terrain.StartCell;               % a fall returns to the start
            probabilities = probs;
            immediate_rewards = zeros(numel(cells), 1);
            for k = 1:numel(cells)
                immediate_rewards(k) = obj.Reward.assess(falls(k), next_states(k));
            end
        end

        function states = get_all_states(obj), states = (1:obj.NumStates)'; end
        function idx = state_to_index(~, state), idx = state(1); end
        function k = action_to_index(~, action), k = action; end
        function actions = get_valid_actions(obj, state), actions = find(obj.ValidMask(state(1), :)); end
        function tf = is_terminal(obj, state), tf = obj.TerminalMask(state(1)); end
    end

    methods (Access = protected)
        function [obs, info] = doReset_(obj)
            rand(obj.stream_());                                 % one draw at reset (the start distribution)
            obj.Terrain.useStream(obj.stream_());                % then exactly one draw per move, by the terrain
            obj.Router_.advanceTo(obj.Router_.Now + 1);
            obj.send_("resetReward", obj.CommandType_, struct('Op', "reset"));
            placed = obj.send_("place", obj.CommandType_, struct('Op', "start"));
            obs = placed.To;
            obj.Live_ = true;
            obj.Started_ = true;
            info = struct('action_mask', obj.ValidMask(obs, :));
        end
    end

    methods (Access = private)
        function v = send_(obj, route, type, value)
            r = obj.Port_.send(route, OceanAgents.Minds.Signals.Signal.make(type, value));
            if isempty(r) || r(1).isRefusal()
                why = "no reply";
                if ~isempty(r), why = r(1).Value.Reason; end
                error('OceanAgents:Devices:Refused', 'DeviceWorld: "%s" was refused: %s', route, why);
            end
            v = r(1).Value;
        end

        function s = stream_(obj)
            s = obj.Stream_;
            if isempty(s), s = RandStream.getGlobalStream(); end   % unseeded: the global stream, as natively
        end
    end

    methods (Static)
        function env = cliffWalking(options)
            % CLIFFWALKING  Sutton & Barto Example 6.6 as devices: start bottom-left, goal bottom-right, the cells
            % between them are the cliff (-100, back to the start), every other step -1.
            arguments
                options.Rows (1,1) double {mustBeInteger, mustBePositive} = 4
                options.Cols (1,1) double {mustBeInteger, mustBePositive} = 12
                options.TimeLimit (1,1) double {mustBePositive} = Inf
                options.Position = []
                options.Forbidden double = zeros(1, 0)
            end
            R = options.Rows;  C = options.Cols;
            if R < 2 || C < 3                                      % as the native class: a start, a cliff, a goal
                error('OceanAgents:Devices:BadConfig', 'Cliff walking needs at least 2 rows and 3 columns.');
            end
            first = (R - 1) * C;
            if any(options.Forbidden == first + C)
                error('OceanAgents:Devices:BadConfig', 'The goal cell cannot be forbidden.');
            end                                   % cells of the bottom row are first + (1:C)
            terrain = OceanAgents.Devices.Terrain(R, C, 'Start', [R 1], 'Cliff', first + (2:C - 1), ...
                'Forbidden', options.Forbidden);
            reward = OceanAgents.Devices.Reward('Goal', first + C, 'StepReward', -1, 'CliffReward', -100, ...
                'TimeLimit', options.TimeLimit);
            env = OceanAgents.Devices.DeviceWorld(terrain, reward, 'Position', options.Position, 'Name', "CliffWalking");
        end
        function env = randomWalk(options)
            % RANDOMWALK  Sutton & Barto Example 6.2 as devices: a row of NumInner cells between two ends, start in the
            % middle, moves left / right, reward RightReward on entering the right end, LeftReward on the left end, 0
            % otherwise; both ends end the episode.
            arguments
                options.NumInner (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.NumInner, 1)} = 5
                options.LeftReward (1,1) double = 0
                options.RightReward (1,1) double = 1
                options.TimeLimit (1,1) double {mustBePositive} = Inf
                options.Position = []
            end
            S = options.NumInner + 2;
            cellReward = NaN(1, S);
            cellReward([1 S]) = [options.LeftReward options.RightReward];
            terrain = OceanAgents.Devices.Terrain(1, S, 'Start', [1 ceil(S / 2)], 'Moves', [0 -1; 0 1]);
            reward = OceanAgents.Devices.Reward('Goal', [1 S], 'StepReward', 0, 'CellReward', cellReward, ...
                'TimeLimit', options.TimeLimit);
            env = OceanAgents.Devices.DeviceWorld(terrain, reward, 'Position', options.Position, 'Name', "RandomWalk");
        end

        function env = windy(options)
            % WINDY  Sutton & Barto Example 6.5 (and Exercises 6.9-6.10) as devices: column wind pushes up, reward -1 per
            % step, the goal ends the episode; optional king moves, a stay action and stochastic wind. Move order: up,
            % right, down, left [, up-right, down-right, down-left, up-left] [, stay] - as WindyGridworldEnv.
            arguments
                options.Wind (1,:) double = [0 0 0 1 1 1 2 2 1 0]
                options.Rows (1,1) double {mustBeInteger, mustBePositive} = 7
                options.Start (1,2) double = [4 1]
                options.Goal (1,2) double = [4 8]
                options.KingMoves (1,1) logical = false
                options.StayAction (1,1) logical = false
                options.StochasticWind (1,1) logical = false
                options.TimeLimit (1,1) double {mustBePositive} = Inf
                options.Position = []
            end
            R = options.Rows;  C = numel(options.Wind);
            inGrid = @(p) all(p == fix(p)) && p(1) >= 1 && p(1) <= R && p(2) >= 1 && p(2) <= C;
            if ~inGrid(options.Start) || ~inGrid(options.Goal)
                error('OceanStats:RL:Env:TabularEnvironment:BadModel', ...
                    'Start %s and Goal %s must be cells of the %d x %d grid.', mat2str(options.Start), ...
                    mat2str(options.Goal), R, C);
            end
            if isequal(options.Start, options.Goal)                 % as natively: a terminal state cannot be a start
                error('OceanStats:RL:Env:TabularEnvironment:BadModel', 'A terminal state cannot be a start state.');
            end
            moves = [-1 0; 0 1; 1 0; 0 -1];
            if options.KingMoves, moves = [moves; -1 1; 1 1; 1 -1; -1 -1]; end
            if options.StayAction, moves = [moves; 0 0]; end
            terrain = OceanAgents.Devices.Terrain(R, C, 'Start', options.Start, 'Moves', moves, ...
                'Wind', options.Wind, 'StochasticWind', options.StochasticWind);
            goal = (options.Goal(1) - 1) * C + options.Goal(2);
            reward = OceanAgents.Devices.Reward('Goal', goal, 'StepReward', -1, 'TimeLimit', options.TimeLimit);
            env = OceanAgents.Devices.DeviceWorld(terrain, reward, 'Position', options.Position, 'Name', "WindyGridworld");
        end
    end
end
