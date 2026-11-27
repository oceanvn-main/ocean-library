classdef RacetrackEnv < OceanStats.RL.Env.IEnvironment & OceanStats.RL.Env.IEnumerableEnvironment
    % RACETRACKENV Discrete Racetrack Environment (Sutton & Barto Exercise 5.12).
    %
    % Models the racetrack driving problem with momentum:
    %
    % STATE: [x, y, v_x, v_y]
    %   x   : integer column in [1, W]
    %   y   : integer row in [1, H]
    %   v_x, v_y : velocity components in [VelocityMin, VelocityMax]
    %     MovementMode "forward-only" (Sutton & Barto): 0 .. MaxVelocity-1   (default 0..4)
    %     MovementMode "bidirectional" (reverse allowed): -(MaxVelocity-1) .. MaxVelocity-1
    %   (v_x, v_y) cannot both be 0, except on the starting line.
    %   MaxVelocity is EXCLUSIVE, as in the Python twin (ocean_rl RacetrackEnv).
    %
    % MAP PROPERTIES: a Tiled map (.tmj) may declare movement_mode, max_velocity,
    %   fail_prob and max_episode_steps. They are the DEFAULTS for that map (explicit
    %   options override them) and are kept in MapProperties; MaxEpisodeSteps is the
    %   map's episode cap (NaN when the map does not say).
    % REACHABILITY: the constructor checks (breadth-first, noise off) whether a finish
    %   cell can be reached from the start line under the movement mode. The result is
    %   FinishReachable; when false a warning (RacetrackEnv:FinishUnreachable) says so --
    %   e.g. a map designed for bidirectional driving run forward-only. (A warning, not
    %   an error, so existing maps keep loading; experiments should check the property.)
    %
    % ACTIONS: 9 discrete accelerations (a_x, a_y) in {-1, 0, 1}^2
    %   1: (-1, -1)   2: (-1,  0)   3: (-1, +1)
    %   4: ( 0, -1)   5: ( 0,  0)   6: ( 0, +1)
    %   7: (+1, -1)   8: (+1,  0)   9: (+1, +1)
    %
    % STOCHASTIC DYNAMICS (10% Acceleration Failure):
    %   With probability FailProb (default 0.1), intended acceleration is
    %   zeroed out during environmental transition.
    %
    % BOUNDARY / CRASH & FINISH LINE:
    %   Projected path from (x,y) to (x',y') is checked via line traversal.
    %   - Finish Line crossed: episode ends immediately (done = true, R = -1).
    %   - Track Boundary crossed (Crash): car resets to a random start cell with
    %     (v_x, v_y) = (0,0), episode continues (done = false, R = -1).
    %
    % SPEEDUP: O(1) state indexing via 4D Lookup Table (LUT).
    %
    % Sample-only (no IModelEnvironment); enumerable state space (IEnumerableEnvironment).
    % Randomness: after setSeed / reset('Seed') the environment owns an independent stream; until then it draws from
    % MATLAB's global stream (so rng(k) still controls an unseeded run).
    % Observation [x y vx vy]; action = acceleration index 1..9 (ActionSpace = Discrete(9)); reward -1 per step.
    % Reference: Sutton & Barto (2018) Exercise 5.12, §5.6-5.7, Fig 5.5

    % -----------------------------------------------------------------------
    properties (Constant)
        MAX_VELOCITY = 4       % Max velocity component (0..4)
        N_ACTIONS    = 9       % 3 x 3 accelerations
        ACTION_GRID  = [ ...   % (9 x 2) matrix of [a_x, a_y]
            -1, -1; -1,  0; -1,  1; ...
             0, -1;  0,  0;  0,  1; ...
             1, -1;  1,  0;  1,  1  ...
        ]
    end

    properties (SetAccess = private)
        TrackName      string   % Name of track ("Track1" or "Track2" or "Custom")
        Grid           double   % (H x W) grid: 0=OffTrack, 1=Track, 2=Start, 3=Finish
        Height         double   % Grid height H (rows y)
        Width          double   % Grid width W (cols x)

        StartCells     double   % (N_start x 2) matrix of [x, y] start line positions
        FinishCells    double   % (N_finish x 2) matrix of [x, y] finish line positions

        StateIndexLUT  double   % (W x H x 5 x 5) 4D lookup table
        ValidActionsLUT cell    % (N_states x 1) cell array of valid action vectors
        AllStatesCache double   % (N_states x 4) matrix of [x, y, v_x, v_y]
        TotalStates    double   % Total valid states N

        FailProb       double   % Probability of acceleration failure (default 0.1)

        MovementMode   string   % "forward-only" | "bidirectional"
        MaxVelocity    double   % exclusive speed bound (5 -> speeds 0..4, or -4..4 bidirectional)
        VelocityMin    double   % smallest velocity component
        VelocityMax    double   % largest velocity component
        MaxEpisodeSteps double = NaN   % the map's episode cap (NaN when the map does not declare one)
        MapProperties  struct   % properties declared by the map file (Tiled), if any
        FinishReachable logical = true  % can any finish cell be reached from the start line?

        % Current episode state
        CurrentState   double   % [x, y, v_x, v_y]
        EpisodeDone    logical  % True if finish line crossed
    end
    properties (Access = private)
        Stream = []              % empty: use the global stream
    end

    % -----------------------------------------------------------------------
    methods
        function obj = RacetrackEnv(track_spec, opts)
            % RACETRACKENV Constructor.
            %
            % Inputs:
            %   track_spec - "Track1" (default), "Track2", string array map, numeric matrix,
            %                or a map file name (.tmj / .map)
            %   opts.FailProb     - acceleration failure probability (map's value, else 0.1)
            %   opts.MovementMode - "forward-only" | "bidirectional" (map's value, else forward-only)
            %   opts.MaxVelocity  - EXCLUSIVE speed bound (map's value, else 5 -> speeds 0..4)
            arguments
                track_spec = "Track1"
                opts.FailProb (1,1) double = NaN
                opts.MovementMode (1,1) string = ""
                opts.MaxVelocity (1,1) double = NaN
            end

            obj.MapProperties = struct();
            obj.parse_track_(track_spec);
            mp = obj.MapProperties;
            fail = opts.FailProb;
            if isnan(fail), fail = 0.1; if isfield(mp, 'fail_prob'), fail = double(mp.fail_prob); end, end
            mode = opts.MovementMode;
            if mode == "", mode = "forward-only"; if isfield(mp, 'movement_mode'), mode = string(mp.movement_mode); end, end
            vmax = opts.MaxVelocity;
            if isnan(vmax), vmax = obj.MAX_VELOCITY + 1; if isfield(mp, 'max_velocity'), vmax = double(mp.max_velocity); end, end
            if isfield(mp, 'max_episode_steps'), obj.MaxEpisodeSteps = double(mp.max_episode_steps); end
            if ~(fail >= 0 && fail <= 1)
                error('RacetrackEnv:FailProb', 'FailProb must be in [0, 1], got %g.', fail);
            end
            if ~ismember(mode, ["forward-only" "bidirectional"])
                error('RacetrackEnv:MovementMode', 'MovementMode must be "forward-only" or "bidirectional", got "%s".', mode);
            end
            if ~(vmax >= 2 && vmax == round(vmax))
                error('RacetrackEnv:MaxVelocity', 'MaxVelocity must be an integer >= 2 (exclusive speed bound), got %g.', vmax);
            end
            obj.FailProb = fail; obj.MovementMode = mode; obj.MaxVelocity = vmax;
            obj.VelocityMax = vmax - 1;
            obj.VelocityMin = 0; if mode == "bidirectional", obj.VelocityMin = 1 - vmax; end
            obj.build_state_lut_();
            obj.check_reachable_();
            obj.EpisodeDone = true;
            obj.ObservationSpace = OceanStats.RL.Env.Spaces.Box([1 1 obj.VelocityMin obj.VelocityMin], ...
                [obj.Width obj.Height obj.VelocityMax obj.VelocityMax]);
            obj.ActionSpace = OceanStats.RL.Env.Spaces.Discrete(obj.N_ACTIONS);
        end

        function meta = getMetadata(obj)
            meta = getMetadata@OceanStats.RL.Env.IEnvironment(obj);
            cm = OceanStats.RL.Env.Classic.RacetrackEnv.classMetadata();
            meta.SymbolicLatex = cm.SymbolicLatex;
            meta.Reference = cm.TextbookRef;
        end

        function setSeed(obj, seed)
            obj.Stream = RandStream('mt19937ar', 'Seed', seed);
        end

        % -------------------------------------------------------------------
        function set_fail_prob(obj, prob)
            % SET_FAIL_PROB Allows turning noise off (prob = 0) for evaluation.
            arguments
                obj
                prob (1,1) double {mustBeNonnegative, mustBeLessThanOrEqual(prob, 1)}
            end
            obj.FailProb = prob;
        end

        function set_state(obj, state)
            % SET_STATE Force environment internal state to [x, y, v_x, v_y].
            obj.CurrentState = state;
            obj.EpisodeDone  = obj.is_terminal(state);
        end

        % -------------------------------------------------------------------
        function [next_state, reward, terminated, truncated, info] = step(obj, action)
            % STEP Applies intended action index (1..9) and computes transition.
            %
            % Outputs:
            %   next_state - [x', y', v'_x, v'_y]
            %   reward     - scalar: -1 for all steps until finish
            %   terminated - logical: true if the finish line was crossed (never truncated)
            if obj.EpisodeDone
                error('RacetrackEnv:EpisodeEnded', ...
                    'Episode has ended. Call reset() before calling step().');
            end

            % Resolve intended acceleration
            if ~obj.ActionSpace.contains(action)
                error('RacetrackEnv:InvalidAction', 'Action index must be an integer in [1, 9].');
            end
            a_vec = obj.ACTION_GRID(action, :);  % [a_x, a_y]

            % Apply 10% acceleration failure rule (environment transition noise)
            if obj.rand_() < obj.FailProb
                realized_a = [0, 0];
            else
                realized_a = a_vec;
            end

            x  = obj.CurrentState(1);
            y  = obj.CurrentState(2);
            vx = obj.CurrentState(3);
            vy = obj.CurrentState(4);

            [outcome, end_x, end_y, vx_next, vy_next] = obj.transition_(x, y, vx, vy, realized_a);

            reward = -1;  % R = -1 for each step

            switch outcome
                case "Finish"
                    obj.EpisodeDone = true;
                    done            = true;
                    next_state      = [end_x, end_y, vx_next, vy_next];

                case "Crash"
                    % Reset to random starting cell with (0,0) velocity, episode continues
                    n_starts  = size(obj.StartCells, 1);
                    idx       = obj.randi_(n_starts);
                    start_pos = obj.StartCells(idx, :);

                    obj.CurrentState = [start_pos(1), start_pos(2), 0, 0];
                    done             = false;
                    next_state       = obj.CurrentState;

                case "Normal"
                    obj.CurrentState = [end_x, end_y, vx_next, vy_next];
                    done             = false;
                    next_state       = obj.CurrentState;
            end
            terminated = done;
            truncated = false;
            info = struct('action_mask', obj.actionMask_(next_state));
        end

        % -------------------------------------------------------------------
        function actions = get_valid_actions(obj, state)
            % GET_VALID_ACTIONS O(1) lookup of valid action indices (1..9).
            s_idx = obj.state_to_index(state);
            if s_idx > 0
                actions = obj.ValidActionsLUT{s_idx};
            else
                actions = 5;  % Default fallback
            end
        end

        function all_states = get_all_states(obj)
            all_states = obj.AllStatesCache;
        end

        function idx = state_to_index(obj, state)
            % STATE_TO_INDEX O(1) state lookup via 4D LUT.
            x  = state(1);
            y  = state(2);
            vx = state(3);
            vy = state(4);

            if x < 1 || x > obj.Width || y < 1 || y > obj.Height || ...
               vx < obj.VelocityMin || vx > obj.VelocityMax || vy < obj.VelocityMin || vy > obj.VelocityMax
                idx = 0;
                return;
            end

            idx = obj.StateIndexLUT(x, y, vx - obj.VelocityMin + 1, vy - obj.VelocityMin + 1);
            if isnan(idx), idx = 0; end
        end

        function action_idx = action_to_index(~, action)
            action_idx = action;
        end

        function action = index_to_action(~, action_idx)
            action = action_idx;
        end

        function n = get_num_actions(obj)
            n = obj.N_ACTIONS;
        end

        function tf = is_terminal(obj, state)
            x = state(1);
            y = state(2);
            if x < 1 || x > obj.Width || y < 1 || y > obj.Height
                tf = false;
                return;
            end
            tf = (obj.Grid(y, x) == 3);  % Finish line cell is terminal
        end
    end

    % -----------------------------------------------------------------------
    methods (Access = protected)
        % -------------------------------------------------------------------
        function [state, info] = doReset_(obj)
            % RESET Starts a new episode from a randomly chosen starting line cell.
            % Initial velocity is (v_x, v_y) = (0, 0).
            n_starts = size(obj.StartCells, 1);
            idx      = obj.randi_(n_starts);
            start_pos = obj.StartCells(idx, :);

            obj.CurrentState = [start_pos(1), start_pos(2), 0, 0];
            obj.EpisodeDone  = false;
            state            = obj.CurrentState;
            info             = struct('action_mask', obj.actionMask_(state));
        end

    end

    % -----------------------------------------------------------------------
    methods (Access = private)
        function u = rand_(obj)
            if isempty(obj.Stream), u = rand(); else, u = rand(obj.Stream); end
        end
        function k = randi_(obj, n)
            if isempty(obj.Stream), k = randi(n); else, k = randi(obj.Stream, n); end
        end
        function m = actionMask_(obj, state)
            m = false(1, obj.N_ACTIONS);
            if ~obj.is_terminal(state), m(obj.get_valid_actions(state)) = true; end
        end

        function parse_track_(obj, track_spec)
            % PARSE_TRACK_ Builds numeric Grid matrix from built-in or custom maps.
            % Grid encoding: 0 = OffTrack, 1 = Track, 2 = Start, 3 = Finish

            if isstring(track_spec) || ischar(track_spec)
                name = string(track_spec);
                if name == "Track1"
                    lines = obj.get_track1_ascii_();
                elseif name == "Track2"
                    lines = obj.get_track2_ascii_();
                else
                    % Search for file on disk or relative to maps directory
                    resolved_file = obj.find_map_file_(name);
                    if strlength(resolved_file) > 0            % (isempty("") is false for strings)
                        [~, ~, ext] = fileparts(resolved_file);
                        if strcmpi(ext, '.tmj') || strcmpi(ext, '.json')
                            obj.parse_tmj_file_(resolved_file, name);
                            return;
                        else
                            lines = readlines(resolved_file);
                            lines = lines(strlength(trim(lines)) > 0 & ~startsWith(trim(lines), "#"));
                        end
                    else
                        lines = string(track_spec);
                    end
                end
                obj.TrackName = name;
            else
                % Numeric matrix passed directly
                obj.Grid      = track_spec;
                obj.Height    = size(track_spec, 1);
                obj.Width     = size(track_spec, 2);
                obj.TrackName = "Custom";
                obj.extract_special_cells_();
                return;
            end

            % Parse ASCII lines (top line is y = Height, bottom line is y = 1)
            H = numel(lines);
            W = max(strlength(lines));
            grid = zeros(H, W);

            for row = 1:H
                str = char(lines(row));
                y = H - row + 1;  % Row 1 of matrix is y=H (top), Row H is y=1 (bottom)
                for col = 1:numel(str)
                    ch = str(col);
                    switch ch
                        case {'S', 's', '2'}, grid(y, col) = 2;  % Start line
                        case {'F', 'f', 'G', 'g', '3', '4'}, grid(y, col) = 3;  % Finish line
                        case {'.', 'O', 'o', '1'}, grid(y, col) = 1;  % Track
                        otherwise, grid(y, col) = 0;  % Off-track / Wall
                    end
                end
            end

            obj.Grid   = grid;
            obj.Height = H;
            obj.Width  = W;

            obj.extract_special_cells_();
        end

        function resolved = find_map_file_(~, name)
            % Map file by name: as given, then in the maps folders -- relative to the current
            % folder AND to the project root (so experiments run from any folder).
            resolved = "";
            if ~isscalar(name), return; end                 % several lines = a map given inline, not a file
            root = string(fileparts(fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))))));
            candidates = [ ...
                name, ...
                name + ".tmj", ...
                name + ".json", ...
                name + ".map", ...
                fullfile("Experiments", "Scripts", "RL", "Python", "maps", "tiled", name), ...
                fullfile("Experiments", "Scripts", "RL", "Python", "maps", "tiled", name + ".tmj"), ...
                fullfile("Experiments", "Scripts", "RL", "Python", "maps", name), ...
                fullfile("Experiments", "Scripts", "RL", "Python", "maps", name + ".map") ...
            ];
            candidates = [candidates, fullfile(root, candidates(5:end))];
            for i = 1:numel(candidates)
                f = candidates(i);
                if isfile(f)
                    resolved = f;
                    return;
                end
            end
        end

        function parse_tmj_file_(obj, file_path, name)
            raw = jsondecode(fileread(file_path));
            if isfield(raw, 'properties')                  % map-level properties (movement_mode, ...)
                pr = raw.properties; if ~iscell(pr), pr = num2cell(pr); end
                for i = 1:numel(pr)
                    obj.MapProperties.(matlab.lang.makeValidName(pr{i}.name)) = pr{i}.value;
                end
            end
            H = raw.height;
            W = raw.width;
            data_vec = raw.layers(1).data;
            raw_matrix = reshape(data_vec, [W, H])';

            grid = zeros(H, W);
            for r = 1:H
                y = H - r + 1;
                for c = 1:W
                    tile_id = raw_matrix(r, c);
                    switch tile_id
                        case 1, grid(y, c) = 0; % Wall / OffTrack
                        case 2, grid(y, c) = 1; % Track
                        case 3, grid(y, c) = 2; % Start line
                        case 4, grid(y, c) = 3; % Finish line
                        otherwise, grid(y, c) = 0;
                    end
                end
            end

            obj.Grid      = grid;
            obj.Height    = H;
            obj.Width     = W;
            obj.TrackName = name;
            obj.extract_special_cells_();
        end

        function extract_special_cells_(obj)
            % EXTRACT_SPECIAL_CELLS_ Caches StartCells and FinishCells coordinates.
            [sy, sx] = find(obj.Grid == 2);
            obj.StartCells = [sx, sy];

            [fy, fx] = find(obj.Grid == 3);
            obj.FinishCells = [fx, fy];

            if isempty(obj.StartCells)
                error('RacetrackEnv:NoStartLine', 'Track map contains no start line cells (code 2 / S).');
            end
            if isempty(obj.FinishCells)
                error('RacetrackEnv:NoFinishLine', 'Track map contains no finish line cells (code 3 / F).');
            end
        end

        function build_state_lut_(obj)
            % BUILD_STATE_LUT_ Pre-computes 4D LUT (x, y, vx-vmin+1, vy-vmin+1) -> linear index.
            vels = obj.VelocityMin:obj.VelocityMax; nv = numel(vels);
            lut   = nan(obj.Width, obj.Height, nv, nv);
            all_s = zeros(obj.Width * obj.Height * nv^2, 4);
            valid_acts_lut = cell(obj.Width * obj.Height * nv^2, 1);
            count = 0;

            for y = 1:obj.Height
                for x = 1:obj.Width
                    cell_type = obj.Grid(y, x);
                    if cell_type == 0 || cell_type == 3
                        continue;  % Skip off-track and terminal finish cells
                    end

                    for vx = vels
                        for vy = vels
                            % Velocity (0,0) is only valid on the starting line (cell_type == 2)
                            if vx == 0 && vy == 0 && cell_type ~= 2
                                continue;
                            end

                            count = count + 1;
                            lut(x, y, vx - obj.VelocityMin + 1, vy - obj.VelocityMin + 1) = count;
                            all_s(count, :) = [x, y, vx, vy];

                            % Pre-compute valid actions for this state
                            is_start = (cell_type == 2);
                            valid_mask = false(1, obj.N_ACTIONS);
                            for a = 1:obj.N_ACTIONS
                                a_vec  = obj.ACTION_GRID(a, :);
                                vx_new = vx + a_vec(1);
                                vy_new = vy + a_vec(2);

                                if vx_new >= obj.VelocityMin && vx_new <= obj.VelocityMax && ...
                                   vy_new >= obj.VelocityMin && vy_new <= obj.VelocityMax
                                    if ~(vx_new == 0 && vy_new == 0 && ~is_start)
                                        valid_mask(a) = true;
                                    end
                                end
                            end
                            acts = find(valid_mask);
                            if isempty(acts), acts = 5; end
                            valid_acts_lut{count} = acts;
                        end
                    end
                end
            end

            obj.TotalStates     = count;
            obj.StateIndexLUT   = lut;
            obj.AllStatesCache  = all_s(1:count, :);
            obj.ValidActionsLUT = valid_acts_lut(1:count);
        end

        function [outcome, end_x, end_y, vx_next, vy_next] = transition_(obj, x, y, vx, vy, realized_a)
            % TRANSITION_ Deterministic part of step: velocity update, no-stop rule, path check.
            vx_next = max(obj.VelocityMin, min(obj.VelocityMax, vx + realized_a(1)));
            vy_next = max(obj.VelocityMin, min(obj.VelocityMax, vy + realized_a(2)));
            if vx_next == 0 && vy_next == 0 && obj.Grid(y, x) ~= 2
                if obj.MovementMode == "bidirectional"
                    vx_next = vx; vy_next = vy;            % keep the previous velocity (as the Python twin)
                elseif vx > 0
                    vx_next = vx;                          % forward-only: the original rule (golden-tested)
                else
                    vy_next = max(1, vy);
                end
            end
            [outcome, end_x, end_y] = obj.check_path_(x, y, x + vx_next, y + vy_next);
        end

        function check_reachable_(obj)
            % CHECK_REACHABLE_ Breadth-first search (noise off) from the start line: some finish
            % cell must be reachable under the movement mode, otherwise no agent can ever finish.
            N = obj.TotalStates; seen = false(N, 1);
            q = zeros(N, 1); head = 1; tail = 0;
            for i = 1:size(obj.StartCells, 1)
                k = obj.state_to_index([obj.StartCells(i, :) 0 0]);
                if k > 0 && ~seen(k), seen(k) = true; tail = tail + 1; q(tail) = k; end
            end
            while head <= tail
                st = obj.AllStatesCache(q(head), :); head = head + 1;
                for a = 1:obj.N_ACTIONS
                    [out, ex, ey, nvx, nvy] = obj.transition_(st(1), st(2), st(3), st(4), obj.ACTION_GRID(a, :));
                    if out == "Finish", obj.FinishReachable = true; return; end
                    if out == "Crash", continue; end
                    k = obj.state_to_index([ex ey nvx nvy]);
                    if k > 0 && ~seen(k), seen(k) = true; tail = tail + 1; q(tail) = k; end
                end
            end
            hint = "";
            if obj.MovementMode == "forward-only"
                hint = " The map may need 'MovementMode', ""bidirectional"" (driving left/down).";
            end
            obj.FinishReachable = false;
            name = obj.TrackName; if ~isscalar(name), name = "inline map"; end
            warning('RacetrackEnv:FinishUnreachable', ...
                'No finish cell of "%s" can be reached from the start line in %s mode (speeds %d..%d; %d states explored).%s', ...
                name, obj.MovementMode, obj.VelocityMin, obj.VelocityMax, tail, hint);
        end

        function [outcome, final_x, final_y] = check_path_(obj, x0, y0, x1, y1)
            % CHECK_PATH_ Sub-pixel line traversal checking projected car path.
            %
            % Returns:
            %   outcome - "Finish", "Crash", or "Normal"
            %   final_x, final_y - resulting position
            dx = x1 - x0;
            dy = y1 - y0;

            if dx == 0 && dy == 0
                outcome = "Normal";
                final_x = x0;
                final_y = y0;
                return;
            end

            n_steps = max(2 * max(abs(dx), abs(dy)), 1);
            t_vec   = (1:n_steps) / n_steps;

            for k = 1:n_steps
                t  = t_vec(k);
                cx = round(x0 + t * dx);
                cy = round(y0 + t * dy);

                % Boundary check
                if cx < 1 || cx > obj.Width || cy < 1 || cy > obj.Height
                    outcome = "Crash";
                    final_x = cx;
                    final_y = cy;
                    return;
                end

                cell_type = obj.Grid(cy, cx);

                if cell_type == 3  % Finish line crossed
                    outcome = "Finish";
                    final_x = cx;
                    final_y = cy;
                    return;
                elseif cell_type == 0  % Off-track wall hit
                    outcome = "Crash";
                    final_x = cx;
                    final_y = cy;
                    return;
                end
            end

            outcome = "Normal";
            final_x = x1;
            final_y = y1;
        end

        function lines = get_track1_ascii_(~)
            % Track 1 layout (32 rows x 17 cols) — Sutton & Barto Fig 5.5 (Left)
            lines = [ ...
                "####..........FFF", ... % 32
                "####..........FFF", ... % 31
                "####..........FFF", ... % 30
                "####..........FFF", ... % 29
                "####..........FFF", ... % 28
                "####..........FFF", ... % 27
                "####........#####", ... % 26
                "####.......######", ... % 25
                "####.......######", ... % 24
                "####.......######", ... % 23
                "####.......######", ... % 22
                "####.......######", ... % 21
                "####.......######", ... % 20
                "####.......######", ... % 19
                "####.......######", ... % 18
                "####.......######", ... % 17
                "####.......######", ... % 16
                "####.......######", ... % 15
                "####.......######", ... % 14
                "####.......######", ... % 13
                "####.......######", ... % 12
                "####.......######", ... % 11
                "####.......######", ... % 10
                "####.......######", ... % 9
                "####.......######", ... % 8
                "####.......######", ... % 7
                "####.......######", ... % 6
                "####.......######", ... % 5
                "####.......######", ... % 4
                "####.......######", ... % 3
                "####.......######", ... % 2
                "####SSSSSS.######"   ... % 1
            ];
        end

        function lines = get_track2_ascii_(~)
            % Track 2 layout (32 rows x 32 cols) — Sutton & Barto Fig 5.5 (Right)
            lines = [ ...
                "#############################FFF", ... % 32
                "#############################FFF", ... % 31
                "#############################FFF", ... % 30
                "#############################FFF", ... % 29
                "#############################FFF", ... % 28
                "#############################FFF", ... % 27
                "#################...........####", ... % 26
                "###############.............####", ... % 25
                "##############..............####", ... % 24
                "#############...............####", ... % 23
                "############................####", ... % 22
                "###########.................####", ... % 21
                "##########..................####", ... % 20
                "#########...................####", ... % 19
                "########....................####", ... % 18
                "#######.....................####", ... % 17
                "######......................####", ... % 16
                "#####.......................####", ... % 15
                "####........................####", ... % 14
                "###.........................####", ... % 13
                "##..........................####", ... % 12
                "#...........................####", ... % 11
                "#...........................####", ... % 10
                "#...........................####", ... % 9
                "#...........................####", ... % 8
                "#...........................####", ... % 7
                "#...........................####", ... % 6
                "#...........................####", ... % 5
                "#...........................####", ... % 4
                "#...........................####", ... % 3
                "#...........................####", ... % 2
                "#SSSSSSSSSSSSSSSSSSSSSSS....####"   ... % 1
            ];
        end
    end

    % -----------------------------------------------------------------------
    methods (Static)
        function meta = classMetadata()
            meta = struct(...
                'ClassName', "OceanStats.RL.Env.Classic.RacetrackEnv", ...
                'Package', "+OceanStats/+RL/+Env/+Classic/RacetrackEnv.m", ...
                'Category', "OceanStats.RL", ...
                'TextbookRef', "Sutton & Barto, Reinforcement Learning: An Introduction (2nd Ed.), Exercise 5.12, Section 5.6", ...
                'TextbookEq', "S_t = (x_t, y_t, v_{x,t}, v_{y,t}), A_t \in \{-1, 0, 1\}^2, R_{t+1} = -1", ...
                'SymbolicLatex', "S_t = (x_t, y_t, v_{x,t}, v_{y,t}), \quad a_x, a_y \in \{-1, 0, +1\}, \quad R = -1", ...
                'Symbols', [ ...
                    struct('Symbol', 'S_t', 'Description', 'State tuple containing position (x_t, y_t) and velocity (v_{x,t}, v_{y,t})'), ...
                    struct('Symbol', 'A_t', 'Description', 'Intended acceleration vector (a_x, a_y) in {-1, 0, 1}^2'), ...
                    struct('Symbol', 'v_x, v_y', 'Description', 'Discrete velocity components restricted to {0, 1, 2, 3, 4}'), ...
                    struct('Symbol', 'R_t', 'Description', 'Immediate reward R = -1 for each step until finish line is crossed'), ...
                    struct('Symbol', 'P(Fail)', 'Description', 'Probability (0.1) of acceleration failure replacing intended acceleration with (0,0)') ...
                ], ...
                'MetadataText', "Discrete Racetrack episodic MDP environment (Sutton & Barto Exercise 5.12) modeling race car driving with momentum, 10% acceleration failure, and line traversal collision detection.", ...
                'Methods', [ ...
                    "RacetrackEnv(track_spec, opts) -> Initialized Environment", ...
                    "reset() -> initial state [x, y, 0, 0]", ...
                    "step(action) -> [next_state, reward, terminated, truncated, info]", ...
                    "get_valid_actions(state) -> valid action indices", ...
                    "state_to_index(state) -> O(1) linear index" ...
                ], ...
                'Capabilities', [ ...
                    "✓ Built-in Track1 and Track2 layouts from Sutton & Barto Fig 5.5", ...
                    "✓ 4D Lookup Table (LUT) O(1) state indexing", ...
                    "✓ Sub-pixel ray grid path traversal collision detection", ...
                    "✓ Stochastic 10% acceleration failure dynamics" ...
                ] ...
            );
        end

        function spec = classPipelineSpec()
            spec = struct(...
                'ComponentName', "RacetrackEnv", ...
                'Category', "Reinforcement Learning", ...
                'Description', "Discrete Racetrack episodic environment (Sutton & Barto Exercise 5.12) with momentum and line-segment collision detection", ...
                'Inputs', struct(...
                    'name', {'Action'}, ...
                    'type', {'double'}, ...
                    'shape', {'1 x 1'}, ...
                    'description', {'Action index in [1, 9]'} ...
                ), ...
                'Outputs', struct(...
                    'name', {'NextState', 'Reward', 'Done'}, ...
                    'type', {'double', 'double', 'logical'}, ...
                    'shape', {'1 x 4', '1 x 1', '1 x 1'}, ...
                    'description', {'Next state [x, y, vx, vy]', 'Step reward (-1)', 'Terminal status'} ...
                ), ...
                'DiagramSymbol', "🏎️ RacetrackEnv" ...
            );
        end
    end
end
