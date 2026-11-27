classdef RepeatedMatrixGameEnv < OceanAgents.Core.IJointEnvironment
    % REPEATEDMATRIXGAMEENV  A normal-form (matrix) game played simultaneously for a fixed number of rounds: the
    % smallest joint world. Every live slot is asked for one action per frame; all actions are revealed together,
    % each slot i receives u_i(profile) = game.payoff(i, profile), and after round T the episode terminates.
    %
    %   g   = OceanStats.GameTheory.Core.NormalFormGame('PayoffA', [1 -1; -1 1], 'PayoffB', [-1 1; 1 -1]);
    %   env = OceanAgents.Env.Reference.RepeatedMatrixGameEnv(g, 'Roster', ["p1" "p2"], 'Rounds', 100, ...
    %             'Observe', "own-payoff");
    %   f   = env.reset(0);                       % frame 1: both slots requested, observation zeros
    %   b   = OceanAgents.Core.ActionBatch('EpisodeID', f.EpisodeID, 'FrameID', f.FrameID, 'Decisions', [d1 d2]);
    %   f   = env.step(b);                        % f.Rewards = [u_1(profile) u_2(profile)]
    %
    % Stage game (Bonanno, Game Theory, ch. 1): G = (N, {S_i}, {u_i}); repeated T times with no discounting inside
    % the world (discounting belongs to each agent's host). Action labels of slot i are 1:|S_i| (strategy indices).
    % Observation of slot i after round t (zeros before round 1):
    %   "own-payoff"      [a_i^t  u_i(a^t)]
    %   "public-profile"  [a_1^t ... a_N^t  u_i(a^t)]
    % A slot never sees an action that has not yet been revealed: views are built only from completed rounds.
    % The finite horizon is part of the task, so the last round is Terminated (not Truncated) for every slot,
    % EpisodeEnded = true, EndReason = "horizon", and no slot is live or requested afterwards.
    % Dynamics are deterministic; reset(seed) records the seed (in ResearchInfo) but must still be called.
    % The whole ActionBatch is validated (ActionBatch.validate) before any state changes.
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 5.2.

    properties (SetAccess = private)
        Game                                    % OceanStats.GameTheory.Core.NormalFormGame (read only: payoff lookups)
        Roster string = strings(1, 0)           % 1 x N AgentIDs, slot k plays player k
        Rounds double = 1                       % horizon T
        Observe string = "own-payoff"
        EpisodeID double = 0                    % episodes started (monotonic)
        FrameID double = 0                      % frames issued in this run (monotonic, never reset)
        Round double = 0                        % completed rounds in the current episode
        Status string = "Idle"                  % "Idle" | "Running" | "Ended" | "Closed"
        Seed double = []                        % seed of the current episode (recorded only)
    end

    properties (Access = private)
        Frame_ = []                             % the current DecisionFrame
        LastProfile_ double = zeros(1, 0)       % last completed joint action (zeros before round 1)
        LastPayoffs_ double = zeros(1, 0)       % payoffs of the last completed round
    end

    methods
        function obj = RepeatedMatrixGameEnv(game, options)
            arguments
                game
                options.Roster = []
                options.Rounds (1,1) double {mustBeInteger, mustBePositive} = 1
                options.Observe (1,1) string {mustBeMember(options.Observe, ["own-payoff" "public-profile"])} = "own-payoff"
            end
            if ~(isa(game, 'OceanStats.GameTheory.Core.NormalFormGame') && isscalar(game))
                error('OceanAgents:Env:UnsupportedSpace', ...
                    'RepeatedMatrixGameEnv plays an OceanStats.GameTheory.Core.NormalFormGame (got %s).', class(game));
            end
            n = game.numPlayers();
            roster = options.Roster;
            if isempty(roster), roster = "p" + (1:n); end
            roster = convertCharsToStrings(roster);
            if ~(isstring(roster) && numel(roster) == n) || any(ismissing(roster)) || any(strlength(roster) == 0) ...
                    || numel(unique(roster)) ~= n
                error('OceanAgents:Core:InvalidRecord', 'Roster must be %d unique non-empty AgentIDs, one per player.', n);
            end
            obj.Game = game;
            obj.Roster = reshape(roster, 1, []);
            obj.Rounds = options.Rounds;
            obj.Observe = options.Observe;
        end

        function spec = describe(obj)
            % DESCRIBE  EnvSpec: simultaneous timing, Discrete(|S_i|) actions, Box observations, horizon T.
            n = numel(obj.Roster);
            obsSpaces = cell(1, n);
            actSpaces = cell(1, n);
            counts = arrayfun(@(i) obj.Game.numStrategies(i), 1:n);
            for i = 1:n
                u = obj.Game.playerTensor(i);
                lo = min(0, min(u(:)));
                hi = max(0, max(u(:)));
                if obj.Observe == "own-payoff"
                    obsSpaces{i} = OceanStats.RL.Env.Spaces.Box([0 lo], [counts(i) hi]);
                else
                    obsSpaces{i} = OceanStats.RL.Env.Spaces.Box([zeros(1, n) lo], [counts hi]);
                end
                actSpaces{i} = OceanStats.RL.Env.Spaces.Discrete(counts(i));
            end
            spec = OceanAgents.Core.EnvSpec('Name', "RepeatedMatrixGame", 'Timing', "simultaneous", ...
                'Roster', obj.Roster, 'Roles', "player" + (1:n), 'ObservationSpaces', obsSpaces, ...
                'ActionSpaces', actSpaces, 'Capabilities', OceanAgents.Core.CapabilitySet(), 'Horizon', obj.Rounds);
        end

        function frame = reset(obj, seed)
            % RESET  Start a new episode at round 0 (seed is recorded; the dynamics are deterministic).
            arguments
                obj
                seed (1,1) double {mustBeInteger, mustBeNonnegative}
            end
            if obj.Status == "Closed"
                error('OceanAgents:Env:NotReady', 'The environment has been closed.');
            end
            n = numel(obj.Roster);
            obj.Seed = seed;
            obj.EpisodeID = obj.EpisodeID + 1;
            obj.Round = 0;
            obj.LastProfile_ = zeros(1, n);
            obj.LastPayoffs_ = zeros(1, n);
            frame = obj.makeFrame_(zeros(1, n), false(1, n), false);
            obj.Status = "Running";
        end

        function frame = step(obj, batch)
            % STEP  Validate the whole batch, then play exactly one round.
            if obj.Status ~= "Running"
                error('OceanAgents:Env:NotReady', 'step called while the environment is %s; call reset(seed) first.', ...
                    obj.Status);
            end
            if ~(isa(batch, 'OceanAgents.Core.ActionBatch') && isscalar(batch))
                error('OceanAgents:Core:InvalidRecord', 'step expects a scalar OceanAgents.Core.ActionBatch.');
            end
            batch.validate(obj.Frame_);                 % throws before any mutation

            n = numel(obj.Roster);
            profile = zeros(1, n);
            for k = 1:n
                profile(k) = batch.decisionFor(obj.Roster(k)).Action;
            end
            payoffs = arrayfun(@(i) obj.Game.payoff(i, profile), 1:n);

            obj.Round = obj.Round + 1;
            obj.LastProfile_ = profile;
            obj.LastPayoffs_ = payoffs;
            done = obj.Round >= obj.Rounds;
            frame = obj.makeFrame_(payoffs, true(1, n), done);
            if done
                obj.Status = "Ended";
            end
        end

        function close(obj)
            % CLOSE  Idempotent; no further reset/step is allowed.
            obj.Status = "Closed";
            obj.Frame_ = [];
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            counts = arrayfun(@(i) obj.Game.numStrategies(i), 1:numel(obj.Roster));
            meta.Assumptions = {sprintf('%d-player normal-form game, strategy counts %s, repeated %d rounds', ...
                numel(obj.Roster), mat2str(counts), obj.Rounds), ...
                sprintf('simultaneous moves; observation "%s" built from completed rounds only', obj.Observe)};
            meta.IsSymbolic = false;
            meta.Reference = "repeated normal-form game (Bonanno, Game Theory, ch. 1); A0 contract section 5.2";
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.ComponentName = "RepeatedMatrixGameEnv";
            spec.Inputs = struct('Name', "batch", 'Domain', "Agent-ActionBatch", ...
                'Dtype', "OceanAgents.Core.ActionBatch", 'Shape', "1x1");
            spec.Outputs = struct('Name', "frame", 'Domain', "Agent-DecisionFrame", ...
                'Dtype', "OceanAgents.Core.DecisionFrame", 'Shape', string(mat2str([1 numel(obj.Roster)])));
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "repeatedmatrixgameenv");
            spec.ComposesFrom = {class(obj.Game)};
        end
    end

    methods (Access = private)
        function frame = makeFrame_(obj, rewards, rewardValid, done)
            n = numel(obj.Roster);
            obj.FrameID = obj.FrameID + 1;
            live = repmat(~done, 1, n);
            views = cell(1, n);
            for k = 1:n
                labels = 1:obj.Game.numStrategies(k);
                views{k} = OceanAgents.Core.ActorView('AgentID', obj.Roster(k), 'EpisodeID', obj.EpisodeID, ...
                    'FrameID', obj.FrameID, 'Tick', obj.Round, 'Observation', obj.observation_(k), ...
                    'ActionLabels', labels, 'ActionMask', repmat(~done, 1, numel(labels)));
            end
            endReason = "";
            if done, endReason = "horizon"; end
            frame = OceanAgents.Core.DecisionFrame('EpisodeID', obj.EpisodeID, 'FrameID', obj.FrameID, ...
                'Tick', obj.Round, 'Roster', obj.Roster, 'Live', live, 'Requested', live, 'Views', views, ...
                'Rewards', rewards, 'RewardValid', rewardValid, 'Terminated', repmat(done, 1, n), ...
                'Truncated', false(1, n), 'EpisodeEnded', done, 'EndReason', endReason, ...
                'ResearchInfo', struct('Round', obj.Round, 'LastProfile', obj.LastProfile_, 'Seed', obj.Seed));
            obj.Frame_ = frame;
        end

        function o = observation_(obj, k)
            % Observation of slot k: built from the last COMPLETED round only.
            if obj.Observe == "own-payoff"
                o = [obj.LastProfile_(k) obj.LastPayoffs_(k)];
            else
                o = [obj.LastProfile_ obj.LastPayoffs_(k)];
            end
        end
    end
end
