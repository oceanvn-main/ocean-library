classdef ParityGuessEnv < OceanAgents.Core.IJointEnvironment
    % PARITYGUESSENV  Two players, simultaneous turns (PG1 contract). Each turn each player PUTS a number (only its
    % parity counts: even / odd) and GUESSES the parity of the other's number; then both numbers are revealed.
    %   exactly one guessed right -> that player wins (+1, the other -1), game over
    %   both guessed right        -> draw (0, 0), game over
    %   neither                   -> next turn;  after MaxTurns turns: draw (0, 0), game over (a rule: terminated)
    %
    %   env = OceanAgents.Env.Reference.ParityGuessEnv('Roster', ["p1" "p2"], 'MaxTurns', 200);
    %   env.Factors                % ActionFactors ["put" "guess"], labels ["even" "odd"] each; joint labels 1..4
    %
    % Observation of player i: the parity the OTHER player put last turn (0 = none yet, 1 = even, 2 = odd). Joint
    % action k = put + 2 (guess - 1) (first factor fastest).

    properties (SetAccess = private)
        Roster string = ["p1" "p2"]
        MaxTurns double = 200
        Factors
        EpisodeID double = 0
        FrameID double = 0
        Turn double = 0
        Status string = "Idle"
        LastPuts double = [0 0]
        Outcome string = ""
    end

    properties (Access = private)
        Frame_ = []
    end

    methods
        function obj = ParityGuessEnv(options)
            arguments
                options.Roster (1,2) string = ["p1" "p2"]
                options.MaxTurns (1,1) double {mustBeInteger, mustBePositive} = 200
            end
            if options.Roster(1) == options.Roster(2)
                error('OceanAgents:Core:InvalidRecord', 'ParityGuessEnv needs two different players.');
            end
            obj.Roster = options.Roster;
            obj.MaxTurns = options.MaxTurns;
            obj.Factors = OceanAgents.Core.ActionFactors(["put" "guess"], {["even" "odd"], ["even" "odd"]});
        end

        function spec = describe(obj)
            obs = {OceanStats.RL.Env.Spaces.Discrete(3, 'Start', 0), OceanStats.RL.Env.Spaces.Discrete(3, 'Start', 0)};
            act = {OceanStats.RL.Env.Spaces.Discrete(4), OceanStats.RL.Env.Spaces.Discrete(4)};
            spec = OceanAgents.Core.EnvSpec('Name', "ParityGuess", 'Timing', "simultaneous", 'Roster', obj.Roster, ...
                'Roles', ["player" "player"], 'ObservationSpaces', obs, 'ActionSpaces', act, ...
                'Capabilities', OceanAgents.Core.CapabilitySet(), 'Horizon', obj.MaxTurns);
        end

        function frame = reset(obj, seed)
            arguments
                obj
                seed (1,1) double {mustBeInteger, mustBeNonnegative} %#ok<INUSA> the game itself is deterministic
            end
            if obj.Status == "Closed", error('OceanAgents:Env:NotReady', 'The environment has been closed.'); end
            obj.EpisodeID = obj.EpisodeID + 1;
            obj.Turn = 0;
            obj.LastPuts = [0 0];
            obj.Outcome = "";
            obj.Status = "Running";
            frame = obj.makeFrame_([0 0], false(1, 2), false);
        end

        function frame = step(obj, batch)
            if obj.Status ~= "Running"
                error('OceanAgents:Env:NotReady', 'step called while the game is %s; call reset(seed) first.', obj.Status);
            end
            batch.validate(obj.Frame_);
            puts = zeros(1, 2);  guesses = zeros(1, 2);
            for k = 1:2
                s = obj.Factors.parts(batch.decisionFor(obj.Roster(k)).Action);
                puts(k) = s(1);  guesses(k) = s(2);
            end
            right = [guesses(1) == puts(2), guesses(2) == puts(1)];
            obj.Turn = obj.Turn + 1;
            obj.LastPuts = puts;
            rewards = [0 0];
            done = true;
            if right(1) && ~right(2)
                rewards = [1 -1];  obj.Outcome = "win:" + obj.Roster(1);
            elseif right(2) && ~right(1)
                rewards = [-1 1];  obj.Outcome = "win:" + obj.Roster(2);
            elseif all(right)
                obj.Outcome = "draw:both-right";
            elseif obj.Turn >= obj.MaxTurns
                obj.Outcome = "draw:turns";
            else
                done = false;
            end
            if done, obj.Status = "Ended"; end
            frame = obj.makeFrame_(rewards, true(1, 2), done);
        end

        function close(obj)
            obj.Status = "Closed";
            obj.Frame_ = [];
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = "P(\text{P2 wins}) = \frac{p_2}{1 + p_2} \text{ when P1 guesses uniformly}";
            meta.Symbols = containers.Map();
            meta.Assumptions = {sprintf('two players, simultaneous; at most %d turns', obj.MaxTurns), ...
                'one right guess wins; both right is a draw; numbers revealed after each turn'};
            meta.IsSymbolic = false;
            meta.Reference = "PG1 parity-guess game (user design, 2026-10-01)";
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.ComponentName = "ParityGuessEnv";
            spec.Inputs = struct('Name', "batch", 'Domain', "Agent-ActionBatch", 'Dtype', "OceanAgents.Core.ActionBatch", ...
                'Shape', "1x1");
            spec.Outputs = struct('Name', "frame", 'Domain', "Agent-DecisionFrame", ...
                'Dtype', "OceanAgents.Core.DecisionFrame", 'Shape', string(mat2str([1 numel(obj.Roster)])));
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "parityguessenv");
            spec.ComposesFrom = {};
        end
    end

    methods (Access = private)
        function frame = makeFrame_(obj, rewards, rewardValid, done)
            obj.FrameID = obj.FrameID + 1;
            views = cell(1, 2);
            for k = 1:2
                views{k} = OceanAgents.Core.ActorView('AgentID', obj.Roster(k), 'EpisodeID', obj.EpisodeID, ...
                    'FrameID', obj.FrameID, 'Tick', obj.Turn, 'Observation', obj.LastPuts(3 - k), ...
                    'ActionLabels', 1:4, 'ActionMask', repmat(~done, 1, 4));
            end
            frame = OceanAgents.Core.DecisionFrame('EpisodeID', obj.EpisodeID, 'FrameID', obj.FrameID, ...
                'Tick', obj.Turn, 'Roster', obj.Roster, 'Live', repmat(~done, 1, 2), 'Requested', repmat(~done, 1, 2), ...
                'Views', views, 'Rewards', rewards, 'RewardValid', rewardValid, 'Terminated', repmat(done, 1, 2), ...
                'Truncated', false(1, 2), 'EpisodeEnded', done, 'EndReason', obj.Outcome, ...
                'ResearchInfo', struct('Turn', obj.Turn, 'Puts', obj.LastPuts));
            obj.Frame_ = frame;
        end
    end
end
