classdef TicTacToeEnv < OceanAgents.Core.IJointEnvironment
    % TICTACTOEENV  Two players take turns on a 3 x 3 board (AB2_Observation_Contract.md success test O6). The Laws:
    % only the player whose turn it is may move, and only onto an empty cell; anything else is refused before the
    % board changes. Three in a row wins (+1 for the winner, -1 for the loser); a full board is a draw (0, 0).
    %
    %   env = OceanAgents.Env.Reference.TicTacToeEnv('Roster', ["x" "o"]);     % "x" moves first
    %   frame = env.reset(7);                     % frame.Requested marks the player to move
    %   frame = env.step(batch);                  % one move by that player
    %
    % Cells are labelled 1..9 row by row. Each player's view: Observation = the board from its own side (1 x 9:
    % 0 empty, 1 own mark, 2 the opponent's), ActionMask = the empty cells on its turn, nothing otherwise.

    properties (SetAccess = private)
        Roster string = ["x" "o"]
        EpisodeID double = 0
        FrameID double = 0
        Board double = zeros(1, 9)                % 0 empty, 1 = Roster(1)'s mark, 2 = Roster(2)'s mark
        Mover double = 1                          % slot whose turn it is
        Moves double = 0                          % moves made in this episode
        Status string = "Idle"
        Winner double = 0                         % 0 none / draw, else the winning slot
    end

    properties (Constant)
        Lines = [1 2 3; 4 5 6; 7 8 9; 1 4 7; 2 5 8; 3 6 9; 1 5 9; 3 5 7]
    end

    properties (Access = private)
        Frame_ = []
    end

    methods
        function obj = TicTacToeEnv(options)
            arguments
                options.Roster (1,2) string = ["x" "o"]
            end
            if numel(unique(options.Roster)) ~= 2
                error('OceanAgents:Core:InvalidRecord', 'TicTacToeEnv needs two different AgentIDs.');
            end
            obj.Roster = options.Roster;
        end

        function spec = describe(obj)
            obs = {OceanStats.RL.Env.Spaces.Box(zeros(1, 9), 2 * ones(1, 9)), ...
                   OceanStats.RL.Env.Spaces.Box(zeros(1, 9), 2 * ones(1, 9))};
            act = {OceanStats.RL.Env.Spaces.Discrete(9), OceanStats.RL.Env.Spaces.Discrete(9)};
            spec = OceanAgents.Core.EnvSpec('Name', "TicTacToe", 'Timing', "turn", 'Roster', obj.Roster, ...
                'Roles', ["first" "second"], 'ObservationSpaces', obs, 'ActionSpaces', act, ...
                'Capabilities', OceanAgents.Core.CapabilitySet(), 'Horizon', 9);
        end

        function frame = reset(obj, seed)
            arguments
                obj
                seed (1,1) double {mustBeInteger, mustBeNonnegative} %#ok<INUSA> deterministic game; seed recorded only
            end
            if obj.Status == "Closed", error('OceanAgents:Env:NotReady', 'The environment has been closed.'); end
            obj.EpisodeID = obj.EpisodeID + 1;
            obj.Board = zeros(1, 9);
            obj.Mover = 1;
            obj.Moves = 0;
            obj.Winner = 0;
            obj.Status = "Running";
            frame = obj.makeFrame_([0 0], false(1, 2), false);
        end

        function frame = step(obj, batch)
            if obj.Status ~= "Running"
                error('OceanAgents:Env:NotReady', 'step called while the game is %s; call reset(seed) first.', obj.Status);
            end
            if ~(isa(batch, 'OceanAgents.Core.ActionBatch') && isscalar(batch))
                error('OceanAgents:Core:InvalidRecord', 'step expects a scalar OceanAgents.Core.ActionBatch.');
            end
            batch.validate(obj.Frame_);               % the Laws: right player, an empty cell; throws before any change
            square = batch.decisionFor(obj.Roster(obj.Mover)).Action;
            obj.Board(square) = obj.Mover;
            obj.Moves = obj.Moves + 1;
            won = any(all(obj.Board(obj.Lines) == obj.Mover, 2));
            rewards = [0 0];
            done = false;
            if won
                obj.Winner = obj.Mover;
                rewards(obj.Mover) = 1;
                rewards(3 - obj.Mover) = -1;
                done = true;
            elseif obj.Moves == 9
                done = true;                          % a draw
            else
                obj.Mover = 3 - obj.Mover;
            end
            if done, obj.Status = "Ended"; end
            frame = obj.makeFrame_(rewards, true(1, 2), done);
        end

        function close(obj)
            obj.Status = "Closed";
            obj.Frame_ = [];
        end

        function meta = getMetadata(~)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = {'two players alternate; the first player in the roster moves first', ...
                'the Laws: only the player to move, only an empty cell; win +1 / loss -1 / draw 0'};
            meta.IsSymbolic = false;
            meta.Reference = "tic-tac-toe (AB2 success test O6)";
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.ComponentName = "TicTacToeEnv";
            spec.Inputs = struct('Name', "batch", 'Domain', "Agent-ActionBatch", 'Dtype', "OceanAgents.Core.ActionBatch", ...
                'Shape', "1x1");
            spec.Outputs = struct('Name', "frame", 'Domain', "Agent-DecisionFrame", ...
                'Dtype', "OceanAgents.Core.DecisionFrame", 'Shape', string(mat2str([1 numel(obj.Roster)])));
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "tictactoeenv");
            spec.ComposesFrom = {};
        end
    end

    methods (Access = private)
        function frame = makeFrame_(obj, rewards, rewardValid, done)
            obj.FrameID = obj.FrameID + 1;
            requested = false(1, 2);
            if ~done, requested(obj.Mover) = true; end
            views = cell(1, 2);
            for k = 1:2
                own = obj.Board;                      % the board from this player's side
                own(obj.Board == k) = 1;
                own(obj.Board == 3 - k) = 2;
                views{k} = OceanAgents.Core.ActorView('AgentID', obj.Roster(k), 'EpisodeID', obj.EpisodeID, ...
                    'FrameID', obj.FrameID, 'Tick', obj.Moves, 'Observation', own, 'ActionLabels', 1:9, ...
                    'ActionMask', requested(k) & obj.Board == 0);
            end
            reason = "";
            if done
                reason = "draw";
                if obj.Winner > 0, reason = "win:" + obj.Roster(obj.Winner); end
            end
            frame = OceanAgents.Core.DecisionFrame('EpisodeID', obj.EpisodeID, 'FrameID', obj.FrameID, ...
                'Tick', obj.Moves, 'Roster', obj.Roster, 'Live', repmat(~done, 1, 2), 'Requested', requested, ...
                'Views', views, 'Rewards', rewards, 'RewardValid', rewardValid, 'Terminated', repmat(done, 1, 2), ...
                'Truncated', false(1, 2), 'EpisodeEnded', done, 'EndReason', reason, ...
                'ResearchInfo', struct('Board', obj.Board, 'Winner', obj.Winner));
            obj.Frame_ = frame;
        end
    end
end
