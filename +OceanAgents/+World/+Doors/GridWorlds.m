classdef GridWorlds
    % GRIDWORLDS  Factories that rebuild the native grid environments from doors (Agency + GridMotion +
    % RewardTermination). Options are exactly the native classes' options plus 'TimeLimit' (default Inf, applied by
    % RewardTermination) and 'AgentID' (default "agent1").
    %
    %   gw = OceanAgents.World.Doors.GridWorlds.cliffWalking();                         % CliffWalkingEnv()
    %   gw = OceanAgents.World.Doors.GridWorlds.randomWalk('NumInner', 19, 'LeftReward', -1, 'RightReward', 1);
    %   gw = OceanAgents.World.Doors.GridWorlds.windy('StochasticWind', true, 'Rows', 5, 'Start', [3 1], 'Goal', [3 8]);
    %   gw.Doors          [Agency, GridMotion, RewardTermination] DoorSpecs
    %   gw.Agency         AgentWorld configuration: ActionLabels, Routing (label k -> event "Move" with payload Label k),
    %                     Observe (the GridMotion Cell projection's Index), ObservationSpace, ActionSpace, Roles
    %   gw.Model          States (S x 1), Terminal (S x 1), ValidMask (S x A), Fn: [next, prob, reward] = Fn(s, a),
    %                     identical to the native TabularEnvironment tables (get_transitions, TerminalMask, ValidMask)
    %   gw.ActionLabels   1 x A
    %   gw.GridMotion, gw.RewardTermination   the door configurations (value data)
    %
    % Native equivalents (W2_Grid_Doors_Contract.md section 3.3):
    %   RandomWalkEnv(NumInner, LeftReward, RightReward): a 1 x (N+2) grid, moves left/right, the two end cells are
    %     terminal Goal cells with CellReward LeftReward / RightReward, StepReward 0, start at cell ceil(S/2).
    %   CliffWalkingEnv(Rows, Cols): reward -1 per step, CliffReward -100 (back to the start, not terminal), cliff = the
    %     bottom row between start and goal; cliff cells are terminal-like in the model exactly as natively.
    %   WindyGridworldEnv(Wind, Rows, Start, Goal, KingMoves, StayAction, StochasticWind): reward -1; move order
    %     up, right, down, left [, 4 diagonals] [, stay].

    methods (Static)
        function gw = cliffWalking(options)
            arguments
                options.Rows (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.Rows, 2)} = 4
                options.Cols (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.Cols, 3)} = 12
                options.TimeLimit (1,1) double = Inf
                options.AgentID (1,1) string = "agent1"
            end
            R = options.Rows;  C = options.Cols;
            cliff = false(R, C);  cliff(R, 2:C-1) = true;
            goal = false(R, C);  goal(R, C) = true;
            g = struct('Rows', R, 'Cols', C, 'Moves', [-1 0; 0 1; 1 0; 0 -1], 'Wind', zeros(1, C), ...
                'StochasticWind', false, 'Cliff', cliff, 'Start', [R 1], 'Terminal', goal | cliff, 'AgentID', options.AgentID);
            rt = struct('StepReward', -1, 'CellReward', NaN(R, C), 'CliffReward', -100, 'Goal', goal, ...
                'TimeLimit', options.TimeLimit);
            gw = OceanAgents.World.Doors.GridWorlds.assemble_("cliff", g, rt);
        end

        function gw = randomWalk(options)
            arguments
                options.NumInner (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.NumInner, 1)} = 5
                options.LeftReward (1,1) double = 0
                options.RightReward (1,1) double = 1
                options.TimeLimit (1,1) double = Inf
                options.AgentID (1,1) string = "agent1"
            end
            S = options.NumInner + 2;
            ends = false(1, S);  ends([1 S]) = true;
            cellReward = NaN(1, S);
            cellReward(1) = options.LeftReward;
            cellReward(S) = options.RightReward;
            g = struct('Rows', 1, 'Cols', S, 'Moves', [0 -1; 0 1], 'Wind', zeros(1, S), 'StochasticWind', false, ...
                'Cliff', false(1, S), 'Start', [1 ceil(S / 2)], 'Terminal', ends, 'AgentID', options.AgentID);
            rt = struct('StepReward', 0, 'CellReward', cellReward, 'CliffReward', 0, 'Goal', ends, ...
                'TimeLimit', options.TimeLimit);
            gw = OceanAgents.World.Doors.GridWorlds.assemble_("walk", g, rt);
        end

        function gw = windy(options)
            arguments
                options.Wind (1,:) double = [0 0 0 1 1 1 2 2 1 0]
                options.Rows (1,1) double {mustBeInteger, mustBePositive} = 7
                options.Start (1,2) double = [4 1]
                options.Goal (1,2) double = [4 8]
                options.KingMoves (1,1) logical = false
                options.StayAction (1,1) logical = false
                options.StochasticWind (1,1) logical = false
                options.TimeLimit (1,1) double = Inf
                options.AgentID (1,1) string = "agent1"
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
            goal = false(R, C);  goal(options.Goal(1), options.Goal(2)) = true;
            g = struct('Rows', R, 'Cols', C, 'Moves', moves, 'Wind', options.Wind, ...
                'StochasticWind', options.StochasticWind, 'Cliff', false(R, C), 'Start', options.Start, ...
                'Terminal', goal, 'AgentID', options.AgentID);
            rt = struct('StepReward', -1, 'CellReward', NaN(R, C), 'CliffReward', 0, 'Goal', goal, ...
                'TimeLimit', options.TimeLimit);
            gw = OceanAgents.World.Doors.GridWorlds.assemble_("windy", g, rt);
        end

        function [next, prob, reward] = modelStep(g, rt, s, a)
            % MODELSTEP  Exact model row of (s, a): next states, probabilities and rewards as column vectors, in the
            % native outcome order. An inadmissible (terminal) state or unknown action raises the native error ID
            % OceanStats:RL:Env:TabularEnvironment:InvalidAction.
            S = g.Rows * g.Cols;  A = size(g.Moves, 1);
            ok = isnumeric(s) && isscalar(s) && s >= 1 && s <= S && s == fix(s) && ...
                isnumeric(a) && isscalar(a) && a >= 1 && a <= A && a == fix(a);
            if ok
                r = floor((s - 1) / g.Cols) + 1;  c = mod(s - 1, g.Cols) + 1;
                ok = ~g.Terminal(r, c);
            end
            if ~ok
                error('OceanStats:RL:Env:TabularEnvironment:InvalidAction', ...
                    'Action %s is not admissible in state %s.', mat2str(a), mat2str(s));
            end
            [next, prob] = OceanAgents.World.Doors.GridMotion.transitions(g, s, a);
            [~, ~, fell] = OceanAgents.World.Doors.GridMotion.outcomesOf(g, s, a);
            reward = zeros(numel(next), 1);
            for k = 1:numel(next)
                reward(k) = OceanAgents.World.Doors.RewardTermination.rewardFor(rt, next(k), fell(k), g.Cols);
            end
        end
    end

    methods (Static, Access = private)
        function gw = assemble_(name, g, rt)
            g = OceanAgents.World.Doors.GridMotion.checkConfig(g);
            rt = OceanAgents.World.Doors.RewardTermination.checkConfig(rt);
            S = g.Rows * g.Cols;
            A = size(g.Moves, 1);
            labels = 1:A;
            doors = [OceanAgents.World.Doors.Agency.spec(), OceanAgents.World.Doors.GridMotion.spec(g), ...
                OceanAgents.World.Doors.RewardTermination.spec(rt)];
            routing = struct('Label', num2cell(labels), 'Type', "Move", 'Payload', num2cell(struct('Label', num2cell(labels))));
            agency = struct('ActionLabels', labels, 'Routing', routing, ...
                'Observe', @(view) view.GridMotion_Cell.Index, ...
                'ObservationSpace', OceanStats.RL.Env.Spaces.Discrete(S), ...
                'ActionSpace', OceanStats.RL.Env.Spaces.Discrete(A), 'Roles', "default", ...
                'Roster', string(g.AgentID));            % AgentWorld checks the world's agents against this roster
            terminal = reshape(g.Terminal', [], 1);                         % row-major cell order = native index
            model = struct('States', (1:S)', 'Terminal', terminal, 'ValidMask', repmat(~terminal, 1, A), ...
                'Fn', @(s, a) OceanAgents.World.Doors.GridWorlds.modelStep(g, rt, s, a));
            gw = struct('Name', name, 'Doors', doors, 'Agency', agency, 'Model', model, 'ActionLabels', labels, ...
                'GridMotion', g, 'RewardTermination', rt);
        end
    end
end
