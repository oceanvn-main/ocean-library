classdef ChangingMazeEnv < OceanStats.RL.Env.Classic.GridTabularEnv
    % CHANGINGMAZEENV Nonstationary maze used in Sutton & Barto Ch. 8.
    % Actions are 1=up, 2=right, 3=down, 4=left. A blocked move leaves
    % the state unchanged. Reaching the goal yields one; all other rewards
    % are zero. The wall changes after ChangeStep real interactions.
    properties (SetAccess = immutable)
        StartState (1,1) double
        GoalState (1,1) double
        ChangeStep (1,1) double
        InitialBlocked (:,1) double
        ChangedBlocked (:,1) double
    end
    properties (SetAccess = private)
        TotalSteps (1,1) double = 0
    end

    methods
        function obj = ChangingMazeEnv(options)
            arguments
                options.Rows (1,1) double {mustBeInteger,mustBePositive} = 6
                options.Cols (1,1) double {mustBeInteger,mustBePositive} = 9
                options.StartRC (1,2) double {mustBeInteger,mustBePositive} = [4 1]
                options.GoalRC (1,2) double {mustBeInteger,mustBePositive} = [1 9]
                options.ChangeStep (1,1) double {mustBeInteger,mustBePositive} = 1000
                options.InitialBlockedRC (:,2) double {mustBeInteger,mustBePositive} = [3*ones(8,1) (2:9)']
                options.ChangedBlockedRC (:,2) double {mustBeInteger,mustBePositive} = [3*ones(8,1) (1:8)']
            end
            obj.Rows = options.Rows;
            obj.Cols = options.Cols;
            obj.validateCoordinates_(options.StartRC, "StartRC");
            obj.validateCoordinates_(options.GoalRC, "GoalRC");
            obj.validateCoordinates_(options.InitialBlockedRC, "InitialBlockedRC");
            obj.validateCoordinates_(options.ChangedBlockedRC, "ChangedBlockedRC");
            obj.StartState = obj.idx(options.StartRC(1), options.StartRC(2));
            obj.GoalState = obj.idx(options.GoalRC(1), options.GoalRC(2));
            obj.ChangeStep = options.ChangeStep;
            obj.InitialBlocked = obj.toIndices_(options.InitialBlockedRC);
            obj.ChangedBlocked = obj.toIndices_(options.ChangedBlockedRC);
            if any([obj.InitialBlocked; obj.ChangedBlocked] == obj.StartState) || ...
                    any([obj.InitialBlocked; obj.ChangedBlocked] == obj.GoalState)
                error('OceanExperiments:Dyna:BlockedEndpoint', ...
                    'The start and goal cells must remain traversable.');
            end

            % The table supplies the stationary state/action enumeration.
            % step() below computes the time-dependent transition explicitly.
            S = obj.Rows * obj.Cols;
            terminal = false(S,1); terminal(obj.GoalState) = true;
            model = cell(S,4);
            for s = 1:S
                if terminal(s), continue; end
                for a = 1:4
                    model{s,a} = struct('Next',s,'Prob',1,'Reward',0);
                end
            end
            start = zeros(S,1); start(obj.StartState) = 1;
            obj.setModel_(model,terminal,start);
        end

        function [obs,reward,terminated,truncated,info] = step(obj,action)
            arguments
                obj
                action (1,1) double {mustBeInteger}
            end
            if ~obj.Ready_
                error('OceanExperiments:Dyna:NotReset','Call reset before step.');
            end
            if ~obj.ActionSpace.contains(action) || obj.Current == obj.GoalState
                error('OceanExperiments:Dyna:InvalidAction','Action is not admissible.');
            end
            obj.TotalSteps = obj.TotalSteps + 1;
            moves = [-1 0; 0 1; 1 0; 0 -1];
            [row,col] = obj.rc(obj.Current);
            nextRow = min(max(row + moves(action,1),1),obj.Rows);
            nextCol = min(max(col + moves(action,2),1),obj.Cols);
            candidate = obj.idx(nextRow,nextCol);
            if obj.TotalSteps <= obj.ChangeStep
                blocked = obj.InitialBlocked;
            else
                blocked = obj.ChangedBlocked;
            end
            if any(candidate == blocked), candidate = obj.Current; end
            obj.Current = candidate;
            obs = candidate;
            terminated = candidate == obj.GoalState;
            truncated = false;
            reward = double(terminated);
            info = struct('action_mask',obj.ValidMask(obs,:));
        end

        function resetClock(obj)
            obj.TotalSteps = 0;
        end

        function snapshot = inspect(obj)
            snapshot = struct('Rows',obj.Rows,'Cols',obj.Cols, ...
                'StartState',obj.StartState,'GoalState',obj.GoalState, ...
                'ChangeStep',obj.ChangeStep,'InitialBlocked',obj.InitialBlocked, ...
                'ChangedBlocked',obj.ChangedBlocked,'TotalSteps',obj.TotalSteps);
        end
    end

    methods (Access = private)
        function validateCoordinates_(obj,rc,name)
            if any(rc(:,1) > obj.Rows) || any(rc(:,2) > obj.Cols)
                error('OceanExperiments:Dyna:CoordinateOutsideMaze', ...
                    '%s contains a coordinate outside the maze.',name);
            end
        end

        function indices = toIndices_(obj,rc)
            indices = (rc(:,1)-1)*obj.Cols + rc(:,2);
        end
    end
end
