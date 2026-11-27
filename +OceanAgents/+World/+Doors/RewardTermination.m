classdef RewardTermination
    % REWARDTERMINATION  The reward-and-ending door of the grid worlds: turns each completed move into a reward and
    % decides whether the agent's episode ended (a goal cell was entered, or the time limit was reached).
    %
    %   cfg  = struct('StepReward', -1, 'CellReward', NaN(4, 12), 'CliffReward', -100, 'Goal', goalMask, 'TimeLimit', Inf);
    %   door = OceanAgents.World.Doors.RewardTermination.spec(cfg);   % DoorSpec "RewardTermination"
    %
    % Owns     Steps: value, double, width 1, dimension N (count), visibility "self".
    % Rule Settle (phase "settle", consumes "Moved" and "CliffFall" from GridMotion), per event in delivery order:
    %          1. reward = CliffReward for a CliffFall; else CellReward(entered cell) if not NaN; else StepReward;
    %          2. emit "Reward" (payload Amount; SourceEntity = the agent);
    %          3. Steps = Steps + 1;
    %          4. entered cell is a Goal cell -> emit "Terminated" (payload Reason "terminal");
    %          5. otherwise Steps >= TimeLimit -> emit "Truncated" (payload Reason "time-limit")
    %             (termination takes precedence over truncation at the limit).
    % The entered cell is the event's To (native index (row - 1) * Cols + col, read with GridMotion's Cols).
    % Reward, Terminated and Truncated are meant for the host (the AgentWorld lists them as 'Outbox' types).
    % Contract: W2_Grid_Doors_Contract.md section 3.2.

    properties (Constant)
        Name = "RewardTermination"
    end

    methods (Static)
        function t = stepsType()
            % STEPSTYPE  The declared type of Steps (GridMotion's Init offers it with this type).
            D = OceanAgents.World.Guard.Dimension;
            t = OceanAgents.World.Guard.ComponentType('Dimension', D.N(), 'Visibility', "self");
        end

        function door = spec(cfg)
            % SPEC  The DoorSpec for a validated configuration.
            cfg = OceanAgents.World.Doors.RewardTermination.checkConfig(cfg);
            res = struct('StepReward', cfg.StepReward, 'CellReward', cfg.CellReward, 'CliffReward', cfg.CliffReward, ...
                'Goal', cfg.Goal, 'TimeLimit', cfg.TimeLimit);
            settle = OceanAgents.World.RuleSpec('Name', "Settle", 'Phase', "settle", 'Requires', "Steps", ...
                'Reads', ["Steps" "StepReward" "CellReward" "CliffReward" "Goal" "TimeLimit" "Cols"], ...
                'Writes', "Steps", 'Consumes', ["Moved" "CliffFall"], 'Emits', ["Reward" "Terminated" "Truncated"], ...
                'Fn', @OceanAgents.World.Doors.RewardTermination.settleRule);
            door = OceanAgents.World.DoorSpec('Name', OceanAgents.World.Doors.RewardTermination.Name, 'Version', "1", ...
                'Owns', struct('Name', "Steps", 'Type', OceanAgents.World.Doors.RewardTermination.stepsType()), ...
                'Resources', res, 'Membership', "Steps", 'Rules', settle, 'Drive', "on-event");
        end

        function cfg = checkConfig(cfg)
            % CHECKCONFIG  Validate a RewardTermination configuration (OceanAgents:World:BadValue).
            bad = @(msg) error('OceanAgents:World:BadValue', 'RewardTermination: %s', msg);
            need = {'StepReward', 'CellReward', 'CliffReward', 'Goal', 'TimeLimit'};
            if ~isstruct(cfg) || ~isscalar(cfg) || ~all(isfield(cfg, need))
                bad(sprintf('cfg needs fields %s.', strjoin(need, ', ')));
            end
            fin = @(x) isnumeric(x) && isscalar(x) && isfinite(x);
            if ~fin(cfg.StepReward) || ~fin(cfg.CliffReward), bad('StepReward and CliffReward must be finite scalars.'); end
            if ~(isnumeric(cfg.CellReward) && ismatrix(cfg.CellReward) && ~any(isinf(cfg.CellReward(:))))
                bad('CellReward must be a Rows x Cols numeric matrix (NaN = use StepReward).');
            end
            if ~(islogical(cfg.Goal) && isequal(size(cfg.Goal), size(cfg.CellReward)))
                bad('Goal must be a logical matrix of the same size as CellReward.');
            end
            L = cfg.TimeLimit;
            if ~(isnumeric(L) && isscalar(L) && L >= 1 && (isinf(L) || L == fix(L)))
                bad('TimeLimit must be a positive integer or Inf.');
            end
            cfg.StepReward = double(cfg.StepReward);  cfg.CliffReward = double(cfg.CliffReward);
            cfg.CellReward = double(cfg.CellReward);  cfg.TimeLimit = double(L);
        end

        function amount = rewardFor(cfg, enteredIndex, isCliffFall, cols)
            % REWARDFOR  The reward of one move (the model and the rule share this definition).
            if isCliffFall
                amount = cfg.CliffReward;
                return
            end
            r = floor((enteredIndex - 1) / cols) + 1;  c = mod(enteredIndex - 1, cols) + 1;
            amount = cfg.CellReward(r, c);
            if isnan(amount), amount = cfg.StepReward; end
        end

        function out = settleRule(ctx)
            % SETTLERULE  Rule Fn of Settle (see the class help).
            g = ctx.Read;
            steps = g.Steps;
            touched = false(numel(ctx.Rows), 1);
            events = OceanAgents.World.Event.empty(1, 0);
            for e = reshape(ctx.Events, 1, [])
                i = find(ctx.IDs == e.SourceEntity, 1);
                if isempty(i)
                    error('OceanAgents:World:BadValue', 'RewardTermination.Settle: entity %d has no Steps.', e.SourceEntity);
                end
                to = e.Payload.To;
                amount = OceanAgents.World.Doors.RewardTermination.rewardFor(g, to, e.Type == "CliffFall", g.Cols);
                events(end + 1) = OceanAgents.World.Event.make("Reward", struct('Amount', amount), ...
                    'SourceEntity', e.SourceEntity); %#ok<AGROW>
                steps(i) = steps(i) + 1;
                touched(i) = true;
                r = floor((to - 1) / g.Cols) + 1;  c = mod(to - 1, g.Cols) + 1;
                if g.Goal(r, c)
                    events(end + 1) = OceanAgents.World.Event.make("Terminated", struct('Reason', "terminal"), ...
                        'SourceEntity', e.SourceEntity); %#ok<AGROW>
                elseif steps(i) >= g.TimeLimit
                    events(end + 1) = OceanAgents.World.Event.make("Truncated", struct('Reason', "time-limit"), ...
                        'SourceEntity', e.SourceEntity); %#ok<AGROW>
                end
            end
            out = struct('Writes', struct('Component', "Steps", 'Rows', ctx.Rows(touched), 'Values', steps(touched), ...
                'Rates', []), 'Events', events);
        end
    end
end
