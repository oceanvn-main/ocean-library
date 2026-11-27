classdef LineDoors
    % LINEDOORS  Test fixture for AgentWorld: agents on an integer line (W2 tests only; independent of the grid doors).
    %
    %   L = OceanTest.OceanAgents.Fixtures.World.Agent.LineDoors;
    %   doors = [OceanAgents.World.Doors.Agency.spec(), L.line(["a1" "a2"], 'Goal', 3, 'Limit', 5)];
    %   aw = OceanAgents.World.AgentWorld(doors, L.cfg(["a1" "a2"]));
    %
    % Door "line" owns Pos (self: each agent sees only its own) and Steps (public; the projection's Requires).
    % Init spawns one entity per AgentID (Pos = Start, Steps = 0). Rule Move (act) consumes "Move" (payload Delta):
    % Pos += Delta, Steps += 1, and emits for the agent: Reward -1; on reaching Pos >= Goal also Reward +Bonus and
    % Terminated ("terminal"); when Steps >= Limit, Truncated ("time-limit") - even at the goal, so termination must win.
    % Action labels [1 2] route to Move with Delta -1 / +1.

    methods (Static)
        function d = line(agentIDs, options)
            arguments
                agentIDs string
                options.Goal (1,1) double = 3
                options.Limit (1,1) double = Inf
                options.Bonus (1,1) double = 10
                options.Start (1,1) double = 0
            end
            T = OceanAgents.World.Guard.ComponentType;
            pos = OceanAgents.World.Guard.ComponentType('Visibility', "self");
            ids = reshape(agentIDs, 1, []);
            start = options.Start;
            L = 'OceanTest.OceanAgents.Fixtures.World.Agent.LineDoors';
            % anonymous (not nested) functions: they capture only these values, never the door spec itself
            init = @(s) feval([L '.spawns'], ids, start, pos, T);
            goal = options.Goal;  limit = options.Limit;  bonus = options.Bonus;
            move = OceanAgents.World.RuleSpec('Name', "Move", 'Phase', "act", 'Consumes', "Move", 'Requires', "Pos", ...
                'Reads', ["Pos" "Steps"], 'Writes', ["Pos" "Steps"], 'Emits', ["Reward" "Terminated" "Truncated"], ...
                'Fn', @(ctx) feval([L '.moveFn'], ctx, goal, limit, bonus));
            see = OceanAgents.World.RuleSpec('Name', "See", 'Phase', "project", 'Kind', "projection", ...
                'Requires', "Steps", 'Reads', "Pos", 'Fn', @(ctx) struct('View', struct('Pos', ctx.Read.Pos(ctx.Read.Pos_Has))));
            d = OceanAgents.World.DoorSpec('Name', "line", 'Owns', struct('Name', {"Pos", "Steps"}, 'Type', {pos, T}), ...
                'Rules', [move see], 'Drive', "on-event", 'Init', init);
        end

        function cmds = spawns(ids, start, pos, T)
            % One agent entity per AgentID: Pos = start, Steps = 0.
            cmds = OceanAgents.World.Command.empty(1, 0);
            for id = ids
                cmds(end + 1) = OceanAgents.World.Command.spawn(struct( ...
                    'AgentID', struct('Value', id, 'Type', OceanAgents.World.Doors.Agency.idType()), ...
                    'Pos', struct('Value', start, 'Type', pos), 'Steps', struct('Value', 0, 'Type', T))); %#ok<AGROW>
            end
        end

        function out = moveFn(ctx, goal, limit, bonus)
            E = @OceanAgents.World.Event.make;
            pos = ctx.Read.Pos;
            steps = ctx.Read.Steps;
            touched = false(numel(ctx.Rows), 1);
            events = OceanAgents.World.Event.empty(1, 0);
            for e = ctx.Events
                k = find(ctx.IDs == e.SourceEntity, 1);
                pos(k) = pos(k) + e.Payload.Delta;
                steps(k) = steps(k) + 1;
                touched(k) = true;
                id = e.SourceEntity;
                events(end + 1) = E("Reward", struct('Amount', -1), 'SourceEntity', id); %#ok<AGROW>
                if pos(k) >= goal
                    events(end + 1) = E("Reward", struct('Amount', bonus), 'SourceEntity', id); %#ok<AGROW>
                    events(end + 1) = E("Terminated", struct('Reason', "terminal"), 'SourceEntity', id); %#ok<AGROW>
                end
                if steps(k) >= limit
                    events(end + 1) = E("Truncated", struct('Reason', "time-limit"), 'SourceEntity', id); %#ok<AGROW>
                end
            end
            out = struct('Writes', [struct('Component', "Pos", 'Rows', ctx.Rows(touched), 'Values', pos(touched)), ...
                struct('Component', "Steps", 'Rows', ctx.Rows(touched), 'Values', steps(touched))], 'Events', events);
        end

        function c = cfg(roster)
            % AgentWorld configuration for the line door.
            c = struct('ActionLabels', [1 2], ...
                'Routing', struct('Label', {1, 2}, 'Type', {"Move", "Move"}, ...
                'Payload', {struct('Delta', -1), struct('Delta', 1)}), ...
                'Observe', @(view) view.line_See.Pos, ...
                'ObservationSpace', OceanStats.RL.Env.Spaces.Box(-100, 100), ...
                'ActionSpace', OceanStats.RL.Env.Spaces.Discrete(2), 'Roster', reshape(string(roster), 1, []));
        end

        function doors = doors(roster, varargin)
            doors = [OceanAgents.World.Doors.Agency.spec(), ...
                OceanTest.OceanAgents.Fixtures.World.Agent.LineDoors.line(roster, varargin{:})];
        end
    end
end
