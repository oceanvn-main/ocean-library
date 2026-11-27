classdef Reward < OceanAgents.Devices.IDevice
    % REWARD  The grid's payoffs and endings (DV1 contract §3): it hears what happened to a body ("moved" from, to,
    % fell) and answers the reward, whether the episode terminated (goal reached) and whether it was cut off (time
    % limit). It counts the steps of the current episode; a reset request starts the count again.
    %
    %   d = OceanAgents.Devices.Reward('Goal', 48, 'StepReward', -1, 'CliffReward', -100);
    %   d = OceanAgents.Devices.Reward('Goal', [1 7], 'StepReward', 0, 'CellReward', [0 NaN NaN NaN NaN NaN 1]);
    %   [r, term] = d.assess(fell, to);      % pure: the rule itself (also used for the exact model)
    % Reward of a move: CliffReward after a fall; otherwise CellReward(entered cell) when given and not NaN; otherwise
    % StepReward (the native rule).

    properties (SetAccess = private)
        StepReward double = -1
        CliffReward double = -100
        Goal double = zeros(1, 0)
        TimeLimit double = Inf
        CellReward double = zeros(1, 0)   % per-cell reward on entering (NaN or none = StepReward)
        Steps double = 0
    end

    methods
        function obj = Reward(options)
            arguments
                options.Goal double = zeros(1, 0)
                options.StepReward (1,1) double = -1
                options.CliffReward (1,1) double = -100
                options.TimeLimit (1,1) double {mustBePositive} = Inf
                options.CellReward double = zeros(1, 0)
                options.ID (1,1) string = "reward"
            end
            ev = OceanAgents.Minds.Signals.SignalType('ValueKind', "record", 'Role', "sense");
            cmd = OceanAgents.Minds.Signals.SignalType('ValueKind', "record", 'Role', "command");
            obj.setDeclarations_(options.ID, [ev cmd], OceanAgents.Minds.Signals.SignalType.reply(), "calendar");
            obj.Goal = reshape(options.Goal, 1, []);
            obj.StepReward = options.StepReward;
            obj.CliffReward = options.CliffReward;
            obj.TimeLimit = options.TimeLimit;
            if any(isinf(options.CellReward))
                error('OceanAgents:Devices:BadConfig', 'Reward: CellReward must be finite or NaN.');
            end
            obj.CellReward = reshape(double(options.CellReward), 1, []);
        end

        function [r, term] = assess(obj, fell, to)
            if fell
                r = obj.CliffReward;
            elseif to <= numel(obj.CellReward) && ~isnan(obj.CellReward(to))
                r = obj.CellReward(to);
            else
                r = obj.StepReward;
            end
            term = any(obj.Goal == to);
        end
    end

    methods (Access = protected)
        function [ok, reason] = verify(~, signal)
            v = signal.Value;
            if signal.Type.Role == "command"
                ok = isfield(v, 'Op') && string(v.Op) == "reset";
                reason = "the reward device takes a reset request";
            else
                ok = all(isfield(v, {'To', 'Fell'}));
                reason = "an event needs To and Fell";
            end
        end

        function out = act(obj, signal)
            if signal.Type.Role == "command"
                obj.Steps = 0;
                out = OceanAgents.Minds.Signals.Signal.ok();
                return
            end
            obj.Steps = obj.Steps + 1;
            [r, term] = obj.assess(signal.Value.Fell, signal.Value.To);
            trunc = ~term && obj.Steps >= obj.TimeLimit;
            out = OceanAgents.Minds.Signals.Signal.ok(struct('Reward', r, 'Terminated', term, 'Truncated', trunc));
        end
    end
end
