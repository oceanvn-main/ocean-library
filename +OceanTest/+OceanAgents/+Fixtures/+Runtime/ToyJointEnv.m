classdef ToyJointEnv < OceanAgents.Core.IJointEnvironment
    % TOYJOINTENV  Test fixture: an N-player simultaneous repeated game, labels [1 2] for every slot, recording
    % every batch it receives. Rounds per episode = 'Rounds'. Payoff:
    %   "coordination"  r_i = #{j ~= i : a_j == a_i} - #{j ~= i : a_j ~= a_i}
    %   "pennies"       (N = 2) r_1 = +1 if a_1 == a_2 else -1,  r_2 = -r_1   (A0 section 9.4)
    % Observation of slot i: [round, own last action]. 'ThrowOnStep', k makes the k-th step throw Fixture:Injected;
    % 'SkipEndingFor', id ends the episode without flagging that slot as ended (a broken environment).
    % 'Roles' (1 x N) declares slot roles; 'DiscreteObs', true makes the observation the round number alone with
    % space Discrete(Rounds + 1, Start = 0), so tabular policy artifacts can be checked against it.
    properties
        Log = []
    end
    properties (SetAccess = private)
        Roster string = ["p1" "p2"]
        Rounds double = 3
        Payoff string = "coordination"
        ThrowOnStep double = 0
        SkipEndingFor string = ""
        Roles string = strings(1, 0)
        DiscreteObs logical = false
        ResetCalls double = 0
        StepCalls double = 0
        Seeds double = zeros(1, 0)
        Batches cell = {}
        Round double = 0
        LastActions double = []
        EpisodeID double = 0
        FrameID double = 0
        Current = []
    end
    methods
        function obj = ToyJointEnv(options)
            arguments
                options.Roster (1,:) string = ["p1" "p2"]
                options.Rounds (1,1) double = 3
                options.Payoff (1,1) string = "coordination"
                options.ThrowOnStep (1,1) double = 0
                options.SkipEndingFor (1,1) string = ""
                options.Roles string = strings(1, 0)
                options.DiscreteObs (1,1) logical = false
                options.Log = []
            end
            obj.Roster = options.Roster;
            obj.Rounds = options.Rounds;
            obj.Payoff = options.Payoff;
            obj.ThrowOnStep = options.ThrowOnStep;
            obj.SkipEndingFor = options.SkipEndingFor;
            obj.Roles = options.Roles;
            obj.DiscreteObs = options.DiscreteObs;
            obj.Log = options.Log;
        end

        function spec = describe(obj)
            n = numel(obj.Roster);
            if obj.DiscreteObs
                obs = repmat({OceanStats.RL.Env.Spaces.Discrete(obj.Rounds + 1, 'Start', 0)}, 1, n);
            else
                obs = repmat({OceanStats.RL.Env.Spaces.Box([0 0], [Inf 2])}, 1, n);
            end
            act = repmat({OceanStats.RL.Env.Spaces.Discrete(2)}, 1, n);
            spec = OceanAgents.Core.EnvSpec('Name', "ToyJoint", 'Timing', "simultaneous", 'Roster', obj.Roster, ...
                'Roles', obj.Roles, 'ObservationSpaces', obs, 'ActionSpaces', act, 'Horizon', obj.Rounds);
        end

        function frame = reset(obj, seed)
            obj.ResetCalls = obj.ResetCalls + 1;
            obj.Seeds(end + 1) = seed;
            obj.Round = 0;
            n = numel(obj.Roster);
            obj.LastActions = zeros(1, n);
            obj.EpisodeID = obj.EpisodeID + 1;
            obj.FrameID = obj.FrameID + 1;
            frame = obj.frame_(false, zeros(1, n), false(1, n));
        end

        function frame = step(obj, batch)
            obj.StepCalls = obj.StepCalls + 1;
            if ~isempty(obj.Log), obj.Log.add("env:step"); end
            if obj.StepCalls == obj.ThrowOnStep
                error('Fixture:Injected', 'Injected joint step fault on call %d.', obj.StepCalls);
            end
            batch.validate(obj.Current);                         % whole batch checked before any mutation
            obj.Batches{end + 1} = batch;
            n = numel(obj.Roster);
            a = zeros(1, n);
            for k = 1:n
                a(k) = batch.decisionFor(obj.Roster(k)).Action;
            end
            r = zeros(1, n);
            if obj.Payoff == "pennies"
                r(1) = 2 * (a(1) == a(2)) - 1;
                r(2) = -r(1);
            else
                for k = 1:n
                    same = sum(a == a(k)) - 1;
                    r(k) = same - (n - 1 - same);
                end
            end
            obj.LastActions = a;
            obj.Round = obj.Round + 1;
            obj.FrameID = obj.FrameID + 1;
            ended = obj.Round >= obj.Rounds;
            terminated = repmat(ended, 1, n) & obj.Roster ~= obj.SkipEndingFor;
            frame = obj.frame_(ended, r, terminated);
        end
    end

    methods (Access = private)
        function frame = frame_(obj, ended, rewards, terminated)
            n = numel(obj.Roster);
            views = cell(1, n);
            for k = 1:n
                obs = [obj.Round obj.LastActions(k)];
                if obj.DiscreteObs, obs = obj.Round; end
                views{k} = OceanAgents.Core.ActorView('AgentID', obj.Roster(k), 'EpisodeID', obj.EpisodeID, ...
                    'FrameID', obj.FrameID, 'Observation', obs, ...
                    'ActionLabels', [1 2], 'ActionMask', repmat(~ended, 1, 2));
            end
            live = repmat(~ended, 1, n);
            reason = "";
            if ended, reason = "horizon"; end
            frame = OceanAgents.Core.DecisionFrame('EpisodeID', obj.EpisodeID, 'FrameID', obj.FrameID, ...
                'Tick', obj.Round, 'Roster', obj.Roster, 'Live', live, 'Requested', live, 'Views', views, ...
                'Rewards', rewards, 'RewardValid', repmat(obj.Round > 0, 1, n), ...
                'Terminated', terminated, 'EpisodeEnded', ended, 'EndReason', reason);
            obj.Current = frame;
        end
    end
end
