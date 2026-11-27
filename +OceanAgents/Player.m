classdef Player < handle
    % PLAYER  One player of a game or world: a mind, its body (an empty one when none is given), their pairing
    % (Dualism) and the host that decides when learning happens. Experiments create players; runners accept them
    % (PG1_Parity_Guess_Contract.md section 4, user's name "Player").
    %
    %   mind = OceanAgents.Minds.PolicySignalMind(OceanAgents.Policies.RandomLegalPolicy(), 1:9);
    %   p1 = OceanAgents.Player("x", mind);                                  % empty body with default sensors
    %   p2 = OceanAgents.Player("o", mind2, 'Body', body, 'Gamma', 0.95);
    %   rec = OceanAgents.Runtime.JointRunner(env, {p1, p2}, 'Seed', 7).run(100);
    %   trace = OceanAgents.Observe.PlayerTrace();  p1.attach(trace);      % for display or recording
    %
    % Display-ready: observers (OceanAgents.Observe.IPlayerObserver) see what the player saw, decided and was paid,
    % as value records only. They cannot change the run, and an observer that fails is detached and recorded.

    properties (SetAccess = private)
        ID string = ""
        Mind
        Body
        Host
    end

    methods
        function obj = Player(id, mind, options)
            arguments
                id (1,1) string {mustBeNonzeroLengthText}
                mind (1,1) OceanAgents.Minds.SignalMind
                options.Body = []
                options.Role (1,1) string = "default"
                options.Gamma (1,1) double = 1
                options.LearningEnabled (1,1) logical = true
            end
            body = options.Body;
            if isempty(body), body = OceanAgents.Body.EmptyBody(mind.CommandType.Labels); end
            OceanAgents.Embodied.Dualism.form(mind, body);
            obj.ID = id;
            obj.Mind = mind;
            obj.Body = body;
            obj.Host = OceanAgents.Runtime.AgentHost(id, mind, 'Role', options.Role, 'Gamma', options.Gamma, ...
                'LearningEnabled', options.LearningEnabled);
        end

        function attach(obj, observer)
            % ATTACH  Let an observer (a display, a recorder) follow this player.
            arguments
                obj
                observer (1,1) OceanAgents.Observe.IPlayerObserver
            end
            % an observer sees value records only; one that holds a player, a mind or its parts could change a run
            % (the isolation check exempts observers, so this is where they are checked)
            held = OceanAgents.Runtime.IsolationChecker.reachable(observer, OceanAgents.Player.AgentParts);
            if ~isempty(held)
                error('OceanAgents:Player:ObserverHoldsAgent', ...
                    'An observer may not hold a player or its parts; this one holds one at %s.', strjoin(held, ", "));
            end
            obj.Host.attachObserver_(observer);
        end

        function detach(obj, observer)
            obj.Host.detachObserver_(observer);
        end

        function n = numObservers(obj)
            n = obj.Host.numObservers();
        end
    end

    properties (Constant)
        AgentParts = ["OceanAgents.Player" "OceanAgents.Runtime.AgentHost" "OceanAgents.Core.IDecisionProcess" ...
            "OceanAgents.Body.IBody" "OceanAgents.Memory.IMemory" "OceanAgents.Core.ILearner" "OceanAgents.Core.IPolicy" ...
            "OceanAgents.Core.IJointEnvironment" "OceanAgents.Core.ISingleAgentEnvironment" "OceanAgents.Devices.IDevice" ...
            "OceanStats.GameTheory.Learning.FictitiousPlay" "OceanStats.RL.TD.Values.TabularQ" "RandStream"]
    end

    methods (Static)
        function h = hostOf(x)
            % HOSTOF  The host behind a Player (anything else is returned unchanged), so runners accept both.
            if isa(x, 'OceanAgents.Player'), h = x.Host; else, h = x; end
        end
    end
end
