classdef SingleAgentRunner < handle
    % SINGLEAGENTRUNNER  Drives one AgentHost through a single-agent environment. The runner is the only caller of
    % reset/step; the host only decides and observes. Randomness: one StreamRegistry per runner, per-episode
    % environment seeds from purpose "env", the host's action draws from purpose "agent:<AgentID>:action".
    %
    %   r   = OceanAgents.Runtime.SingleAgentRunner(env, host, 'Seed', 7, 'MaxStepsPerEpisode', 1e5);
    %   rec = r.run(100);              % RunRecord; never throws for a fault inside the loop
    %
    % Loop per episode (A0 section 4.4):
    %   seed = nextSeed(reg, "env");  view = env.reset(seed);  host.beginEpisode(view)
    %   repeat  dec = host.decide(view);  fb = env.step(dec);  host.observe(fb);  view = fb.NextView
    %   until   ended(fb)
    % Fail-stop: any exception invalidates the host and the runner and is returned in the RunRecord with the
    % stage it happened in ("reset" | "decide" | "step" | "observe"). Running out of MaxStepsPerEpisode
    % without an ending is a fault (OceanAgents:Runtime:StepBudgetExceeded), not a truncation - use a TimeLimit
    % wrapper on the environment when a time limit is part of the task. A second run() on an Invalid runner errors.
    % Preflight at construction, before any reset: roster and role match (a host left at role "default" takes the
    % slot's role), mind.checkCompatible(obsSpace, actSpace, slotRole) rejects an unfit imported policy, the
    % IsolationChecker finds no handle shared between agents or between an agent and the environment
    % (Runtime:SharedComponent; 'IsolationAllow' extends the allowlist), and the host(s) are not bound to another
    % runner. The runner holds its hosts exclusively until release(r) or until it is deleted.
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 4.4.

    properties (SetAccess = private)
        Env                       % OceanAgents.Core.ISingleAgentEnvironment
        Host                      % OceanAgents.Runtime.AgentHost
        Seed double = 0
        MaxStepsPerEpisode double = 1e5
        Registry                  % OceanAgents.Runtime.StreamRegistry
        Status string = "Ready"   % "Ready" | "Running" | "Invalid" | "Released"
        EpisodesRun double = 0    % completed episodes over all run() calls
    end

    properties (Access = private)
        Attached_ logical = false % this runner holds the host's claim (attached its stream)
    end

    methods
        function obj = SingleAgentRunner(env, host, options)
            arguments
                env (1,1) OceanAgents.Core.ISingleAgentEnvironment
                host (1,1)                    % an OceanAgents.Runtime.AgentHost or an OceanAgents.Player
                options.Seed (1,1) double {mustBeInteger, mustBeNonnegative} = 0
                options.MaxStepsPerEpisode (1,1) double {mustBeInteger, mustBePositive} = 1e5
                options.IsolationAllow string = OceanAgents.Runtime.IsolationChecker.DefaultAllow
            end
            host = OceanAgents.Player.hostOf(host);
            if ~(isa(host, 'OceanAgents.Runtime.AgentHost') && isscalar(host))
                error('OceanAgents:Core:InvalidRecord', 'SingleAgentRunner needs an AgentHost or a Player.');
            end
            % Preflight (nothing below touches the environment's episode state):
            spec = env.describe();
            if spec.Roster(1) ~= host.AgentID
                fail_('RosterMismatch', 'The environment serves agent %s; the host is %s.', spec.Roster(1), host.AgentID);
            end
            checkSlotRole_(host, spec.Roles(1), spec.Roster(1));
            % the mind (e.g. an imported policy artifact) must accept this role's spaces
            host.Mind.checkCompatible(spec.ObservationSpaces{1}, spec.ActionSpaces{1}, spec.Roles(1));
            % the agent must not reach any mutable object of the environment (and vice versa)
            shared = OceanAgents.Runtime.IsolationChecker.sharedHandles(host, env, 'Allow', options.IsolationAllow);
            if ~isempty(shared)
                fail_('SharedComponent', 'Host %s shares handle(s) with the environment: %s.', ...
                    host.AgentID, strjoin(shared, ', '));
            end
            obj.Env = env;
            obj.Host = host;
            obj.Seed = options.Seed;
            obj.MaxStepsPerEpisode = options.MaxStepsPerEpisode;
            obj.Registry = OceanAgents.Runtime.StreamRegistry(options.Seed);
            host.attachStream(obj.Registry.stream("agent:" + host.AgentID + ":action"));   % exclusive claim
            obj.Attached_ = true;
        end

        function release(obj)
            % RELEASE  Give the host back (detach its stream) so another runner may drive it; this runner is done.
            if obj.Attached_ && isvalid(obj.Host) && obj.Host.Status ~= "InEpisode"
                obj.Host.detachStream();
                obj.Attached_ = false;
            end
            obj.Status = "Released";
        end

        function delete(obj)
            % DELETE  A destroyed runner releases its host.
            try
                obj.release();
            catch
            end
        end

        function rec = run(obj, numEpisodes)
            % RUN  Run numEpisodes complete episodes (or stop at the first fault); returns a RunRecord.
            arguments
                obj
                numEpisodes (1,1) double {mustBeInteger, mustBeNonnegative}
            end
            if obj.Status ~= "Ready"
                fail_('InvalidState', 'Runner is %s; run() needs a Ready runner.', obj.Status);
            end
            obj.Status = "Running";
            host = obj.Host;
            returns = zeros(numEpisodes, 1);
            lengths = zeros(numEpisodes, 1);
            completed = 0;
            stepCalls = 0;
            updatesBefore = host.UpdateCount;
            stage = "reset";
            try
                for episode = 1:numEpisodes
                    stage = "reset";
                    view = obj.Env.reset(obj.Registry.nextSeed("env"));
                    host.beginEpisode(view);
                    episodeReturn = 0;
                    steps = 0;
                    while true
                        stage = "decide";
                        dec = host.decide(view);
                        stage = "step";
                        stepCalls = stepCalls + 1;
                        fb = obj.Env.step(dec);
                        steps = steps + 1;
                        if fb.RewardValid
                            episodeReturn = episodeReturn + fb.Reward;     % undiscounted return
                        end
                        stage = "observe";
                        host.observe(fb);
                        if fb.ended()
                            break
                        end
                        if steps >= obj.MaxStepsPerEpisode
                            stage = "step";
                            fail_('StepBudgetExceeded', ...
                                'Episode %d did not end within MaxStepsPerEpisode = %d (use a TimeLimit for truncation).', ...
                                episode, obj.MaxStepsPerEpisode);
                        end
                        view = fb.NextView;
                    end
                    completed = completed + 1;
                    returns(completed) = episodeReturn;
                    lengths(completed) = steps;
                end
            catch failure
                obj.failStop_(failure, stage);
                obj.EpisodesRun = obj.EpisodesRun + completed;
                rec = OceanAgents.Runtime.RunRecord('Status', "Failed", 'AgentIDs', host.AgentID, ...
                    'EpisodeReturns', returns(1:completed, :), 'EpisodeLengths', lengths(1:completed), ...
                    'StepCalls', stepCalls, 'UpdateCounts', host.UpdateCount - updatesBefore, ...
                    'FailureStage', stage, 'Failure', failure);
                return
            end
            obj.Status = "Ready";
            obj.EpisodesRun = obj.EpisodesRun + completed;
            rec = OceanAgents.Runtime.RunRecord('Status', "Completed", 'AgentIDs', host.AgentID, ...
                'EpisodeReturns', returns, 'EpisodeLengths', lengths, 'StepCalls', stepCalls, ...
                'UpdateCounts', host.UpdateCount - updatesBefore);
        end

        function meta = getMetadata(obj)
            meta = pillarMeta_(obj, {sprintf('agent %s; run seed %d; MaxStepsPerEpisode %d; status %s', ...
                obj.Host.AgentID, obj.Seed, obj.MaxStepsPerEpisode, obj.Status)}, ...
                "A0 contract section 4.4 (single-agent loop, fail-stop)");
        end

        function spec = getPipelineSpec(obj)
            spec = pillarSpec_(obj, strings(1, 0), strings(1, 0), {class(obj.Env), class(obj.Host)});
        end
    end

    methods (Access = private)
        function failStop_(obj, failure, stage)
            % FAILSTOP_  Invalidate host and runner; nothing after the fault is executed or learned from.
            if obj.Host.Status ~= "Invalid"
                obj.Host.invalidate("run failed at stage " + stage + ": " + string(failure.identifier));
            end
            obj.Status = "Invalid";
        end
    end
end
