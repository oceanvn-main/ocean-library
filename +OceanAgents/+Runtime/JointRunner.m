classdef JointRunner < handle
    % JOINTRUNNER  Drives N AgentHosts through one IJointEnvironment, one committed joint transition at a time.
    %
    %   r   = OceanAgents.Runtime.JointRunner(env, {hostA, hostB}, 'Seed', 7, 'MaxFramesPerEpisode', 1e5);
    %   rec = r.run(10);                  % RunRecord (columns follow describe(env).Roster)
    %
    % Per frame (A0 section 4.5):
    %   1. decide    every requested slot decides on THIS frame's view (evaluated in 'Order', default 1:N);
    %                nothing is stepped until every decision is in
    %   2. validate  ActionBatch (decisions in slot order) is validated against the frame
    %   3. step      exactly one env.step(batch)
    %   4. observe   Feedback goes, in slot order, to every slot that decided on the frame (and to any in-episode
    %                slot with a reward/termination event); learners update only here, after the joint step
    % Hosts are matched to the environment roster by AgentID (any order in the input cell). Each host draws from
    % its own purpose stream "agent:<AgentID>:action", so the evaluation order cannot change any draw (C03);
    % 'Order' exists so a test can permute it and check exactly that.
    % Fail-stop as SingleAgentRunner, with stages "reset" | "decide" | "validate" | "step" | "observe";
    % on a fault every host is invalidated.
    % Preflight at construction, before any reset: roster and role match (a host left at role "default" takes the
    % slot's role), mind.checkCompatible(obsSpace, actSpace, slotRole) rejects an unfit imported policy, the
    % IsolationChecker finds no handle shared between agents or between an agent and the environment
    % (Runtime:SharedComponent; 'IsolationAllow' extends the allowlist), and the host(s) are not bound to another
    % runner. The runner holds its hosts exclusively until release(r) or until it is deleted.
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 4.5.

    properties (SetAccess = private)
        Env                        % OceanAgents.Core.IJointEnvironment
        Hosts cell = {}            % 1 x N, Hosts{k} plays roster slot k
        AgentIDs string = strings(1, 0)
        Seed double = 0
        MaxFramesPerEpisode double = 1e5
        Order double = zeros(1, 0) % evaluation order of the decide stage (a permutation of 1:N)
        Registry                   % OceanAgents.Runtime.StreamRegistry
        Status string = "Ready"    % "Ready" | "Running" | "Invalid" | "Released"
        EpisodesRun double = 0
    end

    properties (Access = private)
        Attached_ logical = false(1, 0)   % hosts whose claim this runner holds
    end

    methods
        function obj = JointRunner(env, hosts, options)
            arguments
                env (1,1) OceanAgents.Core.IJointEnvironment
                hosts cell
                options.Seed (1,1) double {mustBeInteger, mustBeNonnegative} = 0
                options.MaxFramesPerEpisode (1,1) double {mustBeInteger, mustBePositive} = 1e5
                options.Order double = []
                options.IsolationAllow string = OceanAgents.Runtime.IsolationChecker.DefaultAllow
            end
            spec = env.describe();
            roster = spec.Roster;
            n = numel(roster);
            hosts = cellfun(@OceanAgents.Player.hostOf, hosts, 'UniformOutput', false);   % players are accepted
            if ~all(cellfun(@(h) isa(h, 'OceanAgents.Runtime.AgentHost') && isscalar(h), hosts))
                error('OceanAgents:Core:InvalidRecord', 'hosts must be a cell of OceanAgents.Runtime.AgentHost.');
            end
            ids = string(cellfun(@(h) h.AgentID, hosts, 'UniformOutput', false));
            if numel(ids) ~= n || numel(unique(ids)) ~= numel(ids) || ~isempty(setxor(ids, roster))
                fail_('RosterMismatch', 'Hosts [%s] do not match the environment roster [%s].', ...
                    strjoin(ids, ', '), strjoin(roster, ', '));
            end
            order = options.Order;
            if isempty(order), order = 1:n; end
            if ~isequal(sort(reshape(order, 1, [])), 1:n)
                fail_('RosterMismatch', 'Order must be a permutation of the %d roster slots.', n);
            end
            % Preflight (before any reset): roles, imported artifacts/spaces, isolation, free hosts.
            slotted = cell(1, n);
            for k = 1:n
                h = hosts{ids == roster(k)};
                checkSlotRole_(h, spec.Roles(k), roster(k));
                h.Mind.checkCompatible(spec.ObservationSpaces{k}, spec.ActionSpaces{k}, spec.Roles(k));
                slotted{k} = h;
            end
            for i = 1:n
                % C01/C02: no two agents, and no agent and the world, may share a mutable object.
                shared = OceanAgents.Runtime.IsolationChecker.sharedHandles(slotted{i}, env, 'Allow', options.IsolationAllow);
                if ~isempty(shared)
                    fail_('SharedComponent', 'Host %s shares handle(s) with the environment: %s.', ...
                        roster(i), strjoin(shared, ', '));
                end
                for j = i + 1:n
                    [pa, pb] = OceanAgents.Runtime.IsolationChecker.sharedHandles(slotted{i}, slotted{j}, ...
                        'Allow', options.IsolationAllow);
                    if ~isempty(pa)
                        fail_('SharedComponent', 'Hosts %s and %s share handle(s): %s.', roster(i), roster(j), ...
                            strjoin(roster(i) + "." + pa + " == " + roster(j) + "." + pb, '; '));
                    end
                end
            end
            bound = cellfun(@(h) h.isBound(), slotted);
            if any(bound)
                fail_('SharedComponent', 'Host(s) %s already bound to another runner.', strjoin(roster(bound), ', '));
            end
            obj.Env = env;
            obj.Hosts = slotted;
            obj.AgentIDs = roster;
            obj.Seed = options.Seed;
            obj.MaxFramesPerEpisode = options.MaxFramesPerEpisode;
            obj.Order = reshape(order, 1, []);
            obj.Registry = OceanAgents.Runtime.StreamRegistry(options.Seed);
            obj.Attached_ = false(1, n);
            for k = 1:n
                slotted{k}.attachStream(obj.Registry.stream("agent:" + roster(k) + ":action"));   % exclusive claim
                obj.Attached_(k) = true;
            end
        end

        function release(obj)
            % RELEASE  Give every host back (detach streams) so another runner may drive them; this runner is done.
            for k = find(obj.Attached_)
                h = obj.Hosts{k};
                if isvalid(h) && h.Status ~= "InEpisode"
                    h.detachStream();
                    obj.Attached_(k) = false;
                end
            end
            obj.Status = "Released";
        end

        function delete(obj)
            % DELETE  A destroyed runner releases its hosts.
            try
                obj.release();
            catch
            end
        end

        function rec = run(obj, numEpisodes)
            % RUN  Run numEpisodes complete joint episodes (or stop at the first fault); returns a RunRecord.
            arguments
                obj
                numEpisodes (1,1) double {mustBeInteger, mustBeNonnegative}
            end
            if obj.Status ~= "Ready"
                fail_('InvalidState', 'Runner is %s; run() needs a Ready runner.', obj.Status);
            end
            obj.Status = "Running";
            n = numel(obj.Hosts);
            returns = zeros(numEpisodes, n);
            lengths = zeros(numEpisodes, 1);
            completed = 0;
            stepCalls = 0;
            updatesBefore = obj.updateCounts_();
            stage = "reset";
            try
                for episode = 1:numEpisodes
                    stage = "reset";
                    frame = obj.Env.reset(obj.Registry.nextSeed("env"));
                    for k = find(~cellfun(@isempty, frame.Views))
                        obj.Hosts{k}.beginEpisode(frame.Views{k});
                    end
                    episodeReturn = zeros(1, n);
                    frames = 0;
                    while true
                        % 1. decide: every requested slot answers THIS frame before anything moves.
                        stage = "decide";
                        decisions = cell(1, n);
                        for k = obj.Order
                            if ~frame.Requested(k), continue; end
                            host = obj.Hosts{k};
                            if host.Status == "Ready"             % slot joins after the reset frame
                                host.beginEpisode(frame.Views{k});
                            end
                            decisions{k} = host.decide(frame.Views{k});
                        end
                        decided = ~cellfun(@isempty, decisions);
                        % 2. validate the whole batch (slot order, independent of the evaluation order).
                        stage = "validate";
                        submitted = [decisions{decided}];
                        if isempty(submitted), submitted = OceanAgents.Core.Decision.empty(1, 0); end
                        batch = OceanAgents.Core.ActionBatch('EpisodeID', frame.EpisodeID, 'FrameID', frame.FrameID, ...
                            'Phase', frame.Phase, 'Decisions', submitted);
                        batch.validate(frame);
                        % 3. exactly one joint transition.
                        stage = "step";
                        stepCalls = stepCalls + 1;
                        next = obj.Env.step(batch);
                        frames = frames + 1;
                        episodeReturn = episodeReturn + next.Rewards .* next.RewardValid;   % undiscounted
                        % 4. feedback (and learning) only after the committed step, in slot order.
                        stage = "observe";
                        obj.deliverFeedback_(frame, next, decided);
                        if next.EpisodeEnded
                            stillIn = find(cellfun(@(h) h.Status == "InEpisode", obj.Hosts));
                            if ~isempty(stillIn)
                                fail_('InvalidState', 'Episode ended but slot(s) %s received no ending feedback.', ...
                                    strjoin(obj.AgentIDs(stillIn), ', '));
                            end
                            break
                        end
                        if frames >= obj.MaxFramesPerEpisode
                            stage = "step";
                            fail_('StepBudgetExceeded', ...
                                'Episode %d did not end within MaxFramesPerEpisode = %d.', episode, obj.MaxFramesPerEpisode);
                        end
                        frame = next;
                    end
                    completed = completed + 1;
                    returns(completed, :) = episodeReturn;
                    lengths(completed) = frames;
                end
            catch failure
                obj.failStop_(failure, stage);
                obj.EpisodesRun = obj.EpisodesRun + completed;
                rec = OceanAgents.Runtime.RunRecord('Status', "Failed", 'AgentIDs', obj.AgentIDs, ...
                    'EpisodeReturns', returns(1:completed, :), 'EpisodeLengths', lengths(1:completed), ...
                    'StepCalls', stepCalls, 'UpdateCounts', obj.updateCounts_() - updatesBefore, ...
                    'FailureStage', stage, 'Failure', failure);
                return
            end
            obj.Status = "Ready";
            obj.EpisodesRun = obj.EpisodesRun + completed;
            rec = OceanAgents.Runtime.RunRecord('Status', "Completed", 'AgentIDs', obj.AgentIDs, ...
                'EpisodeReturns', returns, 'EpisodeLengths', lengths, 'StepCalls', stepCalls, ...
                'UpdateCounts', obj.updateCounts_() - updatesBefore);
        end

        function meta = getMetadata(obj)
            meta = pillarMeta_(obj, {sprintf('roster [%s]; run seed %d; MaxFramesPerEpisode %d; order %s; status %s', ...
                strjoin(obj.AgentIDs, ', '), obj.Seed, obj.MaxFramesPerEpisode, mat2str(obj.Order), obj.Status)}, ...
                "A0 contract section 4.5 (joint decide-validate-step-observe loop, fail-stop)");
        end

        function spec = getPipelineSpec(obj)
            spec = pillarSpec_(obj, strings(1, 0), strings(1, 0), [{class(obj.Env)}, cellfun(@class, obj.Hosts, 'UniformOutput', false)]);
        end
    end

    methods (Access = private)
        function deliverFeedback_(obj, frame, next, decided)
            % DELIVERFEEDBACK_  Feedback about the transition frame -> next, in slot order.
            for k = 1:numel(obj.Hosts)
                host = obj.Hosts{k};
                event = next.Terminated(k) || next.Truncated(k) || next.RewardValid(k);
                if ~(decided(k) || (event && host.Status == "InEpisode"))
                    continue
                end
                if isempty(next.Views{k})
                    fail_('InvalidState', ['Slot %s decided on frame %d but frame %d has no view for it ' ...
                        '(delayed feedback is outside A1).'], obj.AgentIDs(k), frame.FrameID, next.FrameID);
                end
                host.observe(next.feedbackFor(k, frame.FrameID));
            end
        end

        function counts = updateCounts_(obj)
            counts = cellfun(@(h) h.UpdateCount, obj.Hosts);
        end

        function failStop_(obj, failure, stage)
            % FAILSTOP_  Invalidate every host and the runner; nothing after the fault runs or is learned from.
            for k = 1:numel(obj.Hosts)
                if obj.Hosts{k}.Status ~= "Invalid"
                    obj.Hosts{k}.invalidate("joint run failed at stage " + stage + ": " + string(failure.identifier));
                end
            end
            obj.Status = "Invalid";
        end
    end
end
