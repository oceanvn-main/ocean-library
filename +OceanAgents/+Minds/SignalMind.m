classdef (Abstract) SignalMind < OceanAgents.Core.IDecisionProcess
    % SIGNALMIND  A mind that holds its learners (design D2) and its signal system (D3): it reaches its body and its
    % memories only through its router, and learns only when its host hands it a committed transition (the host gate,
    % AB1_Signal_Body_Contract.md section 7). The A0 contract is kept: decide(view, stream) still returns one Decision.
    %
    %   mind.addLearner(learner, "learner");     % the mind owns it; AgentHost calls mind.learn(t) after a committed step
    %   d = OceanAgents.Embodied.Dualism.form(mind, body);      % join: probe, negotiate, seal
    %   reply = mind.query("values", struct('Op', "best", 'State', 3));

    properties (SetAccess = protected)
        Router = []
        Body = []
        BodySignature = []
        CommandType = OceanAgents.Minds.Signals.SignalType()
        LearnerIDs string = strings(1, 0)
        InEpisode (1,1) logical = false
    end

    properties (Dependent)
        Accepts                                 % the sense types this mind takes: one per perception (AB2)
    end

    properties (Access = protected)
        Learners = {}                           % the mind's own learners (never handed out)
        Memories = {}                           % the mind's memories (reached by signal)
        Perceptions = {}                        % OceanAgents.Minds.Perception.IPerception (AB2 section 6)
    end

    properties (Access = private)
        Pairing_ string = strings(1, 0)         % the body sensor each perception reads ("" = none)
        Perceived_ = []                         % the last perception and the frame it was sensed in (one per frame)
        PerceivedFrame_ = NaN
        PerceivedEpisodeID_ = NaN
        LastState_ = 0                          % the perceived state the last decision was made in (for learning)
        LastDecision_ = []                      % the mind's last decision and its labels (its own record of what it did)
        LastLabels_ = []
        Policies_ = {}                          % the mind's policies (design D29), by name
        PolicyNames_ string = strings(1, 0)
        Selector_ = []                          % OceanAgents.Minds.Selection.IPolicySelector (default: the first policy)
        ActedPolicy_ string = ""
        HasBehaviorRoute_ (1,1) logical = false
        SenseTypes_ = OceanAgents.Minds.Signals.SignalType.empty(1, 0)   % the paired sensors' types (route checks)
        Observations_ = []                      % containers.Map: sensor ID -> last observation value
        Port_ = []
        Boundaries_ string = strings(1, 0)
    end

    methods
        function a = get.Accepts(obj)
            a = obj.SenseTypes_;
            for k = 1:numel(obj.Perceptions), a(end + 1) = obj.Perceptions{k}.Accepts; end %#ok<AGROW>
        end

        function addPolicy(obj, name, policy)
            % ADDPOLICY  Give the mind another policy (D29). The selector decides which one acts.
            arguments
                obj
                name (1,1) string {mustBeNonzeroLengthText}
                policy (1,1) OceanAgents.Core.IPolicy
            end
            obj.requireBetweenEpisodes_('addPolicy');
            if any(obj.PolicyNames_ == name)
                error('OceanAgents:Minds:UnknownPolicy', 'This mind already has a policy "%s".', name);
            end
            obj.Policies_{end + 1} = policy;
            obj.PolicyNames_(end + 1) = name;
        end

        function useSelector(obj, selector)
            % USESELECTOR  How the mind picks the acting policy (settings today, a learned selector later).
            arguments
                obj
                selector (1,1) OceanAgents.Minds.Selection.IPolicySelector
            end
            obj.requireBetweenEpisodes_('useSelector');
            obj.Selector_ = selector;
        end

        function usePolicy(obj, name)
            % USEPOLICY  Settings: always act with the named policy.
            obj.useSelector(OceanAgents.Minds.Selection.FixedSelector(name));
        end

        function n = policyNames(obj)
            n = obj.PolicyNames_;
        end

        function n = actedPolicy(obj)
            % ACTEDPOLICY  The policy that made the last decision ("" for a behavior step).
            n = obj.ActedPolicy_;
        end

        function addPerception(obj, perception)
            arguments
                obj
                perception (1,1) OceanAgents.Minds.Perception.IPerception
            end
            obj.requireBetweenEpisodes_('addPerception');
            obj.Perceptions{end + 1} = perception;
        end

        function addLearner(obj, learner, id)
            arguments
                obj
                learner (1,1) OceanAgents.Core.ILearner
                id (1,1) string {mustBeNonzeroLengthText}
            end
            obj.requireBetweenEpisodes_('addLearner');
            if any(obj.LearnerIDs == id), error('OceanAgents:Memory:WriterBound', 'Learner "%s" already added.', id); end
            % Check everything before changing anything (critic AB1 finding 6): a learner that updates a table must
            % update a table this mind keeps, and that table must not already have a writer.
            bindable = obj.Memories(cellfun(@(m) ismethod(m, 'isUpdatedBy'), obj.Memories));
            mine = bindable(cellfun(@(m) m.isUpdatedBy(learner), bindable));
            tables = bindable(cellfun(@(m) isa(m, 'OceanAgents.Memory.TableMemory'), bindable));
            if isprop(learner, 'Q') && ~isempty(learner.Q) && ~isempty(tables) && isempty(mine)
                error('OceanAgents:Memory:NotWriter', ...
                    'Learner "%s" updates a table this mind does not keep; it would never learn what the mind acts on.', id);
            end
            for k = 1:numel(mine)
                if strlength(mine{k}.WriterID) > 0
                    error('OceanAgents:Memory:WriterBound', 'Table "%s" is already written by "%s".', ...
                        mine{k}.ID, mine{k}.WriterID);
                end
            end
            obj.Learners{end + 1} = learner;
            obj.LearnerIDs(end + 1) = id;
            d = learner.describe();
            b = "step";
            if isfield(d, 'Boundary') && ~isempty(d.Boundary), b = string(d.Boundary); end
            obj.Boundaries_(end + 1) = b;
            for k = 1:numel(mine), mine{k}.bindLearner(learner, id); end
        end

        function addMemory(obj, memory)
            arguments
                obj
                memory (1,1) OceanAgents.Memory.IMemory
            end
            obj.requireBetweenEpisodes_('addMemory');
            if any(cellfun(@(m) m.ID == memory.ID, obj.Memories))
                error('OceanAgents:Signals:BadType', 'This mind already keeps a memory "%s".', memory.ID);
            end
            memory.claimOwner_(obj);                             % refuses another mind's memory (critic AB1 round 3)
            obj.Memories{end + 1} = memory;
        end

        function attachBody(obj, body)
            % ATTACHBODY  Join with a body (called by Dualism.form): a fresh router, routes to the body and to every
            % memory, the command type negotiated once, then sealed.
            arguments
                obj
                body (1,1) OceanAgents.Body.IBody
            end
            obj.requireBetweenEpisodes_('attachBody');
            % Build the whole join locally; the mind changes only after every step succeeded (critic AB1 round 2).
            r = OceanAgents.Minds.Signals.Router();
            r.setOwner_(obj);
            port = r.attach(obj, "mind");
            bodyPort = r.attach(body, body.ID);
            r.addRoute("body", body.ID, "command");
            % every perception reads one sensor whose observation type it accepts (by type only; AB2 I11)
            pairing = strings(1, numel(obj.Perceptions));
            ids = body.sensorIDs();
            for k = 1:numel(obj.Perceptions)
                p = obj.Perceptions{k};
                free = ids(~ismember(ids, pairing));
                hit = free(arrayfun(@(id) p.canRead(body.sensorType(id)), free));
                if numel(hit) > 1
                    error('OceanAgents:Embodied:Incompatible', ['Body "%s": sensors %s all fit perception %s; the ' ...
                        'pairing would be a guess. Give the body one such sensor, or give them different types ' ...
                        '(for example a physical dimension).'], body.ID, strjoin(hit, ", "), p.ID);
                end
                if isempty(hit)
                    if p.Required
                        error('OceanAgents:Embodied:Incompatible', ...
                            'Body "%s" has no sensor this mind can perceive as %s (%s).', body.ID, p.ID, p.Accepts.describe());
                    end
                    continue
                end
                pairing(k) = hit(1);
                r.addRoute("sense:" + hit(1), "mind", "sense");
            end
            % negotiate the sense routes with the paired sensors' own types (the mind accepts exactly those)
            previous = obj.SenseTypes_;
            obj.SenseTypes_ = arrayfun(@(id) body.sensorType(id), pairing(pairing ~= ""));
            if isempty(obj.SenseTypes_), obj.SenseTypes_ = OceanAgents.Minds.Signals.SignalType.empty(1, 0); end
            try
                for id = pairing(pairing ~= "")
                    r.negotiate("sense:" + id, body.sensorType(id));
                end
            catch err
                obj.SenseTypes_ = previous;
                rethrow(err);
            end
            for k = 1:numel(obj.Memories)
                m = obj.Memories{k};
                r.attach(m, m.ID);
                r.addRoute("memory:" + m.ID, m.ID, "query");
                r.negotiate("memory:" + m.ID, OceanAgents.Minds.Signals.SignalType.query());
            end
            r.negotiate("body", obj.CommandType);
            behaviorRoute = body.Signature.accepts(OceanAgents.Body.GridRobotBody.behaviorType());
            if behaviorRoute                                    % the mind starts and stops behaviors (D23)
                r.addRoute("behavior", body.ID, "command");
                r.negotiate("behavior", OceanAgents.Body.GridRobotBody.behaviorType());
            end
            r.seal();
            try
                body.claim_(obj);                                % throws if the body belongs to another mind
            catch err
                obj.SenseTypes_ = previous;
                rethrow(err);
            end
            if ~isempty(obj.Body) && obj.Body ~= body, obj.Body.release_(); end
            body.bindPort_(bodyPort, pairing(pairing ~= ""));
            obj.Pairing_ = pairing;
            obj.HasBehaviorRoute_ = behaviorRoute;
            obj.PerceivedFrame_ = NaN;
            obj.Observations_ = containers.Map('KeyType', 'char', 'ValueType', 'any');
            obj.Port_ = port;
            obj.Router = r;
            obj.Body = body;
            obj.BodySignature = body.probe();
        end

        function tf = hasLearners(obj)
            tf = ~isempty(obj.Learners);
        end

        function n = numLearners(obj)
            n = numel(obj.Learners);
        end

        function d = learnerDescriptions(obj)
            % LEARNERDESCRIPTIONS  describe() of every learner (value data; the learners themselves stay inside).
            d = cellfun(@(l) l.describe(), obj.Learners, 'UniformOutput', false);
        end

    end

    methods (Access = {?OceanAgents.Runtime.AgentHost})
        function n = learn(obj, feedback, gamma)
            % LEARN  Host-only (the learning gate): the step is committed, so the mind turns it into its own experience
            % and every learner learns from it; step-boundary learners update now. The experience is built by the mind
            % from its own records (design D27): the state it perceived, the action it took, the reward, and the next
            % state and permitted moves perceived from the next view by its body's sensors (sensed once per frame).
            n = 0;
            if isempty(obj.Learners), return; end
            transition = obj.experience_(feedback, gamma);
            for k = 1:numel(obj.Learners)
                obj.Learners{k}.consume(transition);
                if obj.Boundaries_(k) == "step", n = n + obj.Learners{k}.update("step"); end
            end
        end

        function n = learnEpisodeEnd(obj)
            n = 0;
            for k = 1:numel(obj.Learners)
                n = n + obj.Learners{k}.update("episode");
            end
        end
    end

    methods

        function out = receive(obj, signal)
            % RECEIVE  Sense signals delivered to the mind: observations are kept by sensor until perceived.
            if startsWith(signal.Route, "sense:")
                obj.Observations_(char(extractAfter(signal.Route, "sense:"))) = signal;
            end
            out = OceanAgents.Minds.Signals.Signal.empty(1, 0);
        end

        function reply = query(obj, memoryID, payload)
            % QUERY  Ask a memory through the router; returns the reply record (throws on a refusal).
            replies = obj.Port_.send("memory:" + memoryID, OceanAgents.Minds.Signals.Signal.make( ...
                OceanAgents.Minds.Signals.SignalType.query(), payload));
            if replies(1).isRefusal()
                error('OceanAgents:Memory:BadQuery', '%s', replies(1).Value.Reason);
            end
            reply = replies(1).Value;
        end

        function beginEpisode(obj, view) %#ok<INUSD>
            obj.requireBody_();
            obj.InEpisode = true;
            obj.Body.beginEpisode();
        end

        function endEpisode(obj)
            obj.InEpisode = false;
        end
    end

    methods (Access = protected)
        function policy = choosePolicy_(obj, p, view)
            if isempty(obj.Policies_)
                error('OceanAgents:Minds:UnknownPolicy', 'This mind has no policy.');
            end
            if isempty(obj.Selector_), obj.Selector_ = OceanAgents.Minds.Selection.FixedSelector(obj.PolicyNames_(1)); end
            name = string(obj.Selector_.select(struct('Perceived', p, 'Names', obj.PolicyNames_, ...
                'EpisodeID', view.EpisodeID, 'Tick', view.Tick)));
            k = find(obj.PolicyNames_ == name, 1);
            if isempty(k)
                error('OceanAgents:Minds:UnknownPolicy', 'The selector chose "%s"; this mind has [%s].', name, ...
                    strjoin(obj.PolicyNames_, ', '));
            end
            obj.ActedPolicy_ = name;
            policy = obj.Policies_{k};
        end

        function remember_(obj, dec, view, policyName)
            obj.LastDecision_ = dec;
            obj.LastLabels_ = view.ActionLabels;
            obj.ActedPolicy_ = policyName;
        end

        function beginPolicies_(obj, view)
            for k = 1:numel(obj.Policies_), obj.Policies_{k}.beginEpisode(obj.blindView_(view)); end
        end

        function endPolicies_(obj)
            for k = 1:numel(obj.Policies_), obj.Policies_{k}.endEpisode(); end
        end

        function replies = sendBehavior_(obj, record)
            % SENDBEHAVIOR_  Start or stop a body behavior through the router (only if the body has behaviors).
            replies = OceanAgents.Minds.Signals.Signal.empty(1, 0);
            if ~obj.HasBehaviorRoute_, return; end
            replies = obj.Port_.send("behavior", OceanAgents.Minds.Signals.Signal.make( ...
                OceanAgents.Body.GridRobotBody.behaviorType(), record));
        end

        function label = issue_(obj, label)
            % ISSUE_  Send a command label to the body through the router and return the body's action for this step.
            obj.requireBody_();
            replies = obj.Port_.send("body", OceanAgents.Minds.Signals.Signal.make(obj.CommandType, label));
            if ~isempty(replies) && replies(1).isRefusal()
                error('OceanAgents:Body:UnknownCommand', 'The body refused "%s": %s', label, replies(1).Value.Reason);
            end
            label = obj.Body.toAction();
        end

        function p = perceive_(obj, view)
            % PERCEIVE_  Ask the body to sense; turn each received observation into what the mind uses (by Use).
            % A frame is sensed once: learning (after the step) and the next decision share the same perception.
            if isequal(obj.PerceivedFrame_, view.FrameID) && view.EpisodeID == obj.PerceivedEpisode_()
                p = obj.Perceived_;
                return
            end
            obj.requireBody_();
            remove(obj.Observations_, keys(obj.Observations_));
            obj.Router.advanceTo(obj.Router.Now + 1);        % one instant per decision (chatter limit is per instant)
            obj.Body.observeAll(view);
            p = struct();
            for k = 1:numel(obj.Perceptions)
                id = obj.Pairing_(k);
                if id == "", continue; end
                if ~isKey(obj.Observations_, char(id))
                    error('OceanAgents:Body:BadObservation', 'No observation arrived from sensor "%s".', id);
                end
                per = obj.Perceptions{k};
                p.(char(per.Use)) = per.perceive(obj.Observations_(char(id)));
            end
            obj.Perceived_ = p;
            obj.PerceivedFrame_ = view.FrameID;
            obj.PerceivedEpisodeID_ = view.EpisodeID;
        end

        function e = PerceivedEpisode_(obj)
            e = obj.PerceivedEpisodeID_;
        end

        function v = blindView_(~, view)
            % BLINDVIEW_  The bookkeeping fields of a view with nothing of the world in it (for policy bookkeeping).
            v = OceanAgents.Core.ActorView('AgentID', view.AgentID, 'EpisodeID', view.EpisodeID, ...
                'FrameID', view.FrameID, 'Tick', view.Tick, 'Phase', view.Phase, 'Observation', 0, ...
                'ActionLabels', view.ActionLabels, 'ActionMask', false(1, numel(view.ActionLabels)));
        end

        function m = permitted_(~, p, labels)
            m = false(1, numel(labels));
            if ~isfield(p, 'mask'), return; end
            [tf, loc] = ismember(string(labels), p.mask.Labels);
            m(tf) = p.mask.Permitted(loc(tf));
        end

        function t = experience_(obj, feedback, gamma)
            % EXPERIENCE_  One step of the mind's own experience, from its own records.
            next = feedback.NextView;
            dec = obj.LastDecision_;
            nextState = obj.LastState_;
            nextLabels = obj.LastLabels_;
            if ~isempty(next), nextLabels = next.ActionLabels; end
            nextMask = false(1, numel(nextLabels));
            sensed = false;
            if ~isempty(next)
                % the last view is sensed too (a final reveal can matter to a learner modelling the others); after a
                % termination nothing is permitted, so the mask stays empty
                q = obj.perceive_(next);
                if isfield(q, 'state'), nextState = q.state; sensed = true; end
                if ~feedback.Terminated, nextMask = obj.permitted_(q, nextLabels); end
            end
            if feedback.Truncated
                % after a time limit nothing is permitted any more; what the Laws WOULD permit next (needed to bootstrap)
                % is the successor mask the environment states for the cut-off episode
                nextMask = feedback.successorMask();
            end
            t = OceanAgents.Core.Transition('AgentID', dec.AgentID, 'EpisodeID', dec.EpisodeID, ...
                'Observation', obj.LastState_, 'Action', dec.Action, 'ActionLabels', obj.LastLabels_, ...
                'Reward', feedback.Reward, 'NextObservation', nextState, 'NextActionLabels', nextLabels, ...
                'NextMask', nextMask, 'Terminated', feedback.Terminated, 'Truncated', feedback.Truncated, ...
                'Duration', feedback.Duration, 'Discount', gamma ^ feedback.Duration, ...
                'PolicyVersion', dec.PolicyVersion, 'Likelihood', dec.Likelihood, 'NextSensed', sensed);
        end

        function dec = decideWith_(obj, view, stream, labels, rawLabels)
            % DECIDEWITH_  Decide from observations only (AB2 I10): the policy sees a view built from the mind's
            % perceptions plus bookkeeping fields; a running body behavior is let continue (D23, I13).
            p = obj.perceive_(view);
            obj.LastState_ = 0;
            if isfield(p, 'state'), obj.LastState_ = p.state; end
            if isfield(p, 'behavior') && p.behavior.Running
                nextMove = obj.Body.nextBehaviorMove();
                allowed = ~isfield(p, 'mask') || any(p.mask.Labels(p.mask.Permitted) == nextMove);
                if allowed
                    label = obj.Body.toAction();             % the behavior's next step; the mind issues nothing
                    dec = OceanAgents.Core.Decision('AgentID', view.AgentID, 'EpisodeID', view.EpisodeID, ...
                        'FrameID', view.FrameID, 'Action', rawLabels(labels == label));
                    obj.remember_(dec, view, "");
                    return
                end
                % the Laws forbid the behavior's next move: stop it (critic AB2 finding 4); a committed behavior
                % that cannot go on is an error, never a silent illegal move
                replies = obj.sendBehavior_(struct('Behavior', "stop"));
                if ~isempty(replies) && replies(1).isRefusal()
                    error('OceanAgents:Body:BehaviorBlocked', ['The Laws forbid move "%s" of a behavior that runs to ' ...
                        'completion.'], nextMove);
                end
            end
            if ~isfield(p, 'mask')
                error('OceanAgents:Embodied:Incompatible', 'This mind needs to perceive which moves the Laws permit.');
            end
            permitted = obj.permitted_(p, view.ActionLabels);
            state = obj.LastState_;
            seen = OceanAgents.Core.ActorView('AgentID', view.AgentID, 'EpisodeID', view.EpisodeID, ...
                'FrameID', view.FrameID, 'Tick', view.Tick, 'Phase', view.Phase, 'Observation', state, ...
                'ActionLabels', view.ActionLabels, 'ActionMask', permitted);
            policy = obj.choosePolicy_(p, view);
            dec = policy.act(seen, stream);
            obj.remember_(dec, view, obj.ActedPolicy_);
            chosen = string(dec.Action);
            done = obj.issue_(chosen);
            if done ~= chosen
                error('OceanAgents:Body:UnknownCommand', 'The body turned "%s" into "%s".', chosen, done);
            end
        end

        function requireBetweenEpisodes_(obj, what)
            if obj.InEpisode
                error('OceanAgents:Embodied:MidEpisode', '%s is allowed only between episodes.', what);
            end
        end

        function requireBody_(obj)
            if isempty(obj.Body)
                error('OceanAgents:Embodied:Incompatible', 'This mind has no body yet (use Dualism.form).');
            end
        end
    end
end
