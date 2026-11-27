classdef AgentHost < handle
    % AGENTHOST  One agent's private container: its mind, its optional learner, its action stream and its
    % transition bookkeeping. Runners talk to hosts; hosts never see an environment, a runner or another agent.
    %
    %   h = OceanAgents.Runtime.AgentHost("alice", mind, 'Learner', learner, 'Role', "default", ...
    %           'Gamma', 1, 'LearningEnabled', true);
    %   h.attachStream(rs);        % runner-only: the host's own action RandStream (one runner at a time)
    %   h.beginEpisode(view);      % Ready -> InEpisode
    %   d = h.decide(view);        % mind decides; the Decision is validated against the view
    %   h.observe(feedback);       % closes the transition -> learner.consume/update (only here, after the world moved)
    %   h.release();               % give the mind/learner back so another host may own them
    %
    % Status machine:  Ready --beginEpisode--> InEpisode --observe(ended feedback)--> Ready
    %                  any --invalidate--> Invalid (terminal: every later public call errors Runtime:InvalidState)
    % Learning: when LearningEnabled is false the learner is never called (C12). Updates happen only inside
    % observe(), i.e. after the environment committed the transition; update("step") after every consumed
    % transition for a "step"-boundary learner, update("episode") when this agent's episode ends.
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 4.1.

    properties (SetAccess = private)
        AgentID string = ""
        Role string = "default"
        Mind = []                 % OceanAgents.Core.IDecisionProcess
        Learner = []              % OceanAgents.Core.ILearner or []
        Gamma double = 1
        LearningEnabled logical = true
        Status string = "Ready"   % "Ready" | "InEpisode" | "Invalid"
        Stream = []               % RandStream attached by the runner
        UpdateCount double = 0    % total parameter updates reported by the learner (for RunRecord.UpdateCounts)
        InvalidReason string = ""
    end

    properties (Access = private)
        Builder_                  % TransitionBuilder (private to this host)
        Boundary_ string = "step" % learner's declared update boundary
        Token_ double = 0         % ownership token in the SharedComponent registry
        Bound_ logical = false    % a runner currently holds this host (attach-once claim)
        SignalMind_ logical = false % the mind owns its learners (AB1 section 7): the host calls mind.learn
        Observers_ = {}             % OceanAgents.Observe.IPlayerObserver, attached through a Player (PG1)
    end

    methods
        function obj = AgentHost(agentID, mind, options)
            arguments
                agentID (1,1) string {mustBeNonzeroLengthText}
                mind (1,1) OceanAgents.Core.IDecisionProcess
                options.Learner = []
                options.Role (1,1) string {mustBeNonzeroLengthText} = "default"
                options.Gamma (1,1) double {mustBeBetween(options.Gamma, 0, 1)} = 1
                options.LearningEnabled (1,1) logical = true
            end
            learner = options.Learner;
            if ~isempty(learner) && ~(isa(learner, 'OceanAgents.Core.ILearner') && isscalar(learner))
                error('OceanAgents:Core:InvalidRecord', 'Learner must be an OceanAgents.Core.ILearner or [].');
            end
            obj.AgentID = agentID;
            obj.Role = options.Role;
            obj.Gamma = options.Gamma;
            obj.LearningEnabled = options.LearningEnabled;
            obj.Mind = mind;
            obj.Learner = learner;
            obj.Builder_ = OceanAgents.Runtime.TransitionBuilder();
            if ~isempty(learner)
                d = learner.describe();
                % Host-learner pairing: the learner's discount must be the host's (A0 section 6.5).
                if isfield(d, 'Gamma') && ~isempty(d.Gamma) && abs(d.Gamma - obj.Gamma) > 0
                    fail_('GammaMismatch', 'Learner %s uses Gamma = %g but host %s has Gamma = %g.', ...
                        string(d.Name), d.Gamma, agentID, obj.Gamma);
                end
                if isfield(d, 'Boundary') && ~isempty(d.Boundary)
                    obj.Boundary_ = string(d.Boundary);
                end
                % The learner checks the acting policy: describe().Components.Policy when the mind exposes one,
                % otherwise the mind's own description (A0 section 11, 2026-09-28).
                md = mind.describe();
                if isfield(md, 'Components') && isstruct(md.Components) && isfield(md.Components, 'Policy')
                    md = md.Components.Policy;
                end
                learner.checkCompatible(md);
            end
            obj.SignalMind_ = isa(mind, 'OceanAgents.Minds.SignalMind');
            if obj.SignalMind_
                % The learner belongs to the mind (design D2); the host keeps only the gate (AB1 section 7).
                ds = mind.learnerDescriptions();
                for k = 1:numel(ds)
                    d = ds{k};
                    if isfield(d, 'Gamma') && ~isempty(d.Gamma) && abs(d.Gamma - obj.Gamma) > 0
                        fail_('GammaMismatch', 'Learner %s of the mind uses Gamma = %g but host %s has Gamma = %g.', ...
                            string(d.Name), d.Gamma, agentID, obj.Gamma);
                    end
                end
                if ~isempty(learner)
                    error('OceanAgents:Runtime:LearnerInMind', ['A signal mind owns its learners (design D27): give ' ...
                        'the learner to the mind with mind.addLearner(learner, id); the host only decides when ' ...
                        'learning happens.']);
                end
            end
            % Cheap first line of isolation: a mind or learner may belong to one live host only.
            components = {mind};
            if ~isempty(learner), components{end + 1} = learner; end
            obj.Token_ = OceanAgents.Runtime.AgentHost.ownership_('claim', components, agentID);
        end

        function attachStream(obj, stream)
            % ATTACHSTREAM  Runner-only: hand the host its own action RandStream (between episodes only).
            % Attaching is the runner's exclusive claim on the host: a host already bound to a runner refuses a
            % second one (Runtime:SharedComponent) until the first runner detaches it (runner release/delete).
            arguments
                obj
                stream (1,1) RandStream
            end
            obj.requireStatus_("Ready", 'attachStream');
            if obj.isBound()
                fail_('SharedComponent', ['Host %s is already bound to a runner (its action stream is attached); ' ...
                    'release or delete that runner first.'], obj.AgentID);
            end
            obj.Stream = stream;
            obj.Bound_ = true;
        end

        function detachStream(obj)
            % DETACHSTREAM  Runner-only: end the runner's claim (called by the runner's release/delete). The last
            % stream stays readable (e.g. to inspect its final State) until the next runner attaches its own.
            % Not allowed in the middle of an episode; an Invalid host may always be detached.
            if obj.Status == "InEpisode"
                fail_('InvalidState', 'Host %s is in an episode; it cannot be detached now.', obj.AgentID);
            end
            obj.Bound_ = false;
        end

        function tf = isBound(obj)
            % ISBOUND  True while a runner holds this host.
            tf = obj.Bound_;
        end

        function beginEpisode(obj, view)
            % BEGINEPISODE  Ready -> InEpisode; tells the mind a new episode starts at view.
            arguments
                obj
                view (1,1) OceanAgents.Core.ActorView
            end
            obj.requireStatus_("Ready", 'beginEpisode');
            obj.requireOwnView_(view);
            obj.Mind.beginEpisode(view);
            obj.Status = "InEpisode";
        end

        function dec = decide(obj, view)
            % DECIDE  Ask the mind for one decision on view, validate it against view, record it as pending.
            arguments
                obj
                view (1,1) OceanAgents.Core.ActorView
            end
            obj.requireStatus_("InEpisode", 'decide');
            obj.requireOwnView_(view);
            if isempty(obj.Stream)
                fail_('InvalidState', 'Host %s has no action stream attached (the runner attaches it).', obj.AgentID);
            end
            if obj.Builder_.IsOpen
                fail_('TransitionOrder', 'Host %s already has a pending decision for frame %d.', ...
                    obj.AgentID, obj.Builder_.Decision.FrameID);
            end
            dec = obj.Mind.decide(view, obj.Stream);
            if ~(isa(dec, 'OceanAgents.Core.Decision') && isscalar(dec))
                error('OceanAgents:Core:InvalidRecord', 'The mind of %s returned a %s, not a Decision.', ...
                    obj.AgentID, class(dec));
            end
            dec.validateAgainst(view);          % legal, same agent/episode/frame, consistent likelihood
            obj.Builder_.open(view, dec);
            obj.notify_("decision", view, dec);
        end

        function observe(obj, feedback)
            % OBSERVE  Deliver the committed outcome: close the pending transition, let the learner consume and
            % update (if enabled), tell the mind; if this agent's episode ended, finish it (-> Ready).
            arguments
                obj
                feedback (1,1) OceanAgents.Core.Feedback
            end
            obj.requireStatus_("InEpisode", 'observe');
            if feedback.AgentID ~= obj.AgentID
                error('OceanAgents:Core:InvalidRecord', 'Feedback for %s delivered to host %s.', ...
                    feedback.AgentID, obj.AgentID);
            end
            % Only a pending decision yields a Transition (a reward/ending event with no decision of this agent
            % behind it is passed to the mind but is not a learning sample).
            if obj.Builder_.IsOpen
                t = obj.Builder_.close(feedback, obj.Gamma);
                if obj.learning_()
                    if obj.SignalMind_
                        obj.addUpdates_(obj.Mind.learn(feedback, obj.Gamma));   % the mind builds its own experience
                    else
                        obj.Learner.consume(t);
                        if obj.Boundary_ == "step"
                            obj.addUpdates_(obj.Learner.update("step"));
                        end
                    end
                end
            end
            obj.Mind.observe(feedback);
            if feedback.ended()
                if obj.learning_()
                    if obj.SignalMind_
                        obj.addUpdates_(obj.Mind.learnEpisodeEnd());
                    else
                        obj.addUpdates_(obj.Learner.update("episode"));
                    end
                end
                obj.Mind.endEpisode();
                obj.Status = "Ready";
            end
            obj.notify_("feedback", [], feedback);
        end

        function setLearningEnabled(obj, tf)
            % SETLEARNINGENABLED  Switch learning on/off (off => the learner is never called).
            arguments
                obj
                tf (1,1) logical
            end
            obj.requireNotInvalid_('setLearningEnabled');
            obj.LearningEnabled = tf;
        end

        function invalidate(obj, reason)
            % INVALIDATE  Fail-stop: Status -> Invalid; any pending decision is dropped (never learned from).
            arguments
                obj
                reason (1,1) string = "invalidated"
            end
            obj.requireNotInvalid_('invalidate');
            obj.Builder_.reset();
            obj.Status = "Invalid";
            obj.InvalidReason = reason;
        end

        function release(obj)
            % RELEASE  Give up ownership of the mind/learner (another host may then own them). The host itself
            % becomes Invalid, so the same components can never be driven by two hosts. Safe to repeat.
            OceanAgents.Runtime.AgentHost.ownership_('release', obj.Token_);
            obj.Token_ = 0;
            if obj.Status ~= "Invalid"
                obj.Builder_.reset();
                obj.Status = "Invalid";
                obj.InvalidReason = "released";
            end
        end

        function s = exportState(obj)
            % EXPORTSTATE  Value snapshot of the host and its components (between episodes only).
            obj.requireStatus_("Ready", 'exportState');
            s = struct('AgentID', obj.AgentID, 'Role', obj.Role, 'Gamma', obj.Gamma, ...
                'LearningEnabled', obj.LearningEnabled, 'UpdateCount', obj.UpdateCount, ...
                'Mind', obj.Mind.exportState(), 'Learner', struct());
            if ~isempty(obj.Learner), s.Learner = obj.Learner.exportState(); end
        end

        function importState(obj, s)
            % IMPORTSTATE  Restore a snapshot made by exportState of a host with the same AgentID/Gamma.
            arguments
                obj
                s (1,1) struct
            end
            obj.requireStatus_("Ready", 'importState');
            if string(s.AgentID) ~= obj.AgentID || s.Gamma ~= obj.Gamma
                fail_('InvalidState', 'State of %s (Gamma %g) cannot be imported into host %s (Gamma %g).', ...
                    string(s.AgentID), s.Gamma, obj.AgentID, obj.Gamma);
            end
            obj.Mind.importState(s.Mind);
            if ~isempty(obj.Learner), obj.Learner.importState(s.Learner); end
            obj.LearningEnabled = s.LearningEnabled;
            obj.UpdateCount = s.UpdateCount;
        end

        function n = numObservers(obj)
            n = numel(obj.Observers_);
        end

        function tf = hasPendingDecision(obj)
            % HASPENDINGDECISION  True between decide() and the observe() that closes it.
            tf = obj.Builder_.IsOpen;
        end

        function delete(obj)
            % DELETE  A destroyed host no longer owns its components.
            if obj.Token_ > 0
                OceanAgents.Runtime.AgentHost.ownership_('release', obj.Token_);
            end
        end

        function meta = getMetadata(obj)
            meta = pillarMeta_(obj, {sprintf('agent %s, role %s, Gamma %g, learning %d, learner boundary "%s"', ...
                obj.AgentID, obj.Role, obj.Gamma, obj.LearningEnabled, obj.Boundary_)}, ...
                "A0 contract section 4.1 (agent host status machine)");
        end

        function spec = getPipelineSpec(obj)
            parts = {class(obj.Mind)};
            if ~isempty(obj.Learner), parts{end + 1} = class(obj.Learner); end
            spec = pillarSpec_(obj, ["Agent-ActorView" "Agent-Feedback"], ["Agent-Decision" "Agent-Transition"], parts);
        end
    end

    methods (Access = {?OceanAgents.Player})
        function attachObserver_(obj, observer)
            if any(cellfun(@(o) o == observer, obj.Observers_)), return; end   % attached once
            obj.Observers_{end + 1} = observer;
        end

        function detachObserver_(obj, observer)
            keep = ~cellfun(@(o) o == observer, obj.Observers_);
            obj.Observers_ = obj.Observers_(keep);
        end
    end

    methods (Access = private)
        function notify_(obj, what, view, record)
            % NOTIFY_  Tell attached observers (value records only). An observer that fails is detached and recorded;
            % it can never stop or change the run (PG1: display must not alter the trajectory).
            k = 1;
            while k <= numel(obj.Observers_)
                o = obj.Observers_{k};
                try
                    if what == "decision", o.onDecision(obj.AgentID, view, record);
                    else, o.onFeedback(obj.AgentID, record); end
                    k = k + 1;
                catch err
                    o.failed_(err.message);
                    obj.Observers_(k) = [];
                end
            end
        end

        function tf = learning_(obj)
            tf = obj.LearningEnabled && (~isempty(obj.Learner) || (obj.SignalMind_ && obj.Mind.hasLearners()));
        end

        function addUpdates_(obj, n)
            if ~(isnumeric(n) && isscalar(n) && isfinite(n) && n >= 0 && n == fix(n))
                error('OceanAgents:Learners:Incompatible', 'Learner update must return a non-negative integer count.');
            end
            obj.UpdateCount = obj.UpdateCount + double(n);
        end

        function requireNotInvalid_(obj, op)
            if obj.Status == "Invalid"
                fail_('InvalidState', 'Host %s is Invalid (%s); %s is not allowed.', obj.AgentID, obj.InvalidReason, op);
            end
        end

        function requireStatus_(obj, status, op)
            obj.requireNotInvalid_(op);
            if obj.Status ~= status
                fail_('InvalidState', 'Host %s is %s; %s requires %s.', obj.AgentID, obj.Status, op, status);
            end
        end

        function requireOwnView_(obj, view)
            if view.AgentID ~= obj.AgentID
                error('OceanAgents:Core:InvalidRecord', 'View for %s handed to host %s.', view.AgentID, obj.AgentID);
            end
        end
    end

    methods (Static, Access = private)
        function out = ownership_(op, varargin)
            % OWNERSHIP_  Process-wide registry of components owned by live hosts.
            %   token = ownership_('claim', components, agentID)   error SharedComponent if any is already owned
            %   ownership_('release', token)
            % Entries hold the components (not the host), so a host that goes out of scope is still destroyed
            % and its delete() releases its entries.
            persistent owned nextToken
            if isempty(owned)
                owned = struct('Component', {}, 'Token', {}, 'AgentID', {});
                nextToken = 0;
            end
            out = [];
            switch op
                case 'claim'
                    [components, agentID] = varargin{:};
                    for c = 1:numel(components)
                        for e = 1:numel(owned)
                            if sameHandle_(owned(e).Component, components{c})
                                fail_('SharedComponent', ...
                                    'The %s given to host %s is already owned by live host %s (release it first).', ...
                                    class(components{c}), agentID, owned(e).AgentID);
                            end
                        end
                    end
                    nextToken = nextToken + 1;
                    for c = 1:numel(components)
                        owned(end + 1) = struct('Component', components{c}, 'Token', nextToken, 'AgentID', agentID); %#ok<AGROW>
                    end
                    out = nextToken;
                case 'release'
                    token = varargin{1};
                    owned = owned([owned.Token] ~= token);
            end
        end
    end
end

function tf = sameHandle_(a, b)
% SAMEHANDLE_  Handle identity for two objects of the same class.
tf = false;
if ~strcmp(class(a), class(b)), return; end
try
    r = (a == b);
    tf = isscalar(r) && logical(r);
catch
    tf = false;
end
end
