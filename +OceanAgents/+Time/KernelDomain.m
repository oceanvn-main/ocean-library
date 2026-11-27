classdef KernelDomain < handle
    % KERNELDOMAIN  A time domain BACKED BY the existing world kernel (TIME_Kernel_Integration_Builder_Plan.md B1-B3).
    % It owns no clock, calendar or event queue: time, due work, action/feedback and RNG stay with World.Timeline,
    % World.Calendar and AgentWorld. It adds (a) the TimeContext of a committed boundary, (b) ONE post-commit capture
    % path into a CommitLog, (c) capability flags, (d) a pause that blocks later transactions, (e) entry points to the
    % world's active registration.
    %
    %   aw  = OceanAgents.World.AgentWorld(doors, cfg);
    %   dom = OceanAgents.Time.KernelDomain(aw, 'Components', ["Pos"], 'DomainID', "cliff");
    %   dom.reset(3);                 % = aw.reset(3); published as commit 1 (tick 0)
    %   dom.step(batch);              % = aw.step(batch): the normal transaction; published as the next commit
    %   dom.Log.at(2)                 % committed history; FrameCoordinator / ViewCursor read it unchanged
    %
    %   w   = OceanAgents.World.World(doors);          % a bare world is autonomous
    %   dom = OceanAgents.Time.KernelDomain(w);  dom.reset(1);  dom.advanceUntil(100);   % one commit per instant processed
    %
    % AgentWorld is ACTION-DRIVEN: advanceUntil is refused (ActionDriven) before anything changes - a step needs the
    % action batch and returns the feedback; use step(batch). Pausing refuses later steps; a step already in progress
    % is never interrupted (it is atomic). Handler registration, scheduling, per-owner cancellation, checkpoint export
    % and cross-domain messages are Unsupported (capabilities say so) - the kernel's own calendar semantics are kept.
    % A failed step publishes nothing. A recording failure never breaks the committed world: it sets HistoryGap and
    % keeps the error (PublishError); history then has a stated gap, never a silent one.
    properties (SetAccess = private)
        DomainID string
        Unit string = "tick"
        RunID string = "run"
        BranchID string = "main"
        Paused logical = false
        Commit double = 0            % commits published (reset and every completed advance/step)
        Visits double = 0            % kernel instants processed through advanceUntil or steps taken through step
        Components string = strings(1, 0)
        Log OceanAgents.Time.CommitLog
        HistoryGap logical = false
        PublishError = []
    end
    properties (Dependent)
        Now                          % kernel authority: the clock of the world at its committed boundary
        Version                      % source version of the history (CommitLog.Version)
    end
    properties (Access = private)
        World_
        IsAgent_ logical
        Episode_ double = 0
        StateFcn_
    end
    methods
        function obj = KernelDomain(world, opts)
            arguments
                world
                opts.DomainID (1,1) string = "kernel"
                opts.Components string = strings(1, 0)
                opts.Log = []
                opts.RunID (1,1) string = "run"
                opts.StateFcn = []
            end
            obj.IsAgent_ = isa(world, 'OceanAgents.World.AgentWorld');
            if ~obj.IsAgent_ && ~isa(world, 'OceanAgents.World.World')
                error('OceanAgents:Time:BadValue', 'KernelDomain needs a World or an AgentWorld.');
            end
            obj.World_ = world;
            obj.DomainID = opts.DomainID;
            obj.RunID = opts.RunID;
            obj.Components = reshape(opts.Components, 1, []);
            obj.StateFcn_ = opts.StateFcn;
            if isempty(opts.Log), obj.Log = OceanAgents.Time.CommitLog(); else, obj.Log = opts.Log; end
            if obj.IsAgent_
                world.onCommit(@(aw) obj.publishAgent_(aw));
            end
        end

        function c = capabilities(obj)
            c = struct('Backend', "kernel", 'Phased', true, 'Autonomous', ~obj.IsAgent_, 'CommittedQuery', true, ...
                'MessageInjection', false, 'Restore', false, 'ActiveRegistration', true, ...
                'PredictionCancellation', "per-entity-predictor");
        end

        function t = get.Now(obj)
            if obj.IsAgent_, t = obj.World_.tick(); else, t = max(obj.World_.Now, 0); end
        end

        function v = get.Version(obj), v = obj.Log.Version; end

        function ctx = context(obj)
            ctx = OceanAgents.Time.TimeContext('RunID', obj.RunID, 'BranchID', obj.BranchID, 'Episode', obj.Episode_, ...
                'DomainID', obj.DomainID, 'Time', obj.Now, 'Unit', obj.Unit, 'Commit', obj.Commit, ...
                'SourceVersion', obj.Log.Version, 'ActiveSetRevision', obj.activeRevision());
        end

        function r = activeRevision(obj)
            if obj.IsAgent_, r = obj.World_.activeRevision(); else, r = obj.World_.ActiveRevision; end
        end

        function t = nextTime(obj)
            if obj.IsAgent_, t = Inf; else, t = obj.World_.NextTime; end
        end

        function pause(obj), obj.Paused = true; end
        function resume(obj), obj.Paused = false; end

        % ---- the normal drivers ----
        function out = reset(obj, seed)
            arguments
                obj
                seed double = []
            end
            if obj.Paused, error('OceanAgents:Time:DomainPaused', 'Domain %s is paused.', obj.DomainID); end
            if obj.IsAgent_
                out = obj.World_.reset(seed);                 % published by the AgentWorld commit listener
            else
                obj.World_.reset(seed);
                out = [];
                obj.Episode_ = obj.Episode_ + 1;
                obj.publish_(struct());
            end
        end

        function frame = step(obj, batch)
            % STEP  AgentWorld only: the unchanged action/feedback transaction (one World.advance), then one commit.
            if ~obj.IsAgent_
                error('OceanAgents:Time:Unsupported', 'step(batch) belongs to an AgentWorld; use advanceUntil for a bare world.');
            end
            if obj.Paused, error('OceanAgents:Time:DomainPaused', 'Domain %s is paused; no new transaction starts.', obj.DomainID); end
            frame = obj.World_.step(batch);
            obj.Visits = obj.Visits + 1;
        end

        function advanceUntil(obj, target)
            % ADVANCEUNTIL  Bare world: process every kernel instant <= target through World.advance (all phases, the
            % full ordering key); one commit per instant processed. No commit is made at the target itself.
            arguments
                obj
                target (1,1) double
            end
            if obj.IsAgent_
                error('OceanAgents:Time:ActionDriven', ['Domain %s is an AgentWorld: it advances only through step(batch) ' ...
                    '(action and feedback transaction).'], obj.DomainID);
            end
            if obj.Paused, error('OceanAgents:Time:DomainPaused', 'Domain %s is paused.', obj.DomainID); end
            w = obj.World_;
            if target < obj.Now
                error('OceanAgents:Time:Retroactive', 'Domain %s: Now = %g, cannot advance to %g.', obj.DomainID, obj.Now, target);
            end
            while isfinite(w.NextTime) && w.NextTime <= target
                w.advance();
                obj.Visits = obj.Visits + 1;
                obj.publish_(struct());
            end
        end

        % ---- active registration (the world's own operations) ----
        function ids = liveEntityIDs(obj), ids = obj.World_.liveEntityIDs(); end
        function ids = inactiveEntityIDs(obj), ids = obj.World_.inactiveEntityIDs(); end
        function ids = pinnedEntityIDs(obj)
            if obj.IsAgent_, ids = obj.World_.entityIDs(); else, ids = zeros(1, 0); end
        end
        function r = deactivate(obj, ids), r = obj.World_.deactivate(ids); end
        function r = activate(obj, ids, varargin), r = obj.World_.activate(ids, varargin{:}); end

        % ---- explicitly unsupported (refused before any side effect) ----
        function handler(obj, varargin), obj.unsupported_('handler registration'); end
        function schedule(obj, varargin), obj.unsupported_('scheduling (the kernel calendar is not a callback queue)'); end
        function inject(obj, varargin), obj.unsupported_('external schedule injection'); end
        function cancelOwner(obj, varargin), obj.unsupported_('cancel-by-owner (the kernel cancels per entity and predictor)'); end
        function exportState(obj), obj.unsupported_('checkpoint export (kernel restore is not supported)'); end
        function importState(obj, varargin), obj.unsupported_('checkpoint import (kernel restore is not supported)'); end
    end
    methods (Access = private)
        function unsupported_(obj, what)
            error('OceanAgents:Time:Unsupported', 'Kernel-backed domain %s does not support %s.', obj.DomainID, what);
        end

        function publishAgent_(obj, aw)
            obj.Episode_ = aw.EpisodeID;
            state = struct();
            try
                for c = obj.Components
                    v = aw.researchView(c);
                    state.(char(c)) = v.Value;
                    state.Roster = v.Roster; state.Ended = v.Ended; state.FrameID = v.FrameID;
                end
            catch err
                obj.recordFailure_(err);
                return
            end
            obj.publish_(state);
        end

        function publish_(obj, state)
            if ~obj.IsAgent_ && ~isempty(obj.StateFcn_)
                try
                    state = obj.StateFcn_(obj.World_);
                catch err
                    obj.recordFailure_(err); return
                end
            end
            obj.Commit = obj.Commit + 1;
            ctx = obj.context();
            try
                obj.Log.append(ctx, state);
            catch err
                obj.recordFailure_(err);
            end
        end

        function recordFailure_(obj, err)
            % The world has committed; a recording failure must not corrupt or abort it. History states its gap.
            obj.HistoryGap = true;
            obj.PublishError = err;
        end
    end
end
