classdef World < handle
    % WORLD  The world kernel's loop: it moves the clock, releases due discontinuities, runs the rules of the driven
    % doors phase by phase, and applies their collected outputs at each phase's synchronisation point - all or nothing.
    %
    %   doors = [moverDoor, batteryDoor, agerDoor];                     % OceanAgents.World.DoorSpec array
    %   w = OceanAgents.World.World(doors, 'Phases', ["resolve" "act" "settle" "project"], 'SkipIdle', true);
    %   w.reset(7);                     % Init spawns go through the guard; Now = 0
    %   r = w.advance();                % exactly one instant (an InstantReport struct)
    %   reports = w.run(200);           % instants until Now >= 200 or no future work
    %   w.inject(OceanAgents.World.Event.make("Ping"));     % external event, delivered at the next instant
    %   w.reset([]);                    % W2: new episode, every door stream CONTINUES (native env.reset() semantics)
    %   ev = w.takeOutbox();            % W2: host-only record of the 'Outbox' event types (e.g. rewards, endings)
    %   v = w.project(entityID);        % projection rules, visibility-filtered for that viewer
    %
    % ONE INSTANT t (W1_Kernel_Contract.md section 7.1): Now = t, then for each phase p in order:
    %   1 due       Calendar.due(t, p): predictions become events of their DueEvent type (SourceEntity = the entity);
    %               scheduled commands join this phase's commands
    %   2 deliver   events whose delivery point is (t, p) go to their consuming rules
    %   3 run       each door (rank order) runs its rules of phase p if the door is driven at t and the rule has rows
    %               (a rule with no Requires runs whenever its door is driven), or if events were delivered to the rule.
    %               Rules read the state as it was at the start of the phase; nothing is written while rules run.
    %   4 collect   writes, events, commands (checked against each RuleSpec: OceanAgents:World:Undeclared)
    %   5 sync      a) write conflicts / Store.checkWrites; b) apply the writes; (attach/detach may name only the
    %                  issuing door's own components: otherwise refused, Reason "not-owner")
    %               c) commands in tie-break order, each evaluated on the state after the writes and every earlier
    %                  admitted command of this sync, and applied if admitted: guard (false -> recorded no-op); spawn/
    %                  attach through admission; an attach of a component the entity already has is refused
    %                  ("already-has": commands never change values); refusals are recorded and the run continues;
    %               d) future commands onto the calendar;
    %               e) monitor axioms on the POST-command state (entities removed here are not monitored; new ones
    %                  passed admission) - a violation raises OceanAgents:Guard:AxiomViolated naming the axiom and the
    %                  entity IDs, and the fail-stop undoes the whole phase; f) predictors and skip-safe monitor
    %                  checks on changed/new rows -> Calendar; g) stamp emitted events for delivery.
    % SKIP-SAFE MONITORS (section 12): an entity RANGE monitor on a linear or clock component gets, per member row, a
    %   calendar check at the first integer instant its value leaves [lo, hi] (closed form, corrected by exact
    %   evaluation; recomputed when the row's time function changes; cancelled with the entity), so a skipping world
    %   processes that instant and fails exactly where a stepping world fails. A PREDICATE monitor reading a linear or
    %   clock component is refused at construction when SkipIdle is true (OceanAgents:World:BadAssembly); it is
    %   allowed when stepping. Monitor checks are Calendar items of Kind "monitor" (a flag; no name is reserved).
    %   Fail-stop: any error restores Store, Registry, Calendar (and resources, pending events) to the last completed
    %   synchronisation point, sets Status "Failed", records Failure and rethrows; later calls raise
    %   OceanAgents:World:InvalidState until reset.
    % EVENT DELIVERY (7.2): an event emitted at (t, p) reaches a consumer of phase q at t if q comes after p, else at t+1,
    %   so a rule never sees an event from its own phase at the same instant. Delivered events are ordered by
    %   (time, emission phase, source door rank, source rule rank, source entity, seq); injected events have rank 0.
    % NEXT INSTANT (7.3): with SkipIdle, min(next driven instant, next calendar item, next pending delivery, Horizon);
    %   otherwise t + 1. Skipping is exact because nothing changes at a skipped instant (K9).
    %
    % W2 KERNEL ADDITIONS (W2_Grid_Doors_Contract.md section 2):
    %   - a door with Stream "legacy-mt19937ar" gets RandStream('mt19937ar', 'Seed', seed) at reset(seed) (bit-for-bit
    %     reproduction of existing environments only); "keyed" doors keep StreamRegistry purpose "door:<Name>";
    %   - reset([]) keeps every door stream as it is (no reseeding) and clears everything else as a seeded reset does;
    %     before any seeded reset it raises OceanAgents:World:NoSeed;
    %   - 'Outbox', types: every event of those types emitted during an advance (by a rule or as a due prediction) is
    %     also recorded for the HOST; takeOutbox() returns them in tie-break order (time, emission phase, source door
    %     rank, source rule rank, source entity, seq) and clears the record. Rules and projections never see it. For
    %     construction check V4 an outbox type counts as consumed: the World validates the assembly together with a
    %     validation-only sink door consuming the outbox types (and lists them as injectable for V5, since a type may
    %     be recorded before any rule emits it); the sink never exists in the running world.
    %
    % Kernel clarifications (reported with the W1 build):
    %   - ctx.Read also carries <name>_Rate for linear components (a predictor needs the rate to solve its crossing);
    %   - predictors run at EVERY synchronisation point on the rows whose predicted component changed
    %     (Store.takeDirty, which includes new rows); rows that lost the component get their prediction removed;
    %     a prediction is scheduled at the earliest phase that consumes its DueEvent; a crossing at or before the
    %     current point is delivered at the next reachable point of that phase;
    %   - commands issued by rules carry IssuerEntity 0; a command whose target is not alive is refused ("not-alive");
    %   - projections get ctx.Stream = [] (a view must not consume world randomness); visibility applies to every
    %     projection: "hidden" never visible, "self" only on the viewer's row (defaults with <name>_Has false);
    %   - reset processes the Init spawns at a synchronisation point before instant 0 (report in ResetReport);
    %     an Init spawn with a future 'At' is scheduled on the Calendar and applied at that instant;
    %     the first advance() processes the first instant >= 0 with work (0 when stepping).

    properties (SetAccess = private)
        Status string = "Idle"          % "Idle" | "Running" | "Failed"
        Failure = []                    % MException of the failure, or []
        Phases string = strings(1, 0)
        SkipIdle logical = true
        Horizon double = Inf
        Injectable string = strings(1, 0)
        Doors = OceanAgents.World.DoorSpec.empty(1, 0)   % flattened, in rank order
        Store                           % OceanAgents.World.Store
        Registry                        % OceanAgents.World.Registry
        Calendar                        % OceanAgents.World.Calendar
        Timeline                        % OceanAgents.World.Timeline
        Participants                    % OceanAgents.World.Participants: who is due when (GOD does not know)
        Resources struct = struct()     % current values of every door resource
        NextTime double = 0
        ResetReport struct = struct()   % the Init synchronisation of the last reset (same form as a phase report)
        Seed double = 0                 % seed of the last SEEDED reset
        Outbox string = strings(1, 0)   % event types recorded for the host (takeOutbox)
        ActiveRevision double = 0       % incremented at every effective change of the inactive set (deactivate/activate)
    end

    properties (Dependent)
        Now
    end

    properties (Access = private)
        Rules_ struct                   % one entry per rule, in (door rank, rule rank) order
        Consumers_                      % containers.Map event type -> rule indices (kind "rule")
        DuePhase_                       % containers.Map DueEvent -> earliest consumer phase index
        Types_ struct = struct()        % component -> ComponentType
        Owner_                          % containers.Map component/resource -> door rank
        InitialResources_ struct = struct()
        Streams_ cell = {}              % door rank -> RandStream
        Pending_ struct                 % queued event deliveries (Events, Rule, Time, Phase, Key)
        EvSeq_ double = 0
        CmdSeq_ double = 0
        LastProcessed_ double = -1
        CoveredThrough_ double = -1     % run(tEnd) covered every instant <= this (idle ones skipped)
        Seeded_ logical = false         % a seeded reset has happened (reset([]) needs one)
        OutboxEvents_ = OceanAgents.World.Event.empty(1, 0)
        OutboxKey_ double = zeros(0, 6)
        GuardIndex_ struct = struct()   % GuardEngine.buildIndex(Doors), built once at construction
        MonRange_ struct                % entity range monitors on linear/clock components (skip-safe checks)
        TimeWatched_ string = strings(1, 0)   % linear/clock components watched by predictors or range monitors
        Inactive_ double = zeros(0, 1)  % IDs of entities outside the active set (empty = full registration)
        Stash_ struct = struct('ID', {}, 'FrozenAt', {}, 'Items', {})   % calendar commands of inactive entities
    end


    methods
        function obj = World(doors, options)
            arguments
                doors
                options.Phases string = ["resolve" "act" "settle" "project"]
                options.SkipIdle (1,1) logical = true
                options.Horizon (1,1) double {mustBeNonnegative} = Inf
                options.Injectable string = strings(1, 0)
                options.Outbox string = strings(1, 0)
            end
            G = OceanAgents.World.Guard.GuardEngine;
            flat = G.flattenDoors(doors);
            phases = reshape(options.Phases, 1, []);
            injectable = reshape(options.Injectable, 1, []);
            if isempty(injectable), injectable = strings(1, 0); end
            outbox = reshape(unique(options.Outbox, 'stable'), 1, []);
            if isempty(outbox), outbox = strings(1, 0); end
            % Every construction check (sections 8, 12) throws. Outbox types count as consumed for V4: validate with a
            % validation-only sink door consuming them (and list them as injectable for V5). It is not part of the world.
            checked = flat;
            if ~isempty(outbox) && ~isempty(phases)
                sinkName = "World_outbox_host";
                while any([flat.Name] == sinkName), sinkName = sinkName + "_"; end
                sink = OceanAgents.World.DoorSpec('Name', sinkName, 'Drive', "on-event", 'Rules', ...
                    OceanAgents.World.RuleSpec('Name', "Host", 'Phase', phases(end), 'Consumes', outbox, ...
                    'Fn', @(ctx) struct()));
                checked = [flat sink];
            end
            G.validateAssembly(checked, phases, unique([injectable outbox], 'stable'));
            obj.GuardIndex_ = G.buildIndex(flat);               % admission/monitor index, built once
            obj.Doors = flat;
            obj.Phases = phases;
            obj.SkipIdle = options.SkipIdle;
            if ~(isinf(options.Horizon) || options.Horizon == fix(options.Horizon))
                fail_('BadValue', 'Horizon must be a non-negative integer tick or Inf, got %g.', options.Horizon);
            end
            obj.Horizon = options.Horizon;
            obj.Injectable = injectable;
            obj.Outbox = outbox;
            obj.buildIndex_();
            obj.buildMonitorIndex_();
            obj.Store = OceanAgents.World.Store(obj.Types_);
            obj.Registry = OceanAgents.World.Registry();
            obj.Calendar = OceanAgents.World.Calendar('Phases', phases, 'Owners', [flat.Name]);   % owners = registered participants
            obj.Timeline = OceanAgents.World.Timeline(phases);                % GOD: time only (S1.2b-1)
            obj.Participants = OceanAgents.World.Participants();              % who is due when
            for d = reshape(flat, 1, [])                                      % today's participants: the doors' rule sets
                obj.Participants.register(d.Name, d.Drive, d.DriveK);
            end
            obj.Pending_ = emptyPending_();
        end

        function t = get.Now(obj)
            t = obj.Timeline.Now;
        end

        % ============================================================== public loop
        function reset(obj, seed)
            % RESET  New episode: clear state (the entity ID counter continues), fresh door streams, Init spawns
            % through normal admission at a synchronisation point before instant 0. reset([]) keeps every door
            % stream as it is (continuing streams; needs an earlier seeded reset, else OceanAgents:World:NoSeed).
            arguments
                obj
                seed double = 0
            end
            if isempty(seed)
                if ~obj.Seeded_
                    fail_('NoSeed', 'reset([]) continues the door streams; it needs an earlier seeded reset.');
                end
            elseif ~(isscalar(seed) && isfinite(seed) && seed >= 0 && seed == fix(seed))
                fail_('BadValue', 'reset(seed) needs a non-negative integer seed or [].');
            end
            obj.Store.clearAll();
            obj.Registry.clearEpisode();
            obj.Calendar.clear();
            obj.Timeline.reset();
            if ~isempty(obj.Inactive_)
                obj.Inactive_ = zeros(0, 1);
                obj.Stash_ = struct('ID', {}, 'FrozenAt', {}, 'Items', {});
                obj.ActiveRevision = obj.ActiveRevision + 1;
            end
            obj.Resources = obj.InitialResources_;
            obj.Pending_ = emptyPending_();
            obj.EvSeq_ = 0;
            obj.CmdSeq_ = 0;
            obj.LastProcessed_ = -1;
            obj.CoveredThrough_ = -1;
            obj.OutboxEvents_ = OceanAgents.World.Event.empty(1, 0);
            obj.OutboxKey_ = zeros(0, 6);
            if ~isempty(seed)                            % fresh streams; reset([]) keeps the current ones
                obj.Seed = seed;
                obj.Seeded_ = true;
                reg = OceanAgents.Runtime.StreamRegistry(seed);
                obj.Streams_ = cell(1, numel(obj.Doors));
                for d = 1:numel(obj.Doors)
                    if obj.Doors(d).Stream == "legacy-mt19937ar"
                        obj.Streams_{d} = RandStream('mt19937ar', 'Seed', seed);   % bit-for-bit legacy only
                    else
                        obj.Streams_{d} = reg.stream("door:" + obj.Doors(d).Name);
                    end
                end
            end
            obj.Status = "Running";
            obj.Failure = [];
            snap = obj.capture_();
            try
                cmds = obj.emptyCommands_();
                future = struct('Cmd', {}, 'DoorRank', {}, 'RuleRank', {}, 'Time', {}, 'PhaseIdx', {});
                for d = 1:numel(obj.Doors)
                    fn = obj.Doors(d).Init;
                    if isempty(fn), continue; end
                    out = fn(obj.Streams_{d});
                    if isempty(out), continue; end
                    if ~isa(out, 'OceanAgents.World.Command')
                        fail_('Undeclared', 'Init of door %s must return Command objects.', obj.Doors(d).Name);
                    end
                    for c = reshape(out, 1, [])
                        if c.Kind ~= "spawn"
                            fail_('Undeclared', 'Init of door %s may only spawn (got %s).', obj.Doors(d).Name, c.Kind);
                        end
                        obj.CmdSeq_ = obj.CmdSeq_ + 1;
                        if isempty(c.Time)
                            stamped = c.stamp(obj.Doors(d).Name + "/Init", 0, obj.CmdSeq_, 0, "");
                            cmds(end + 1) = struct('Cmd', stamped, 'DoorRank', d, 'RuleRank', 0); %#ok<AGROW>
                        else                                     % a future spawn: scheduled, applied at its instant
                            q = 1;
                            if strlength(c.Phase) > 0, q = obj.Timeline.phaseIndex(c.Phase); end
                            stamped = c.stamp(obj.Doors(d).Name + "/Init", 0, obj.CmdSeq_, c.Time, obj.Phases(q));
                            future(end + 1) = struct('Cmd', stamped, 'DoorRank', d, 'RuleRank', 0, ...
                                'Time', c.Time, 'PhaseIdx', q); %#ok<AGROW>
                        end
                    end
                end
                col = obj.emptyCollect_();
                col.Commands = cmds;
                col.Future = future;
                obj.ResetReport = obj.synchronise_(0, 0, col);
                obj.ResetReport.Phase = "reset";
            catch err
                obj.failStop_(err, snap);
            end
            obj.NextTime = obj.computeNext_(-1);
        end

        function report = advance(obj)
            % ADVANCE  Process exactly one instant (the current NextTime); returns an InstantReport.
            obj.requireRunning_('advance');
            t = obj.NextTime;
            if ~isfinite(t)
                fail_('InvalidState', 'No future work: the world has nothing scheduled, driven or pending.');
            end
            % Safe outside the fail-stop: t = NextTime > every processed instant, so setNow cannot fail, and the
            % clock is not part of the restored state (Now stays at a failed instant). The Calendar cursor moves
            % with due(t, p) at each phase.
            obj.Timeline.setNow(t);
            phases = repmat(emptyPhaseReport_(), 1, numel(obj.Phases));
            for p = 1:numel(obj.Phases)
                snap = obj.capture_();
                try
                    phases(p) = obj.runPhase_(t, p);
                catch err
                    obj.failStop_(err, snap);
                end
            end
            obj.LastProcessed_ = t;
            obj.NextTime = obj.computeNext_(t);
            report = struct('Time', t, 'Phases', phases, 'NextTime', obj.NextTime);
        end

        function reports = run(obj, tEnd)
            % RUN  advance() while the next instant is <= tEnd; stops when there is no future work (NextTime = Inf).
            % tEnd: a non-negative integer tick or Inf. run(Inf) needs an end: SkipIdle (stops at no future work) or a
            % finite Horizon - a stepping world without a horizon always has a next instant (World:BadValue).
            % Afterwards time counts as covered through min(tEnd, Horizon) when that is finite, so a later inject is
            % delivered at the same instant whether the world skipped or stepped.
            arguments
                obj
                tEnd (1,1) double
            end
            obj.requireRunning_('run');
            if isnan(tEnd) || tEnd < 0 || (isfinite(tEnd) && tEnd ~= fix(tEnd))
                fail_('BadValue', 'run(tEnd) needs a non-negative integer tick or Inf, got %g.', tEnd);
            end
            if isinf(tEnd) && ~obj.SkipIdle && isinf(obj.Horizon)
                fail_('BadValue', 'run(Inf) would never end: a stepping world without a Horizon always has a next instant.');
            end
            reports = struct('Time', {}, 'Phases', {}, 'NextTime', {});
            while isfinite(obj.NextTime) && obj.NextTime <= tEnd
                reports(end + 1) = obj.advance(); %#ok<AGROW>
            end
            covered = min(tEnd, obj.Horizon);
            if isfinite(covered)
                obj.CoveredThrough_ = max(obj.CoveredThrough_, covered);
            end
        end

        function inject(obj, events)
            % INJECT  External events (declared 'Injectable' types), delivered to their consumers at the next instant
            % after everything processed or covered by run(tEnd); beyond the Horizon -> OceanAgents:World:InvalidState
            % (the world does not fail).
            arguments
                obj
                events                                   % OceanAgents.World.Event array; [] is a no-op
            end
            obj.requireRunning_('inject');
            if isempty(events), return; end                          % inject([]) is a no-op
            if ~isa(events, 'OceanAgents.World.Event')
                fail_('BadValue', 'inject needs OceanAgents.World.Event objects, got %s.', class(events));
            end
            % the next instant after everything processed or covered by run(tEnd): the same in skipping and stepping
            at = max(obj.LastProcessed_, obj.CoveredThrough_) + 1;
            if at > obj.Horizon
                fail_('InvalidState', 'Cannot inject: the next instant (%g) is beyond the horizon (%g).', at, obj.Horizon);
            end
            events = reshape(events, 1, []);
            bad = events(~ismember([events.Type], obj.Injectable));   % validate ALL before queueing any (atomic)
            if ~isempty(bad)
                fail_('UnknownEvent', 'Event type(s) %s not declared Injectable; nothing was injected.', ...
                    strjoin(unique([bad.Type]), ', '));
            end
            for e = events
                obj.EvSeq_ = obj.EvSeq_ + 1;
                stamped = e.stamp(at, "", "", obj.EvSeq_);
                obj.queueDelivery_(stamped, at, 0, 0, 0);
            end
            obj.NextTime = min(obj.NextTime, at);
        end

        function ev = takeOutbox(obj)
            % TAKEOUTBOX  Host-only: the recorded 'Outbox' events since the last take (or reset), in tie-break order
            % (time, emission phase, source door rank, source rule rank, source entity, seq); clears the record.
            ev = obj.OutboxEvents_;
            if ~isempty(ev)
                [~, order] = sortrows(obj.OutboxKey_);
                ev = ev(order);
            end
            obj.OutboxEvents_ = OceanAgents.World.Event.empty(1, 0);
            obj.OutboxKey_ = zeros(0, 6);
        end

        % ============================================================== active registration (TIME integration)
        function ids = liveEntityIDs(obj)
            % LIVEENTITYIDS  IDs of every alive entity (ascending).
            ids = sort(reshape(obj.Registry.idsOf(obj.Store.rowsWith()), [], 1));
        end

        function ids = pinnedEntityIDs(~)
            % PINNEDENTITYIDS  Entities that can never be deactivated (none in a bare world).
            ids = zeros(1, 0);
        end

        function ids = inactiveEntityIDs(obj)
            % INACTIVEENTITYIDS  IDs currently outside the active set (empty: full registration).
            ids = obj.Inactive_;
        end

        function tf = isActive(obj, ids)
            % ISACTIVE  Alive entities outside the inactive set (logical per id; dead ids are false).
            ids = reshape(ids, [], 1);
            tf = obj.Registry.isAlive(ids) & ~ismember(ids, obj.Inactive_);
        end

        function report = deactivate(obj, ids)
            % DEACTIVATE  Take alive entities out of the active set, at a commit boundary (between advances, never
            % during one). They receive no rule runs, predictions or monitor checks; their linear/clock values are held
            % at the current instant; commands targeting them are refused ("inactive"); their scheduled commands are
            % set aside (see activate). Validated entirely before anything changes.
            arguments
                obj
                ids (:,1) double
            end
            obj.requireRunning_('deactivate');
            ids = unique(ids);
            if isempty(ids), report = struct('Changed', false, 'Revision', obj.ActiveRevision); return; end
            if ~all(obj.Registry.isAlive(ids)), fail_('BadValue', 'deactivate: every entity must be alive.'); end
            if any(ismember(ids, obj.Inactive_)), fail_('BadValue', 'deactivate: an entity is already inactive.'); end
            rows = obj.Registry.rowsOf(ids);
            now = obj.boundary_();
            items = obj.Calendar.peek();
            stash = struct('ID', num2cell(ids'), 'FrozenAt', now, 'Items', {{}});
            for k = 1:numel(ids)
                stash(k).Items = items([items.EntityID] == ids(k) & [items.Kind] == "command");
            end
            obj.Store.freeze(rows, now);
            obj.Calendar.cancelEntity(ids);          % their predictions, monitor checks and commands (re-derived on activate)
            obj.Stash_ = [obj.Stash_ stash];
            obj.Inactive_ = [obj.Inactive_; ids];
            obj.ActiveRevision = obj.ActiveRevision + 1;
            obj.NextTime = obj.computeNext_(obj.LastProcessed_);
            report = struct('Changed', true, 'Revision', obj.ActiveRevision);
        end

        function report = activate(obj, ids, options)
            % ACTIVATE  Return inactive entities to the active set at the current commit boundary. Their time functions
            % resume from the held values (no jump) and predictors schedule from now: nothing is scheduled in the past.
            % Calendar commands that were set aside need a DECLARED policy (an undeclared one is refused):
            %   Deadlines "shift"  every set-aside command moves later by the inactive duration (deadlines paused too)
            %   Deadlines "drop"   they are discarded
            %   Deadlines "refuse" they are reinstated unchanged; any that is already overdue refuses the activation
            arguments
                obj
                ids (:,1) double
                options.Deadlines (1,1) string = ""
            end
            obj.requireRunning_('activate');
            ids = unique(ids);
            if isempty(ids), report = struct('Changed', false, 'Revision', obj.ActiveRevision); return; end
            if ~all(ismember(ids, obj.Inactive_)), fail_('BadValue', 'activate: every entity must be inactive.'); end
            now = obj.boundary_();
            sel = ismember([obj.Stash_.ID], ids);
            pol = options.Deadlines;
            allItems = [obj.Stash_(sel).Items];
            if ~isempty(allItems)
                if ~any(pol == ["shift" "drop" "refuse"])
                    fail_('UndeclaredPolicy', 'activate: %d scheduled command(s) were set aside; declare Deadlines as "shift", "drop" or "refuse".', numel(allItems));
                end
                if pol == "refuse" && any([allItems.Time] <= now)
                    fail_('OverdueWork', 'activate: a set-aside command is overdue at %g and Deadlines is "refuse".', now);
                end
            elseif pol ~= "" && ~any(pol == ["shift" "drop" "refuse"])
                fail_('UndeclaredPolicy', 'activate: Deadlines must be "shift", "drop" or "refuse".');
            end
            rows = obj.Registry.rowsOf(ids);
            snap = obj.capture_();
            try
                for k = find(sel)
                    st = obj.Stash_(k);
                    for it = reshape(st.Items, 1, [])
                        when = it.Time;
                        if pol == "shift", when = it.Time + (now - st.FrozenAt); end
                        if pol == "drop", continue; end
                        obj.Calendar.scheduleCommand(it.Command, when, it.PhaseIndex, it.OwnerRank, 'Rank', it.Rank);
                    end
                end
                obj.Store.unfreeze(rows, now);
                taken = struct();
                for comp = obj.TimeWatched_
                    taken.(char(comp)) = obj.Store.takeDirty(comp);
                end
                P = numel(obj.Phases);
                obj.runPredictors_(now, P, taken, rows);
                obj.scheduleMonitorChecks_(now, P, taken, rows);
            catch err
                obj.failStop_(err, snap);     % Inactive_ and Stash_ are untouched until everything above succeeded
            end
            obj.Stash_(sel) = [];
            obj.Inactive_ = setdiff(obj.Inactive_, ids, 'stable');
            obj.ActiveRevision = obj.ActiveRevision + 1;
            obj.NextTime = obj.computeNext_(obj.LastProcessed_);
            report = struct('Changed', true, 'Revision', obj.ActiveRevision);
        end

        function views = project(obj, viewerID)
            % PROJECT  Run every projection rule (phase order, then rank order) for viewerID (0 = no entity) and
            % return struct(<door>_<rule> -> out.View). Visibility, for EVERY projection (its own door's included):
            % "public" on every row, "self" only on the viewer's row, "hidden" on no row (defaults, <name>_Has false;
            % another door's hidden component is refused at construction, V6). Linear reads also get <name>_Rate.
            arguments
                obj
                viewerID (1,1) double {mustBeInteger, mustBeNonnegative} = 0
            end
            obj.requireRunning_('project');
            viewerRow = [];
            if viewerID > 0, viewerRow = obj.Registry.rowsOf(viewerID); end
            views = struct();
            t = obj.Now;
            proj = find(arrayfun(@(r) r.Kind == "projection", obj.Rules_));
            [~, order] = sortrows([[obj.Rules_(proj).PhaseIdx]' [obj.Rules_(proj).DoorRank]' [obj.Rules_(proj).RuleRank]']);
            for i = reshape(proj(order), 1, [])
                r = obj.Rules_(i);
                spec = r.Spec;
                rows = obj.Store.rowsWith(spec.Requires);
                read = struct();
                for nm = spec.Reads
                    if ~isfield(obj.Types_, nm)
                        read.(char(nm)) = obj.Resources.(char(nm));
                        continue
                    end
                    ty = obj.Types_.(char(nm));
                    v = obj.Store.valueAt(nm, rows, t);
                    has = obj.Store.has(nm, rows);
                    % visibility applies to every projection, its own door's included: hidden -> no row,
                    % self -> the viewer's own row only, public -> every row
                    switch ty.Visibility
                        case "hidden", masked = true(numel(rows), 1);
                        case "self"
                            if isempty(viewerRow), masked = true(numel(rows), 1); else, masked = rows ~= viewerRow; end
                        otherwise, masked = false(numel(rows), 1);
                    end
                    v(masked, :) = defaultsLike_(v(masked, :));
                    has(masked) = false;
                    read.(char(nm)) = v;
                    read.([char(nm) '_Has']) = has;
                    if ty.Kind == "linear"
                        raw = obj.Store.raw(nm);
                        read.([char(nm) '_Rate']) = raw.Rate(rows, :) .* ~masked;
                    end
                end
                ctx = struct('Now', t, 'Phase', obj.Phases(r.PhaseIdx), 'Door', r.Door, 'Rows', rows, ...
                    'IDs', obj.Registry.idsOf(rows), 'Read', read, 'Events', OceanAgents.World.Event.empty(1, 0), ...
                    'Stream', []);
                out = spec.Fn(ctx);
                if isempty(out), out = struct(); end
                if ~isstruct(out) || ~isscalar(out) || ~all(ismember(fieldnames(out), {'View'}))
                    fail_('Undeclared', 'Projection %s may only return a View.', r.Key);
                end
                view = struct();
                if isfield(out, 'View'), view = out.View; end
                views.(char(r.Door + "_" + spec.Name)) = view;
            end
        end

        % ============================================================== pillars
        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = struct('Symbol', {}, 'Description', {});
            meta.Assumptions = {sprintf('%d doors, phases [%s], SkipIdle %d, horizon %g', numel(obj.Doors), ...
                strjoin(obj.Phases, ' '), obj.SkipIdle, obj.Horizon), ...
                'instant = due -> deliver -> run -> collect -> all-or-nothing sync, per phase; skipping is exact'};
            meta.IsSymbolic = false;
            meta.Reference = "W1_Kernel_Contract.md sections 5.5 and 7 (world loop); DEVS-style tie-breaking";
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.ComponentName = "World";
            spec.Inputs = struct('Name', "events", 'Domain', "World-Event", 'Dtype', "OceanAgents.World.Event", 'Shape', "1 x k");
            spec.Outputs = struct('Name', "report", 'Domain', "World-InstantReport", 'Dtype', "struct", 'Shape', "1 x 1");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "world");
            spec.ComposesFrom = {"OceanAgents.World.Timeline", "OceanAgents.World.Store", "OceanAgents.World.Registry", ...
                "OceanAgents.World.Calendar", "OceanAgents.World.Guard.GuardEngine"};
            spec.Doors = [obj.Doors.Name];
        end
    end

    % ================================================================== the instant
    methods (Access = private)
        function rep = runPhase_(obj, t, p)
            phaseName = obj.Phases(p);
            col = obj.emptyCollect_();
            % 1. due discontinuities
            due = obj.Calendar.due(t, p);
            dueEvents = OceanAgents.World.Event.empty(1, 0);
            dueRanks = zeros(1, 0);
            for it = reshape(due, 1, [])
                if it.Kind == "monitor"
                    continue           % a skip-safe monitor check: it only makes this instant happen (checked at sync)
                elseif it.Kind == "prediction"
                    obj.EvSeq_ = obj.EvSeq_ + 1;
                    e = OceanAgents.World.Event('Type', it.DueEvent, 'Time', t, 'Phase', phaseName, ...
                        'SourceDoor', it.Owner, 'SourceEntity', it.EntityID, 'Seq', obj.EvSeq_);
                    dueEvents(end + 1) = e; %#ok<AGROW>
                    obj.record_(e, p, it.OwnerRank, it.Rank);
                    dueRanks(end + 1) = it.Rank; %#ok<AGROW>
                else
                    col.Commands(end + 1) = struct('Cmd', it.Command, 'DoorRank', it.OwnerRank, 'RuleRank', it.Rank);
                end
            end
            for k = 1:numel(dueEvents)                          % consumers of later phases get it later this instant
                obj.queueDelivery_(dueEvents(k), t, p, dueRanks(k), t);
            end
            % 2. deliver
            delivered = obj.takeDeliveries_(t, p);
            % 3. run
            driven = obj.Participants.due(t);
            rulesRun = strings(1, 0);
            for i = 1:numel(obj.Rules_)
                r = obj.Rules_(i);
                if r.Kind ~= "rule" || r.PhaseIdx ~= p, continue; end
                events = delivered{i};
                spec = r.Spec;
                rows = obj.Store.rowsWith(spec.Requires);
                if ~isempty(obj.Inactive_), rows = obj.activeRows_(rows); end
                runs = ~isempty(events) || (driven(r.DoorRank) && (isempty(spec.Requires) || ~isempty(rows)));
                if ~runs, continue; end
                ctx = obj.context_(r, rows, t, events);
                out = spec.Fn(ctx);
                col = obj.collect_(col, r, out, rows, t, p);
                rulesRun(end + 1) = r.Key; %#ok<AGROW>
            end
            % 4-5. synchronise
            rep = obj.synchronise_(t, p, col);
            rep.Phase = phaseName;
            rep.DueItems = numel(due);
            rep.RulesRun = rulesRun;
        end

        function ctx = context_(obj, r, rows, t, events)
            read = struct();
            for nm = r.Spec.Reads
                if isfield(obj.Types_, nm)
                    read.(char(nm)) = obj.Store.valueAt(nm, rows, t);
                    read.([char(nm) '_Has']) = obj.Store.has(nm, rows);
                    if obj.Types_.(char(nm)).Kind == "linear"
                        raw = obj.Store.raw(nm);
                        read.([char(nm) '_Rate']) = raw.Rate(rows, :);
                    end
                else
                    read.(char(nm)) = obj.Resources.(char(nm));
                end
            end
            ctx = struct('Now', t, 'Phase', obj.Phases(r.PhaseIdx), 'Door', r.Door, 'Rows', rows, ...
                'IDs', obj.Registry.idsOf(rows), 'Read', read, 'Events', events, 'Stream', obj.Streams_{r.DoorRank});
        end

        function col = collect_(obj, col, r, out, rows, t, p)
            % Check a rule's output against its RuleSpec and add it to this phase's collection (nothing applied).
            spec = r.Spec;
            if isempty(out), return; end
            if ~isstruct(out) || ~isscalar(out)
                fail_('BadValue', 'Rule %s must return a scalar struct.', r.Key);
            end
            f = fieldnames(out);
            extra = setdiff(f, {'Writes', 'Events', 'Commands'});
            if ~isempty(extra)
                fail_('Undeclared', 'Rule %s returned field(s) %s (a rule returns Writes, Events, Commands).', ...
                    r.Key, strjoin(extra, ', '));
            end
            if isfield(out, 'Writes')
                for w = reshape(out.Writes, 1, [])
                    if isfield(w, 'Resource') && ~isempty(w.Resource)
                        nm = string(w.Resource);
                        if ~any(spec.Writes == nm) || isfield(obj.Types_, nm)
                            fail_('Undeclared', 'Rule %s writes resource %s, which it does not declare.', r.Key, nm);
                        end
                        OceanAgents.Core.assertValueData(struct('v', {w.Values}), "resource " + nm);
                        col.ResWrites(end + 1) = struct('Resource', nm, 'Value', {w.Values}, 'Rule', r.Key);
                        continue
                    end
                    if ~isfield(w, 'Component'), fail_('BadValue', 'Rule %s: a write needs Component or Resource.', r.Key); end
                    nm = string(w.Component);
                    if ~any(spec.Writes == nm) || ~isfield(obj.Types_, nm)
                        fail_('Undeclared', 'Rule %s writes component %s, which it does not declare.', r.Key, nm);
                    end
                    wr = double(reshape(w.Rows, [], 1));
                    if ~all(ismember(wr, rows))
                        fail_('Undeclared', 'Rule %s writes %s on rows outside its ctx.Rows.', r.Key, nm);
                    end
                    rates = [];
                    if isfield(w, 'Rates'), rates = w.Rates; end
                    col.Writes(end + 1) = struct('Component', nm, 'Rows', wr, 'Values', {w.Values}, 'Rates', {rates}, ...
                        'Rule', r.Key);
                end
            end
            if isfield(out, 'Events') && ~isempty(out.Events)
                if ~isa(out.Events, 'OceanAgents.World.Event')
                    fail_('BadValue', 'Rule %s: Events must be OceanAgents.World.Event objects.', r.Key);
                end
                for e = reshape(out.Events, 1, [])
                    if ~any(spec.Emits == e.Type)
                        fail_('Undeclared', 'Rule %s emits %s, which it does not declare.', r.Key, e.Type);
                    end
                    col.Events(end + 1) = struct('Event', e, 'Door', r.Door, 'DoorRank', r.DoorRank, 'RuleRank', r.RuleRank);
                end
            end
            if isfield(out, 'Commands') && ~isempty(out.Commands)
                if ~isa(out.Commands, 'OceanAgents.World.Command')
                    fail_('BadValue', 'Rule %s: Commands must be OceanAgents.World.Command objects.', r.Key);
                end
                for c = reshape(out.Commands, 1, [])
                    if ~any(spec.Commands == c.Kind)
                        fail_('Undeclared', 'Rule %s issues a %s command, which it does not declare.', r.Key, c.Kind);
                    end
                    T = t;
                    if ~isempty(c.Time), T = c.Time; end
                    q = p;
                    if strlength(c.Phase) > 0, q = obj.Timeline.phaseIndex(c.Phase); end
                    if T < t || (T == t && q < p)
                        fail_('Retroactive', 'Rule %s schedules a %s at (%g, %s), before now (%g, %s).', r.Key, ...
                            c.Kind, T, obj.Phases(q), t, obj.Phases(p));
                    end
                    obj.CmdSeq_ = obj.CmdSeq_ + 1;
                    stamped = c.stamp(r.Key, 0, obj.CmdSeq_, T, obj.Phases(q));
                    entry = struct('Cmd', stamped, 'DoorRank', r.DoorRank, 'RuleRank', r.RuleRank);
                    if T == t && q == p
                        col.Commands(end + 1) = entry;                 % this phase's synchronisation point
                    else
                        col.Future(end + 1) = struct('Cmd', stamped, 'DoorRank', r.DoorRank, 'RuleRank', r.RuleRank, ...
                            'Time', T, 'PhaseIdx', q);
                    end
                end
            end
        end

        function rep = synchronise_(obj, t, p, col)
            % All-or-nothing application of one phase's collected outputs (section 7.1 step 5).
            rep = emptyPhaseReport_();
            G = OceanAgents.World.Guard.GuardEngine;
            % a) declarations: one writer per (component, row) and per resource; data validity
            obj.checkConflicts_(col);
            writes = rmfield(col.Writes, 'Rule');
            obj.Store.checkWrites(writes, t);
            % b) apply the writes and resource writes (validated above); the phase snapshot undoes them on failure
            obj.Store.applyWrites(writes, t);
            for w = reshape(col.ResWrites, 1, [])
                obj.Resources.(char(w.Resource)) = w.Value;
            end
            % c) commands in tie-break order, each evaluated on the state after the writes AND every earlier admitted
            %    command of this synchronisation point (so a split attach cannot bypass admission), then applied
            cmds = col.Commands;
            if ~isempty(cmds)
                key = [[cmds.DoorRank]' [cmds.RuleRank]' arrayfun(@(c) c.Cmd.IssuerEntity, cmds)' ...
                    arrayfun(@(c) c.Cmd.Seq, cmds)'];
                [~, order] = sortrows(key);
                cmds = cmds(order);
            end
            touched = zeros(0, 1);                                   % rows whose membership may have changed
            for c = reshape(cmds, 1, [])
                cmd = c.Cmd;
                alive = cmd.Target > 0 && obj.Registry.isAlive(cmd.Target);
                if ~isempty(cmd.Guard) && ~logical(cmd.Guard(struct('TargetAlive', alive, 'Now', t)))
                    rep.GuardNoOps = rep.GuardNoOps + 1;
                    continue
                end
                if cmd.Kind ~= "spawn" && ~alive
                    rep.CommandsRefused(end + 1) = refusal_(cmd, "not-alive", strings(1, 0), ...
                        sprintf('target entity %d is not alive', cmd.Target));
                    continue
                end
                if cmd.Kind ~= "spawn" && ~isempty(obj.Inactive_) && any(obj.Inactive_ == cmd.Target)
                    rep.CommandsRefused(end + 1) = refusal_(cmd, "inactive", strings(1, 0), ...
                        sprintf('target entity %d is outside the active set', cmd.Target));
                    continue
                end
                switch cmd.Kind
                    case "spawn"
                        [ok, reasons] = G.admitEntity(cmd.Offer, obj.Doors, obj.Store, t, 'Index', obj.GuardIndex_);
                    case "attach"
                        row = obj.Registry.rowsOf(cmd.Target);
                        offered = string(fieldnames(cmd.Offer))';
                        has = offered(arrayfun(@(nm) isfield(obj.Types_, nm) && obj.Store.has(nm, row), offered));
                        foreign = obj.foreignNames_(offered, c.DoorRank);
                        if ~isempty(foreign)                         % only the owning door attaches its components
                            ok = false;
                            reasons = struct('Reason', "not-owner", 'AxiomID', "", 'Door', "", ...
                                'Detail', sprintf('door %s does not own %s', obj.Doors(c.DoorRank).Name, strjoin(foreign, ', ')));
                        elseif ~isempty(has)                         % commands never change values
                            ok = false;
                            reasons = struct('Reason', "already-has", 'AxiomID', "", 'Door', "", ...
                                'Detail', sprintf('entity %d already has %s', cmd.Target, strjoin(has, ', ')));
                        else
                            [ok, reasons] = G.admitEntity(cmd.Offer, obj.Doors, obj.Store, t, 'Row', row, ...
                                'Index', obj.GuardIndex_);
                        end
                    case "detach"
                        bad = cmd.Detach(~arrayfun(@(nm) isfield(obj.Types_, nm), cmd.Detach));
                        foreign = obj.foreignNames_(cmd.Detach, c.DoorRank);
                        ok = isempty(bad) && isempty(foreign);
                        if ~isempty(bad)
                            reasons = struct('Reason', "no-owner", 'AxiomID', "", 'Door', "", ...
                                'Detail', "no door owns " + strjoin(bad, ", "));
                        else
                            reasons = struct('Reason', "not-owner", 'AxiomID', "", 'Door', "", ...
                                'Detail', sprintf('door %s does not own %s', obj.Doors(c.DoorRank).Name, strjoin(foreign, ', ')));
                        end
                    case "remove"
                        ok = true;
                end
                if ok
                    touched = [touched; obj.applyCommand_(cmd, t)]; %#ok<AGROW>
                    rep.CommandsApplied = rep.CommandsApplied + 1;
                else
                    ids = [reasons.AxiomID];
                    rep.CommandsRefused(end + 1) = refusal_(cmd, strjoin(unique([reasons.Reason], 'stable'), ","), ...
                        ids(strlength(ids) > 0), strjoin(string({reasons.Detail}), "; "));
                end
            end
            % d) future commands onto the calendar
            for f = reshape(col.Future, 1, [])
                obj.Calendar.scheduleCommand(f.Cmd, f.Time, f.PhaseIdx, f.DoorRank, 'Rank', f.RuleRank);
            end
            % e) entity monitors on the POST-command state (removed entities are no longer members; new ones were
            %    admitted). A violation throws and the fail-stop restores the phase's starting state: nothing applied.
            viol = G.checkMonitors(obj.Doors, obj.Store, struct(), t, 'Index', obj.GuardIndex_);
            if ~isempty(viol)
                parts = strings(1, numel(viol));
                for k = 1:numel(viol)
                    ids = obj.Registry.idsOf(viol(k).EntityRows);
                    parts(k) = sprintf('%s (door %s) violated by entity %s: %s', viol(k).AxiomID, viol(k).Door, ...
                        mat2str(reshape(ids, 1, [])), viol(k).Detail);
                end
                error('OceanAgents:Guard:AxiomViolated', 'At t = %g, phase %d: %s. Nothing applied.', t, p, ...
                    strjoin(parts, '; '));
            end
            % f) predictors and skip-safe range-monitor checks on rows whose time function changed (incl. new rows)
            taken = struct();
            for comp = obj.TimeWatched_
                taken.(char(comp)) = obj.Store.takeDirty(comp);
                if ~isempty(obj.Inactive_), taken.(char(comp)) = obj.activeRows_(taken.(char(comp))); end
            end
            touched = unique(touched(obj.Store.isAlive(touched)));
            if ~isempty(obj.Inactive_), touched = obj.activeRows_(touched); end
            rep.PredictionsUpdated = obj.runPredictors_(t, p, taken, touched);
            obj.scheduleMonitorChecks_(t, p, taken, touched);
            % g) stamp emitted events for delivery
            for k = 1:numel(col.Events)
                obj.EvSeq_ = obj.EvSeq_ + 1;
                name = "";
                if p > 0, name = obj.Phases(p); end
                e = col.Events(k).Event.stamp(t, name, col.Events(k).Door, obj.EvSeq_);
                obj.record_(e, p, col.Events(k).DoorRank, col.Events(k).RuleRank);
                obj.queueDelivery_(e, t, p, col.Events(k).RuleRank, []);
            end
            rep.EventsEmitted = numel(col.Events);
        end

        function t = boundary_(obj)
            % The commit boundary at which registration changes apply: the last processed instant, or the instant
            % run(tEnd) covered when idle instants were skipped (so skipping and stepping agree).
            t = max(obj.Timeline.Now, obj.CoveredThrough_);
        end

        function rows = activeRows_(obj, rows)
            % Rows of entities outside the inactive set (frozen rows are exactly the inactive ones).
            if isempty(rows), return; end
            rows = rows(~obj.Store.isFrozen(rows));
        end

        function names = foreignNames_(obj, names, doorRank)
            % Components among names that exist but are owned by a door other than doorRank (attach/detach may
            % only name the issuing door's own components; section 12, critic round 2 N1).
            names = reshape(string(names), 1, []);
            owned = arrayfun(@(nm) isfield(obj.Types_, nm) && obj.Owner_(char(nm)) == doorRank, names);
            known = arrayfun(@(nm) isfield(obj.Types_, nm), names);
            names = names(known & ~owned);
        end

        function checkConflicts_(~, col)
            % Two different rules writing one (component, row), or one resource: OceanAgents:World:WriteConflict.
            if ~isempty(col.Writes)
                names = unique([col.Writes.Component]);
                for nm = names
                    ws = col.Writes([col.Writes.Component] == nm);
                    rows = vertcat(ws.Rows);
                    who = repelem([ws.Rule], arrayfun(@(w) numel(w.Rows), ws));
                    [u, ~, j] = unique(rows);
                    for k = find(accumarray(j, 1) > 1)'
                        writers = unique(who(rows == u(k)));
                        if numel(writers) > 1
                            error('OceanAgents:World:WriteConflict', ...
                                'Component %s, row %d is written by %s in one phase.', nm, u(k), strjoin(writers, ' and '));
                        end
                    end
                end
            end
            if ~isempty(col.ResWrites)
                [u, ~, j] = unique([col.ResWrites.Resource]);
                dup = find(accumarray(j(:), 1) > 1, 1);
                if ~isempty(dup)
                    error('OceanAgents:World:WriteConflict', 'Resource %s is written twice in one phase (%s).', ...
                        u(dup), strjoin([col.ResWrites(j == dup).Rule], ' and '));
                end
            end
        end

        function row = applyCommand_(obj, cmd, t)
            % Apply one admitted command; returns the store row it touched.
            switch cmd.Kind
                case "spawn"
                    id = obj.Registry.allocate(1);
                    row = obj.Store.addRows(1);
                    obj.Registry.bind(id, row);
                    obj.Store.setOffer(row, cmd.Offer, t);
                case "attach"
                    row = obj.Registry.rowsOf(cmd.Target);
                    obj.Store.setOffer(row, cmd.Offer, t);
                case "detach"
                    row = obj.Registry.rowsOf(cmd.Target);
                    for nm = cmd.Detach
                        obj.Store.clearComponent(row, nm);
                    end
                case "remove"
                    row = obj.Registry.rowsOf(cmd.Target);
                    obj.Registry.retire(cmd.Target);
                    obj.Store.killRows(row);
                    obj.Calendar.cancelEntity(cmd.Target);
            end
        end

        function n = runPredictors_(obj, t, p, taken, touched)
            % Every predictor, on the rows whose predicted component changed since the last synchronisation
            % (taken: component -> dirty rows, taken once per synchronisation point and shared with the monitors).
            n = 0;
            for i = 1:numel(obj.Rules_)
                r = obj.Rules_(i);
                if r.Kind ~= "predictor", continue; end
                spec = r.Spec;
                comp = char(spec.Predicts);
                % rows whose predicted function changed, plus rows whose components changed (spawn/attach/detach):
                % they may newly satisfy Requires (predict) or stop satisfying it (cancel)
                dirty = unique([taken.(comp); touched]);
                if isempty(dirty), continue; end
                eligible = obj.Store.rowsWith(spec.Requires);
                if ~isempty(obj.Inactive_), eligible = obj.activeRows_(eligible); end
                rows = dirty(ismember(dirty, eligible));
                gone = dirty(~ismember(dirty, eligible));        % not (or no longer) eligible: drop their predictions
                times = inf(numel(rows), 1);
                if ~isempty(rows)
                    ctx = obj.context_(r, rows, t, OceanAgents.World.Event.empty(1, 0));
                    out = spec.Fn(ctx);
                    if isempty(out), out = struct(); end
                    if ~isstruct(out) || ~isscalar(out) || ~all(ismember(fieldnames(out), {'Predictions'}))
                        fail_('Undeclared', 'Predictor %s may only return Predictions.', r.Key);
                    end
                    if isfield(out, 'Predictions') && ~isempty(out.Predictions)
                        pr = out.Predictions;
                        prow = double(reshape(pr.Rows, [], 1));
                        ptime = double(reshape(pr.Times, [], 1));
                        [inCtx, at] = ismember(prow, rows);
                        if ~all(inCtx) || numel(prow) ~= numel(ptime)
                            fail_('Undeclared', 'Predictor %s predicts rows outside its ctx.Rows.', r.Key);
                        end
                        times(at) = ptime;
                    end
                end
                allRows = [rows; gone];
                allTimes = [times; inf(numel(gone), 1)];
                q = obj.DuePhase_(char(spec.DueEvent));
                fin = isfinite(allTimes);
                late = fin & allTimes <= t;                      % crossing at or before now
                allTimes(late) = t + double(q <= p);             % next reachable (time, q)
                obj.Calendar.schedulePrediction(r.Key, obj.Registry.idsOf(allRows), allTimes, q, r.DoorRank, ...
                    spec.DueEvent, 'Rank', r.RuleRank);
                n = n + numel(rows);
            end
        end

        % ============================================================== events
        function queueDelivery_(obj, e, t, p, ruleRank, onlyLater)
            % Queue e for every consuming rule: phase q at t if q comes after p, else at t + 1. For a due prediction
            % (onlyLater = t) consumers of phase p receive it now (returned by takeDeliveries_) - queued at (t, p).
            if ~isKey(obj.Consumers_, char(e.Type)), return; end
            if isempty(e.Phase) || strlength(e.Phase) == 0, pe = 0; else, pe = find(obj.Phases == e.Phase, 1); end
            srcRank = 0;
            if strlength(e.SourceDoor) > 0, srcRank = find([obj.Doors.Name] == e.SourceDoor, 1); end
            % delivery tie-break key: (time, emission phase, source door rank, source rule rank, source entity, seq)
            key = [e.Time pe srcRank ruleRank e.SourceEntity e.Seq];
            for i = obj.Consumers_(char(e.Type))
                q = obj.Rules_(i).PhaseIdx;
                if ~isempty(onlyLater)
                    when = t;                                    % due prediction: this instant, phase q >= p
                    if q < p, when = t + 1; end
                elseif p == 0
                    when = t;                                    % injected (or reset): at the delivery instant
                elseif q > p
                    when = t;
                else
                    when = t + 1;
                end
                obj.Pending_.Events(end + 1) = e;
                obj.Pending_.Rule(end + 1, 1) = i;
                obj.Pending_.Time(end + 1, 1) = when;
                obj.Pending_.Phase(end + 1, 1) = q;
                obj.Pending_.Key(end + 1, :) = key;
            end
        end

        function delivered = takeDeliveries_(obj, t, p)
            % Remove and return the deliveries due at (t, p), per rule, in tie-break order.
            delivered = repmat({OceanAgents.World.Event.empty(1, 0)}, 1, numel(obj.Rules_));
            P = obj.Pending_;
            if isempty(P.Time), return; end
            if any(P.Time < t | (P.Time == t & P.Phase < p))
                fail_('InvalidState', 'An event delivery was due before (%g, %d) and never delivered.', t, p);
            end
            now = find(P.Time == t & P.Phase == p);
            if isempty(now), return; end
            for i = unique(P.Rule(now))'
                sel = now(P.Rule(now) == i);
                [~, order] = sortrows(P.Key(sel, :));
                delivered{i} = P.Events(sel(order));
            end
            keep = true(numel(P.Time), 1);
            keep(now) = false;
            obj.Pending_.Events = P.Events(keep);
            obj.Pending_.Rule = P.Rule(keep);
            obj.Pending_.Time = P.Time(keep);
            obj.Pending_.Phase = P.Phase(keep);
            obj.Pending_.Key = P.Key(keep, :);
        end

        % ============================================================== bookkeeping
        function t = computeNext_(obj, t)
            if obj.Status ~= "Running", t = Inf; return; end
            if t >= obj.Horizon
                t = Inf;
                return
            end
            if ~obj.SkipIdle
                t = t + 1;
            else
                cands = [obj.Participants.nextDue(t), obj.Calendar.nextTime()];
                if ~isempty(obj.Pending_.Time), cands(end + 1) = min(obj.Pending_.Time); end
                t = max(min(cands), t + 1);
            end
            t = min(t, obj.Horizon);
        end

        function scheduleMonitorChecks_(obj, t, p, taken, touched)
            % Skip-safe range monitors (section 12): for each member row whose time function changed, schedule a
            % monitor check at the first integer instant its value leaves [lo, hi] (Inf = never). At that instant the
            % normal monitor evaluation runs, exactly as it does at every instant of a stepping world.
            for m = reshape(obj.MonRange_, 1, [])
                dirty = unique([taken.(char(m.Component)); touched]);
                if isempty(dirty), continue; end
                members = obj.Store.rowsWith(m.Membership);
                rows = dirty(ismember(dirty, members));
                gone = dirty(~ismember(dirty, members));          % no longer members: drop their checks
                times = inf(numel(rows), 1);
                if ~isempty(rows)
                    times = exitTime_(obj.Store.raw(m.Component), m.Kind, rows, m.Lo, m.Hi, t);
                end
                allRows = [rows; gone];
                allTimes = [times; inf(numel(gone), 1)];
                late = isfinite(allTimes) & (allTimes < t | (allTimes == t & p >= 1));
                allTimes(late) = t + 1;                      % cannot happen for a member that just passed its monitor
                obj.Calendar.scheduleMonitorCheck(m.AxiomID, obj.Registry.idsOf(allRows), allTimes, 1, m.DoorRank);
            end
        end

        function buildMonitorIndex_(obj)
            % Range monitors on linear/clock components get skip-safe checks; a predicate monitor reading one needs a
            % predictor of its door on it when skipping (section 12).
            mon = struct('AxiomID', {}, 'DoorRank', {}, 'Component', {}, 'Kind', {}, 'Lo', {}, 'Hi', {}, 'Membership', {});
            watched = strings(1, 0);
            for d = 1:numel(obj.Doors)
                door = obj.Doors(d);
                predicted = strings(1, 0);
                for r = door.Rules
                    if r.Kind == "predictor", predicted(end + 1) = r.Predicts; end %#ok<AGROW>
                end
                watched = [watched predicted]; %#ok<AGROW>
                for a = door.Axioms
                    if a.Scope ~= "entity" || a.Mode ~= "monitor", continue; end
                    reads = a.readsAll();
                    kinds = arrayfun(@(nm) obj.Types_.(char(nm)).Kind, reads);
                    timeVarying = reads(kinds ~= "value");
                    if isempty(timeVarying), continue; end
                    if a.isRange()
                        c = a.Component;
                        mon(end + 1) = struct('AxiomID', a.ID, 'DoorRank', d, 'Component', c, ...
                            'Kind', obj.Types_.(char(c)).Kind, 'Lo', a.Range(1), 'Hi', a.Range(2), ...
                            'Membership', door.Membership); %#ok<AGROW>
                        watched(end + 1) = c; %#ok<AGROW>
                    elseif obj.SkipIdle
                        fail_('BadAssembly', ['Monitor %s (door %s) is a predicate on time-varying component(s) %s: ' ...
                            'with SkipIdle it could be violated at an instant the world skips. Use a range axiom ' ...
                            '(skip-safe) or set SkipIdle = false.'], a.ID, door.Name, strjoin(timeVarying, ', '));
                    end
                end
            end
            obj.MonRange_ = mon;
            obj.TimeWatched_ = unique(watched, 'stable');
        end

        function record_(obj, e, p, doorRank, ruleRank)
            % Host outbox: record an event of an 'Outbox' type (never visible to rules or projections).
            if isempty(obj.Outbox) || ~any(obj.Outbox == e.Type), return; end
            obj.OutboxEvents_(end + 1) = e;
            obj.OutboxKey_(end + 1, :) = [e.Time p doorRank ruleRank e.SourceEntity e.Seq];
        end

        function s = capture_(obj)
            s = struct('Store', obj.Store.snapshot(), 'Registry', obj.Registry.snapshot(), ...
                'Calendar', obj.Calendar.snapshot(), 'Resources', obj.Resources, 'Pending', obj.Pending_, ...
                'EvSeq', obj.EvSeq_, 'CmdSeq', obj.CmdSeq_, 'Outbox', {obj.OutboxEvents_}, 'OutboxKey', obj.OutboxKey_);
        end

        function failStop_(obj, err, snap)
            % Fail-stop: back to the last completed synchronisation point, then rethrow.
            obj.Store.restore(snap.Store);
            obj.Registry.restore(snap.Registry);
            obj.Calendar.restore(snap.Calendar);
            obj.Resources = snap.Resources;
            obj.Pending_ = snap.Pending;
            obj.EvSeq_ = snap.EvSeq;
            obj.CmdSeq_ = snap.CmdSeq;
            obj.OutboxEvents_ = snap.Outbox;
            obj.OutboxKey_ = snap.OutboxKey;
            obj.Status = "Failed";
            obj.Failure = err;
            obj.NextTime = Inf;
            rethrow(err);
        end

        function requireRunning_(obj, op)
            if obj.Status ~= "Running"
                fail_('InvalidState', 'World is %s: %s is not allowed (reset first).', obj.Status, op);
            end
        end

        function col = emptyCollect_(obj)
            col = struct('Writes', struct('Component', {}, 'Rows', {}, 'Values', {}, 'Rates', {}, 'Rule', {}), ...
                'ResWrites', struct('Resource', {}, 'Value', {}, 'Rule', {}), ...
                'Events', struct('Event', {}, 'Door', {}, 'DoorRank', {}, 'RuleRank', {}), ...
                'Commands', obj.emptyCommands_(), ...
                'Future', struct('Cmd', {}, 'DoorRank', {}, 'RuleRank', {}, 'Time', {}, 'PhaseIdx', {}));
        end

        function c = emptyCommands_(~)
            c = struct('Cmd', {}, 'DoorRank', {}, 'RuleRank', {});
        end

        function buildIndex_(obj)
            flat = obj.Doors;
            obj.Owner_ = containers.Map('KeyType', 'char', 'ValueType', 'double');
            rules = struct('Key', {}, 'Door', {}, 'DoorRank', {}, 'RuleRank', {}, 'PhaseIdx', {}, 'Kind', {}, 'Spec', {});
            res = struct();
            for d = 1:numel(flat)
                for c = flat(d).Owns
                    obj.Types_.(char(c.Name)) = c.Type;
                    obj.Owner_(char(c.Name)) = d;
                end
                for f = string(fieldnames(flat(d).Resources))'
                    res.(char(f)) = flat(d).Resources.(char(f));
                    obj.Owner_(char(f)) = d;
                end
                for k = 1:numel(flat(d).Rules)
                    s = flat(d).Rules(k);
                    rules(end + 1) = struct('Key', flat(d).Name + "/" + s.Name, 'Door', flat(d).Name, 'DoorRank', d, ...
                        'RuleRank', k, 'PhaseIdx', find(obj.Phases == s.Phase, 1), 'Kind', s.Kind, 'Spec', s); %#ok<AGROW>
                end
            end
            obj.Rules_ = rules;
            obj.InitialResources_ = res;
            obj.Resources = res;
            obj.Consumers_ = containers.Map('KeyType', 'char', 'ValueType', 'any');
            obj.DuePhase_ = containers.Map('KeyType', 'char', 'ValueType', 'double');
            for i = 1:numel(rules)
                if rules(i).Kind ~= "rule", continue; end
                for ty = rules(i).Spec.Consumes
                    k = char(ty);
                    if isKey(obj.Consumers_, k), obj.Consumers_(k) = [obj.Consumers_(k) i];
                    else, obj.Consumers_(k) = i; end
                    if isKey(obj.DuePhase_, k), obj.DuePhase_(k) = min(obj.DuePhase_(k), rules(i).PhaseIdx);
                    else, obj.DuePhase_(k) = rules(i).PhaseIdx; end
                end
            end
        end
    end
end

% ====================================================================== helpers
function P = emptyPending_()
P = struct('Events', OceanAgents.World.Event.empty(1, 0), 'Rule', zeros(0, 1), 'Time', zeros(0, 1), ...
    'Phase', zeros(0, 1), 'Key', zeros(0, 6));
end

function r = emptyPhaseReport_()
r = struct('Phase', "", 'DueItems', 0, 'RulesRun', strings(1, 0), 'EventsEmitted', 0, 'CommandsApplied', 0, ...
    'CommandsRefused', struct('Kind', {}, 'Reason', {}, 'AxiomIDs', {}, 'Issuer', {}, 'Target', {}, 'Detail', {}), ...
    'GuardNoOps', 0, 'PredictionsUpdated', 0);
end

function r = refusal_(cmd, reason, axiomIDs, detail)
r = struct('Kind', cmd.Kind, 'Reason', string(reason), 'AxiomIDs', reshape(string(axiomIDs), 1, []), ...
    'Issuer', cmd.Issuer, 'Target', cmd.Target, 'Detail', string(detail));
end

function fail_(reason, fmt, varargin)
error(sprintf('OceanAgents:World:%s', reason), fmt, varargin{:});
end

function T = exitTime_(raw, kind, rows, lo, hi, t)
% First integer instant > t at which a linear/clock value leaves [lo, hi] (Inf if never), per row over all columns.
% Closed form (linear: value(tau) = Base + Rate (tau - T0) crosses a bound at tau = T0 + (bound - Base) / Rate;
% clock: value(tau) = tau - T0), then corrected by evaluating exactly as Store.valueAt does, so the scheduled instant
% is the first one at which the monitor, evaluated normally, fails.
T0 = raw.T0(rows);
if kind == "clock"
    B = zeros(numel(rows), 1);  R = ones(numel(rows), 1);
    value = @(tau) tau - T0;
else
    B = raw.Base(rows, :);  R = raw.Rate(rows, :);
    value = @(tau) B + R .* (tau - T0);
end
cand = inf(size(R));
if kind == "clock", Bc = -T0; else, Bc = B - R .* T0; end   % value(tau) = Bc + R tau
if isfinite(hi)
    up = R > 0;
    x = (hi - Bc) ./ R;
    cand(up) = floor(x(up)) + 1;
end
if isfinite(lo)
    down = R < 0;
    x = (lo - Bc) ./ R;
    cand(down) = min(cand(down), floor(x(down)) + 1);
end
T = min(cand, [], 2);
fin = isfinite(T);
isOut = @(tau) any(value(tau) > hi | value(tau) < lo, 2);
for pass = 1:3                                               % rounding at the boundary, in both directions
    tau = T;  tau(~fin) = 0;
    inside = fin & ~isOut(tau);
    T(inside) = T(inside) + 1;
    tau = T - 1;  tau(~fin) = 0;
    early = fin & (T - 1) > t & isOut(tau);
    T(early) = T(early) - 1;
end
T(fin) = max(T(fin), t + 1);
end

function v = defaultsLike_(v)
% Type defaults for masked (not visible) rows: 0 / false / "".
if isstring(v), v(:) = ""; elseif islogical(v), v(:) = false; else, v(:) = 0; end
end
