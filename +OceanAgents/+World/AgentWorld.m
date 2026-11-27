classdef AgentWorld < OceanAgents.Core.IJointEnvironment
    % AGENTWORLD  The agent face of a World: an OceanAgents.Core.IJointEnvironment, so the existing JointRunner can
    % drive a door-built world. Agents are the alive members of the Agency door (AgentID); their actions become
    % routed events; their rewards and endings come back through the World's host-only outbox.
    %
    %   cfg = struct('ActionLabels', [1 2], ...
    %                'Routing', struct('Label', {1, 2}, 'Type', {"Move", "Move"}, 'Payload', {struct('Label', 1), ...
    %                                  struct('Label', 2)}), ...
    %                'Observe', @(view) view.grid_Cell.Index, ...
    %                'ObservationSpace', OceanStats.RL.Env.Spaces.Discrete(48), ...
    %                'ActionSpace', OceanStats.RL.Env.Spaces.Discrete(4), 'Roster', "agent1");
    %   aw = OceanAgents.World.AgentWorld(doors, cfg, 'SkipIdle', false, 'Name', "cliff");
    %   frame = aw.reset(7);  frame = aw.step(batch);
    %
    % reset(seed): World.reset(seed) (or World.reset([]) for seed = [], continuing streams); the roster is the
    %   AgentID values of the alive Agency members in entity-ID order and must equal the declared roster
    %   (OceanAgents:Runtime:RosterMismatch otherwise). The first frame has every label legal, RewardValid false.
    % step(batch): validate the batch against the current frame; inject one routed event per decision (SourceEntity =
    %   that agent's entity); EXACTLY ONE World.advance(); takeOutbox(): reward = sum of Reward.Amount per agent,
    %   endings from Terminated/Truncated (termination wins). An ended agent's view has an all-false mask (plus
    %   Info.successor_action_mask all true on truncation); when every agent has ended, EpisodeEnded is true and
    %   EndReason is the first ending's reason. step after EpisodeEnded -> OceanAgents:Env:NotReady.
    % Outbox types: ["Reward" "Terminated" "Truncated"]; the World's Injectable types are the routed types;
    % EnvSpec.Horizon = Inf; the World itself is never exposed.
    % The declared roster is cfg.Roster (or cfg.AgentID); default "agent1".
    % See Planning/System/Game Theory/W2_Grid_Doors_Contract.md section 4.

    properties (Constant)
        OutboxTypes = ["Reward" "Terminated" "Truncated"]
    end

    properties (SetAccess = private)
        Name string = "agentworld"
        Roster string = strings(1, 0)
        Roles string = strings(1, 0)
        ActionLabels double = zeros(1, 0)
        EpisodeID double = 0
        FrameID double = 0
        NumAdvances double = 0          % World.advance calls made by step (exactly one per step)
        ListenerFailures cell = {}      % errors thrown by commit listeners (recorded, never rethrown)
    end

    properties (Access = private)
        World_                          % OceanAgents.World.World (never exposed)
        Cfg_ struct
        Entities_ double = zeros(1, 0)  % entity ID of each roster slot
        Frame_ = []                     % current DecisionFrame
        Ended_ logical = false(1, 0)
        Ready_ logical = false
        FirstReason_ string = ""        % reason of the episode's first ending (EndReason)
        Listeners_ cell = {}            % commit listeners: fn(aw), called after reset and after each completed step
    end

    methods
        function obj = AgentWorld(doors, agencyCfg, options)
            arguments
                doors
                agencyCfg (1,1) struct
                options.Phases string = ["resolve" "act" "settle" "project"]
                options.SkipIdle (1,1) logical = false
                options.Name (1,1) string = "agentworld"
            end
            need = {'ActionLabels', 'Routing', 'Observe', 'ObservationSpace', 'ActionSpace'};
            missing = need(~isfield(agencyCfg, need));
            if ~isempty(missing)
                error('OceanAgents:World:BadAssembly', 'AgentWorld: agencyCfg lacks %s.', strjoin(missing, ', '));
            end
            labels = reshape(double(agencyCfg.ActionLabels), 1, []);
            routing = agencyCfg.Routing;
            if ~(isstruct(routing) && all(isfield(routing, {'Label', 'Type', 'Payload'})))
                error('OceanAgents:World:BadAssembly', 'AgentWorld: Routing must be a struct array (Label, Type, Payload).');
            end
            if ~isequal(sort([routing.Label]), sort(labels)) || numel(routing) ~= numel(labels)
                error('OceanAgents:World:BadAssembly', 'AgentWorld: Routing must give exactly one event per action label.');
            end
            if isfield(agencyCfg, 'Roster'), roster = string(agencyCfg.Roster);
            elseif isfield(agencyCfg, 'AgentID'), roster = string(agencyCfg.AgentID);
            else, roster = "agent1";
            end
            roster = reshape(roster, 1, []);
            roles = repmat("default", 1, numel(roster));
            if isfield(agencyCfg, 'Roles') && ~isempty(agencyCfg.Roles)
                roles = reshape(string(agencyCfg.Roles), 1, []);
                if isscalar(roles), roles = repmat(roles, 1, numel(roster)); end
            end
            routed = unique(string({routing.Type}), 'stable');
            obj.World_ = OceanAgents.World.World(doors, 'Phases', options.Phases, 'SkipIdle', options.SkipIdle, ...
                'Injectable', routed, 'Outbox', obj.OutboxTypes);
            if ~any(arrayfun(@(d) any(string({d.Owns.Name}) == "AgentID"), obj.World_.Doors))
                error('OceanAgents:World:BadAssembly', 'AgentWorld: no door owns AgentID (include Doors.Agency.spec()).');
            end
            obj.Name = options.Name;
            obj.Roster = roster;
            obj.Roles = roles;
            obj.ActionLabels = labels;
            obj.Cfg_ = agencyCfg;
        end

        function spec = describe(obj)
            n = numel(obj.Roster);
            spec = OceanAgents.Core.EnvSpec('Name', obj.Name, 'Timing', "simultaneous", 'Roster', obj.Roster, ...
                'Roles', obj.Roles, 'ObservationSpaces', repmat({obj.Cfg_.ObservationSpace}, 1, n), ...
                'ActionSpaces', repmat({obj.Cfg_.ActionSpace}, 1, n), 'Capabilities', OceanAgents.Core.CapabilitySet(), ...
                'Horizon', Inf);
        end

        function frame = reset(obj, seed)
            % RESET  New episode (seed [] = continue the World's door streams); returns the first DecisionFrame.
            arguments
                obj
                seed double = []
            end
            w = obj.World_;
            w.reset(seed);
            rows = w.Store.rowsWith("AgentID");
            ids = w.Registry.idsOf(rows);
            [ids, order] = sort(ids);
            names = reshape(w.Store.valueAt("AgentID", rows(order), w.Now), 1, []);
            if ~isequal(names, obj.Roster)
                error('OceanAgents:Runtime:RosterMismatch', 'AgentWorld %s: the world''s agents [%s] differ from the declared roster [%s].', ...
                    obj.Name, strjoin(names, ', '), strjoin(obj.Roster, ', '));
            end
            obj.Entities_ = reshape(ids, 1, []);
            obj.Ended_ = false(1, numel(obj.Roster));
            obj.FirstReason_ = "";
            obj.EpisodeID = obj.EpisodeID + 1;
            obj.FrameID = obj.FrameID + 1;
            obj.Ready_ = true;
            n = numel(obj.Roster);
            obj.Frame_ = obj.frame_(true(1, n), false(1, n), zeros(1, n), false(1, n), false(1, n), false, "");
            frame = obj.Frame_;
            obj.publish_();
        end

        function frame = step(obj, batch)
            % STEP  One joint transition: routed events -> exactly one World.advance -> outbox -> next frame.
            if ~obj.Ready_ || isempty(obj.Frame_) || obj.Frame_.EpisodeEnded
                error('OceanAgents:Env:NotReady', 'AgentWorld %s: step before reset or after the episode ended.', obj.Name);
            end
            batch.validate(obj.Frame_);                             % nothing happens to the world on a bad batch
            w = obj.World_;
            events = OceanAgents.World.Event.empty(1, 0);
            for k = find(obj.Frame_.Requested)
                d = batch.decisionFor(obj.Roster(k));
                route = obj.Cfg_.Routing([obj.Cfg_.Routing.Label] == d.Action);
                events(end + 1) = OceanAgents.World.Event.make(route.Type, route.Payload, ...
                    'SourceEntity', obj.Entities_(k)); %#ok<AGROW>
            end
            w.inject(events);
            w.advance();
            obj.NumAdvances = obj.NumAdvances + 1;
            out = w.takeOutbox();
            n = numel(obj.Roster);
            reward = zeros(1, n);
            term = false(1, n);
            trunc = false(1, n);
            reasons = strings(1, n);
            acted = obj.Frame_.Requested;
            for e = out
                k = find(obj.Entities_ == e.SourceEntity, 1);
                if isempty(k) || obj.Ended_(k), continue; end
                switch e.Type
                    case "Reward"
                        reward(k) = reward(k) + e.Payload.Amount;
                    case "Terminated"
                        if ~term(k), reasons(k) = reasonOf_(e, "terminal"); end
                        term(k) = true;
                    case "Truncated"
                        if ~term(k) && ~trunc(k), reasons(k) = reasonOf_(e, "time-limit"); end
                        trunc(k) = true;
                end
            end
            trunc = trunc & ~term;                                   % termination wins
            reasons(term & reasons == "") = "terminal";
            endedNow = (term | trunc) & ~obj.Ended_;
            if any(endedNow) && strlength(obj.FirstReason_) == 0
                obj.FirstReason_ = reasons(find(endedNow, 1));       % the episode's first ending
            end
            obj.Ended_ = obj.Ended_ | endedNow;
            allEnded = all(obj.Ended_);
            endReason = "";
            if allEnded, endReason = obj.FirstReason_; end
            obj.FrameID = obj.FrameID + 1;
            obj.Frame_ = obj.frame_(~obj.Ended_, acted, reward .* acted, term & endedNow, trunc & endedNow, ...
                allEnded, endReason, endedNow);
            frame = obj.Frame_;
            obj.publish_();
        end

        function v = researchView(obj, component)
            % RESEARCHVIEW  Privileged read-only snapshot of one component of the roster's entities at the committed
            % boundary (after reset or after a step's single advance). For observers (recording, rendering), never for
            % agents: it ignores visibility, changes nothing and returns plain copies.
            %   v = struct(EpisodeID, FrameID, Tick, Roster, Ended, Value)   Value: numel(Roster) x width, roster order
            arguments
                obj
                component (1,1) string
            end
            if ~obj.Ready_
                error('OceanAgents:Env:NotReady', 'AgentWorld %s: researchView before reset.', obj.Name);
            end
            w = obj.World_;
            rows = w.Registry.rowsOf(obj.Entities_);
            v = struct('EpisodeID', obj.EpisodeID, 'FrameID', obj.FrameID, 'Tick', max(w.Now, 0), ...
                'Roster', obj.Roster, 'Ended', obj.Ended_, 'Value', double(w.Store.valueAt(component, rows, w.Now)));
        end

        function onCommit(obj, fn)
            % ONCOMMIT  Register a listener fn(aw) called ONCE per committed boundary: after reset and after each step
            % that completed (a failed or refused step publishes nothing). Listeners read through researchView and the
            % accessors below; they must not drive the world.
            arguments
                obj
                fn (1,1) function_handle
            end
            obj.Listeners_{end + 1} = fn;
        end

        function t = tick(obj)
            % TICK  The world clock at the committed boundary (max(Now, 0), as in researchView and the frames).
            t = max(obj.World_.Now, 0);
        end

        function ids = entityIDs(obj)
            % ENTITYIDS  Entity IDs of the roster slots (agents), in roster order; empty before reset.
            ids = obj.Entities_;
        end

        function ids = liveEntityIDs(obj)
            % LIVEENTITYIDS  IDs of every alive entity of the world (agents and others).
            obj.requireReady_('liveEntityIDs');
            ids = obj.World_.liveEntityIDs();
        end

        function ids = pinnedEntityIDs(obj)
            % PINNEDENTITYIDS  Roster agents: action-driven, never deactivated.
            ids = obj.Entities_;
        end

        function ids = inactiveEntityIDs(obj)
            ids = obj.World_.inactiveEntityIDs();
        end

        function r = activeRevision(obj)
            r = obj.World_.ActiveRevision;
        end

        function report = deactivate(obj, ids)
            % DEACTIVATE  Take non-agent entities out of the active set at the committed boundary (see World.deactivate).
            % Roster agents are action-driven and cannot be deactivated here (OceanAgents:World:RosterEntity).
            obj.requireReady_('deactivate');
            if any(ismember(ids, obj.Entities_))
                error('OceanAgents:World:RosterEntity', ...
                    'AgentWorld %s: roster agents are action-driven; deactivating one is not supported.', obj.Name);
            end
            report = obj.World_.deactivate(ids);
        end

        function report = activate(obj, ids, varargin)
            obj.requireReady_('activate');
            report = obj.World_.activate(ids, varargin{:});
        end

        function close(~)
            % CLOSE  Nothing to release.
        end

        function meta = getMetadata(obj)
            meta = struct('SymbolicLatex', strings(1, 0), 'Symbols', struct('Symbol', {}, 'Description', {}), ...
                'Assumptions', {{sprintf('agent face of a World with %d doors; roster [%s]; one advance per step', ...
                numel(obj.World_.Doors), strjoin(obj.Roster, ', '))}}, 'IsSymbolic', false, ...
                'Reference', "W2_Grid_Doors_Contract.md section 4");
        end

        function spec = getPipelineSpec(obj)
            spec = struct('ComponentName', "AgentWorld", ...
                'Inputs', struct('Name', "batch", 'Domain', "Agent-ActionBatch", 'Dtype', "OceanAgents.Core.ActionBatch", 'Shape', "1 x 1"), ...
                'Outputs', struct('Name', "frame", 'Domain', "Agent-DecisionFrame", 'Dtype', "OceanAgents.Core.DecisionFrame", 'Shape', "1 x 1"), ...
                'Kind', "NONVIRTUAL", 'DiagramSymbol', struct('Type', "default", 'Label', lower(obj.Name)), ...
                'ComposesFrom', {{"OceanAgents.World.World"}});
        end
    end

    methods (Access = private)
        function publish_(obj)
            % The world has already committed: a failing listener must not turn a completed transaction into an error.
            for k = 1:numel(obj.Listeners_)
                try
                    obj.Listeners_{k}(obj);
                catch err
                    obj.ListenerFailures = [obj.ListenerFailures {err}];
                end
            end
        end

        function requireReady_(obj, op)
            if ~obj.Ready_
                error('OceanAgents:Env:NotReady', 'AgentWorld %s: %s before reset.', obj.Name, op);
            end
        end

        function f = frame_(obj, live, rewardValid, rewards, term, trunc, ended, endReason, endedNow)
            % Build a DecisionFrame: views for live slots (requested) and for slots that ended at this step.
            if nargin < 9, endedNow = false(size(live)); end
            n = numel(obj.Roster);
            A = numel(obj.ActionLabels);
            views = cell(1, n);
            w = obj.World_;
            for k = 1:n
                if ~(live(k) || endedNow(k)), continue; end
                info = struct();
                mask = true(1, A);
                if endedNow(k)
                    mask = false(1, A);
                    if trunc(k), info.successor_action_mask = true(1, A); end
                end
                obs = double(obj.Cfg_.Observe(w.project(obj.Entities_(k))));
                views{k} = OceanAgents.Core.ActorView('AgentID', obj.Roster(k), 'EpisodeID', obj.EpisodeID, ...
                    'FrameID', obj.FrameID, 'Tick', max(w.Now, 0), 'Observation', obs, 'ActionLabels', obj.ActionLabels, ...
                    'ActionMask', mask, 'Info', info);
            end
            f = OceanAgents.Core.DecisionFrame('EpisodeID', obj.EpisodeID, 'FrameID', obj.FrameID, ...
                'Tick', max(w.Now, 0), 'Roster', obj.Roster, 'Live', live, 'Requested', live, 'Views', views, ...
                'Rewards', rewards, 'RewardValid', rewardValid, 'Terminated', term, 'Truncated', trunc, ...
                'EpisodeEnded', ended, 'EndReason', endReason);
        end
    end
end

function r = reasonOf_(e, default)
r = default;
if isfield(e.Payload, 'Reason') && strlength(string(e.Payload.Reason)) > 0, r = string(e.Payload.Reason); end
end
