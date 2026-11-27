classdef NativeSingleAgentBridge < OceanAgents.Core.ISingleAgentEnvironment
    % NATIVESINGLEAGENTBRIDGE  Runs an existing single-agent OceanStats.RL.Env.IEnvironment in agent records:
    % reset gives an ActorView, step takes a Decision and returns the agent's Feedback. The native environment's
    % observations, rewards and terminated/truncated flags pass through unchanged, so a native loop and a bridged loop
    % with the same seed and the same actions see exactly the same trajectory (A0 C32 / IG01).
    %
    %   env  = OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner', 5);
    %   b    = OceanAgents.Env.NativeSingleAgentBridge(env, 'AgentID', "agent1");
    %   spec = b.describe();                    % EnvSpec: Timing "single", capabilities derived from the env
    %   view = b.reset(7);                      % env.reset('Seed', 7); EpisodeID and FrameID advance
    %   d    = OceanAgents.Core.Decision('AgentID', "agent1", 'EpisodeID', view.EpisodeID, ...
    %              'FrameID', view.FrameID, 'Action', 2);
    %   fb   = b.step(d);                       % validated against view BEFORE env.step; fb.NextView is the next view
    %
    % Action labels.  The native ActionSpace must be Discrete(N, 'Start', s0); the view's ActionLabels are
    %   s0:(s0+N-1) and ActionMask(k) = info.action_mask(k) refers to ActionLabels(k). The decision's LABEL (never a
    %   column index) is what is passed to env.step.
    % Masks.  A missing info.action_mask means "all legal" at a non-ended observation and "none legal" at an ended one.
    %   After an ended step the NextView mask is all false; after a truncation the native successor mask is kept in
    %   NextView.Info.successor_action_mask so a learner can still bootstrap (Sutton & Barto 2018, sec. 3.4 / Pardo
    %   et al. 2018 "Time limits in RL": truncation is not termination).
    % Info.  The view's Info keeps only the native info fields named in 'InfoAllow' (action_mask itself is consumed
    %   into ActionMask and never copied), so diagnostic fields of the native env do not leak to the agent.
    % Capabilities (A0 5.1).  Enumerable <- unwrapped() is an IEnumerableEnvironment; ExactModel <- the OUTER env is an
    %   IModelEnvironment (a TimeLimit-wrapped model env does not claim it: the horizon changes the model); every
    %   social capability is false. Horizon is the smallest TimeLimit.MaxSteps in the wrapper chain (Inf if none).
    % Errors.  step before reset or after an ended episode -> OceanAgents:Env:NotReady; stale decision ->
    %   OceanAgents:Core:StaleFrame; illegal label -> OceanAgents:Core:IllegalAction; non-Discrete action space ->
    %   OceanAgents:Env:UnsupportedSpace. A rejected decision never reaches env.step.
    % close() is a documented no-op: IEnvironment has no close.
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 5.1.

    properties (SetAccess = private)
        AgentID string = "agent1"
        Role string = "default"
        InfoAllow string = "action_mask"
        ActionLabels double = zeros(1, 0)       % 1 x A external labels, column order of every view
        EpisodeID double = 0                    % episodes started in this run (monotonic)
        FrameID double = 0                      % views issued in this run (monotonic, never reset)
        Status string = "Idle"                  % "Idle" (before reset) | "Running" | "Ended"
        NativeStepCalls double = 0              % total env.step calls made by this bridge
    end

    properties (Access = private)
        Env_                                    % the wrapped OceanStats.RL.Env.IEnvironment
        View_ = []                              % current ActorView (the one a decision must answer)
        Tick_ double = 0                        % native steps taken in the current episode
    end

    methods
        function obj = NativeSingleAgentBridge(env, options)
            arguments
                env
                options.AgentID = "agent1"
                options.Role = "default"
                options.InfoAllow = "action_mask"
            end
            if ~(isa(env, 'OceanStats.RL.Env.IEnvironment') && isscalar(env))
                error('OceanAgents:Env:UnsupportedSpace', ...
                    'NativeSingleAgentBridge wraps a scalar OceanStats.RL.Env.IEnvironment (got %s).', class(env));
            end
            space = env.ActionSpace;
            if ~isa(space, 'OceanStats.RL.Env.Spaces.Discrete')
                error('OceanAgents:Env:UnsupportedSpace', ...
                    'NativeSingleAgentBridge needs a Discrete action space (got %s).', space.describe());
            end
            obj.Env_ = env;
            obj.AgentID = textScalar_(options.AgentID, 'AgentID');
            obj.Role = textScalar_(options.Role, 'Role');
            allow = options.InfoAllow;
            allow = convertCharsToStrings(allow);
            if ~isstring(allow) || any(ismissing(allow))
                error('OceanAgents:Core:InvalidRecord', 'InfoAllow must be a string array of info field names.');
            end
            obj.InfoAllow = reshape(allow, 1, []);
            obj.ActionLabels = space.Start + (0:space.N - 1);
        end

        function spec = describe(obj)
            % DESCRIBE  EnvSpec of the wrapped environment (no episode mutation).
            env = obj.Env_;
            caps = OceanAgents.Core.CapabilitySet( ...
                'Enumerable', isa(env.unwrapped(), 'OceanStats.RL.Env.IEnumerableEnvironment'), ...
                'ExactModel', isa(env, 'OceanStats.RL.Env.IModelEnvironment'));
            spec = OceanAgents.Core.EnvSpec('Name', string(class(env.unwrapped())), 'Timing', "single", ...
                'Roster', obj.AgentID, 'Roles', obj.Role, 'ObservationSpaces', {env.ObservationSpace}, ...
                'ActionSpaces', {env.ActionSpace}, 'Capabilities', caps, 'Horizon', obj.horizon_());
        end

        function view = reset(obj, seed)
            % RESET  Start a new episode: env.reset('Seed', seed), then the first ActorView.
            arguments
                obj
                seed (1,1) double {mustBeInteger, mustBeNonnegative}
            end
            obj.Status = "Idle";                        % a failing native reset leaves the bridge not ready
            [obs, info] = obj.Env_.reset('Seed', seed);
            obj.EpisodeID = obj.EpisodeID + 1;
            obj.Tick_ = 0;
            mask = obj.maskFrom_(info, false);
            view = obj.makeView_(obs, mask, obj.allowedInfo_(info));
            obj.Status = "Running";
        end

        function fb = step(obj, decision)
            % STEP  Validate decision against the current view, then exactly one native env.step(label).
            if obj.Status ~= "Running"
                error('OceanAgents:Env:NotReady', 'step called while the bridge is %s; call reset(seed) first.', ...
                    obj.Status);
            end
            if ~(isa(decision, 'OceanAgents.Core.Decision') && isscalar(decision))
                error('OceanAgents:Core:InvalidRecord', 'step expects a scalar OceanAgents.Core.Decision.');
            end
            decided = obj.View_;
            decision.validateAgainst(decided);          % throws BEFORE the world moves

            obj.Status = "Ended";                       % if the native step throws, the episode is unusable
            obj.NativeStepCalls = obj.NativeStepCalls + 1;
            [obs, reward, terminated, truncated, info] = obj.Env_.step(decision.Action);
            obj.Tick_ = obj.Tick_ + 1;
            terminated = logical(terminated);
            truncated = logical(truncated) && ~terminated;   % termination takes precedence (A0 2.3)
            ended = terminated || truncated;

            nativeMask = obj.maskFrom_(info, terminated);
            extra = obj.allowedInfo_(info);
            if truncated
                extra.successor_action_mask = nativeMask;
            end
            if ended
                mask = false(1, numel(obj.ActionLabels));
                reason = "terminal";
                if truncated, reason = "time-limit"; end
            else
                mask = nativeMask;
                reason = "";
            end
            next = obj.makeView_(obs, mask, extra);
            fb = OceanAgents.Core.Feedback('AgentID', obj.AgentID, 'EpisodeID', decided.EpisodeID, ...
                'FrameID', decided.FrameID, 'Reward', reward, 'Duration', 1, ...
                'Terminated', terminated, 'Truncated', truncated, 'NextView', next, 'Reason', reason);
            if ~ended
                obj.Status = "Running";
            end
        end

        function close(~)
            % CLOSE  No-op: OceanStats.RL.Env.IEnvironment owns no external resources and has no close.
        end

        function view = currentView(obj)
            % CURRENTVIEW  The view the next decision must answer ([] before the first reset).
            view = obj.View_;
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = {sprintf('bridges %s (observation %s, action %s) to agent records', ...
                class(obj.Env_), obj.Env_.ObservationSpace.describe(), obj.Env_.ActionSpace.describe()), ...
                sprintf('action labels %s; info allowlist [%s]', mat2str(obj.ActionLabels), strjoin(obj.InfoAllow, ', '))};
            meta.IsSymbolic = false;
            meta.Reference = "native single-agent loop in agent records (A0 contract section 5.1)";
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.ComponentName = "NativeSingleAgentBridge";
            spec.Inputs = struct('Name', "decision", 'Domain', "Agent-Decision", ...
                'Dtype', "OceanAgents.Core.Decision", 'Shape', "1x1");
            spec.Outputs = struct('Name', {"view", "feedback"}, 'Domain', {"Agent-ActorView", "Agent-Feedback"}, ...
                'Dtype', {"OceanAgents.Core.ActorView", "OceanAgents.Core.Feedback"}, 'Shape', {"1x1", "1x1"});
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "nativesingleagentbridge");
            spec.ComposesFrom = {class(obj.Env_)};
        end
    end

    methods (Access = private)
        function view = makeView_(obj, obs, mask, info)
            obj.FrameID = obj.FrameID + 1;
            view = OceanAgents.Core.ActorView('AgentID', obj.AgentID, 'EpisodeID', obj.EpisodeID, ...
                'FrameID', obj.FrameID, 'Tick', obj.Tick_, 'Observation', obs, ...
                'ActionLabels', obj.ActionLabels, 'ActionMask', mask, 'Info', info);
            obj.View_ = view;
        end

        function mask = maskFrom_(obj, info, ended)
            % Native info.action_mask as a 1 x A logical row; when absent: all legal unless the episode ended.
            a = numel(obj.ActionLabels);
            if isstruct(info) && isfield(info, 'action_mask')
                mask = logical(reshape(info.action_mask, 1, []));
                if numel(mask) ~= a
                    error('OceanAgents:Core:InvalidRecord', 'info.action_mask has %d entries; the action space has %d.', ...
                        numel(mask), a);
                end
            else
                mask = repmat(~ended, 1, a);
            end
        end

        function out = allowedInfo_(obj, info)
            % Keep only allowlisted native info fields; action_mask (already in ActionMask) and the bridge-owned
            % successor_action_mask are never copied.
            out = struct();
            if ~isstruct(info), return; end
            keep = setdiff(obj.InfoAllow, ["action_mask" "successor_action_mask"]);
            for name = keep
                if isfield(info, name)
                    out.(char(name)) = info.(char(name));
                end
            end
        end

        function h = horizon_(obj)
            % Smallest TimeLimit in the wrapper chain (Inf when there is none).
            h = Inf;
            e = obj.Env_;
            while isa(e, 'OceanStats.RL.Env.EnvWrapper')
                if isa(e, 'OceanStats.RL.Env.Wrappers.TimeLimit')
                    h = min(h, e.MaxSteps);
                end
                e = e.Inner;
            end
        end
    end
end

function s = textScalar_(s, name)
if ischar(s), s = string(s); end
if ~(isstring(s) && isscalar(s)) || ismissing(s) || strlength(s) == 0
    error('OceanAgents:Core:InvalidRecord', '%s must be a non-empty string scalar.', name);
end
end
