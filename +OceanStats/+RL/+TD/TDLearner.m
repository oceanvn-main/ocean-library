classdef TDLearner < handle
    % TDLEARNER  Temporal-difference learner: a thin orchestrator that wires independent, replaceable parts and runs the
    % episodes. The TD rule, the update scheme, the estimates being learned, the behaviour and the environment interaction
    % are each a strategy in their own package, so a new algorithm is ONE new file:
    %
    %   Target        +Targets/     the number an estimate moves toward (TD0, Sarsa, ExpectedSarsa, QLearning, ...)
    %   Backup        +Backups/     the update scheme, chosen by Mode (OneStep, NStep, Trace, LambdaReturn, ...)
    %   StepSize      +StepSize/    alpha as a function of the visit count
    %   Exploration   +Exploration/ epsilon-greedy, softmax, ... (control)      Trace  +Traces/  accumulating, replacing, ...
    %   value function +Values/     TabularV, TabularQ, DoubleTabularQ (chosen from Task / Double_learning)
    %   behaviour     +Policies/    ExplorationPolicy (control) or FixedPolicy (prediction)
    %   Sampler                     the ONLY code that touches the environment (reset/step, indices, MaxSteps truncation)
    %   observer      +Observers/   Recorder (Record = true), ReturnLogger, ... any IObserver via 'Observer' (one or a cell)
    %
    %   learner = OceanStats.RL.TD.TDLearner('Env', env, 'Target', "qlearning", ...
    %                 'StepSize', 0.5, 'Exploration', 0.1, 'Gamma', 1, 'Episodes', 500, 'Seed', 1);
    %   [V, Q, pi, history] = learner.run();
    %
    % Env is an OceanStats.RL.Env environment (or anything with the older sampling interface).
    % TASK / MODE (what is learned; how it is updated):
    %   Task "control"      learns Q(s,a); behaviour = Exploration strategy. Default Target "qlearning".
    %   Task "prediction"   learns V(s) for the fixed policy PolicyProbs ((S x A), default uniform over admissible
    %                       actions). Default Target "td0".
    %   Mode "onestep"      update right after every step with Target.compute (TD(0), Sarsa, Expected Sarsa,
    %                       Q-learning; 'Double_learning',true = Double Q-learning).
    %   Mode "nstep"        buffered n-step return (N steps): n-step TD prediction / n-step Sarsa.
    %   Mode "trace"        backward-view eligibility traces (Lambda, Trace strategy): TD(lambda), Sarsa(lambda),
    %                       Watkins's Q(lambda) (off-policy Target => traces cut after exploratory actions).
    %                       'Offline',true accumulates the episode's updates and applies them at its end.
    %   Mode "lambdareturn" forward-view offline lambda-return algorithm (prediction); equals the OFFLINE
    %                       accumulating-trace backward view exactly (a tested theorem).
    %   Any other Mode is looked up in +Backups by name (Factory): drop a new IBackup file there to add a scheme.
    %   Extension hooks (each a new file, no edit here): 'ValueFunction' = @(sampler, initialValue) IValueFunction,
    %   'Policy' = @(sampler) IBehaviourPolicy, 'Observer' = IObserver (or a cell of them), 'Extra' = struct passed to the
    %   backup as ctx.extra. A hook is checked against what the run actually uses (value kind, one-step-only estimators,
    %   greedy-target estimators). A 'Policy' hook replaces the Exploration / PolicyProbs behaviour entirely.
    %   An observer instance is reset by startRun() at the start of every run().
    %
    % TERMINATION vs TRUNCATION (Gymnasium semantics): bootstrapping stops ONLY on true termination. The environment's
    % time limit and MaxSteps (per episode) are truncations.
    % REPRODUCIBILITY: every random draw comes from this run's RandStream (Seed); the environment's own stream is
    % re-seeded from it when the environment supports it. run() restarts from the Seed every time.
    % 3 pillars: getMetadata() / getPipelineSpec() (VIRTUAL: composes the strategies).

    properties (SetAccess = private)
        Env
        Task (1,1) string = "control"
        Mode (1,1) string = "onestep"
        Target
        Backup
        StepSize
        Exploration
        Trace
        Gamma (1,1) double = 1
        Lambda (1,1) double = 0.9
        N (1,1) double = 1
        Episodes (1,1) double = 100
        MaxSteps (1,1) double = 1e5
        Double_learning (1,1) logical = false
        Offline (1,1) logical = false
        Record (1,1) logical = false
        PolicyProbs double = []
        InitialValue (1,1) double = 0
        Seed (1,1) double = 0
        ValueFunction = []
        Policy = []
        Observer = []
        Extra struct = struct()
    end

    methods
        function obj = TDLearner(options)
            arguments
                options.Env
                options.Task (1,1) string {mustBeMember(options.Task, ["control", "prediction"])} = "control"
                options.Mode {OceanStats.RL.TD.TDLearner.mustBeModeSpec} = "onestep"
                options.Target = ""
                options.StepSize = 0.1
                options.Exploration = 0.1
                options.Trace = "accumulating"
                options.Gamma (1,1) double {mustBeGreaterThanOrEqual(options.Gamma, 0), mustBeLessThanOrEqual(options.Gamma, 1)} = 1
                options.Lambda (1,1) double {mustBeGreaterThanOrEqual(options.Lambda, 0), mustBeLessThanOrEqual(options.Lambda, 1)} = 0.9
                options.N (1,1) double {mustBeInteger, mustBePositive} = 1
                options.Episodes (1,1) double {mustBeInteger, mustBePositive} = 100
                options.MaxSteps (1,1) double {mustBeInteger, mustBePositive} = 1e5
                options.Double_learning (1,1) logical = false
                options.Offline (1,1) logical = false
                options.Record (1,1) logical = false
                options.PolicyProbs double = []
                options.InitialValue (1,1) double = 0
                options.Seed (1,1) double = 0
                options.ValueFunction = []
                options.Policy = []
                options.Observer = []
                options.Extra (1,1) struct = struct()
            end
            obj.Env = OceanStats.RL.Compat.LegacyEnvironment.adapt(options.Env, 'OceanStats.RL.MonteCarlo.IEpisodeEnvironment');
            obj.Task = options.Task;
            if isa(options.Mode, 'OceanStats.RL.TD.Backups.IBackup')
                obj.Backup = options.Mode;  obj.Mode = string(options.Mode.name());
            else
                obj.Mode = string(options.Mode);
                try
                    obj.Backup = OceanStats.RL.TD.Factory.create("backup", obj.Mode);
                catch ME
                    if strcmp(ME.identifier, 'OceanNLP:Text:StrategyResolver:Unknown')
                        error('OceanStats:RL:TD:TDLearner:UnknownMode', 'Unknown Mode "%s". Available: %s.', obj.Mode, ...
                            strjoin(string(OceanStats.RL.TD.Factory.validNames("backup")), ', '));
                    end
                    rethrow(ME);
                end
            end
            if isequal(options.Target, "")
                if obj.Task == "prediction", options.Target = "td0"; else, options.Target = "qlearning"; end
            end
            obj.Target = OceanStats.RL.TD.TDLearner.asComponent_(options.Target, "target", 'OceanStats.RL.TD.Targets.ITarget');
            obj.StepSize = OceanStats.RL.TD.TDLearner.asStepSize_(options.StepSize);
            obj.Exploration = OceanStats.RL.TD.TDLearner.asExploration_(options.Exploration);
            obj.Trace = OceanStats.RL.TD.TDLearner.asComponent_(options.Trace, "trace", 'OceanStats.RL.TD.Traces.ITrace');
            obj.Gamma = options.Gamma;  obj.Lambda = options.Lambda;  obj.N = options.N;
            obj.Episodes = options.Episodes;  obj.MaxSteps = options.MaxSteps;
            obj.Double_learning = options.Double_learning;  obj.Offline = options.Offline;  obj.Record = options.Record;
            obj.PolicyProbs = options.PolicyProbs;
            obj.InitialValue = options.InitialValue;
            obj.Seed = options.Seed;
            obj.ValueFunction = options.ValueFunction;
            obj.Policy = options.Policy;
            obj.Observer = options.Observer;
            obj.Extra = options.Extra;
            obj.validateCombination_();
        end

        function [V, Q, pi, history] = run(obj)
            % every call restarts from the Seed: the run's stream AND the environment's stream are reset here,
            % so run() twice (or two learners sharing an environment) gives identical results
            stream = RandStream('mt19937ar', 'Seed', obj.Seed);
            sm = OceanStats.RL.TD.Sampler(obj.Env, obj.MaxSteps);
            sm.seed(obj.Seed + 1);
            vf = obj.buildValues_(sm);
            policy = obj.buildPolicy_(sm);
            obj.validateParts_(vf);
            [observer, recorder] = obj.buildObserver_();
            observer.startRun();
            ctx = OceanStats.RL.TD.Context();
            ctx.extra = obj.Extra;
            ctx.sampler = sm;  ctx.values = vf;  ctx.policy = policy;  ctx.target = obj.Target;
            ctx.stepSize = obj.StepSize;  ctx.trace = obj.Trace;  ctx.observer = observer;  ctx.stream = stream;
            ctx.gamma = obj.Gamma;  ctx.lambda = obj.Lambda;  ctx.n = obj.N;  ctx.offline = obj.Offline;
            returns = zeros(obj.Episodes, 1);  lengths = zeros(obj.Episodes, 1);
            for ep = 1:obj.Episodes
                ctx.progress = (ep - 1) / max(obj.Episodes - 1, 1);
                [returns(ep), lengths(ep)] = obj.Backup.runEpisode(ctx);
            end
            [V, Q, pi] = obj.summarise_(vf, policy, sm);
            history = struct('EpisodeReturns', returns, 'EpisodeLengths', lengths, 'TotalSteps', sm.Steps);
            if obj.Record, history.Episodes = recorder.episodes(); end
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = { ...
                'VIRTUAL composition: the loop lives in the Backup, the update rule in the Target, the estimates in the value function; the pure functions are in +Functional', ...
                'IsSymbolic=false at this level: the loop is procedural; the closed-form update rule is in obj.Target.getMetadata().SymbolicLatex', ...
                'bootstrapping stops only on true termination; truncation (env time limit / MaxSteps) keeps the bootstrap', ...
                sprintf('task=%s mode=%s target=%s N=%d lambda=%.3g gamma=%.3g', obj.Task, obj.Mode, obj.Target.name(), obj.N, obj.Lambda, obj.Gamma)};
            meta.IsSymbolic = false;
            tm = obj.Target.getMetadata();
            assert(strlength(tm.Reference) > 0 && ~isempty(tm.SymbolicLatex), ...
                'OceanStats:RL:TD:TDLearner:TargetMetadataUnresolved', 'Target metadata must resolve to a formula.');
            meta.Reference = "VIRTUAL composite: inspect obj.Target.getMetadata() for the closed-form update rule";
            meta.ComposedPartReference = struct('TargetSymbolicLatex', tm.SymbolicLatex, 'TargetReference', tm.Reference);
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.Inputs = struct('Name', "env", 'Domain', "RL-Environment", 'Dtype', "object", 'Shape', "1 x 1");
            spec.Outputs = [struct('Name', "V", 'Domain', "RL-Value", 'Dtype', "double", 'Shape', "S x 1"), ...
                            struct('Name', "Q", 'Domain', "RL-Value", 'Dtype', "double", 'Shape', "S x A")];
            spec.Kind = "VIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "td-learner");
            spec.ComposesFrom = {class(obj.Target), class(obj.Backup), class(obj.StepSize), class(obj.Exploration), class(obj.Trace), class(obj.Env)};
        end
    end

    % ------------------------------------------------------------------ private machinery
    methods (Access = private)
        function validateCombination_(obj)
            % compatibility is decided from declared capabilities (target kind, backup and value-function abilities)
            kind = obj.Target.valueKind();
            if obj.Task == "prediction" && kind ~= "state"
                error('OceanStats:RL:TD:TDLearner:BadTarget', 'Prediction (state values) needs a state-value Target (e.g. "td0"), got "%s".', obj.Target.name());
            end
            if obj.Task == "control" && kind ~= "action"
                error('OceanStats:RL:TD:TDLearner:BadTarget', 'Target "%s" estimates state values; use Task "prediction".', obj.Target.name());
            end
            vkind = kind;                                              % the value kind learned is the target's
            if ~obj.Backup.supportsValueKind(vkind)
                error('OceanStats:RL:TD:TDLearner:BadMode', 'Mode "%s" is not defined for %s.', obj.Mode, obj.Task);
            end
            if obj.Target.IsOffPolicy && ~obj.Backup.allowsOffPolicyTarget(vkind)
                error('OceanStats:RL:TD:TDLearner:BadTarget', ...
                    'Mode "%s" needs an on-policy target (e.g. "sarsa", "expectedsarsa"); off-policy n-step needs importance sampling or tree backup, which is not implemented.', obj.Mode);
            end
            if obj.Double_learning && ~(obj.Task == "control" && obj.Backup.supportsUpdateFirstValues() && obj.Target.supportsDouble())
                error('OceanStats:RL:TD:TDLearner:BadCombination', 'Double learning is defined for one-step control with a target whose supportsDouble() is true (Q-learning).');
            end
            if obj.Offline && ~obj.Backup.supportsOffline(vkind)
                error('OceanStats:RL:TD:TDLearner:BadCombination', '"Offline" applies to prediction with Mode "trace".');
            end
            if obj.Task == "control" && ~isempty(obj.PolicyProbs)
                error('OceanStats:RL:TD:TDLearner:BadCombination', 'PolicyProbs is the policy being evaluated: it applies to Task "prediction" only (control uses the Exploration strategy).');
            end
            if obj.Double_learning && ~isempty(obj.ValueFunction)
                error('OceanStats:RL:TD:TDLearner:BadCombination', '"Double_learning" selects the built-in double estimator; do not combine it with a ValueFunction hook (supply a double estimator as the hook instead).');
            end
        end

        function validateParts_(obj, vf)
            % checked against the parts actually built (a hook may replace the defaults)
            if vf.kind() ~= obj.Target.valueKind()
                error('OceanStats:RL:TD:TDLearner:BadValueFunction', 'The value function holds %s values but Target "%s" needs %s values.', ...
                    vf.kind(), obj.Target.name(), obj.Target.valueKind());
            end
            if vf.requiresGreedyTarget() && ~obj.Target.supportsDouble()
                error('OceanStats:RL:TD:TDLearner:BadCombination', '%s ignores the target and needs a greedy target (supportsDouble), not "%s".', class(vf), obj.Target.name());
            end
            if vf.requiresOneStepBackup() && ~obj.Backup.supportsUpdateFirstValues()
                error('OceanStats:RL:TD:TDLearner:BadCombination', '%s can only be driven by the one-step backup, not Mode "%s".', class(vf), obj.Mode);
            end
        end

        function [observer, recorder] = buildObserver_(obj)
            list = {};
            recorder = OceanStats.RL.TD.Observers.NullObserver();
            if obj.Record, recorder = OceanStats.RL.TD.Observers.Recorder();  list{end+1} = recorder; end
            if ~isempty(obj.Observer)
                extra = obj.Observer;
                if ~iscell(extra), extra = {extra}; end
                for k = 1:numel(extra)
                    if ~isa(extra{k}, 'OceanStats.RL.TD.Observers.IObserver')
                        error('OceanStats:RL:TD:TDLearner:BadObserver', 'Observer %d is not an OceanStats.RL.TD.Observers.IObserver (got %s).', k, class(extra{k}));
                    end
                end
                list = [list, extra(:)'];
            end
            if isempty(list)
                observer = OceanStats.RL.TD.Observers.NullObserver();
            elseif isscalar(list)
                observer = list{1};
            else
                observer = OceanStats.RL.TD.Observers.CompositeObserver(list);
            end
        end

        function vf = buildValues_(obj, sm)
            if ~isempty(obj.ValueFunction)
                vf = obj.ValueFunction(sm, obj.InitialValue);
                if ~isa(vf, 'OceanStats.RL.TD.Values.IValueFunction')
                    error('OceanStats:RL:TD:TDLearner:BadValueFunction', 'The ValueFunction hook must return an IValueFunction, got %s.', class(vf));
                end
            elseif obj.Task == "prediction"
                vf = OceanStats.RL.TD.Values.TabularV(sm.NumS, obj.InitialValue);
            elseif obj.Double_learning
                vf = OceanStats.RL.TD.Values.DoubleTabularQ(sm.ValidMask, obj.InitialValue);
            else
                vf = OceanStats.RL.TD.Values.TabularQ(sm.ValidMask, obj.InitialValue);
            end
        end

        function policy = buildPolicy_(obj, sm)
            if ~isempty(obj.Policy)
                policy = obj.Policy(sm);
                if ~isa(policy, 'OceanStats.RL.TD.Policies.IBehaviourPolicy')
                    error('OceanStats:RL:TD:TDLearner:BadPolicy', 'The Policy hook must return an IBehaviourPolicy, got %s.', class(policy));
                end
            elseif obj.Task == "prediction"
                policy = OceanStats.RL.TD.Policies.FixedPolicy(obj.PolicyProbs, sm.ValidMask);
            else
                policy = OceanStats.RL.TD.Policies.ExplorationPolicy(obj.Exploration);
            end
        end

        function [V, Q, pi] = summarise_(obj, vf, policy, sm)
            pi = zeros(sm.NumS, 1);
            if obj.Task == "control"
                Q = vf.finalQ();
                V = zeros(sm.NumS, 1);
                for s = 1:sm.NumS
                    v = sm.ValidMask(s, :);
                    if ~any(v), continue; end
                    q = Q(s, :);  q(~v) = -Inf;
                    [m, a] = max(q);
                    V(s) = m;  pi(s) = sm.ActionValues(a);
                end
            else
                V = vf.finalV();
                Q = obj.InitialValue * ones(sm.NumS, sm.NumA);
                Q(~sm.ValidMask) = 0;
                for s = 1:sm.NumS
                    v = sm.ValidMask(s, :);
                    if any(v), pi(s) = sm.ActionValues(policy.mostLikelyAction(vf, s, v)); end
                end
            end
        end
    end

    methods (Static)
        function mustBeModeSpec(x)
            if ~(isa(x, 'OceanStats.RL.TD.Backups.IBackup') || ((ischar(x) || (isstring(x) && isscalar(x))) && strlength(string(x)) > 0))
                error('OceanStats:RL:TD:TDLearner:BadMode', 'Mode must be the name of an update scheme (text) or an IBackup object.');
            end
        end
    end

    methods (Static, Access = private)
        function c = asComponent_(x, kind, base)
            if isa(x, base)
                c = x;
            else
                c = OceanStats.RL.TD.Factory.create(kind, string(x));
            end
        end
        function c = asStepSize_(x)
            if isa(x, 'OceanStats.RL.TD.StepSize.IStepSize'), c = x;
            elseif isnumeric(x), c = OceanStats.RL.TD.StepSize.ConstantStepSize(x);
            else, c = OceanStats.RL.TD.Factory.create("stepsize", string(x));
            end
        end
        function c = asExploration_(x)
            if isa(x, 'OceanStats.RL.TD.Exploration.IExploration'), c = x;
            elseif isnumeric(x), c = OceanStats.RL.TD.Exploration.EpsilonGreedyExploration(x);
            else, c = OceanStats.RL.TD.Factory.create("exploration", string(x));
            end
        end
    end
end
