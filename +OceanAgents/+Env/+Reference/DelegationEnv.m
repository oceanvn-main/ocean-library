classdef DelegationEnv < OceanAgents.Core.IJointEnvironment
    % DELEGATIONENV  Repeated delegation world (A2 reference world). Each round has one common task context; every
    % agent either delegates the task to one provider or takes the outside option. Each delegation succeeds with the
    % provider's hidden reliability for that context, and the outcome becomes an EvidenceRecord about that provider.
    %
    %   pay = struct('cheap', [10 -20 0], 'expensive', [30 -50 5]);           % [Success Failure Outside] per context
    %   env = OceanAgents.Env.Reference.DelegationEnv('Roster', ["fixed" "adaptive"], 'Providers', ["p1" "p2" "p3"], ...
    %             'Contexts', ["cheap" "expensive"], 'ContextProbs', [0.5 0.5], 'Reliability', [0.9 0.6; 0.5 0.5; 0.2 0.8], ...
    %             'Payoffs', pay, 'Visibility', "public", 'Rounds', 50);
    %   f = env.reset(7);          % owned RandStream('mrg32k3a', 'Seed', 7); round 1 context drawn here
    %   f = env.step(batch);       % one round resolved; f.Views{k}.Evidence holds what slot k may observe
    %
    % Actions (every slot): labels 1:P+1. Label k <= P delegates to Providers(k); label P+1 is the outside option.
    %   view.Info.action_subjects = [Providers "outside"] (aligned with ActionLabels); view.Info.context = context name.
    % Observation of the view on which round t is decided: [contextIndex t]. The final view (after round T) has
    %   observation [0 T+1] and Info.context = "" (no round left to play); its mask is all false.
    % Draw order (the only randomness, all from the env-owned stream, never the global one):
    %   reset:    one draw u for the context of round 1: c = first index with u < cumsum(ContextProbs)
    %   step(t):  for each DELEGATING slot in slot order (never decide order): u1; if u1 < ObstructionProb the outcome
    %             is "unknown" (reward ObstructedPayoff; no u2 is drawn), else u2 and success iff u2 < R_t(k, c);
    %             then, if t < T, one draw for the context of round t+1.
    %   Outside option: reward Payoffs.(c)(3), no draw, no evidence. Success/failure: Payoffs.(c)(1) / (2).
    %   R_t is Reliability, replaced from round FromRound on by the latest Schedule entry with FromRound <= t.
    % Evidence IDs are scoped by RunTag, a tag unique to this env instance (option 'RunTag'; default a fresh UUID-based
    %   tag), so a belief store that outlives this world (checkpoint, reuse with a new world) never mistakes a new event
    %   for an old one (critic A2 F1).
    % Evidence. A delegation by agent a in round t of episode e yields EvidenceID "<RunTag>/E<e>-<t>-<a>", SubjectID = provider,
    %   Context = c, Quantity "completion", SourceKind "direct", SourceID = a, EpisodeID = e, Tick = t. It is carried
    %   by the NEXT view (the frame returned by step(t)): Visibility "own" -> only the delegator's view; "public" ->
    %   every slot's view gets the same records (same IDs). Obstruction gives Outcome "unknown", never "failure".
    % Reports. Reports(j) (Round, ReporterID, SubjectID, Context, Outcome, EvidenceID) is a scripted claim delivered to
    %   EVERY slot, whatever the Visibility, in the view on which round Round is decided (for Round = 1 that is the
    %   initial view from reset). It arrives as SourceKind "report", SourceID = ReporterID, Tick = Round, and, because
    %   scripted reports recur every episode while IDs must be unique per run, EvidenceID "<RunTag>/<EvidenceID>@<episode>".
    %   Within a view, the previous round's direct evidence (slot order) comes first, then this round's reports.
    % Hidden truth (reliabilities, draws, schedule, seed) appears only in DecisionFrame.ResearchInfo, never in a view.
    % Horizon: after round T every slot is Terminated, EpisodeEnded = true, EndReason "horizon", no slot is live or
    %   requested, and the final views carry round T's evidence.
    % See Planning/System/Game Theory/A2_Beliefs_Contract.md section 5.

    properties (SetAccess = private)
        RunTag string = ""                        % scopes every EvidenceID this instance emits
        Roster string = strings(1, 0)           % 1 x N AgentIDs (delegators)
        Roles string = strings(1, 0)            % 1 x N roles (default "delegator")
        Providers string = strings(1, 0)        % 1 x P provider IDs (evidence subjects)
        Contexts string = strings(1, 0)         % 1 x C task contexts
        ContextProbs double = zeros(1, 0)       % 1 x C context distribution
        Payoffs struct = struct()               % Payoffs.(context) = [Success Failure Outside]
        ObstructedPayoff double = 0
        ObstructionProb double = 0
        Visibility string = "own"
        Rounds double = 1                       % horizon T
        EpisodeID double = 0                    % episodes started (monotonic)
        FrameID double = 0                      % frames issued in this run (monotonic, never reset)
        Round double = 0                        % completed rounds in the current episode
        Status string = "Idle"                  % "Idle" | "Running" | "Ended" | "Closed"
    end

    properties (Access = private)
        Reliability_ double = zeros(0, 0)       % P x C base reliabilities (hidden truth)
        Schedule_ struct = struct('FromRound', {}, 'Reliability', {})   % sorted by FromRound (hidden truth)
        Reports_ struct = struct('Round', {}, 'ReporterID', {}, 'SubjectID', {}, 'Context', {}, 'Outcome', {}, ...
            'EvidenceID', {})
        Stream_ = []                            % env-owned RandStream, reseeded by reset
        Seed_ double = []
        Context_ double = 0                     % context index of the round about to be played
        Frame_ = []                             % current DecisionFrame
    end

    methods
        function obj = DelegationEnv(options)
            arguments
                options.Roster
                options.Roles = []
                options.Providers
                options.Contexts
                options.ContextProbs = []
                options.Reliability
                options.Schedule = []
                options.Payoffs
                options.ObstructedPayoff (1,1) double {mustBeFinite} = 0
                options.ObstructionProb (1,1) double {mustBeBetween(options.ObstructionProb, 0, 1)} = 0
                options.Visibility (1,1) string {mustBeMember(options.Visibility, ["own" "public"])} = "own"
                options.Reports = []
                options.Rounds (1,1) double {mustBeInteger, mustBePositive} = 1
                options.RunTag = []
            end
            missing = setdiff(["Roster" "Providers" "Contexts" "Reliability" "Payoffs"], string(fieldnames(options))');
            if ~isempty(missing)
                bad_('DelegationEnv requires: %s.', strjoin(missing, ', '));
            end
            obj.Roster = idRow_(options.Roster, 'Roster');
            tag = options.RunTag;
            if isempty(tag)
                u = char(java.util.UUID.randomUUID().toString());      % not MATLAB's RNG: no stream is touched
                tag = "W" + string(u(1:8)) + string(u(10:13));
            end
            tag = string(tag);
            if ~isscalar(tag) || strlength(tag) == 0 || contains(tag, ["/" "@"])
                bad_('RunTag must be a non-empty string without "/" or "@".');
            end
            obj.RunTag = tag;
            n = numel(obj.Roster);
            roles = options.Roles;
            if isempty(roles), roles = repmat("delegator", 1, n); end
            roles = convertCharsToStrings(roles);
            if ~(isstring(roles) && numel(roles) == n), bad_('Roles must have one entry per roster slot.'); end
            obj.Roles = reshape(roles, 1, []);
            obj.Providers = idRow_(options.Providers, 'Providers');
            if any(obj.Providers == "outside"), bad_('"outside" is reserved for the outside option.'); end
            obj.Contexts = idRow_(options.Contexts, 'Contexts');
            p = numel(obj.Providers);
            c = numel(obj.Contexts);

            probs = options.ContextProbs;
            if isempty(probs), probs = ones(1, c) / c; end
            if ~(isnumeric(probs) && numel(probs) == c && all(isfinite(probs)) && all(probs >= 0)) ...
                    || abs(sum(probs) - 1) > 1e-12
                bad_('ContextProbs must be %d non-negative finite values summing to 1.', c);
            end
            obj.ContextProbs = reshape(double(probs), 1, []);

            obj.Reliability_ = reliability_(options.Reliability, p, c, 'Reliability');
            obj.Schedule_ = schedule_(options.Schedule, p, c);
            obj.Payoffs = payoffs_(options.Payoffs, obj.Contexts);
            obj.ObstructedPayoff = options.ObstructedPayoff;
            obj.ObstructionProb = options.ObstructionProb;
            obj.Visibility = options.Visibility;
            obj.Rounds = options.Rounds;
            obj.Reports_ = reports_(options.Reports, obj.Rounds);
        end

        function spec = describe(obj)
            % DESCRIBE  EnvSpec: simultaneous timing, Discrete(P+1) actions, Box([0 1], [C T+1]) observations.
            n = numel(obj.Roster);
            obsSpace = OceanStats.RL.Env.Spaces.Box([0 1], [numel(obj.Contexts) obj.Rounds + 1]);
            actSpace = OceanStats.RL.Env.Spaces.Discrete(numel(obj.Providers) + 1);
            spec = OceanAgents.Core.EnvSpec('Name', "Delegation", 'Timing', "simultaneous", 'Roster', obj.Roster, ...
                'Roles', obj.Roles, 'ObservationSpaces', repmat({obsSpace}, 1, n), ...
                'ActionSpaces', repmat({actSpace}, 1, n), 'Capabilities', OceanAgents.Core.CapabilitySet(), ...
                'Horizon', obj.Rounds);
        end

        function frame = reset(obj, seed)
            % RESET  Reseed the owned stream, draw round 1's context, and issue the initial frame (with round-1 reports).
            arguments
                obj
                seed (1,1) double {mustBeInteger, mustBeNonnegative}
            end
            if obj.Status == "Closed"
                error('OceanAgents:Env:NotReady', 'The environment has been closed.');
            end
            obj.Stream_ = RandStream('mrg32k3a', 'Seed', seed);
            obj.Seed_ = seed;
            obj.EpisodeID = obj.EpisodeID + 1;
            obj.Round = 0;
            obj.Context_ = obj.drawContext_();
            n = numel(obj.Roster);
            frame = obj.makeFrame_(zeros(1, n), false(1, n), false, cell(1, n), struct());
            obj.Status = "Running";
        end

        function frame = step(obj, batch)
            % STEP  Validate the whole batch, then resolve one round in slot order.
            if obj.Status ~= "Running"
                error('OceanAgents:Env:NotReady', 'step called while the environment is %s; call reset(seed) first.', ...
                    obj.Status);
            end
            if ~(isa(batch, 'OceanAgents.Core.ActionBatch') && isscalar(batch))
                error('OceanAgents:Core:InvalidRecord', 'step expects a scalar OceanAgents.Core.ActionBatch.');
            end
            batch.validate(obj.Frame_);                 % throws before any draw or state change

            n = numel(obj.Roster);
            p = numel(obj.Providers);
            t = obj.Round + 1;
            c = obj.Context_;
            ctx = obj.Contexts(c);
            pay = obj.Payoffs.(char(ctx));
            R = obj.reliabilityAt_(t);
            actions = zeros(1, n);
            rewards = zeros(1, n);
            outcomes = repmat("outside", 1, n);
            u1 = zeros(1, n);                           % 0 where no draw was made (see Drawn)
            u2 = zeros(1, n);
            drawn = zeros(1, n);                        % number of uniforms drawn for the slot (0, 1 or 2)
            direct = OceanAgents.Core.EvidenceRecord.empty(1, 0);
            owner = zeros(1, 0);
            for k = 1:n                                  % slot order, never decide order
                a = batch.decisionFor(obj.Roster(k)).Action;
                actions(k) = a;
                if a == p + 1
                    rewards(k) = pay(3);
                    continue
                end
                u1(k) = rand(obj.Stream_);
                drawn(k) = 1;
                if u1(k) < obj.ObstructionProb
                    outcomes(k) = "unknown";
                    rewards(k) = obj.ObstructedPayoff;
                else
                    u2(k) = rand(obj.Stream_);
                    drawn(k) = 2;
                    if u2(k) < R(a, c)
                        outcomes(k) = "success";
                        rewards(k) = pay(1);
                    else
                        outcomes(k) = "failure";
                        rewards(k) = pay(2);
                    end
                end
                direct(end + 1) = OceanAgents.Core.EvidenceRecord( ...
                    'EvidenceID', sprintf("%s/E%d-%d-%s", obj.RunTag, obj.EpisodeID, t, obj.Roster(k)), ...
                    'SubjectID', obj.Providers(a), 'Context', ctx, 'Quantity', "completion", ...
                    'Outcome', outcomes(k), 'SourceKind', "direct", 'SourceID', obj.Roster(k), ...
                    'EpisodeID', obj.EpisodeID, 'Tick', t); %#ok<AGROW>
                owner(end + 1) = k; %#ok<AGROW>
            end

            done = t >= obj.Rounds;
            nextContext = 0;
            if ~done
                nextContext = obj.drawContext_();
            end
            % commit
            obj.Round = t;
            obj.Context_ = nextContext;
            carried = cell(1, n);
            for k = 1:n
                if obj.Visibility == "public"
                    carried{k} = direct;
                else
                    carried{k} = direct(owner == k);
                end
            end
            resolved = struct('Round', t, 'Context', c, 'ContextName', ctx, 'Reliability', R, 'Actions', actions, ...
                'Outcomes', outcomes, 'U1', u1, 'U2', u2, 'Drawn', drawn);
            frame = obj.makeFrame_(rewards, true(1, n), done, carried, resolved);
            if done
                obj.Status = "Ended";
            end
        end

        function close(obj)
            % CLOSE  Idempotent; no further reset/step is allowed.
            obj.Status = "Closed";
            obj.Frame_ = [];
            obj.Stream_ = [];
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = {sprintf('%d delegators, %d providers, contexts [%s], %d rounds, visibility "%s"', ...
                numel(obj.Roster), numel(obj.Providers), strjoin(obj.Contexts, ', '), obj.Rounds, obj.Visibility), ...
                sprintf('obstruction probability %g (outcome "unknown", reward %g); %d scheduled reliability change(s); %d scripted report(s)', ...
                obj.ObstructionProb, obj.ObstructedPayoff, numel(obj.Schedule_), numel(obj.Reports_)), ...
                'reliabilities are hidden truth: reported only in DecisionFrame.ResearchInfo'};
            meta.IsSymbolic = false;
            meta.Reference = "repeated delegation world (A2 contract section 5)";
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.ComponentName = "DelegationEnv";
            spec.Inputs = struct('Name', "batch", 'Domain', "Agent-ActionBatch", ...
                'Dtype', "OceanAgents.Core.ActionBatch", 'Shape', "1x1");
            spec.Outputs = struct('Name', "frame", 'Domain', "Agent-DecisionFrame", ...
                'Dtype', "OceanAgents.Core.DecisionFrame", 'Shape', string(mat2str([1 numel(obj.Roster)])));
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "delegationenv");
            spec.ComposesFrom = {};
        end
    end

    methods (Access = private)
        function c = drawContext_(obj)
            % Inverse CDF with one draw from the owned stream.
            u = rand(obj.Stream_);
            c = find(u < cumsum(obj.ContextProbs), 1);
            if isempty(c), c = find(obj.ContextProbs > 0, 1, 'last'); end   % cumsum rounding just below 1
        end

        function R = reliabilityAt_(obj, t)
            R = obj.Reliability_;
            for s = obj.Schedule_
                if s.FromRound <= t, R = s.Reliability; end
            end
        end

        function ev = reportsFor_(obj, t)
            ev = OceanAgents.Core.EvidenceRecord.empty(1, 0);
            for r = obj.Reports_([obj.Reports_.Round] == t)
                ev(end + 1) = OceanAgents.Core.EvidenceRecord( ...
                    'EvidenceID', obj.RunTag + "/" + r.EvidenceID + "@" + obj.EpisodeID, 'SubjectID', r.SubjectID, ...
                    'Context', r.Context, 'Quantity', "completion", 'Outcome', r.Outcome, 'SourceKind', "report", ...
                    'SourceID', r.ReporterID, 'EpisodeID', obj.EpisodeID, 'Tick', t); %#ok<AGROW>
            end
        end

        function frame = makeFrame_(obj, rewards, rewardValid, done, carried, resolved)
            n = numel(obj.Roster);
            labels = 1:numel(obj.Providers) + 1;
            obj.FrameID = obj.FrameID + 1;
            if done
                obs = [0 obj.Round + 1];
                ctxName = "";
                reports = OceanAgents.Core.EvidenceRecord.empty(1, 0);
            else
                obs = [obj.Context_ obj.Round + 1];
                ctxName = obj.Contexts(obj.Context_);
                reports = obj.reportsFor_(obj.Round + 1);
            end
            info = struct('context', ctxName, 'action_subjects', [obj.Providers "outside"]);
            views = cell(1, n);
            for k = 1:n
                ev = carried{k};
                if isempty(ev), ev = OceanAgents.Core.EvidenceRecord.empty(1, 0); end
                views{k} = OceanAgents.Core.ActorView('AgentID', obj.Roster(k), 'EpisodeID', obj.EpisodeID, ...
                    'FrameID', obj.FrameID, 'Tick', obj.Round, 'Observation', obs, 'ActionLabels', labels, ...
                    'ActionMask', repmat(~done, 1, numel(labels)), 'Info', info, 'Evidence', [ev reports]);
            end
            live = repmat(~done, 1, n);
            endReason = "";
            if done, endReason = "horizon"; end
            research = struct('Seed', obj.Seed_, 'Round', obj.Round, 'NextContext', obj.Context_, ...
                'Reliability', obj.Reliability_, 'Schedule', obj.Schedule_, ...
                'NextReliability', obj.reliabilityAt_(obj.Round + 1), 'Resolved', resolved);
            frame = OceanAgents.Core.DecisionFrame('EpisodeID', obj.EpisodeID, 'FrameID', obj.FrameID, ...
                'Tick', obj.Round, 'Roster', obj.Roster, 'Live', live, 'Requested', live, 'Views', views, ...
                'Rewards', rewards, 'RewardValid', rewardValid, 'Terminated', repmat(done, 1, n), ...
                'Truncated', false(1, n), 'EpisodeEnded', done, 'EndReason', endReason, 'ResearchInfo', research);
            obj.Frame_ = frame;
        end
    end
end

% ---- construction helpers (configuration errors use OceanAgents:Core:InvalidRecord)
function bad_(fmt, varargin)
error('OceanAgents:Core:InvalidRecord', fmt, varargin{:});
end

function s = idRow_(s, name)
s = convertCharsToStrings(s);
if ~(isstring(s) && ~isempty(s)) || any(ismissing(s)) || any(strlength(s) == 0) || numel(unique(s)) ~= numel(s)
    bad_('%s must be a non-empty row of unique non-empty strings.', name);
end
s = reshape(s, 1, []);
end

function R = reliability_(R, p, c, name)
if ~(isnumeric(R) && isequal(size(R), [p c]) && all(isfinite(R(:))) && all(R(:) >= 0) && all(R(:) <= 1))
    bad_('%s must be a %d x %d matrix of probabilities (providers x contexts).', name, p, c);
end
R = double(R);
end

function s = schedule_(s, p, c)
if isempty(s)
    s = struct('FromRound', {}, 'Reliability', {});
    return
end
if ~(isstruct(s) && all(isfield(s, {'FromRound', 'Reliability'})))
    bad_('Schedule must be a struct array with fields FromRound and Reliability.');
end
s = reshape(s, 1, []);
out = struct('FromRound', {}, 'Reliability', {});
for j = 1:numel(s)
    f = s(j).FromRound;
    if ~(isnumeric(f) && isscalar(f) && f == fix(f) && f >= 1)
        bad_('Schedule(%d).FromRound must be a positive integer.', j);
    end
    out(j).FromRound = double(f);
    out(j).Reliability = reliability_(s(j).Reliability, p, c, sprintf('Schedule(%d).Reliability', j));
end
[from, order] = sort([out.FromRound]);
if numel(unique(from)) ~= numel(from), bad_('Schedule FromRound values must be distinct.'); end
s = out(order);
end

function pay = payoffs_(pay, contexts)
if ~(isstruct(pay) && isscalar(pay))
    bad_('Payoffs must be a scalar struct keyed by context name.');
end
for ctx = contexts
    if ~isvarname(ctx), bad_('Context "%s" is not a valid struct field name.', ctx); end
    if ~isfield(pay, ctx), bad_('Payoffs has no entry for context "%s".', ctx); end
    v = pay.(char(ctx));
    if ~(isnumeric(v) && numel(v) == 3 && all(isfinite(v)))
        bad_('Payoffs.%s must be [Success Failure Outside] (3 finite values).', ctx);
    end
    pay.(char(ctx)) = reshape(double(v), 1, 3);
end
extra = setdiff(string(fieldnames(pay))', contexts);
if ~isempty(extra), bad_('Payoffs has entries for unknown context(s): %s.', strjoin(extra, ', ')); end
end

function r = reports_(r, rounds)
fields = {'Round', 'ReporterID', 'SubjectID', 'Context', 'Outcome', 'EvidenceID'};
if isempty(r)
    r = cell2struct(cell(numel(fields), 0), fields, 1)';
    return
end
if ~(isstruct(r) && all(isfield(r, fields)))
    bad_('Reports must be a struct array with fields %s.', strjoin(fields, ', '));
end
r = reshape(r, 1, []);
for j = 1:numel(r)
    t = r(j).Round;
    if ~(isnumeric(t) && isscalar(t) && t == fix(t) && t >= 1 && t <= rounds)
        bad_('Reports(%d).Round must be an integer in 1..%d.', j, rounds);
    end
    for f = fields(2:end)
        r(j).(f{1}) = convertCharsToStrings(r(j).(f{1}));
    end
    if ~isempty(regexp(r(j).EvidenceID, '^E\d+-\d+-', 'once'))
        bad_('Reports(%d).EvidenceID must not use the direct-evidence form "E<episode>-<round>-<agent>".', j);
    end
    % validate the remaining fields exactly as the delivered record will be validated
    OceanAgents.Core.EvidenceRecord('EvidenceID', r(j).EvidenceID, 'SubjectID', r(j).SubjectID, ...
        'Context', r(j).Context, 'Outcome', r(j).Outcome, 'SourceKind', "report", 'SourceID', r(j).ReporterID, ...
        'EpisodeID', 1, 'Tick', t);
end
ids = [r.EvidenceID];
if numel(unique(ids)) ~= numel(ids), bad_('Reports must have distinct EvidenceIDs.'); end
end
