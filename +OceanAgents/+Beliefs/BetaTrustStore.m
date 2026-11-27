classdef BetaTrustStore < OceanAgents.Beliefs.IBeliefStore
    % BETATRUSTSTORE  Evidence-adaptive trust: one Beta-Bernoulli belief per (subject, context) about the probability
    % that the subject completes a delegated task, updated only from eligible DIRECT evidence.
    %
    %   success:  alpha <- alpha + 1          failure:  beta <- beta + 1
    %   Estimate = alpha / (alpha + beta)     Variance = alpha beta / ((alpha + beta)^2 (alpha + beta + 1))
    %
    %   s = OceanAgents.Beliefs.BetaTrustStore('Prior', [1 1]);
    %   d = s.observe(view.Evidence)       % 1 x K dispositions, e.g. ["applied" "duplicate" "unknown"]
    %   r = s.belief("p1", "cheap")        % BeliefRecord: Estimate, Variance, Alpha, Beta, EvidenceMass, ...
    %   s.version()                        % +1 per applied update
    %
    % What does NOT move the posterior (each still consumes its EvidenceID exactly once): a duplicate delivery, a
    % different Quantity ("ineligible"), a report ("report"; indirect reputation is out of scope in A2), an unknown
    % outcome ("unknown"; never converted to failure), and any event while updates are disabled ("frozen").
    % Contexts are independent and beliefs are keyed by SubjectID, so a new provider starts from the prior. The
    % variance can RISE after contradicting evidence (Beta(9,1) + failure -> Beta(9,2): 9/1100 -> 3/242).
    %
    % Options: 'Prior' [alpha0 beta0] (default [1 1]); 'Priors' struct array (SubjectID, Context, Alpha, Beta, SourceID)
    % of per-(subject, context) overrides; 'Quantity' ("completion"); 'Persist' "run" | "episode" (episode: posteriors and
    % disposition counts return to the priors at endEpisode, the seen-ID set is kept because IDs are unique per run);
    % 'MaxEvidence' (seen IDs), 'MaxSubjects' (distinct subjects with a stored belief or prior override), 'MaxTrace'
    % (EvidenceIDs kept per belief, most recent last). Exceeding a capacity raises OceanAgents:Beliefs:CapacityExceeded
    % and the offending observe() call changes NOTHING (the whole batch is planned on value copies, then committed).
    % See Planning/System/Game Theory/A2_Beliefs_Contract.md sections 3.2, 3.4 and 8; meta/BetaTrustStore.meta.json.

    properties (SetAccess = private)
        Prior (1,2) double = [1 1]          % default [alpha0 beta0]
        Priors struct                       % 1 x P per-(subject, context) overrides (normalised)
        Quantity (1,1) string = "completion"
        Persist (1,1) string = "run"
        MaxEvidence (1,1) double = 1e5
        MaxSubjects (1,1) double = 1e4
        MaxTrace (1,1) double = 20
        UpdatesEnabled (1,1) logical = true
    end

    properties (Constant)
        DefaultPriorSource = "default-prior"
    end

    properties (Access = private)
        Seen_                   % dictionary EvidenceID -> canonicalKey (value type)
        Entries_                % struct of columns: Key, Subject, Context, Alpha, Beta, Mass, Trace (cell)
        Index_                  % dictionary Key -> row of Entries_
        PriorIndex_             % dictionary Key -> index into Priors
        Subjects_               % dictionary SubjectID -> true, for subjects with an entry or a prior override
        Version_ (1,1) double = 0
        Counts_ struct
    end

    methods
        function obj = BetaTrustStore(options)
            arguments
                options.Prior (1,2) double {mustBePositive, mustBeFinite} = [1 1]
                options.Priors = []
                options.Quantity (1,1) string {mustBeNonzeroLengthText} = "completion"
                options.Persist (1,1) string {mustBeMember(options.Persist, ["run", "episode"])} = "run"
                options.MaxEvidence (1,1) double {mustBeInteger, mustBePositive} = 1e5
                options.MaxSubjects (1,1) double {mustBeInteger, mustBePositive} = 1e4
                options.MaxTrace (1,1) double {mustBeInteger, mustBeNonnegative} = 20
            end
            obj.Prior = options.Prior;
            obj.Priors = normalisePriors_(options.Priors);
            obj.Quantity = options.Quantity;
            obj.Persist = options.Persist;
            obj.MaxEvidence = options.MaxEvidence;
            obj.MaxSubjects = options.MaxSubjects;
            obj.MaxTrace = options.MaxTrace;
            obj.PriorIndex_ = dictionary(string.empty(0, 1), zeros(0, 1));
            for k = 1:numel(obj.Priors)
                obj.PriorIndex_(key_(obj.Priors(k).SubjectID, obj.Priors(k).Context)) = k;
            end
            if numel(unique([obj.Priors.SubjectID])) > obj.MaxSubjects
                error('OceanAgents:Beliefs:CapacityExceeded', 'The Priors name %d subjects; MaxSubjects is %d.', ...
                    numel(unique([obj.Priors.SubjectID])), obj.MaxSubjects);
            end
            obj.Seen_ = seenState_();
            obj.Counts_ = dispositionCounts_();
            obj.resetPosteriors_();
        end

        function d = describe(obj)
            d = struct('Name', "BetaTrustStore", 'UpdateRule', "beta-bernoulli", 'Quantity', obj.Quantity, ...
                'Persist', obj.Persist, 'UpdatesEnabled', obj.UpdatesEnabled, 'Prior', obj.Prior, ...
                'MaxEvidence', obj.MaxEvidence, 'MaxSubjects', obj.MaxSubjects, 'MaxTrace', obj.MaxTrace);
        end

        function rec = belief(obj, subjectID, context)
            % BELIEF  Posterior of (subjectID, context) for the store's Quantity; the prior when nothing was applied.
            arguments
                obj
                subjectID (1,1) string {mustBeNonzeroLengthText}
                context (1,1) string {mustBeNonzeroLengthText}
            end
            key = key_(subjectID, context);
            [a, b, src] = obj.priorFor_(key);
            mass = 0;
            trace = strings(1, 0);
            if isKey(obj.Index_, key)
                r = obj.Index_(key);
                a = obj.Entries_.Alpha(r);
                b = obj.Entries_.Beta(r);
                mass = obj.Entries_.Mass(r);
                trace = obj.Entries_.Trace{r};
            end
            kind = "prior";
            if mass > 0, kind = "direct"; end
            rec = OceanAgents.Beliefs.BeliefRecord('SubjectID', subjectID, 'Context', context, ...
                'Quantity', obj.Quantity, 'Estimate', a / (a + b), ...
                'Variance', a * b / ((a + b)^2 * (a + b + 1)), 'Alpha', a, 'Beta', b, 'EvidenceMass', mass, ...
                'SourceKind', kind, 'SourceID', src, 'EvidenceIDs', trace, 'UpdateRule', "beta-bernoulli", ...
                'Version', obj.Version_);
        end

        function dispositions = observe(obj, evidence)
            % OBSERVE  Classify and (when eligible) apply each record in order. All-or-nothing: the batch is planned on
            % value copies and committed only if no record raised EvidenceConflict or CapacityExceeded.
            evidence = evidenceList_(evidence);
            [dispositions, seen] = classifyEvidence_(evidence, obj.Seen_, obj.Quantity, obj.UpdatesEnabled, ...
                "applied", obj.MaxEvidence);
            E = obj.Entries_;  index = obj.Index_;  subjects = obj.Subjects_;
            for k = find(dispositions == "applied")
                e = evidence(k);
                key = key_(e.SubjectID, e.Context);
                if isKey(index, key)
                    r = index(key);
                else
                    if ~isKey(subjects, e.SubjectID)
                        subjects(e.SubjectID) = true;
                        if numEntries(subjects) > obj.MaxSubjects
                            error('OceanAgents:Beliefs:CapacityExceeded', ...
                                'Subject capacity MaxSubjects = %d exceeded by "%s"; nothing is evicted.', ...
                                obj.MaxSubjects, e.SubjectID);
                        end
                    end
                    [a0, b0] = obj.priorFor_(key);
                    r = numel(E.Key) + 1;
                    E.Key(r, 1) = key;  E.Subject(r, 1) = e.SubjectID;  E.Context(r, 1) = e.Context;
                    E.Alpha(r, 1) = a0;  E.Beta(r, 1) = b0;  E.Mass(r, 1) = 0;  E.Trace{r, 1} = strings(1, 0);
                    index(key) = r;
                end
                if e.Outcome == "success"
                    E.Alpha(r) = E.Alpha(r) + 1;
                else
                    E.Beta(r) = E.Beta(r) + 1;
                end
                E.Mass(r) = E.Mass(r) + 1;
                E.Trace{r} = trimTrace_([E.Trace{r}, e.EvidenceID], obj.MaxTrace);
            end
            % commit (nothing above touched obj)
            obj.Seen_ = seen;
            obj.Entries_ = E;  obj.Index_ = index;  obj.Subjects_ = subjects;
            obj.Version_ = obj.Version_ + nnz(dispositions == "applied");
            obj.Counts_ = dispositionCounts_(obj.Counts_, dispositions);
        end

        function v = version(obj)
            v = obj.Version_;
        end

        function n = counts(obj)
            n = obj.Counts_;
        end

        function setUpdatesEnabled(obj, tf)
            arguments
                obj
                tf (1,1) logical
            end
            obj.UpdatesEnabled = tf;
        end

        function endEpisode(obj)
            % ENDEPISODE  Persist "episode": posteriors and disposition counts back to the priors; seen IDs kept.
            % The version is NOT reset or incremented (it counts applied updates only).
            if obj.Persist == "episode"
                obj.resetPosteriors_();
                obj.Counts_ = dispositionCounts_();
            end
        end

        function s = exportState(obj)
            % EXPORTSTATE  Value-data snapshot (numeric/string copies only).
            [ids, ks] = seenState_(obj.Seen_);
            s = struct('Name', "BetaTrustStore", 'Config', obj.config_(), 'UpdatesEnabled', obj.UpdatesEnabled, ...
                'Version', obj.Version_, 'Counts', obj.Counts_, 'SeenIDs', ids, 'SeenKeys', ks, ...
                'Entries', obj.Entries_);
        end

        function importState(obj, s)
            % IMPORTSTATE  Replace this store's state with s (validated first; on any error nothing changes).
            % The store must have been built with the same configuration (Prior, Priors, Quantity, Persist, capacities).
            arguments
                obj
                s (1,1) struct
            end
            need = {'Name', 'Config', 'UpdatesEnabled', 'Version', 'Counts', 'SeenIDs', 'SeenKeys', 'Entries'};
            if ~all(isfield(s, need)) || ~isequal(s.Name, "BetaTrustStore")
                error('OceanAgents:Beliefs:BadImport', 'State is not a BetaTrustStore state.');
            end
            if ~isequal(s.Config, obj.config_())
                error('OceanAgents:Beliefs:BadImport', 'State was exported by a store with a different configuration.');
            end
            OceanAgents.Core.assertValueData(s, "state");
            if ~(islogical(s.UpdatesEnabled) && isscalar(s.UpdatesEnabled))
                error('OceanAgents:Beliefs:BadImport', 'UpdatesEnabled must be a logical scalar.');
            end
            if ~(isnumeric(s.Version) && isscalar(s.Version) && s.Version >= 0 && s.Version == fix(s.Version))
                error('OceanAgents:Beliefs:BadImport', 'Version must be a non-negative integer.');
            end
            n = validateCounts_(s.Counts);
            seen = seenState_(s.SeenIDs, s.SeenKeys, obj.MaxEvidence);
            [E, index, subjects] = obj.validateEntries_(s.Entries, seen);
            obj.UpdatesEnabled = s.UpdatesEnabled;
            obj.Version_ = double(s.Version);
            obj.Counts_ = n;
            obj.Seen_ = seen;
            obj.Entries_ = E;  obj.Index_ = index;  obj.Subjects_ = subjects;
        end

        function spec = getPipelineSpec(~)
            % GETPIPELINESPEC  Pillar 3: evidence records in, belief records out.
            spec = struct();
            spec.ComponentName = "BetaTrustStore";
            spec.Inputs = struct('Name', "evidence", 'Domain', "Agent-Evidence", 'Dtype', "OceanAgents.Core.EvidenceRecord", 'Shape', "1 x K");
            spec.Outputs = struct('Name', "belief", 'Domain', "Agent-Belief", 'Dtype', "OceanAgents.Beliefs.BeliefRecord", 'Shape', "1 x 1");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "betatrust");
            spec.ComposesFrom = {};
        end
    end

    methods (Static)
        function meta = getMetadata()
            % GETMETADATA  Pillar 1: pedagogical metadata from meta/BetaTrustStore.meta.json plus the OceanAgents pillar
            % fields (SymbolicLatex, Symbols, Assumptions, IsSymbolic, Reference; A2 contract section 7).
            here = fileparts(mfilename('fullpath'));
            meta = jsondecode(fileread(fullfile(here, 'meta', 'BetaTrustStore.meta.json')));
            meta.SymbolicLatex = string(meta.equations.update.latex);
            meta.Assumptions = meta.pedagogy.assumptions;
            meta.IsSymbolic = true;
            meta.Reference = string(meta.textbook.reference);
        end
    end

    methods (Access = private)
        function [a, b, src] = priorFor_(obj, key)
            if isKey(obj.PriorIndex_, key)
                p = obj.Priors(obj.PriorIndex_(key));
                a = p.Alpha;  b = p.Beta;  src = p.SourceID;
            else
                a = obj.Prior(1);  b = obj.Prior(2);  src = obj.DefaultPriorSource;
            end
        end

        function resetPosteriors_(obj)
            obj.Entries_ = emptyEntries_();
            obj.Index_ = dictionary(string.empty(0, 1), zeros(0, 1));
            obj.Subjects_ = dictionary(string.empty(0, 1), false(0, 1));
            for k = 1:numel(obj.Priors)
                obj.Subjects_(obj.Priors(k).SubjectID) = true;
            end
        end

        function c = config_(obj)
            c = struct('Prior', obj.Prior, 'Priors', obj.Priors, 'Quantity', obj.Quantity, 'Persist', obj.Persist, ...
                'MaxEvidence', obj.MaxEvidence, 'MaxSubjects', obj.MaxSubjects, 'MaxTrace', obj.MaxTrace);
        end

        function [E, index, subjects] = validateEntries_(obj, E, seen)
            % Entries must be exactly what observe() could have produced from this store's priors.
            bad = @(msg) error('OceanAgents:Beliefs:BadImport', 'Entries: %s', msg);
            want = fieldnames(emptyEntries_());
            if ~(isstruct(E) && isscalar(E)) || ~isempty(setxor(fieldnames(E), want))
                bad('wrong fields.');
            end
            n = numel(E.Key);
            if ~(isstring(E.Key) && isstring(E.Subject) && isstring(E.Context) && isnumeric(E.Alpha) && ...
                    isnumeric(E.Beta) && isnumeric(E.Mass) && iscell(E.Trace)) || ...
                    ~all(cellfun(@numel, {E.Subject, E.Context, E.Alpha, E.Beta, E.Mass, E.Trace}) == n)
                bad('columns must be string/double/cell columns of equal length.');
            end
            E = structfun(@(c) reshape(c, [], 1), E, 'UniformOutput', false);
            E.Alpha = double(E.Alpha);  E.Beta = double(E.Beta);  E.Mass = double(E.Mass);
            if any(E.Subject == "" | E.Context == "") || numel(unique(E.Key)) ~= n || ...
                    any(E.Key ~= key_(E.Subject, E.Context))
                bad('keys must be unique and match Subject/Context.');
            end
            index = dictionary(string.empty(0, 1), zeros(0, 1));
            for r = 1:n
                [a0, b0] = obj.priorFor_(E.Key(r));
                incA = round(E.Alpha(r) - a0);  incB = round(E.Beta(r) - b0);
                tol = 1e-9 * max(1, E.Alpha(r) + E.Beta(r));
                t = E.Trace{r};
                if ~(isfinite(E.Alpha(r)) && isfinite(E.Beta(r)) && incA >= 0 && incB >= 0 && ...
                        abs(E.Alpha(r) - (a0 + incA)) <= tol && abs(E.Beta(r) - (b0 + incB)) <= tol && ...
                        E.Mass(r) == incA + incB && E.Mass(r) >= 1)
                    bad(sprintf('row %d is not its prior plus whole success/failure counts equal to Mass.', r));
                end
                if ~(isstring(t) && (isempty(t) || isrow(t)) && numel(t) <= min(obj.MaxTrace, E.Mass(r)) && ...
                        all(isKey(seen, t)))
                    bad(sprintf('row %d trace must be at most min(MaxTrace, Mass) seen EvidenceIDs.', r));
                end
                E.Trace{r} = reshape(t, 1, []);
                index(E.Key(r)) = r;
            end
            subjects = dictionary(string.empty(0, 1), false(0, 1));
            allSubjects = unique([E.Subject; reshape([obj.Priors.SubjectID], [], 1)]);
            if numel(allSubjects) > obj.MaxSubjects
                error('OceanAgents:Beliefs:CapacityExceeded', 'State holds %d subjects; MaxSubjects is %d.', ...
                    numel(allSubjects), obj.MaxSubjects);
            end
            if ~isempty(allSubjects), subjects(allSubjects) = true; end
        end
    end
end

function k = key_(subjectID, context)
k = subjectID + string(char(31)) + context;
end

function E = emptyEntries_()
E = struct('Key', strings(0, 1), 'Subject', strings(0, 1), 'Context', strings(0, 1), 'Alpha', zeros(0, 1), ...
    'Beta', zeros(0, 1), 'Mass', zeros(0, 1), 'Trace', {cell(0, 1)});
end

function t = trimTrace_(t, maxTrace)
if numel(t) > maxTrace, t = t(end - maxTrace + 1:end); end
end

function P = normalisePriors_(P)
template = struct('SubjectID', {}, 'Context', {}, 'Alpha', {}, 'Beta', {}, 'SourceID', {});
if isempty(P), P = reshape(template, 1, 0); return; end
bad = @(msg) error('OceanAgents:Beliefs:BadImport', 'Priors: %s', msg);
if ~isstruct(P) || ~isempty(setxor(fieldnames(P), fieldnames(template)))
    bad('must be a struct array with fields SubjectID, Context, Alpha, Beta, SourceID.');
end
out = template;
for k = 1:numel(P)
    p = P(k);
    try
        s = string(p.SubjectID);  c = string(p.Context);  src = string(p.SourceID);
    catch
        bad(sprintf('entry %d has non-text identifiers.', k));
    end
    if ~(isscalar(s) && isscalar(c) && isscalar(src)) || any(ismissing([s c src])) || s == "" || c == ""
        bad(sprintf('entry %d needs non-empty scalar SubjectID and Context and a scalar SourceID.', k));
    end
    a = p.Alpha;  b = p.Beta;
    if ~(isnumeric(a) && isnumeric(b) && isscalar(a) && isscalar(b) && isreal(a) && isreal(b) && ...
            isfinite(a) && isfinite(b) && a > 0 && b > 0)
        bad(sprintf('entry %d needs positive finite scalar Alpha and Beta.', k));
    end
    out(k) = struct('SubjectID', s, 'Context', c, 'Alpha', double(a), 'Beta', double(b), 'SourceID', src);
end
out = reshape(out, 1, []);
keys = key_([out.SubjectID], [out.Context]);
if numel(unique(keys)) ~= numel(keys)
    bad('each (SubjectID, Context) may appear only once.');
end
P = out;
end
