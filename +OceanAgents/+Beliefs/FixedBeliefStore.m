classdef FixedBeliefStore < OceanAgents.Beliefs.IBeliefStore
    % FIXEDBELIEFSTORE  Imported, never-fitted scores: the belief memory of a "fixed-belief-reactive" mind.
    % Each (subject, context) has one declared score with its scale and an explicit conversion to a probability.
    % Evidence is seen and counted (so every event is still consumed exactly once) but NEVER changes an estimate.
    %
    %   scores = struct('SubjectID', "p1", 'Context', "cheap", 'Score', 80, 'ScaleMin', 0, 'ScaleMax', 100, ...
    %                   'Conversion', "linear-scale", 'SourceID', "config");
    %   s = OceanAgents.Beliefs.FixedBeliefStore(scores, 'Default', []);
    %   r = s.belief("p1", "cheap")       % Estimate 0.8, RawScore 80, Conversion "linear-scale", Variance []
    %   d = s.observe(evidence)           % eligible direct evidence -> "recorded-only"; the estimate is unchanged
    %
    % Conversion "none" means the score already IS a probability and requires the scale [0 1]; "linear-scale" maps
    % Estimate = (Score - ScaleMin) / (ScaleMax - ScaleMin). A score outside its scale, or a 0-100 score declared
    % "none", raises OceanAgents:Beliefs:BadImport (a 0-100 score is not a probability, AP07).
    % Unknown (subject, context): OceanAgents:Beliefs:UnknownSubject when 'Default' is empty, else an import record with
    % Estimate = Default and SourceID "default".
    % Persist "episode" resets only the disposition counts at endEpisode (estimates never change).
    % Options 'MaxEvidence' / 'MaxSubjects' bound the seen-ID set and the imported subject table as in every store.
    % See Planning/System/Game Theory/A2_Beliefs_Contract.md sections 3.2, 3.3 and 8 (H10, H11).

    properties (SetAccess = private)
        Scores struct                   % 1 x S normalised import table (as given, strings/doubles)
        Default double = []             % [] or a probability used for unknown (subject, context)
        Quantity (1,1) string = "completion"
        Persist (1,1) string = "run"
        MaxEvidence (1,1) double = 1e5
        MaxSubjects (1,1) double = 1e4
        UpdatesEnabled (1,1) logical = true
    end

    properties (Access = private)
        Estimates_ double = zeros(1, 0) % converted probability per Scores entry
        Index_                          % dictionary key -> index into Scores
        Seen_                           % dictionary EvidenceID -> canonicalKey
        Counts_ struct
    end

    methods
        function obj = FixedBeliefStore(scores, options)
            arguments
                scores = []
                options.Quantity (1,1) string {mustBeNonzeroLengthText} = "completion"
                options.Default = []
                options.Persist (1,1) string {mustBeMember(options.Persist, ["run", "episode"])} = "run"
                options.MaxEvidence (1,1) double {mustBeInteger, mustBePositive} = 1e5
                options.MaxSubjects (1,1) double {mustBeInteger, mustBePositive} = 1e4
            end
            dflt = options.Default;
            if ~isempty(dflt) && ~(isnumeric(dflt) && isreal(dflt) && isscalar(dflt) && isfinite(dflt) && ...
                    dflt >= 0 && dflt <= 1)
                error('OceanAgents:Beliefs:BadImport', 'Default must be empty or a probability in [0, 1].');
            end
            [obj.Scores, obj.Estimates_] = normaliseScores_(scores);
            if numel(unique([obj.Scores.SubjectID])) > options.MaxSubjects
                error('OceanAgents:Beliefs:CapacityExceeded', 'The scores name %d subjects; MaxSubjects is %d.', ...
                    numel(unique([obj.Scores.SubjectID])), options.MaxSubjects);
            end
            obj.Default = double(dflt);
            obj.Quantity = options.Quantity;
            obj.Persist = options.Persist;
            obj.MaxEvidence = options.MaxEvidence;
            obj.MaxSubjects = options.MaxSubjects;
            obj.Index_ = dictionary(string.empty(0, 1), zeros(0, 1));
            for k = 1:numel(obj.Scores)
                obj.Index_(key_(obj.Scores(k).SubjectID, obj.Scores(k).Context)) = k;
            end
            obj.Seen_ = seenState_();
            obj.Counts_ = dispositionCounts_();
        end

        function d = describe(obj)
            d = struct('Name', "FixedBeliefStore", 'UpdateRule', "fixed", 'Quantity', obj.Quantity, ...
                'Persist', obj.Persist, 'UpdatesEnabled', obj.UpdatesEnabled, 'NumScores', numel(obj.Scores), ...
                'HasDefault', ~isempty(obj.Default));
        end

        function rec = belief(obj, subjectID, context)
            % BELIEF  The imported score of (subjectID, context) as a probability, with its raw score and scale.
            arguments
                obj
                subjectID (1,1) string {mustBeNonzeroLengthText}
                context (1,1) string {mustBeNonzeroLengthText}
            end
            key = key_(subjectID, context);
            if isKey(obj.Index_, key)
                k = obj.Index_(key);
                s = obj.Scores(k);
                est = obj.Estimates_(k);
                raw = s.Score;  lo = s.ScaleMin;  hi = s.ScaleMax;  conv = s.Conversion;  src = s.SourceID;
            elseif ~isempty(obj.Default)
                est = obj.Default;
                raw = obj.Default;  lo = 0;  hi = 1;  conv = "none";  src = "default";
            else
                error('OceanAgents:Beliefs:UnknownSubject', 'No imported score for subject "%s" in context "%s" and no Default.', ...
                    subjectID, context);
            end
            rec = OceanAgents.Beliefs.BeliefRecord('SubjectID', subjectID, 'Context', context, ...
                'Quantity', obj.Quantity, 'Estimate', est, 'EvidenceMass', 0, 'SourceKind', "import", ...
                'SourceID', src, 'RawScore', raw, 'ScaleMin', lo, 'ScaleMax', hi, 'Conversion', conv, ...
                'UpdateRule', "fixed", 'Version', 0);
        end

        function dispositions = observe(obj, evidence)
            % OBSERVE  Classify each record in order; eligible direct evidence is "recorded-only". All-or-nothing.
            evidence = evidenceList_(evidence);
            [dispositions, seen] = classifyEvidence_(evidence, obj.Seen_, obj.Quantity, obj.UpdatesEnabled, ...
                "recorded-only", obj.MaxEvidence);
            obj.Seen_ = seen;
            obj.Counts_ = dispositionCounts_(obj.Counts_, dispositions);
        end

        function v = version(~)
            % VERSION  Always 0: a fixed store never applies an update.
            v = 0;
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
            % ENDEPISODE  Persist "episode": reset the recorded-evidence (disposition) counts; seen IDs are kept.
            if obj.Persist == "episode"
                obj.Counts_ = dispositionCounts_();
            end
        end

        function s = exportState(obj)
            [ids, ks] = seenState_(obj.Seen_);
            s = struct('Name', "FixedBeliefStore", 'Config', obj.config_(), 'UpdatesEnabled', obj.UpdatesEnabled, ...
                'Counts', obj.Counts_, 'SeenIDs', ids, 'SeenKeys', ks);
        end

        function importState(obj, s)
            % IMPORTSTATE  Restore seen IDs, counts and the update switch (validated first; nothing changes on error).
            % The store must have been built with the same scores and options.
            arguments
                obj
                s (1,1) struct
            end
            need = {'Name', 'Config', 'UpdatesEnabled', 'Counts', 'SeenIDs', 'SeenKeys'};
            if ~all(isfield(s, need)) || ~isequal(s.Name, "FixedBeliefStore")
                error('OceanAgents:Beliefs:BadImport', 'State is not a FixedBeliefStore state.');
            end
            if ~isequal(s.Config, obj.config_())
                error('OceanAgents:Beliefs:BadImport', 'State was exported by a store with different scores or options.');
            end
            OceanAgents.Core.assertValueData(s, "state");
            if ~(islogical(s.UpdatesEnabled) && isscalar(s.UpdatesEnabled))
                error('OceanAgents:Beliefs:BadImport', 'UpdatesEnabled must be a logical scalar.');
            end
            n = validateCounts_(s.Counts);
            seen = seenState_(s.SeenIDs, s.SeenKeys, obj.MaxEvidence);
            obj.UpdatesEnabled = s.UpdatesEnabled;
            obj.Counts_ = n;
            obj.Seen_ = seen;
        end

        function meta = getMetadata(obj)
            % GETMETADATA  Pillar 1: a container of imported scores; no formula (IsSymbolic = false).
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = struct('Symbol', {}, 'Description', {});
            meta.Assumptions = {sprintf('%d imported (subject, context) scores; never fitted', numel(obj.Scores)), ...
                'conversion to a probability is declared per score ("none" needs scale [0 1], or "linear-scale")'};
            meta.IsSymbolic = false;
            meta.Reference = "fixed imported scores (A2_Beliefs_Contract.md section 3.3)";
        end

        function spec = getPipelineSpec(~)
            spec = struct();
            spec.ComponentName = "FixedBeliefStore";
            spec.Inputs = struct('Name', "evidence", 'Domain', "Agent-Evidence", 'Dtype', "OceanAgents.Core.EvidenceRecord", 'Shape', "1 x K");
            spec.Outputs = struct('Name', "belief", 'Domain', "Agent-Belief", 'Dtype', "OceanAgents.Beliefs.BeliefRecord", 'Shape', "1 x 1");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "fixedbelief");
            spec.ComposesFrom = {};
        end
    end

    methods (Access = private)
        function c = config_(obj)
            c = struct('Scores', obj.Scores, 'Default', obj.Default, 'Quantity', obj.Quantity, ...
                'Persist', obj.Persist, 'MaxEvidence', obj.MaxEvidence, 'MaxSubjects', obj.MaxSubjects);
        end
    end
end

function k = key_(subjectID, context)
k = subjectID + string(char(31)) + context;
end

function [out, est] = normaliseScores_(scores)
% Validate the import table (errors OceanAgents:Beliefs:BadImport) and convert each score to a probability.
template = struct('SubjectID', {}, 'Context', {}, 'Score', {}, 'ScaleMin', {}, 'ScaleMax', {}, ...
    'Conversion', {}, 'SourceID', {});
out = reshape(template, 1, 0);
est = zeros(1, 0);
if isempty(scores), return; end
bad = @(msg) error('OceanAgents:Beliefs:BadImport', 'Scores: %s', msg);
if ~isstruct(scores) || ~isempty(setxor(fieldnames(scores), fieldnames(template)))
    bad('must be a struct array with fields SubjectID, Context, Score, ScaleMin, ScaleMax, Conversion, SourceID.');
end
for k = 1:numel(scores)
    s = scores(k);
    try
        id = string(s.SubjectID);  ctx = string(s.Context);  conv = string(s.Conversion);  src = string(s.SourceID);
    catch
        bad(sprintf('entry %d has non-text identifiers.', k));
    end
    if ~all([isscalar(id) isscalar(ctx) isscalar(conv) isscalar(src)]) || any(ismissing([id ctx conv src])) || ...
            id == "" || ctx == ""
        bad(sprintf('entry %d needs non-empty scalar SubjectID and Context, and scalar Conversion and SourceID.', k));
    end
    x = s.Score;  lo = s.ScaleMin;  hi = s.ScaleMax;
    if ~all(cellfun(@(v) isnumeric(v) && isreal(v) && isscalar(v) && isfinite(v), {x, lo, hi})) || ~(lo < hi)
        bad(sprintf('entry %d (%s, %s) needs finite scalar Score and a scale with ScaleMin < ScaleMax.', k, id, ctx));
    end
    if x < lo || x > hi
        bad(sprintf('score %g of (%s, %s) lies outside its declared scale [%g %g].', x, id, ctx, lo, hi));
    end
    switch conv
        case "none"
            if lo ~= 0 || hi ~= 1
                bad(sprintf(['(%s, %s) declares Conversion "none" on scale [%g %g]: only a [0 1] score is a ' ...
                    'probability; declare "linear-scale" to convert it.'], id, ctx, lo, hi));
            end
            p = double(x);
        case "linear-scale"
            p = (double(x) - double(lo)) / (double(hi) - double(lo));
        otherwise
            bad(sprintf('(%s, %s) has unknown Conversion "%s" (use "none" or "linear-scale").', id, ctx, conv));
    end
    out(k) = struct('SubjectID', id, 'Context', ctx, 'Score', double(x), 'ScaleMin', double(lo), ...
        'ScaleMax', double(hi), 'Conversion', conv, 'SourceID', src);
    est(k) = p;
end
keys = key_([out.SubjectID], [out.Context]);
if numel(unique(keys)) ~= numel(keys)
    bad('each (SubjectID, Context) may appear only once.');
end
end
