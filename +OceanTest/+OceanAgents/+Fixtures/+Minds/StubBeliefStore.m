classdef StubBeliefStore < OceanAgents.Beliefs.IBeliefStore
    % STUBBELIEFSTORE  Test fixture belief store (not a real estimator): returns the BeliefRecords a test sets and
    % logs every call, so TrustMind can be tested without the real stores.
    %
    %   s = StubBeliefStore("beta-bernoulli");        % or "fixed"
    %   s.setBeta("p1", "cheap", 3, 2);               % Beta(3,2) record (mean 0.6, var 0.04)
    %   s.setFixed("p2", "cheap", 0.8);               % fixed-score record (no variance)
    %   'CountOutcomes', true: observe() adds direct success/failure outcomes to the Beta counts of an existing
    %   (subject, context) record (a minimal Beta-Bernoulli update, alpha + 1 / beta + 1), so a test can show that
    %   evidence fed through the mind changes its next decision. Duplicate IDs are reported "duplicate".
    properties (SetAccess = private)
        UpdateRule string = "fixed"
        CountOutcomes logical = false
        ObservedIDs string = strings(1, 0)     % every EvidenceID handed to observe(), in order (duplicates included)
        ObserveCalls double = 0
        BeliefCalls string = strings(1, 0)     % "subject|context" of every belief() call
        BeginCalls double = 0
        EndCalls double = 0
        Version_ double = 0
        Records_                               % containers.Map "subject|context" -> BeliefRecord
        Seen_ string = strings(1, 0)
        Enabled_ logical = true
    end
    methods
        function obj = StubBeliefStore(updateRule, options)
            arguments
                updateRule (1,1) string = "fixed"
                options.CountOutcomes (1,1) logical = false
            end
            obj.UpdateRule = updateRule;
            obj.CountOutcomes = options.CountOutcomes;
            obj.Records_ = containers.Map('KeyType', 'char', 'ValueType', 'any');
        end

        function setBeta(obj, subject, context, a, b, mass)
            if nargin < 6, mass = 0; end
            kind = "prior";
            if mass > 0, kind = "direct"; end
            obj.Records_(char(subject + "|" + context)) = OceanAgents.Beliefs.BeliefRecord('SubjectID', subject, ...
                'Context', context, 'Estimate', a / (a + b), 'Variance', a * b / ((a + b)^2 * (a + b + 1)), ...
                'Alpha', a, 'Beta', b, 'EvidenceMass', mass, 'SourceKind', kind, 'UpdateRule', "beta-bernoulli", ...
                'Version', obj.Version_);
        end

        function setFixed(obj, subject, context, p)
            obj.Records_(char(subject + "|" + context)) = OceanAgents.Beliefs.BeliefRecord('SubjectID', subject, ...
                'Context', context, 'Estimate', p, 'SourceKind', "import", 'SourceID', "test", 'UpdateRule', "fixed");
        end

        function d = describe(obj)
            d = struct('Name', "StubBeliefStore", 'UpdateRule', obj.UpdateRule, 'Quantity', "completion", ...
                'Persist', "run", 'UpdatesEnabled', obj.Enabled_);
        end

        function rec = belief(obj, subjectID, context)
            key = string(subjectID) + "|" + string(context);
            obj.BeliefCalls(end + 1) = key;
            if ~isKey(obj.Records_, char(key))
                error('OceanAgents:Beliefs:UnknownSubject', 'Stub has no belief for %s.', key);
            end
            rec = obj.Records_(char(key));
        end

        function disp = observe(obj, evidence)
            obj.ObserveCalls = obj.ObserveCalls + 1;
            disp = strings(1, numel(evidence));
            for k = 1:numel(evidence)
                e = evidence(k);
                obj.ObservedIDs(end + 1) = e.EvidenceID;
                if any(obj.Seen_ == e.EvidenceID), disp(k) = "duplicate"; continue; end
                obj.Seen_(end + 1) = e.EvidenceID;
                key = char(e.SubjectID + "|" + e.Context);
                if obj.CountOutcomes && obj.Enabled_ && e.SourceKind == "direct" && e.Outcome ~= "unknown" ...
                        && isKey(obj.Records_, key)
                    r = obj.Records_(key);
                    a = r.Alpha + (e.Outcome == "success");
                    b = r.Beta + (e.Outcome == "failure");
                    obj.Version_ = obj.Version_ + 1;
                    obj.setBeta(e.SubjectID, e.Context, a, b, r.EvidenceMass + 1);
                    disp(k) = "applied";
                else
                    disp(k) = "recorded-only";
                end
            end
        end

        function v = version(obj)
            v = obj.Version_;
        end

        function n = counts(obj)
            n = struct('Applied', obj.Version_, 'Duplicate', 0, 'Unknown', 0, 'Report', 0, 'Ineligible', 0, ...
                'Frozen', 0, 'RecordedOnly', 0);
        end

        function setUpdatesEnabled(obj, tf)
            obj.Enabled_ = tf;
        end

        function beginEpisode(obj)
            obj.BeginCalls = obj.BeginCalls + 1;
        end

        function endEpisode(obj)
            obj.EndCalls = obj.EndCalls + 1;
        end

        function s = exportState(obj)
            s = struct('Version', obj.Version_, 'Seen', obj.Seen_);
        end

        function importState(obj, s)
            obj.Version_ = s.Version;
            obj.Seen_ = s.Seen;
        end
    end
end
