classdef BetaTrustStoreTest < OceanTest.BaseTestCase
    % BETATRUSTSTORETEST  OceanAgents.Beliefs.BetaTrustStore (A2_Beliefs_Contract.md sections 3.2, 3.4, 8 H1-H6).
    % Gates: C42 (variance may rise), C25, AP09-AP11 (dispositions), AP05 (keyed by SubjectID), AP39 (capacity).
    % The posterior is also checked against an independent plain-counting reference over a random evidence stream.

    methods (Static, Access = private)
        function e = ev(id, outcome, varargin)
            % Direct "completion" evidence about p1 in context "cheap" unless overridden.
            e = OceanAgents.Core.EvidenceRecord('EvidenceID', id, 'SubjectID', "p1", 'Context', "cheap", ...
                'Outcome', outcome, 'SourceID', "alice", 'EpisodeID', 1, 'Tick', 1, varargin{:});
        end
    end

    methods (Access = private)
        function verifyBeta(tc, rec, a, b, mass)
            tc.verifyEqual([rec.Alpha rec.Beta rec.EvidenceMass], [a b mass]);
            tc.verifyEqual(rec.Estimate, a / (a + b), 'AbsTol', 1e-15);
            tc.verifyEqual(rec.Variance, a * b / ((a + b)^2 * (a + b + 1)), 'AbsTol', 1e-15);
        end
    end

    methods (Test)
        % ---------------- hand values H1-H6 ----------------
        function h1VarianceRisesAfterContradiction(tc)
            % H1 / C42: Beta(9,1) + one failure = Beta(9,2); mean 9/11; variance 9/1100 -> 3/242 (it RISES).
            s = OceanAgents.Beliefs.BetaTrustStore('Prior', [9 1]);
            before = s.belief("p1", "cheap");
            tc.verifyEqual(before.Variance, 9 / 1100, 'AbsTol', 1e-15);
            tc.verifyEqual(before.SourceKind, "prior");
            tc.verifyEqual(s.observe(tc.ev("E1", "failure")), "applied");
            after = s.belief("p1", "cheap");
            tc.verifyEqual([after.Alpha after.Beta], [9 2]);
            tc.verifyEqual(after.Estimate, 9 / 11, 'AbsTol', 1e-15);
            tc.verifyEqual(after.Variance, 3 / 242, 'AbsTol', 1e-15);
            % exact, statistics-free: 3/242 - 9/1100 = (3*1100 - 9*242)/(242*1100) = 1122/266200 > 0
            tc.verifyGreaterThan(after.Variance - before.Variance, 1122 / 266200 - 1e-15);
            tc.verifyGreaterThan(after.Variance, before.Variance);
        end

        function h2SuccessSuccessFailure(tc)
            s = OceanAgents.Beliefs.BetaTrustStore();
            r0 = s.belief("p1", "cheap");
            tc.verifyBeta(r0, 1, 1, 0);
            tc.verifyEqual([r0.SourceKind r0.SourceID r0.UpdateRule], ["prior" "default-prior" "beta-bernoulli"]);
            d = s.observe([tc.ev("E1", "success"), tc.ev("E2", "success"), tc.ev("E3", "failure")]);
            tc.verifyEqual(d, ["applied" "applied" "applied"]);
            r = s.belief("p1", "cheap");
            tc.verifyBeta(r, 3, 2, 3);
            tc.verifyEqual(r.Estimate, 0.6, 'AbsTol', 1e-15);
            tc.verifyEqual(r.Variance, 0.04, 'AbsTol', 1e-15);
            tc.verifyEqual(r.SourceKind, "direct");
            tc.verifyEqual(r.EvidenceIDs, ["E1" "E2" "E3"]);
            tc.verifyEqual([s.version() r.Version], [3 3]);
        end

        function h3DuplicateDelivery(tc)
            s = OceanAgents.Beliefs.BetaTrustStore();
            s.observe([tc.ev("E1", "success"), tc.ev("E2", "success"), tc.ev("E3", "failure")]);
            tc.verifyEqual(s.observe(tc.ev("E3", "failure")), "duplicate");
            tc.verifyEqual(s.observe([tc.ev("E4", "success"), tc.ev("E4", "success")]), ["applied" "duplicate"]);
            tc.verifyBeta(s.belief("p1", "cheap"), 4, 2, 4);
            tc.verifyEqual(s.version(), 4);
            tc.verifyEqual(s.counts().Duplicate, 2);
        end

        function h4ConflictIsAFaultAndChangesNothing(tc)
            s = OceanAgents.Beliefs.BetaTrustStore();
            s.observe([tc.ev("E1", "success"), tc.ev("E2", "failure")]);
            before = s.exportState();
            tc.verifyError(@() s.observe(tc.ev("E1", "failure")), 'OceanAgents:Beliefs:EvidenceConflict');
            tc.verifyError(@() s.observe(tc.ev("E1", "success", 'Tick', 2)), 'OceanAgents:Beliefs:EvidenceConflict');
            % a conflict late in a batch rolls back the whole batch (E9 is neither applied nor marked seen)
            tc.verifyError(@() s.observe([tc.ev("E9", "success"), tc.ev("E2", "success")]), ...
                'OceanAgents:Beliefs:EvidenceConflict');
            tc.verifyEqual(s.exportState(), before);
            tc.verifyEqual(s.observe(tc.ev("E9", "success")), "applied");
        end

        function h5UnknownThenReport(tc)
            s = OceanAgents.Beliefs.BetaTrustStore();
            s.observe([tc.ev("E1", "success"), tc.ev("E2", "success"), tc.ev("E3", "failure")]);
            r0 = s.belief("p1", "cheap");
            d = s.observe([tc.ev("E4", "unknown"), ...
                tc.ev("R1", "success", 'SourceKind', "report", 'SourceID', "bob")]);
            tc.verifyEqual(d, ["unknown" "report"]);
            tc.verifyEqual(s.belief("p1", "cheap"), r0);          % posterior AND version unchanged
            tc.verifyEqual(s.version(), 3);
            n = s.counts();
            tc.verifyEqual([n.Unknown n.Report n.Applied], [1 1 3]);
        end

        function h6ContextsAreIndependent(tc)
            s = OceanAgents.Beliefs.BetaTrustStore('Prior', [2 5]);
            s.observe(tc.ev("E1", "success"));
            tc.verifyBeta(s.belief("p1", "cheap"), 3, 5, 1);
            tc.verifyBeta(s.belief("p1", "expensive"), 2, 5, 0);
            tc.verifyEqual(s.belief("p1", "expensive").SourceKind, "prior");
        end

        % ---------------- disposition order and bookkeeping ----------------
        function dispositionOrderFollowsTheTable(tc)
            s = OceanAgents.Beliefs.BetaTrustStore();
            d = s.observe([ ...
                tc.ev("A", "success", 'Quantity', "quality", 'SourceKind', "report"), ...  % ineligible before report
                tc.ev("B", "unknown", 'SourceKind', "report"), ...                        % report before unknown
                tc.ev("C", "failure", 'Quantity', "quality")]);
            tc.verifyEqual(d, ["ineligible" "report" "ineligible"]);
            s.setUpdatesEnabled(false);
            d = s.observe([tc.ev("D", "unknown"), tc.ev("E", "success"), ...
                tc.ev("F", "success", 'SourceKind', "report")]);
            tc.verifyEqual(d, ["unknown" "frozen" "report"]);           % unknown / report precede frozen
            tc.verifyEqual(s.version(), 0);
            n = s.counts();
            tc.verifyEqual(string(fieldnames(n))', ...
                ["Applied" "Duplicate" "Unknown" "Report" "Ineligible" "Frozen" "RecordedOnly"]);
            tc.verifyEqual([n.Applied n.Duplicate n.Unknown n.Report n.Ineligible n.Frozen n.RecordedOnly], ...
                [0 0 1 2 2 1 0]);
        end

        function frozenEvidenceIsConsumedOnce(tc)
            s = OceanAgents.Beliefs.BetaTrustStore();
            s.observe(tc.ev("E1", "success"));
            r0 = s.belief("p1", "cheap");
            s.setUpdatesEnabled(false);
            tc.verifyFalse(s.describe().UpdatesEnabled);
            tc.verifyEqual(s.observe(tc.ev("E2", "failure")), "frozen");
            tc.verifyEqual(s.belief("p1", "cheap"), r0);
            tc.verifyEqual(s.version(), 1);
            s.setUpdatesEnabled(true);
            tc.verifyEqual(s.observe(tc.ev("E2", "failure")), "duplicate");   % not replayed after unfreezing
            tc.verifyEqual(s.belief("p1", "cheap"), r0);
            tc.verifyEqual(s.observe(tc.ev("E3", "failure")), "applied");
            tc.verifyEqual(s.version(), 2);
        end

        function versionCountsAppliedUpdatesOnly(tc)
            s = OceanAgents.Beliefs.BetaTrustStore();
            tc.verifyEqual(s.version(), 0);
            s.observe([tc.ev("a", "success"), tc.ev("b", "unknown"), tc.ev("c", "failure"), tc.ev("a", "success"), ...
                tc.ev("d", "success", 'SubjectID', "p2")]);
            tc.verifyEqual(s.version(), 3);
            tc.verifyEqual(s.observe([]), strings(1, 0));
            tc.verifyEqual(s.observe(OceanAgents.Core.EvidenceRecord.empty(1, 0)), strings(1, 0));
            tc.verifyEqual(s.version(), 3);
            tc.verifyError(@() s.observe(struct('EvidenceID', "x")), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() s.belief("", "cheap"), 'MATLAB:validators:mustBeNonzeroLengthText');
        end

        function keyedBySubjectIDNotSlot(tc)
            % AP05: a new provider (even one listed in the same position as an old one) inherits nothing.
            s = OceanAgents.Beliefs.BetaTrustStore();
            s.observe([tc.ev("E1", "success"), tc.ev("E2", "success")]);
            tc.verifyBeta(s.belief("p1", "cheap"), 3, 1, 2);
            tc.verifyBeta(s.belief("p1-replacement", "cheap"), 1, 1, 0);
            tc.verifyBeta(s.belief("P1", "cheap"), 1, 1, 0);           % IDs are exact strings
        end

        function perSubjectPriorOverrides(tc)
            pri = struct('SubjectID', "p1", 'Context', "cheap", 'Alpha', 5, 'Beta', 2, 'SourceID', "expert");
            s = OceanAgents.Beliefs.BetaTrustStore('Prior', [1 1], 'Priors', pri);
            r = s.belief("p1", "cheap");
            tc.verifyBeta(r, 5, 2, 0);
            tc.verifyEqual([r.SourceKind r.SourceID], ["prior" "expert"]);
            tc.verifyEqual(s.belief("p1", "expensive").SourceID, "default-prior");
            s.observe(tc.ev("E1", "success"));
            r = s.belief("p1", "cheap");
            tc.verifyBeta(r, 6, 2, 1);
            tc.verifyEqual([r.SourceKind r.SourceID], ["direct" "expert"]);
            id = 'OceanAgents:Beliefs:BadImport';
            tc.verifyError(@() OceanAgents.Beliefs.BetaTrustStore('Priors', [pri pri]), id);          % duplicate
            tc.verifyError(@() OceanAgents.Beliefs.BetaTrustStore('Priors', setfield(pri, 'Alpha', 0)), id);
            tc.verifyError(@() OceanAgents.Beliefs.BetaTrustStore('Priors', rmfield(pri, 'SourceID')), id);
            tc.verifyError(@() OceanAgents.Beliefs.BetaTrustStore('Prior', [0 1]), 'MATLAB:validators:mustBePositive');
        end

        % ---------------- persistence ----------------
        function persistEpisodeResetsPosteriorsButKeepsSeenIDs(tc)
            pri = struct('SubjectID', "p1", 'Context', "cheap", 'Alpha', 5, 'Beta', 2, 'SourceID', "expert");
            s = OceanAgents.Beliefs.BetaTrustStore('Persist', "episode", 'Priors', pri);
            s.beginEpisode();
            s.observe([tc.ev("E1", "success"), tc.ev("E2", "failure", 'SubjectID', "p2"), tc.ev("E3", "unknown")]);
            s.endEpisode();
            tc.verifyBeta(s.belief("p1", "cheap"), 5, 2, 0);
            tc.verifyEqual(s.belief("p1", "cheap").SourceID, "expert");
            tc.verifyBeta(s.belief("p2", "cheap"), 1, 1, 0);
            n = s.counts();
            tc.verifyEqual(sum(cell2mat(struct2cell(n))), 0);
            tc.verifyEqual(s.version(), 2, 'the version counts applied updates; a reset is not one');
            s.beginEpisode();
            tc.verifyEqual(s.observe(tc.ev("E1", "success")), "duplicate");   % IDs are unique per run
            tc.verifyBeta(s.belief("p1", "cheap"), 5, 2, 0);
        end

        function persistRunKeepsEverything(tc)
            s = OceanAgents.Beliefs.BetaTrustStore();
            s.beginEpisode();
            s.observe(tc.ev("E1", "success"));
            s.endEpisode();
            s.beginEpisode();
            tc.verifyBeta(s.belief("p1", "cheap"), 2, 1, 1);
            tc.verifyEqual(s.counts().Applied, 1);
        end

        % ---------------- capacity ----------------
        function evidenceCapacityIsAtomic(tc)
            s = OceanAgents.Beliefs.BetaTrustStore('MaxEvidence', 3);
            s.observe(tc.ev("E1", "success"));
            before = s.exportState();
            tc.verifyError(@() s.observe([tc.ev("E2", "success"), tc.ev("E3", "failure"), tc.ev("E4", "success")]), ...
                'OceanAgents:Beliefs:CapacityExceeded');
            tc.verifyEqual(s.exportState(), before, 'the offending call changes nothing');
            s.observe([tc.ev("E2", "success"), tc.ev("E3", "unknown")]);           % exactly at capacity
            tc.verifyEqual(s.observe(tc.ev("E1", "success")), "duplicate");        % re-deliveries still fine
            tc.verifyError(@() s.observe(tc.ev("E5", "success")), 'OceanAgents:Beliefs:CapacityExceeded');
            tc.verifyBeta(s.belief("p1", "cheap"), 3, 1, 2);
        end

        function subjectCapacityIsAtomic(tc)
            s = OceanAgents.Beliefs.BetaTrustStore('MaxSubjects', 2);
            s.observe([tc.ev("E1", "success"), tc.ev("E2", "success", 'SubjectID', "p2")]);
            s.observe(tc.ev("E3", "success", 'SubjectID', "p2", 'Context', "expensive"));  % same subject, new context
            % events that are not applied never occupy a subject slot
            s.observe([tc.ev("E4", "unknown", 'SubjectID', "p3"), tc.ev("E5", "success", 'SubjectID', "p4", ...
                'SourceKind', "report")]);
            before = s.exportState();
            tc.verifyError(@() s.observe([tc.ev("E6", "success"), tc.ev("E7", "failure", 'SubjectID', "p3")]), ...
                'OceanAgents:Beliefs:CapacityExceeded');
            tc.verifyEqual(s.exportState(), before);
            tc.verifyEqual(s.observe(tc.ev("E6", "success")), "applied");
            pri = struct('SubjectID', {"a", "b", "c"}, 'Context', "x", 'Alpha', 1, 'Beta', 1, 'SourceID', "cfg");
            tc.verifyError(@() OceanAgents.Beliefs.BetaTrustStore('MaxSubjects', 2, 'Priors', pri), ...
                'OceanAgents:Beliefs:CapacityExceeded');
        end

        function traceIsBounded(tc)
            s = OceanAgents.Beliefs.BetaTrustStore('MaxTrace', 2);
            s.observe([tc.ev("E1", "success"), tc.ev("E2", "failure"), tc.ev("E3", "success")]);
            r = s.belief("p1", "cheap");
            tc.verifyEqual(r.EvidenceIDs, ["E2" "E3"]);            % most recent last
            tc.verifyEqual(r.EvidenceMass, 3);
            s0 = OceanAgents.Beliefs.BetaTrustStore('MaxTrace', 0);
            s0.observe(tc.ev("E1", "success"));
            tc.verifyEqual(s0.belief("p1", "cheap").EvidenceIDs, strings(1, 0));
        end

        % ---------------- export / import ----------------
        function exportImportRoundTrip(tc)
            pri = struct('SubjectID', "p2", 'Context', "cheap", 'Alpha', 0.5, 'Beta', 0.5, 'SourceID', "jeffreys");
            opts = {'Prior', [2 3], 'Priors', pri, 'MaxTrace', 3};
            s = OceanAgents.Beliefs.BetaTrustStore(opts{:});
            stream = OceanTest.OceanAgents.Beliefs.BetaTrustStoreTest.randomStream(200, 7);
            for c = 1:numel(stream)
                s.setUpdatesEnabled(stream(c).Enabled);
                s.observe(stream(c).Evidence);
            end
            st = s.exportState();
            OceanAgents.Core.assertValueData(st, "state");
            t = OceanAgents.Beliefs.BetaTrustStore(opts{:});
            t.importState(st);
            tc.verifyEqual(t.exportState(), st);                         % bit-exact
            tc.verifyEqual([t.version() s.version()], [st.Version st.Version]);
            for subj = ["p1" "p2" "p3" "p4"]
                for ctx = ["cheap" "expensive"]
                    tc.verifyEqual(t.belief(subj, ctx), s.belief(subj, ctx));
                end
            end
            % independent afterwards: the original learns, the copy and the exported struct do not change
            s.setUpdatesEnabled(true);
            s.observe(tc.ev("NEW", "success"));
            tc.verifyEqual(t.exportState(), st);
            tc.verifyNotEqual(s.exportState(), st);
            tc.verifyEqual(t.observe(tc.ev("NEW", "success")), "applied");
            tc.verifyEqual(t.belief("p1", "cheap"), s.belief("p1", "cheap"));
        end

        function importIsValidatedAndAtomic(tc)
            s = OceanAgents.Beliefs.BetaTrustStore();
            s.observe([tc.ev("E1", "success"), tc.ev("E2", "failure", 'SubjectID', "p2"), tc.ev("E3", "unknown")]);
            st = s.exportState();
            target = OceanAgents.Beliefs.BetaTrustStore();
            target.observe(tc.ev("T1", "success"));
            before = target.exportState();
            id = 'OceanAgents:Beliefs:BadImport';
            bads = {};
            b = st;  b.Name = "FixedBeliefStore";                    bads{end + 1} = b;
            b = st;  b.Config.Prior = [2 2];                         bads{end + 1} = b;   % other configuration
            b = st;  b.Entries.Alpha(1) = 2.5;                       bads{end + 1} = b;   % not prior + whole count
            b = st;  b.Entries.Mass(1) = 5;                          bads{end + 1} = b;   % mass mismatch
            b = st;  b.Entries.Trace{1} = "E99";                     bads{end + 1} = b;   % unseen ID in trace
            b = st;  b.Entries.Key(1) = "p9";                        bads{end + 1} = b;   % key != subject|context
            b = st;  b.Counts = rmfield(b.Counts, 'Frozen');         bads{end + 1} = b;
            b = st;  b.Counts.Applied = -1;                          bads{end + 1} = b;
            b = st;  b.SeenIDs(2) = b.SeenIDs(1);                    bads{end + 1} = b;   % duplicate IDs
            b = st;  b.SeenKeys = b.SeenKeys(1:end - 1);             bads{end + 1} = b;
            b = st;  b.Version = 1.5;                                bads{end + 1} = b;
            b = st;  b.UpdatesEnabled = 1;                           bads{end + 1} = b;
            for k = 1:numel(bads)
                tc.verifyError(@() target.importState(bads{k}), id, sprintf('bad state %d', k));
                tc.verifyEqual(target.exportState(), before, sprintf('bad state %d changed the store', k));
            end
            b = st;  b.Entries.Trace{1} = containers.Map();
            tc.verifyError(@() target.importState(b), 'OceanAgents:Core:InvalidRecord');
            small = OceanAgents.Beliefs.BetaTrustStore('MaxEvidence', 2);
            tc.verifyError(@() small.importState(st), id);                % other configuration (capacity)
        end

        % ---------------- independent reference ----------------
        function matchesIndependentCountingReference(tc)
            % 400 observe() calls over a seeded random stream with duplicates, reports, unknown outcomes, other
            % quantities, 4 subjects x 2 contexts and frozen intervals. Posteriors must match the plain count EXACTLY.
            prior = [2 3];
            s = OceanAgents.Beliefs.BetaTrustStore('Prior', prior, 'MaxTrace', 1e6);
            stream = OceanTest.OceanAgents.Beliefs.BetaTrustStoreTest.randomStream(400, 20260928);
            got = strings(1, 0);
            for c = 1:numel(stream)
                s.setUpdatesEnabled(stream(c).Enabled);
                got = [got, s.observe(stream(c).Evidence)]; %#ok<AGROW>
            end
            ref = OceanTest.OceanAgents.Fixtures.Beliefs.referenceBetaCounts(stream, prior, "completion");
            tc.verifyEqual(got, ref.Dispositions);
            tc.verifyEqual(s.version(), ref.Applied);
            n = s.counts();
            for d = ["applied" "duplicate" "unknown" "report" "ineligible" "frozen"]
                f = regexprep(char(d), '^(.)', '${upper($1)}');
                tc.verifyEqual(n.(f), nnz(ref.Dispositions == d), d);
            end
            tc.assertGreaterThan(numel(ref.Keys), 5);
            for d = ["duplicate" "unknown" "report" "ineligible" "frozen"]
                tc.assertGreaterThan(nnz(ref.Dispositions == d), 0, "stream must exercise " + d);
            end
            for k = 1:numel(ref.Keys)
                parts = split(ref.Keys(k), "|");
                r = s.belief(parts(1), parts(2));
                tc.verifyEqual([r.Alpha r.Beta r.EvidenceMass], [ref.Alpha(k) ref.Beta(k) ref.Mass(k)], ref.Keys(k));
                a = ref.Alpha(k);  b = ref.Beta(k);
                tc.verifyEqual(r.Estimate, a / (a + b), 'AbsTol', 1e-15);
                tc.verifyEqual(r.Variance, a * b / ((a + b)^2 * (a + b + 1)), 'AbsTol', 1e-15);
            end
            % pairs never applied stay exactly at the prior
            for subj = ["p1" "p2" "p3" "p4" "p5"]
                for ctx = ["cheap" "expensive"]
                    if ~any(ref.Keys == subj + "|" + ctx)
                        r = s.belief(subj, ctx);
                        tc.verifyEqual([r.Alpha r.Beta r.EvidenceMass], [prior 0]);
                    end
                end
            end
        end

        % ---------------- pillars ----------------
        function pillars(tc)
            s = OceanAgents.Beliefs.BetaTrustStore();
            d = s.describe();
            tc.verifyEqual([d.Name d.UpdateRule d.Quantity d.Persist], ...
                ["BetaTrustStore" "beta-bernoulli" "completion" "run"]);
            tc.verifyTrue(d.UpdatesEnabled);
            m = OceanAgents.Beliefs.BetaTrustStore.getMetadata();
            tc.verifyTrue(m.IsSymbolic);
            tc.verifyEqual(m.SymbolicLatex, ...
                "\alpha' = \alpha + \mathbb{1}[\text{success}],\ \beta' = \beta + \mathbb{1}[\text{failure}],\ \hat p = \alpha/(\alpha+\beta),\ \mathrm{Var} = \alpha\beta/((\alpha+\beta)^2(\alpha+\beta+1))");
            tc.verifyTrue(all(isfield(m, {'id', 'display_name', 'short_description', 'category', 'textbook', ...
                'pedagogy', 'equations', 'Symbols', 'parameters', 'complexity', 'strengths', 'weaknesses', ...
                'when_to_use', 'references', 'Assumptions', 'Reference'})));
            tc.verifyTrue(all(isfield(m.Symbols, {'Symbol', 'Description'})));
            tc.verifyEqual(s.getMetadata().SymbolicLatex, m.SymbolicLatex);
            spec = s.getPipelineSpec();
            tc.verifyEqual([spec.Inputs.Domain spec.Outputs.Domain], ["Agent-Evidence" "Agent-Belief"]);
        end
    end

    methods (Static)
        function stream = randomStream(numCalls, seed)
            % Seeded random evidence stream (no conflicts): new direct outcomes, re-deliveries of earlier records,
            % reports, unknown outcomes, another quantity, and frozen calls. Uses only its own RandStream.
            rs = RandStream('mrg32k3a', 'Seed', seed);
            subjects = ["p1" "p2" "p3" "p4"];
            contexts = ["cheap" "expensive"];
            outcomes = ["success" "failure" "unknown"];
            history = OceanAgents.Core.EvidenceRecord.empty(1, 0);
            stream = struct('Evidence', {}, 'Enabled', {});
            next = 0;
            for c = 1:numCalls
                k = randi(rs, 4) - 1;                                  % 0..3 records per call
                batch = OceanAgents.Core.EvidenceRecord.empty(1, 0);
                ids = strings(1, 0);
                for j = 1:k
                    if ~isempty(history) && rand(rs) < 0.2
                        e = history(randi(rs, numel(history)));      % re-delivery: same ID, same content
                        if any(ids == e.EvidenceID), continue; end
                    else
                        next = next + 1;
                        kind = "direct";  qty = "completion";
                        u = rand(rs);
                        if u < 0.1, kind = "report"; elseif u < 0.15, qty = "quality"; end
                        u2 = rand(rs);
                        if u2 < 0.1, o = outcomes(3); elseif u2 < 0.55, o = outcomes(2); else, o = outcomes(1); end
                        e = OceanAgents.Core.EvidenceRecord('EvidenceID', "R" + next, ...
                            'SubjectID', subjects(randi(rs, 4)), 'Context', contexts(randi(rs, 2)), ...
                            'Quantity', qty, 'Outcome', o, 'SourceKind', kind, 'SourceID', "src" + randi(rs, 3), ...
                            'EpisodeID', 1 + floor(c / 100), 'Tick', c);
                        history(end + 1) = e; %#ok<AGROW>
                    end
                    batch(end + 1) = e; %#ok<AGROW>
                    ids(end + 1) = e.EvidenceID; %#ok<AGROW>
                end
                stream(c).Evidence = batch;
                stream(c).Enabled = rand(rs) >= 0.1;
            end
        end
    end
end
