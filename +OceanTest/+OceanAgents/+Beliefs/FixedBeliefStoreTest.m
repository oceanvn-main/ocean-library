classdef FixedBeliefStoreTest < OceanTest.BaseTestCase
    % FIXEDBELIEFSTORETEST  OceanAgents.Beliefs.FixedBeliefStore (A2_Beliefs_Contract.md sections 3.2, 3.3, 8 H10-H11).
    % Gates: AP07 (a 0-100 score is not a probability), AP08 (fixed beliefs never move), unknown subject / Default.

    methods (Static, Access = private)
        function s = score(subject, context, x, lo, hi, conversion)
            s = struct('SubjectID', subject, 'Context', context, 'Score', x, 'ScaleMin', lo, 'ScaleMax', hi, ...
                'Conversion', conversion, 'SourceID', "config");
        end
        function e = ev(id, outcome, varargin)
            e = OceanAgents.Core.EvidenceRecord('EvidenceID', id, 'SubjectID', "p1", 'Context', "cheap", ...
                'Outcome', outcome, 'SourceID', "alice", 'EpisodeID', 1, 'Tick', 1, varargin{:});
        end
        function s = twoScores()
            s = [OceanTest.OceanAgents.Beliefs.FixedBeliefStoreTest.score("p1", "cheap", 80, 0, 100, "linear-scale"), ...
                OceanTest.OceanAgents.Beliefs.FixedBeliefStoreTest.score("p2", "cheap", 0.3, 0, 1, "none")];
        end
    end

    methods (Test)
        function h10LinearScaleConversion(tc)
            s = OceanAgents.Beliefs.FixedBeliefStore(tc.score("p1", "cheap", 80, 0, 100, "linear-scale"));
            r = s.belief("p1", "cheap");
            tc.verifyEqual(r.Estimate, 0.8);
            tc.verifyEqual([r.RawScore r.ScaleMin r.ScaleMax], [80 0 100]);
            tc.verifyEqual([r.Conversion r.SourceKind r.SourceID r.UpdateRule], ["linear-scale" "import" "config" "fixed"]);
            tc.verifyEmpty(r.Variance);                           % undefined, not zero
            tc.verifyEmpty(r.Alpha);
            tc.verifyEmpty(r.Beta);
            tc.verifyEqual([r.EvidenceMass r.Version], [0 0]);
            % the same score declared as already being a probability is refused (AP07)
            tc.verifyError(@() OceanAgents.Beliefs.FixedBeliefStore(tc.score("p1", "cheap", 80, 0, 100, "none")), ...
                'OceanAgents:Beliefs:BadImport');
            % a non-zero ScaleMin
            s2 = OceanAgents.Beliefs.FixedBeliefStore(tc.score("p1", "cheap", 3, 1, 5, "linear-scale"));
            tc.verifyEqual(s2.belief("p1", "cheap").Estimate, 0.5);
        end

        function noConversionNeedsUnitScale(tc)
            s = OceanAgents.Beliefs.FixedBeliefStore(tc.score("p1", "cheap", 0.7, 0, 1, "none"));
            r = s.belief("p1", "cheap");
            tc.verifyEqual([r.Estimate r.RawScore], [0.7 0.7]);
            tc.verifyEqual(r.Conversion, "none");
        end

        function badImportsAreRefused(tc)
            id = 'OceanAgents:Beliefs:BadImport';
            mk = @(varargin) OceanAgents.Beliefs.FixedBeliefStore(varargin{:});
            tc.verifyError(@() mk(tc.score("p1", "cheap", 120, 0, 100, "linear-scale")), id);   % outside scale
            tc.verifyError(@() mk(tc.score("p1", "cheap", -0.1, 0, 1, "none")), id);
            tc.verifyError(@() mk(tc.score("p1", "cheap", 5, 5, 5, "linear-scale")), id);        % empty scale
            tc.verifyError(@() mk(tc.score("p1", "cheap", NaN, 0, 1, "none")), id);
            tc.verifyError(@() mk(tc.score("p1", "cheap", 0.5, 0, 1, "logit")), id);            % unknown conversion
            tc.verifyError(@() mk(tc.score("", "cheap", 0.5, 0, 1, "none")), id);
            tc.verifyError(@() mk([tc.score("p1", "cheap", 0.5, 0, 1, "none"), tc.score("p1", "cheap", 0.6, 0, 1, "none")]), id);
            tc.verifyError(@() mk(rmfield(tc.score("p1", "cheap", 0.5, 0, 1, "none"), 'SourceID')), id);
            tc.verifyError(@() mk([], 'Default', 1.2), id);
            tc.verifyError(@() mk([], 'Default', [0.2 0.3]), id);
        end

        function unknownSubjectAndDefault(tc)
            s = OceanAgents.Beliefs.FixedBeliefStore(tc.twoScores());
            tc.verifyError(@() s.belief("p3", "cheap"), 'OceanAgents:Beliefs:UnknownSubject');
            tc.verifyError(@() s.belief("p1", "expensive"), 'OceanAgents:Beliefs:UnknownSubject');
            d = OceanAgents.Beliefs.FixedBeliefStore(tc.twoScores(), 'Default', 0.5);
            r = d.belief("p3", "cheap");
            tc.verifyEqual([r.Estimate r.EvidenceMass], [0.5 0]);
            tc.verifyEqual([r.SourceKind r.SourceID], ["import" "default"]);
            tc.verifyEqual(d.belief("p2", "cheap").Estimate, 0.3);       % known scores still win
            e = OceanAgents.Beliefs.FixedBeliefStore([], 'Default', 0);  % no scores at all
            tc.verifyEqual(e.belief("anyone", "any").Estimate, 0);
        end

        function h11EvidenceIsRecordedButNeverApplied(tc)
            s = OceanAgents.Beliefs.FixedBeliefStore(tc.twoScores());
            r0 = s.belief("p1", "cheap");
            d = s.observe([tc.ev("E1", "success"), tc.ev("E2", "failure"), tc.ev("E1", "success"), ...
                tc.ev("E3", "unknown"), tc.ev("E4", "success", 'SourceKind', "report"), ...
                tc.ev("E5", "success", 'Quantity', "quality")]);
            tc.verifyEqual(d, ["recorded-only" "recorded-only" "duplicate" "unknown" "report" "ineligible"]);
            r1 = s.belief("p1", "cheap");
            tc.verifyTrue(isequal(r1, r0));                                % bitwise unchanged
            tc.verifyEqual(typecast(r1.Estimate, 'uint64'), typecast(r0.Estimate, 'uint64'));
            tc.verifyEqual(s.version(), 0);
            n = s.counts();
            tc.verifyEqual([n.RecordedOnly n.Duplicate n.Unknown n.Report n.Ineligible n.Applied n.Frozen], ...
                [2 1 1 1 1 0 0]);
            s.setUpdatesEnabled(false);
            tc.verifyEqual(s.observe(tc.ev("E6", "failure")), "frozen");
            s.setUpdatesEnabled(true);
            tc.verifyEqual(s.observe(tc.ev("E6", "failure")), "duplicate");
            tc.verifyError(@() s.observe(tc.ev("E1", "failure")), 'OceanAgents:Beliefs:EvidenceConflict');
            tc.verifyTrue(isequal(s.belief("p1", "cheap"), r0));
        end

        function persistEpisodeResetsCountsOnly(tc)
            s = OceanAgents.Beliefs.FixedBeliefStore(tc.twoScores(), 'Persist', "episode");
            s.beginEpisode();
            s.observe(tc.ev("E1", "success"));
            s.endEpisode();
            tc.verifyEqual(s.counts().RecordedOnly, 0);
            tc.verifyEqual(s.observe(tc.ev("E1", "success")), "duplicate");   % seen IDs kept
            tc.verifyEqual(s.belief("p1", "cheap").Estimate, 0.8);
            r = OceanAgents.Beliefs.FixedBeliefStore(tc.twoScores());
            r.observe(tc.ev("E1", "success"));
            r.endEpisode();
            tc.verifyEqual(r.counts().RecordedOnly, 1);
        end

        function capacity(tc)
            s = OceanAgents.Beliefs.FixedBeliefStore(tc.twoScores(), 'MaxEvidence', 2);
            s.observe(tc.ev("E1", "success"));
            before = s.exportState();
            tc.verifyError(@() s.observe([tc.ev("E2", "success"), tc.ev("E3", "success")]), ...
                'OceanAgents:Beliefs:CapacityExceeded');
            tc.verifyEqual(s.exportState(), before);
            tc.verifyError(@() OceanAgents.Beliefs.FixedBeliefStore(tc.twoScores(), 'MaxSubjects', 1), ...
                'OceanAgents:Beliefs:CapacityExceeded');
        end

        function exportImportRoundTrip(tc)
            s = OceanAgents.Beliefs.FixedBeliefStore(tc.twoScores(), 'Default', 0.25);
            s.observe([tc.ev("E1", "success"), tc.ev("E2", "unknown")]);
            s.setUpdatesEnabled(false);
            st = s.exportState();
            OceanAgents.Core.assertValueData(st, "state");
            t = OceanAgents.Beliefs.FixedBeliefStore(tc.twoScores(), 'Default', 0.25);
            t.importState(st);
            tc.verifyEqual(t.exportState(), st);
            tc.verifyEqual(t.counts(), s.counts());
            tc.verifyFalse(t.describe().UpdatesEnabled);
            tc.verifyEqual(t.observe(tc.ev("E1", "success")), "duplicate");
            tc.verifyEqual(s.exportState(), st, 'the copy is independent of the original');
            id = 'OceanAgents:Beliefs:BadImport';
            other = OceanAgents.Beliefs.FixedBeliefStore(tc.twoScores());               % different Default
            tc.verifyError(@() other.importState(st), id);
            b = st;  b.Counts.RecordedOnly = 0.5;
            tc.verifyError(@() t.importState(b), id);
            b = st;  b.SeenIDs = [b.SeenIDs; "E9"];
            tc.verifyError(@() t.importState(b), id);
            b = st;  b.Name = "BetaTrustStore";
            tc.verifyError(@() t.importState(b), id);
        end

        function pillars(tc)
            s = OceanAgents.Beliefs.FixedBeliefStore(tc.twoScores());
            d = s.describe();
            tc.verifyEqual([d.Name d.UpdateRule d.Quantity d.Persist], ["FixedBeliefStore" "fixed" "completion" "run"]);
            tc.verifyTrue(d.UpdatesEnabled);
            m = s.getMetadata();
            tc.verifyFalse(m.IsSymbolic);
            tc.verifyTrue(all(isfield(m, {'SymbolicLatex', 'Symbols', 'Assumptions', 'Reference'})));
            spec = s.getPipelineSpec();
            tc.verifyEqual([spec.Inputs.Domain spec.Outputs.Domain], ["Agent-Evidence" "Agent-Belief"]);
        end
    end
end
