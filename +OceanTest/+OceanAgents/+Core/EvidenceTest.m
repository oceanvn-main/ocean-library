classdef EvidenceTest < OceanTest.BaseTestCase
    % EVIDENCETEST  OceanAgents.Core.EvidenceRecord and ActorView.Evidence (A2_Beliefs_Contract.md section 2).

    methods (Static, Access = private)
        function e = ev(id, varargin)
            e = OceanAgents.Core.EvidenceRecord('EvidenceID', id, 'SubjectID', "p1", 'Context', "cheap", ...
                'Outcome', "success", 'SourceID', "alice", 'EpisodeID', 1, 'Tick', 3, varargin{:});
        end
        function v = view(evidence)
            v = OceanAgents.Core.ActorView('AgentID', "alice", 'EpisodeID', 1, 'FrameID', 2, 'Observation', [1 3], ...
                'ActionLabels', 1:4, 'ActionMask', true(1, 4), 'Evidence', evidence);
        end
    end

    methods (Test)
        function defaultsAndFields(tc)
            e = tc.ev("E1");
            tc.verifyEqual(e.Quantity, "completion");
            tc.verifyEqual(e.SourceKind, "direct");
            tc.verifyEqual(e.Tick, 3);
            tc.verifyFalse(isa(e, 'handle'));
        end

        function rejectsBadValues(tc)
            id = 'OceanAgents:Core:InvalidRecord';
            tc.verifyError(@() tc.ev("E1", 'Outcome', "betrayal"), id);
            tc.verifyError(@() tc.ev("E1", 'SourceKind', "rumour"), id);
            tc.verifyError(@() tc.ev(""), id);
            tc.verifyError(@() tc.ev("E1", 'EpisodeID', -1), id);
            tc.verifyError(@() tc.ev("E1", 'Tick', NaN), 'OceanAgents:Core:NonFinite');
            tc.verifyError(@() OceanAgents.Core.EvidenceRecord('EvidenceID', "E1", 'SubjectID', "p1", ...
                'Context', "cheap", 'Outcome', "success", 'EpisodeID', 1), id);           % SourceID missing
        end

        function canonicalKeyDetectsChangedContent(tc)
            a = tc.ev("E1");
            tc.verifyEqual(tc.ev("E1").canonicalKey(), a.canonicalKey());
            tc.verifyNotEqual(tc.ev("E1", 'Outcome', "failure").canonicalKey(), a.canonicalKey());
            tc.verifyNotEqual(tc.ev("E1", 'SourceKind', "report").canonicalKey(), a.canonicalKey());
            tc.verifyNotEqual(tc.ev("E1", 'Tick', 4).canonicalKey(), a.canonicalKey());
        end

        function viewCarriesEvidence(tc)
            v = tc.view([tc.ev("E1"), tc.ev("E2", 'Outcome', "unknown")]);
            tc.verifyEqual(v.evidenceIDs(), ["E1" "E2"]);
            tc.verifyEqual(v.Evidence(2).Outcome, "unknown");
            none = tc.view([]);
            tc.verifyEqual(none.evidenceIDs(), strings(1, 0));
            tc.verifySize(none.Evidence, [1 0]);
        end

        function viewRejectsDuplicateOrForeignEvidence(tc)
            tc.verifyError(@() tc.view([tc.ev("E1"), tc.ev("E1")]), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() tc.view(struct('EvidenceID', "E1")), 'OceanAgents:Core:InvalidRecord');
            m = [tc.ev("E1") tc.ev("E2"); tc.ev("E3") tc.ev("E4")];
            tc.verifyError(@() tc.view(m), 'OceanAgents:Core:InvalidRecord');           % not 1 x K (critic A2 F6)
            tc.verifyError(@() tc.view([tc.ev("E1"); tc.ev("E2")]), 'OceanAgents:Core:InvalidRecord');
        end

        function identifiersRejectControlCharacters(tc)
            % a separator character inside an ID could make two (subject, context) keys collide (critic A2 F5)
            us = char(31);
            tc.verifyError(@() tc.ev("E1", 'SubjectID', "a" + us + "b"), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() tc.ev("E1", 'Context', "c" + newline), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() tc.ev("E" + char(0)), 'OceanAgents:Core:InvalidRecord');
            tc.ev("E-1/x@2", 'SubjectID', "p 1", 'Context', "very cheap");                  % printable text is fine
        end
    end
end
