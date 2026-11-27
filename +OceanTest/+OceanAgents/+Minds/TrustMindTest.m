classdef TrustMindTest < OceanTest.BaseTestCase
    % TRUSTMINDTEST  OceanAgents.Minds.TrustMind (A2 contract section 4): evidence consumed exactly once through
    % beginEpisode/observe and never in decide; beliefs read only for legal delegate actions; provenance fields and
    % values; probabilities only when the rule gives them; profile naming; incompatibility errors; a belief changed by
    % evidence fed through observe() changes the next decision; state delegation; pillars.
    % Uses a stub belief store (+OceanTest/+OceanAgents/+Fixtures/+Minds/StubBeliefStore) and views built here.

    properties (Constant)
        Payoffs = struct('cheap', [10 -20 0], 'expensive', [30 -40 5])
    end

    methods (Static, Access = private)
        function e = ev(id, subject, outcome)
            e = OceanAgents.Core.EvidenceRecord('EvidenceID', id, 'SubjectID', subject, 'Context', "cheap", ...
                'Outcome', outcome, 'SourceKind', "direct", 'SourceID', "me", 'EpisodeID', 1);
        end
        function v = view(frameID, evidence, mask, info)
            % Labels [1 2 3] = p1, p2, outside, context "cheap" unless info overrides.
            if nargin < 2 || isempty(evidence), evidence = OceanAgents.Core.EvidenceRecord.empty(1, 0); end
            if nargin < 3, mask = true(1, 3); end
            if nargin < 4
                info = struct('context', "cheap", 'action_subjects', ["p1" "p2" "outside"]);
            end
            v = OceanAgents.Core.ActorView('AgentID', "me", 'EpisodeID', 1, 'FrameID', frameID, 'Observation', [1 frameID], ...
                'ActionLabels', [1 2 3], 'ActionMask', mask, 'Info', info, 'Evidence', evidence);
        end
        function fb = feedback(frameID, nextView)
            fb = OceanAgents.Core.Feedback('AgentID', "me", 'EpisodeID', 1, 'FrameID', frameID, 'Reward', 0, ...
                'NextView', nextView);
        end
        function [mind, store] = h7Mind(varargin)
            % H7 beliefs: p1 Beta(3,2), p2 fixed 0.8 (a mixed stub, typed "beta-bernoulli"), EU rule.
            store = OceanTest.OceanAgents.Fixtures.Minds.StubBeliefStore("beta-bernoulli", varargin{:});
            store.setBeta("p1", "cheap", 3, 2);
            store.setFixed("p2", "cheap", 0.8);
            mind = OceanAgents.Minds.TrustMind(store, OceanAgents.Minds.TrustRules.ExpectedUtilityRule(), ...
                OceanTest.OceanAgents.Minds.TrustMindTest.Payoffs);
        end
    end

    methods (Test)
        function evidenceConsumedExactlyOnce(tc)
            [mind, store] = tc.h7Mind();
            rs = RandStream('mrg32k3a', 'Seed', 1);
            v1 = tc.view(1, tc.ev("E1", "p1", "success"));
            mind.beginEpisode(v1);
            tc.verifyEqual([store.BeginCalls store.ObserveCalls], [1 1]);
            tc.verifyEqual(store.ObservedIDs, "E1");
            mind.decide(v1, rs);                              % the same view: its evidence is NOT consumed again
            tc.verifyEqual(store.ObserveCalls, 1);
            v2 = tc.view(2, [tc.ev("E2", "p2", "failure") tc.ev("E3", "p1", "unknown")]);
            mind.observe(tc.feedback(1, v2));
            tc.verifyEqual(store.ObservedIDs, ["E1" "E2" "E3"]);
            mind.decide(v2, rs);
            mind.observe(tc.feedback(2, tc.view(3)));         % a view without evidence: nothing handed over
            tc.verifyEqual(store.ObserveCalls, 2);
            tc.verifyEqual(mind.EvidenceConsumed, 3);
            mind.endEpisode();
            tc.verifyEqual(store.EndCalls, 1);
        end

        function decisionAndProvenanceH7(tc)
            [mind, store] = tc.h7Mind();
            rs = RandStream('mrg32k3a', 'Seed', 1);
            before = rs.State;
            v = tc.view(1);
            mind.beginEpisode(v);
            d = mind.decide(v, rs);
            d.validateAgainst(v);
            tc.verifyEqual(d.Action, 2);                      % H7: U = [-2 4 0]
            tc.verifyEqual(d.Probabilities, [0 1 0]);
            tc.verifyEqual(d.Likelihood, 1);
            tc.verifyEqual(rs.State, before);
            p = d.Provenance;
            tc.verifyEqual(sort(string(fieldnames(p)))', sort(["Rule" "Context" "BeliefVersion" "Labels" "Subjects" "Estimates" ...
                "Variances" "EvidenceMass" "SourceKinds" "Scores" "Chosen"]));
            tc.verifyEqual(p.Rule, "ExpectedUtilityRule");
            tc.verifyEqual(p.Context, "cheap");
            tc.verifyEqual(p.BeliefVersion, store.version());
            tc.verifyEqual(p.Labels, [1 2 3]);
            tc.verifyEqual(p.Subjects, ["p1" "p2" "outside"]);
            tc.verifyEqual(p.Estimates, [0.6 0.8 NaN]);
            tc.verifyEqual(p.Variances, [3 * 2 / (25 * 6) NaN NaN]);
            tc.verifyEqual(p.EvidenceMass, [0 0 0]);
            tc.verifyEqual(p.SourceKinds, ["prior" "import" "outside"]);
            tc.verifyEqual(p.Scores, [-2 4 0]);
            tc.verifyEqual(p.Chosen, 2);
        end

        function onlyLegalDelegatesAreRead(tc)
            [mind, store] = tc.h7Mind();
            v = tc.view(1, [], [true false true]);            % p2 illegal
            mind.beginEpisode(v);
            d = mind.decide(v, RandStream('mrg32k3a'));
            tc.verifyEqual(store.BeliefCalls, "p1|cheap", 'no belief read for illegal or outside actions');
            tc.verifyEqual(d.Action, 3);                      % U(p1) = -2 < 0 = outside
            tc.verifyEqual(d.Probabilities, [0 0 1]);
            tc.verifyEqual(d.Provenance.Subjects, ["p1" "outside"]);
            tc.verifyEqual(d.Provenance.Labels, [1 3]);
        end

        function thompsonGivesNoLikelihood(tc)
            store = OceanTest.OceanAgents.Fixtures.Minds.StubBeliefStore("beta-bernoulli");
            store.setBeta("p1", "cheap", 3, 2);
            store.setBeta("p2", "cheap", 1, 1);
            mind = OceanAgents.Minds.TrustMind(store, OceanAgents.Minds.TrustRules.ThompsonRule(), tc.Payoffs);
            v = tc.view(1);
            mind.beginEpisode(v);
            rs = RandStream('mrg32k3a', 'Seed', 4);
            s0 = rs.State;
            d = mind.decide(v, rs);
            tc.verifyEmpty(d.Probabilities);
            tc.verifyFalse(d.hasLikelihood());
            tc.verifyEmpty(d.Provenance.Scores);
            tc.verifyEqual(d.Provenance.Rule, "ThompsonRule");
            rs.State = s0;
            tc.verifyEqual(mind.decide(v, rs).Action, d.Action, 'same stream state -> same decision');
            tc.verifyEqual(mind.describe().Profile, "evidence-trust-adaptive");
        end

        function evidenceThroughObserveChangesNextDecision(tc)
            % Beta(1,1) for both providers: U = 0.5*10 + 0.5*(-20) = -5 < 0 -> outside (label 3).
            % Four successes of p1 delivered in the next view -> Beta(5,1), U = (5/6)*10 - (1/6)*20 = 5 > 0 -> label 1.
            store = OceanTest.OceanAgents.Fixtures.Minds.StubBeliefStore("beta-bernoulli", 'CountOutcomes', true);
            store.setBeta("p1", "cheap", 1, 1);
            store.setBeta("p2", "cheap", 1, 1);
            mind = OceanAgents.Minds.TrustMind(store, OceanAgents.Minds.TrustRules.ExpectedUtilityRule(), tc.Payoffs);
            rs = RandStream('mrg32k3a', 'Seed', 1);
            v1 = tc.view(1);
            mind.beginEpisode(v1);
            d1 = mind.decide(v1, rs);
            tc.verifyEqual(d1.Action, 3);
            ev = [tc.ev("E1", "p1", "success") tc.ev("E2", "p1", "success") tc.ev("E3", "p1", "success") ...
                tc.ev("E4", "p1", "success")];
            v2 = tc.view(2, ev);
            d2before = mind.decide(v2, RandStream('mrg32k3a', 'Seed', 1));   % decide never consumes evidence
            tc.verifyEqual(d2before.Action, 3);
            mind.observe(tc.feedback(1, v2));
            tc.verifyEqual(mind.LastDispositions, repmat("applied", 1, 4));
            d2 = mind.decide(v2, rs);
            tc.verifyEqual(d2.Action, 1);
            tc.verifyEqual(d2.Provenance.Scores(1), 5, 'AbsTol', 1e-12);
            tc.verifyEqual(d2.Provenance.EvidenceMass, [4 0 0]);
            tc.verifyEqual(d2.Provenance.BeliefVersion, 4);
            % the same evidence delivered again is a duplicate and changes nothing
            mind.observe(tc.feedback(2, tc.view(3, ev)));
            tc.verifyEqual(mind.LastDispositions, repmat("duplicate", 1, 4));
            tc.verifyEqual(store.version(), 4);
        end

        function profileNaming(tc)
            S = @OceanTest.OceanAgents.Fixtures.Minds.StubBeliefStore;
            EU = OceanAgents.Minds.TrustRules.ExpectedUtilityRule();
            m = OceanAgents.Minds.TrustMind(S("fixed"), EU, tc.Payoffs);
            tc.verifyEqual(m.describe().Profile, "fixed-belief-reactive");
            d = m.describe();
            tc.verifyEqual(sort(string(fieldnames(d.Components)))', ["Rule" "Store"]);
            tc.verifyEqual(d.Components.Rule.Name, "ExpectedUtilityRule");
            m2 = OceanAgents.Minds.TrustMind(S("beta-bernoulli"), OceanAgents.Minds.TrustRules.ExpectedUtilityRule(), tc.Payoffs);
            tc.verifyEqual(m2.Profile, "evidence-trust-adaptive");
            tc.verifyError(@() OceanAgents.Minds.TrustMind(S("fixed"), OceanAgents.Minds.TrustRules.ExpectedUtilityRule(), ...
                tc.Payoffs, 'Profile', "evidence-trust-adaptive"), 'OceanAgents:Minds:Incompatible');
            tc.verifyError(@() OceanAgents.Minds.TrustMind(S("windowed"), OceanAgents.Minds.TrustRules.ExpectedUtilityRule(), ...
                tc.Payoffs), 'OceanAgents:Minds:Incompatible');
            m3 = OceanAgents.Minds.TrustMind(S("windowed"), OceanAgents.Minds.TrustRules.ExpectedUtilityRule(), ...
                tc.Payoffs, 'Profile', "custom");
            tc.verifyEqual(m3.Profile, "custom");
        end

        function incompatibilities(tc)
            S = @OceanTest.OceanAgents.Fixtures.Minds.StubBeliefStore;
            ID = 'OceanAgents:Minds:Incompatible';
            % rule/store mismatch at construction
            tc.verifyError(@() OceanAgents.Minds.TrustMind(S("fixed"), OceanAgents.Minds.TrustRules.ThompsonRule(), tc.Payoffs), ID);
            tc.verifyError(@() OceanAgents.Minds.TrustMind(S("fixed"), ...
                OceanAgents.Minds.TrustRules.ExpectedUtilityRule('RiskAversion', 1), tc.Payoffs), ID);
            tc.verifyError(@() OceanAgents.Minds.TrustMind(S("fixed"), OceanAgents.Minds.TrustRules.ExpectedUtilityRule(), ...
                struct('cheap', [1 2])), ID);
            % per-view world contract
            [mind, ~] = tc.h7Mind();
            rs = RandStream('mrg32k3a');
            mind.beginEpisode(tc.view(1));
            subj = ["p1" "p2" "outside"];
            tc.verifyError(@() mind.decide(tc.view(2, [], true(1, 3), struct('action_subjects', subj)), rs), ID);
            tc.verifyError(@() mind.decide(tc.view(2, [], true(1, 3), struct('context', "cheap")), rs), ID);
            tc.verifyError(@() mind.decide(tc.view(2, [], true(1, 3), ...
                struct('context', "luxury", 'action_subjects', subj)), rs), ID);
            tc.verifyError(@() mind.decide(tc.view(2, [], true(1, 3), ...
                struct('context', "cheap", 'action_subjects', ["p1" "outside"])), rs), ID);
            % action space
            tc.verifyError(@() mind.checkCompatible([], OceanStats.RL.Env.Spaces.Box(0, 1), "default"), ID);
            mind.checkCompatible([], OceanStats.RL.Env.Spaces.Discrete(3), "default");   % accepted
            % a mixed store with a fixed score under Thompson is refused per decision
            st = S("beta-bernoulli");
            st.setBeta("p1", "cheap", 1, 1);
            st.setFixed("p2", "cheap", 0.7);
            th = OceanAgents.Minds.TrustMind(st, OceanAgents.Minds.TrustRules.ThompsonRule(), tc.Payoffs);
            th.beginEpisode(tc.view(1));
            tc.verifyError(@() th.decide(tc.view(1), rs), ID);
        end

        function secondContextUsesItsPayoffs(tc)
            store = OceanTest.OceanAgents.Fixtures.Minds.StubBeliefStore("fixed");
            store.setFixed("p1", "expensive", 0.5);
            store.setFixed("p2", "expensive", 0.2);
            mind = OceanAgents.Minds.TrustMind(store, OceanAgents.Minds.TrustRules.ExpectedUtilityRule(), tc.Payoffs);
            v = tc.view(1, [], true(1, 3), struct('context', "expensive", 'action_subjects', ["p1" "p2" "outside"]));
            mind.beginEpisode(v);
            d = mind.decide(v, RandStream('mrg32k3a'));
            % [30 -40 5]: U(p1) = 15 - 20 = -5, U(p2) = 6 - 32 = -26, outside 5
            tc.verifyEqual(d.Provenance.Scores, [-5 -26 5], 'AbsTol', 1e-12);
            tc.verifyEqual(d.Action, 3);
            tc.verifyEqual(d.Provenance.Variances, [NaN NaN NaN]);
        end

        function stateDelegatesToStore(tc)
            [mind, store] = tc.h7Mind('CountOutcomes', true);
            mind.beginEpisode(tc.view(1, tc.ev("E1", "p1", "success")));
            s = mind.exportState();
            tc.verifyEqual(s.Name, "TrustMind");
            tc.verifyEqual(s.Store.Version, 1);
            [mind2, store2] = tc.h7Mind();
            mind2.importState(s);
            tc.verifyEqual(store2.version(), store.version());
            tc.verifyError(@() mind2.importState(struct('Name', "PolicyMind")), 'OceanAgents:Minds:Incompatible');
        end

        function pillars(tc)
            [mind, ~] = tc.h7Mind();
            meta = mind.getMetadata();
            tc.verifyFalse(meta.IsSymbolic);
            spec = mind.getPipelineSpec();
            tc.verifyEqual([spec.Inputs.Domain], ["Agent-ActorView" "Agent-Feedback"]);
            tc.verifyEqual(spec.Outputs.Domain, "Agent-Decision");
        end
    end
end
