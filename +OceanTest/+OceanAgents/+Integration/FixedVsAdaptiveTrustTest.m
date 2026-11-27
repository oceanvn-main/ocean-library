classdef FixedVsAdaptiveTrustTest < OceanTest.BaseTestCase
    % FIXEDVSADAPTIVETRUSTTEST  C21/S08 end to end: a fixed-belief mind and an evidence-adaptive mind share one
    % DelegationEnv under JointRunner with public outcomes, so both are shown exactly the same evidence.
    %
    % Evidence, not self-consistency:
    %   - both agents receive the identical evidence ID sequence (recorded at the mind boundary);
    %   - the fixed store's beliefs stay bitwise equal to the imported scores and its version stays 0;
    %   - the adaptive posteriors equal an independent plain count (Fixtures.Beliefs.referenceBetaCounts) over exactly
    %     the evidence that agent was shown, across two episodes with run-long persistence;
    %   - the fixed agent's choice depends only on the context (its beliefs never move).
    % The adaptive agent's share of delegations to p1 before/after p1's reliability collapses is REPORTED, not asserted.
    % A2_Beliefs_Contract.md section 9.

    properties (Constant)
        Providers = ["p1" "p2" "p3"]
        Contexts = ["cheap" "expensive"]
        R = [0.9 0.9; 0.6 0.4; 0.3 0.8]         % providers x contexts, hidden from agents
        Rounds = 120
        SwitchRound = 61                        % p1 collapses from this round on, every episode
    end

    methods (Static)
        function recs = fixedBeliefs(store)
            C = OceanTest.OceanAgents.Integration.FixedVsAdaptiveTrustTest;
            recs = {};
            for i = 1:3
                for j = 1:2
                    recs{end + 1} = store.belief(C.Providers(i), C.Contexts(j)); %#ok<AGROW>
                end
            end
        end

        function p = payoffs()
            p = struct('cheap', [2 -1 0], 'expensive', [10 -20 0]);
        end

        function [runner, fixedRec, adaptRec, fixedStore, betaStore, env] = build(seed, betaStore)
            C = OceanTest.OceanAgents.Integration.FixedVsAdaptiveTrustTest;
            sched = struct('FromRound', C.SwitchRound, 'Reliability', [0.1 0.1; C.R(2:3, :)]);
            env = OceanAgents.Env.Reference.DelegationEnv('Roster', ["fixed" "adaptive"], 'Providers', C.Providers, ...
                'Contexts', C.Contexts, 'ContextProbs', [0.5 0.5], 'Reliability', C.R, 'Schedule', sched, ...
                'Payoffs', C.payoffs(), 'ObstructionProb', 0.1, 'ObstructedPayoff', 0, ...
                'Visibility', "public", 'Rounds', C.Rounds);
            % imported scores on a 0-100 scale (declared conversion), deliberately not the hidden truth
            scores = struct('SubjectID', {}, 'Context', {}, 'Score', {}, 'ScaleMin', {}, 'ScaleMax', {}, ...
                'Conversion', {}, 'SourceID', {});
            imported = [90 90; 60 60; 50 50];
            for i = 1:3
                for j = 1:2
                    scores(end + 1) = struct('SubjectID', C.Providers(i), 'Context', C.Contexts(j), ...
                        'Score', imported(i, j), 'ScaleMin', 0, 'ScaleMax', 100, 'Conversion', "linear-scale", ...
                        'SourceID', "config"); %#ok<AGROW>
                end
            end
            fixedStore = OceanAgents.Beliefs.FixedBeliefStore(scores);
            if nargin < 2
                betaStore = OceanAgents.Beliefs.BetaTrustStore('Prior', [1 1], 'Persist', "run");
            end
            fixedRec = OceanTest.OceanAgents.Fixtures.Integration.RecordingMind(OceanAgents.Minds.TrustMind( ...
                fixedStore, OceanAgents.Minds.TrustRules.ExpectedUtilityRule(), C.payoffs()));
            adaptRec = OceanTest.OceanAgents.Fixtures.Integration.RecordingMind(OceanAgents.Minds.TrustMind( ...
                betaStore, OceanAgents.Minds.TrustRules.ThompsonRule(), C.payoffs()));
            hosts = {OceanAgents.Runtime.AgentHost("fixed", fixedRec), OceanAgents.Runtime.AgentHost("adaptive", adaptRec)};
            runner = OceanAgents.Runtime.JointRunner(env, hosts, 'Seed', seed);
        end
    end

    methods (Test)
        function sameEvidenceFixedStaysAdaptiveFollowsCounts(tc)
            [runner, fixedRec, adaptRec, fixedStore, betaStore, env] = tc.build(2026);
            fixedBefore = fixedStore.exportState();
            recsBefore = tc.fixedBeliefs(fixedStore);
            % the mind is authorised to know the declared game payoffs; they must be the world's (critic A2 F2)
            tc.verifyEqual(fixedRec.Inner.Payoffs, env.Payoffs);
            tc.verifyEqual(adaptRec.Inner.Payoffs, env.Payoffs);
            r = runner.run(2);
            tc.assertEqual(r.Status, "Completed", 'run failed');
            tc.verifyEqual(r.NumEpisodes, 2);

            % C21: identical evidence shown to both minds (public visibility), nothing invented
            evF = fixedRec.allEvidence();
            evA = adaptRec.allEvidence();
            tc.verifyEqual([evA.EvidenceID], [evF.EvidenceID]);
            tc.verifyGreaterThan(numel(evA), 100, 'the run should produce plenty of evidence');
            tc.verifyTrue(all([evA.SourceKind] == "direct"));

            % fixed profile: imported beliefs bitwise unchanged, version 0, every event consumed once.
            % (Its event memory - seen IDs and disposition counts - legitimately grows; only beliefs are frozen.)
            fixedAfter = fixedStore.exportState();
            tc.verifyEqual(fixedAfter.Config, fixedBefore.Config);
            tc.verifyEqual(numel(fixedAfter.SeenIDs), numel(evF));
            tc.verifyEqual(fixedStore.version(), 0);
            tc.verifyEqual(tc.fixedBeliefs(fixedStore), recsBefore, 'fixed beliefs must be bitwise unchanged');
            importedP = [0.9 0.6 0.5];
            for i = 1:3
                b = fixedStore.belief(tc.Providers(i), "cheap");
                tc.verifyEqual(b.Estimate, importedP(i), 'AbsTol', 1e-15);    % declared linear-scale conversion
                tc.verifyEqual(b.SourceKind, "import");
            end
            nF = fixedStore.counts();
            tc.verifyEqual(nF.Applied, 0);
            tc.verifyEqual(nF.RecordedOnly + nF.Unknown, numel(evF));

            % adaptive profile: posterior = independent plain count over exactly what this agent was shown
            stream = struct('Evidence', adaptRec.Delivered, 'Enabled', true);
            ref = OceanTest.OceanAgents.Fixtures.Beliefs.referenceBetaCounts(stream, [1 1], "completion");
            for k = 1:numel(ref.Keys)
                parts = split(ref.Keys(k), "|");
                b = betaStore.belief(parts(1), parts(2));
                tc.verifyEqual([b.Alpha b.Beta], [ref.Alpha(k) ref.Beta(k)], sprintf('posterior %s', ref.Keys(k)));
                tc.verifyEqual(b.EvidenceMass, ref.Mass(k));
            end
            tc.verifyEqual(betaStore.version(), ref.Applied);
            nA = betaStore.counts();
            tc.verifyEqual(nA.Applied + nA.Unknown, numel(evA));
            tc.verifyGreaterThan(nA.Unknown, 0, 'obstructed outcomes must stay unknown, not failures');

            % the fixed agent's choice is a function of the context alone
            ctx = fixedRec.Contexts;
            acts = cellfun(@(d) d.Action, fixedRec.Decisions);
            for c = tc.Contexts
                tc.verifyNumElements(unique(acts(ctx == c)), 1, sprintf('fixed agent changed its choice in %s', c));
            end

            % REPORT (not asserted): adaptive delegation share to p1 before/after p1 collapses, per episode
            actsA = cellfun(@(d) d.Action, adaptRec.Decisions);
            T = tc.Rounds;
            for e = 1:2
                idx = (e - 1) * T + (1:T);
                before = mean(actsA(idx(1:tc.SwitchRound - 1)) == 1);
                after = mean(actsA(idx(tc.SwitchRound:end)) == 1);
                tc.log(matlab.unittest.Verbosity.Terse, sprintf(['episode %d: adaptive chose p1 in %.0f%% of rounds ' ...
                    'before the collapse, %.0f%% after; fixed chose p1 in %.0f%% overall'], e, 100 * before, ...
                    100 * after, 100 * mean(acts(idx) == 1)));
            end
            tc.log(matlab.unittest.Verbosity.Terse, sprintf('returns (fixed, adaptive) per episode: %s', ...
                mat2str(r.EpisodeReturns)));
        end

        function provenanceTracesThePosteriorAtDecisionTime(tc)
            % every adaptive decision's provenance estimates equal the posterior implied by the evidence delivered
            % BEFORE that decision (independent count), so the trace is the store's actual state, not a later one
            [runner, ~, adaptRec] = tc.build(7);
            runner.run(1);
            T = numel(adaptRec.Decisions);
            tc.assertEqual(numel(adaptRec.Delivered), T + 1);        % beginEpisode view + one observe per round
            for i = 1:T
                d = adaptRec.Decisions{i};
                p = d.Provenance;
                tc.verifyEqual(p.Rule, "ThompsonRule");
                tc.verifyEqual(p.Chosen, d.Action);
                tc.verifyEmpty(d.Probabilities, 'Thompson has no closed-form likelihood; none may be invented');
                ref = OceanTest.OceanAgents.Fixtures.Beliefs.referenceBetaCounts( ...
                    struct('Evidence', adaptRec.Delivered(1:i), 'Enabled', true), [1 1], "completion");
                for j = find(p.Subjects ~= "outside")
                    k = find(ref.Keys == p.Subjects(j) + "|" + adaptRec.Contexts(i), 1);
                    ab = [1 1];
                    if ~isempty(k), ab = [ref.Alpha(k) ref.Beta(k)]; end
                    tc.verifyEqual(p.Estimates(j), ab(1) / sum(ab), 'AbsTol', 1e-15, ...
                        sprintf('decision %d, %s: stale or invented estimate', i, p.Subjects(j)));
                end
            end
        end

        function reusedStoreLearnsInAFreshWorld(tc)
            % critic A2 F1: a trained store reused with a NEW world instance must apply the new world's events,
            % not reject them as conflicts or drop them as duplicates (evidence IDs are scoped by the world's RunTag)
            [r1, ~, ~, ~, store] = tc.build(5);
            tc.assertEqual(r1.run(1).Status, "Completed");
            r1.release();
            before = store.counts();
            v0 = store.version();
            [r2, ~, adaptRec2] = tc.build(5, store);                  % same seed, same roster, fresh world
            res = r2.run(1);
            tc.assertEqual(res.Status, "Completed");
            after = store.counts();
            tc.verifyEqual(after.Duplicate, before.Duplicate, 'new events were silently dropped as duplicates');
            known = sum([adaptRec2.allEvidence().Outcome] ~= "unknown");
            tc.verifyEqual(store.version() - v0, known);
        end

        function sameSeedSameRun(tc)
            [rA, ~, aA] = tc.build(11);
            [rB, ~, aB] = tc.build(11);
            xA = rA.run(1);
            xB = rB.run(1);
            tc.verifyEqual(xB.EpisodeReturns, xA.EpisodeReturns);
            tc.verifyEqual(cellfun(@(d) d.Action, aB.Decisions), cellfun(@(d) d.Action, aA.Decisions));
        end
    end
end
