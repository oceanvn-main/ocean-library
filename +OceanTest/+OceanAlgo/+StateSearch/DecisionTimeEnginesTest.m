classdef DecisionTimeEnginesTest < OceanTest.BaseTestCase
    % DECISIONTIMEENGINESTEST  S&B 8.9-8.11 engines against exact oracles: synchronous value-iteration sweeps,
    % exact policy evaluation (q_pi) and converged value iteration (q*).

    methods (Test)
        % ---------------- Depth-limited search (8.9) ----------------
        function depthOneIsOneStepLookahead(tc)
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP.tiny();
            V0 = [0; 8; 99];
            r = OceanAlgo.StateSearch.DepthLimitedSearch('Depth', 1, 'LeafValue', @(s) V0(s), ...
                'DiscountFactor', 0.5).search(m, 1);
            % q(1,1) = .25 (2 + .5 * 8) + .75 * 4 = 4.5;  q(1,2) = 3  (hand values, as in RTDPSolverTest)
            tc.verifyEqual(r.ActionValues, [4.5 3], 'AbsTol', 1e-12);
            tc.verifyEqual([r.Action, r.Value], [1 4.5], 'AbsTol', 1e-12);
        end

        function depthKEqualsKValueIterationSweeps(tc)
            rs = RandStream('mrg32k3a', 'Seed', 7);
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP.random(9, 3, 2, rs);
            hv = randn(rs, 9, 1); gamma = 0.9;
            for k = 1:5
                [~, Q] = m.valueIterationSweeps(hv, gamma, k);
                for s0 = [1 4 9]
                    r = OceanAlgo.StateSearch.DepthLimitedSearch('Depth', k, 'LeafValue', @(s) hv(s), ...
                        'DiscountFactor', gamma).search(m, s0);
                    tc.verifyEqual(r.ActionValues, Q(s0, r.Actions), 'AbsTol', 1e-10);
                end
            end
        end

        function transpositionsChangeWorkNotValues(tc)
            rs = RandStream('mrg32k3a', 'Seed', 8);
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP.random(6, 2, 2, rs);
            hv = randn(rs, 6, 1);
            on = OceanAlgo.StateSearch.DepthLimitedSearch('Depth', 5, 'LeafValue', @(s) hv(s), 'DiscountFactor', 0.9, ...
                'RecordBackups', true).search(m, 1);
            off = OceanAlgo.StateSearch.DepthLimitedSearch('Depth', 5, 'LeafValue', @(s) hv(s), 'DiscountFactor', 0.9, ...
                'Transpositions', false).search(m, 1);
            tc.verifyEqual(on.ActionValues, off.ActionValues, 'AbsTol', 1e-12);
            tc.verifyLessThan(on.Work.Expansions, off.Work.Expansions);
            tc.verifyGreaterThan(on.Work.TranspositionHits, 0);
            tc.verifyEqual(height(on.Trace), on.Work.Expansions);
            tc.verifyEqual(on.Trace.Remaining(end), 5);                 % the root backup comes last
        end

        function depthBudgetReturnsDeepestCompleted(tc)
            rs = RandStream('mrg32k3a', 'Seed', 9);
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP.random(30, 3, 3, rs);
            hv = randn(rs, 30, 1);
            r = OceanAlgo.StateSearch.DepthLimitedSearch('Depth', 6, 'LeafValue', @(s) hv(s), 'DiscountFactor', 0.9, ...
                'MaxExpansions', 40).search(m, 1);
            tc.verifyEqual(r.Status, "budget_exhausted");
            d = r.Work.CompletedDepth;
            tc.verifyGreaterThanOrEqual(d, 1); tc.verifyLessThan(d, 6);
            ref = OceanAlgo.StateSearch.DepthLimitedSearch('Depth', d, 'LeafValue', @(s) hv(s), ...
                'DiscountFactor', 0.9).search(m, 1);
            tc.verifyEqual(r.ActionValues, ref.ActionValues, 'AbsTol', 1e-12);
        end

        % ---------------- Rollout (8.10) ----------------
        function rolloutIsExactForDeterministicModelAndPolicy(tc)
            rs = RandStream('mrg32k3a', 'Seed', 10);
            m = tc.deterministicMDP_(8, 3, rs);
            pi = arrayfun(@(s) tc.firstAction_(m, s), (1:8)');
            gamma = 0.9; v = m.policyValue(pi, gamma, @(s) 0);
            pol = OceanTest.OceanAlgo.StateSearch.Fixtures.FixedPolicy(pi);
            for s0 = 1:8
                if isempty(m.actions(s0)), continue; end
                r = OceanAlgo.StateSearch.RolloutSearch('RolloutPolicy', pol, 'NumRollouts', 2, 'Horizon', 400, ...
                    'DiscountFactor', gamma).search(m, s0);
                q = arrayfun(@(a) tc.q_(m, s0, a, v, gamma), r.Actions);
                tc.verifyEqual(r.ActionValues, q, 'AbsTol', 1e-9);
            end
        end

        function rolloutImprovesOnItsPolicy(tc)
            % Policy improvement theorem (S&B 4.2): pi'(s) = rollout action at every s; v_pi' >= v_pi.
            rs = RandStream('mrg32k3a', 'Seed', 11);
            for trial = 1:5
                m = tc.deterministicMDP_(10, 3, rs);
                pi = arrayfun(@(s) tc.firstAction_(m, s), (1:10)');
                gamma = 0.9; v = m.policyValue(pi, gamma, @(s) 0);
                eng = OceanAlgo.StateSearch.RolloutSearch('RolloutPolicy', ...
                    OceanTest.OceanAlgo.StateSearch.Fixtures.FixedPolicy(pi), 'NumRollouts', 1, 'Horizon', 400, ...
                    'DiscountFactor', gamma);
                pi2 = pi;
                for s = 1:10
                    if ~isempty(m.actions(s)), pi2(s) = eng.search(m, s).Action; end
                end
                v2 = m.policyValue(pi2, gamma, @(s) 0);
                tc.verifyGreaterThanOrEqual(v2 - v, -1e-9);
            end
        end

        function rolloutStochasticEstimatesWithinStandardErrors(tc)
            rs = RandStream('mrg32k3a', 'Seed', 12);
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP.random(6, 2, 2, rs);
            pi = ones(6, 1); gamma = 0.9; v = m.policyValue(pi, gamma, @(s) 0);
            r = OceanAlgo.StateSearch.RolloutSearch('RolloutPolicy', OceanTest.OceanAlgo.StateSearch.Fixtures.FixedPolicy(pi), ...
                'NumRollouts', 4000, 'Horizon', 300, 'DiscountFactor', gamma, 'Seed', 3).search(m, 1);
            q = arrayfun(@(a) tc.q_(m, 1, a, v, gamma), r.Actions);
            tc.verifyLessThan(abs(r.ActionValues - q), 5 * r.Work.StandardErrors);
        end

        function rolloutBudgetSeedAndGlobalRng(tc)
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP.tiny();
            original = rng; tc.addTeardown(@() rng(original));
            rng(17); expected = rand(); rng(17);
            a = OceanAlgo.StateSearch.RolloutSearch('Seed', 4, 'RecordReturns', true).search(m, 1);
            b = OceanAlgo.StateSearch.RolloutSearch('Seed', 4, 'RecordReturns', true).search(m, 1);
            tc.verifyEqual(a.Trace, b.Trace);
            tc.verifyEqual(rand(), expected);
            c = OceanAlgo.StateSearch.RolloutSearch('MaxSimulatedSteps', 5).search(m, 1);
            tc.verifyEqual(c.Status, "budget_exhausted");
            tc.verifyLessThanOrEqual(max(c.ActionVisits) - min(c.ActionVisits), 1);
            tc.verifyTrue(ismember(c.Action, [1 2]));
        end

        % ---------------- MCTS (8.11) ----------------
        function mctsRootValuesApproachQStarOnTinyMDP(tc)
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP.tiny();
            r = OceanAlgo.StateSearch.MonteCarloTreeSearch('NumIterations', 4000, 'ExplorationConstant', 1, ...
                'DiscountFactor', 0.5, 'Seed', 2).search(m, 1);
            % q*(1,1) = .25 (2 + .5 * 1) + .75 * 4 = 3.625;  q*(1,2) = 3
            tc.verifyEqual(r.Action, 1);
            tc.verifyEqual(r.ActionValues(2), 3, 'AbsTol', 1e-12);
            tc.verifyEqual(r.ActionValues(1), 3.625, 'AbsTol', 0.05);
            tc.verifyEqual(sum(r.ActionVisits), 4000);
        end

        function mctsPicksOptimalActionOnRandomMDPs(tc)
            rs = RandStream('mrg32k3a', 'Seed', 13);
            checked = 0;
            while checked < 3
                m = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP.random(6, 3, 2, rs);
                gamma = 0.8;
                [~, Q] = m.valueIterationSweeps(zeros(6, 1), gamma, 300);
                q = sort(Q(1, :), 'descend');
                if q(1) - q(2) < 0.5, continue; end                    % clear optimum only
                [~, best] = max(Q(1, :));
                % Returns here spread over about +-10 (N(0,1) rewards, γ = 0.8). c must match that scale: with
                % c = 2 an arm whose first random rollouts were poor stayed locked out (diagnosed 2026-10-11).
                r = OceanAlgo.StateSearch.MonteCarloTreeSearch('NumIterations', 3000, 'ExplorationConstant', 5, ...
                    'MaxDepth', 40, 'DiscountFactor', gamma, 'Seed', checked + 1).search(m, 1);
                tc.verifyEqual(r.Action, best);
                checked = checked + 1;
            end
        end

        function mctsTreeInvariantsAndTrace(tc)
            rs = RandStream('mrg32k3a', 'Seed', 14);
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP.random(5, 2, 2, rs);
            eng = OceanAlgo.StateSearch.MonteCarloTreeSearch('NumIterations', 300, 'MaxDepth', 20, ...
                'DiscountFactor', 0.9, 'RecordIterations', true);
            r = eng.search(m, 1);
            t = eng.getTree();
            tc.verifyEqual(t.Visits(1), 300);
            tc.verifyEqual(arrayfun(@(i) sum(t.EdgeVisits{i}), (1:numel(t.Visits))'), t.Visits);
            tc.verifyLessThanOrEqual(r.Work.NodeCount, 301);            % at most one new node per iteration
            tc.verifyEqual(numel(r.Trace), 300);
            tc.verifyEqual(r.Trace(1).Nodes(1), 1);
        end

        function mctsTreeReuseAfterAdvance(tc)
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP.tiny();
            eng = OceanAlgo.StateSearch.MonteCarloTreeSearch('NumIterations', 200, 'DiscountFactor', 0.5);
            eng.search(m, 1);
            before = eng.getTree();
            child = find(before.Parent == 1 & before.ParentAction == 1 & strcmp(before.Key, '2'));
            tc.assertNotEmpty(child);
            tc.verifyTrue(eng.advance(1, 2));
            r = eng.search(m, 2);
            tc.verifyEqual(r.Work.ReusedVisits, before.Visits(child));
            tc.verifyEqual(eng.getTree().Visits(1), before.Visits(child) + 200);
            tc.verifyFalse(eng.advance(2, 3));                         % not in the tree: cleared
            tc.verifyEqual(eng.search(m, 1).Work.ReusedVisits, 0);
        end

        function mctsSeedReproducibleAndGlobalRngUntouched(tc)
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP.tiny();
            original = rng; tc.addTeardown(@() rng(original));
            rng(23); expected = rand(); rng(23);
            a = OceanAlgo.StateSearch.MonteCarloTreeSearch('NumIterations', 100, 'Seed', 5).search(m, 1);
            b = OceanAlgo.StateSearch.MonteCarloTreeSearch('NumIterations', 100, 'Seed', 5).search(m, 1);
            tc.verifyEqual(a.ActionVisits, b.ActionVisits);
            tc.verifyEqual(a.ActionValues, b.ActionValues);
            tc.verifyEqual(rand(), expected);
            tc.verifyEqual(OceanAlgo.StateSearch.MonteCarloTreeSearch().search(m, 3).Status, "no_legal_action");
        end
    end

    methods (Static, Access = private)
        function m = deterministicMDP_(S, A, rs)
            o = @(n, r, t) struct('Next', n, 'Reward', r, 'Terminated', t, 'Probability', 1);
            T = cell(S, A);
            for s = 1:S
                for a = 1:A
                    if a > 1 && rand(rs) < 0.2, continue; end          % action 1 always legal
                    T{s, a} = o(randi(rs, S), randn(rs), rand(rs) < 0.1);
                end
            end
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP(T);
        end
        function a = firstAction_(m, s)
            acts = m.actions(s); a = acts(1);
        end
        function q = q_(m, s, a, v, gamma)
            d = m.outcomes(s, a); q = 0;
            for k = 1:numel(d)
                c = 0; if ~d(k).Terminated, c = v(d(k).Next); end
                q = q + d(k).Probability * (d(k).Reward + gamma * c);
            end
        end
    end
end
