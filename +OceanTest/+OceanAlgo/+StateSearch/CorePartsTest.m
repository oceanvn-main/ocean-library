classdef CorePartsTest < OceanTest.BaseTestCase
    % COREPARTSTEST  S0 parts of OceanAlgo.StateSearch: heap, state table, outcome checks, sample model,
    % rollout return, UCB1 selection, SearchResult.
    methods (Test)
        function heapPopsInLexicographicOrder(tc)
            rs = RandStream('mrg32k3a', 'Seed', 3);
            keys = [randi(rs, 5, 200, 1), randi(rs, 5, 200, 1), (1:200)'];
            h = OceanAlgo.StateSearch.MinHeap(3, 4);           % small capacity: forces doubling
            for i = 1:200, h.push(i, keys(i, :)); end
            got = zeros(200, 3);
            for i = 1:200, [~, got(i, :)] = h.pop(); end
            tc.verifyEqual(got, sortrows(keys));
            tc.verifyTrue(h.isEmpty());
            tc.verifyError(@() h.pop(), 'OceanAlgo:StateSearch:EmptyHeap');
            tc.verifyError(@() h.push(1, [1 NaN 0]), 'OceanAlgo:StateSearch:BadPriority');
        end

        function stateTableNumericAndString(tc)
            t = OceanAlgo.StateSearch.StateTable();
            tc.verifyEqual(t.get(7), 0);
            t.set(7, 3); t.set(2.5, 4);
            tc.verifyEqual([t.get(7), t.get(2.5), t.count()], [3 4 2]);
            tc.verifyError(@() t.set("a", 1), 'OceanAlgo:StateSearch:BadKey');
            u = OceanAlgo.StateSearch.StateTable();
            u.set("r1c2", 5);
            tc.verifyEqual(u.get('r1c2'), 5);
            tc.verifyError(@() u.get(3), 'OceanAlgo:StateSearch:BadKey');
            tc.verifyError(@() u.set([1 2], 1), 'OceanAlgo:StateSearch:BadKey');
        end

        function outcomeChecks(tc)
            o = @(p) struct('Next', 1, 'Reward', 0, 'Terminated', false, 'Probability', p);
            tc.verifyError(@() OceanAlgo.StateSearch.checkOutcomes([o(0.5) o(0.4)]), 'OceanAlgo:StateSearch:BadOutcomes');
            tc.verifyError(@() OceanAlgo.StateSearch.checkOutcomes([o(0.5) o(0.5)], true), 'OceanAlgo:StateSearch:NotDeterministic');
            tc.verifyError(@() OceanAlgo.StateSearch.checkOutcomes(struct('Next', 1)), 'OceanAlgo:StateSearch:BadOutcomes');
            tc.verifyEqual(numel(OceanAlgo.StateSearch.checkOutcomes([o(0.5) o(0.5)])), 2);
        end

        function sampleModelMatchesDistribution(tc)
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP.tiny();
            sm = OceanAlgo.StateSearch.DistributionSampleModel(m);
            rs = RandStream('mrg32k3a', 'Seed', 11);
            n = 20000; term = 0;
            for i = 1:n, x = sm.sample(1, 1, rs); term = term + x.Terminated; end
            % P(terminate) = 0.75; 5 standard errors.
            tc.verifyLessThan(abs(term / n - 0.75), 5 * sqrt(0.75 * 0.25 / n));
            tc.verifyEqual(sm.actions(1), [1 2]);
        end

        function rolloutReturnHandValues(tc)
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP.tiny();
            pol = OceanAlgo.StateSearch.UniformRandomRolloutPolicy();
            rs = RandStream('mrg32k3a', 'Seed', 1);
            h = @(s) 10 * s;
            [G, k, why] = OceanAlgo.StateSearch.rolloutReturn(m, 2, NaN, pol, rs, 0.5, 10, h);
            tc.verifyEqual([G k], [1 1]); tc.verifyEqual(why, "terminated");
            tc.verifyEqual(OceanAlgo.StateSearch.rolloutReturn(m, 1, NaN, pol, rs, 0.5, 0, h), 10);
            [G, ~, why] = OceanAlgo.StateSearch.rolloutReturn(m, 3, NaN, pol, rs, 0.5, 5, h);
            tc.verifyEqual(G, 30); tc.verifyEqual(why, "no_action");
            % Truncation after one step from (1, a1): G = 4 (terminated) or 2 + 0.5 * h(2) = 12.
            seen = zeros(1, 0);
            for i = 1:50
                seen(end+1) = OceanAlgo.StateSearch.rolloutReturn(m, 1, 1, pol, rs, 0.5, 1, h); %#ok<AGROW>
            end
            tc.verifyEqual(unique(seen), [4 12]);
        end

        function rolloutDoesNotTouchGlobalRng(tc)
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP.tiny();
            pol = OceanAlgo.StateSearch.UniformRandomRolloutPolicy();
            original = rng; tc.addTeardown(@() rng(original));
            rng(5); expected = rand(); rng(5);
            for i = 1:20
                OceanAlgo.StateSearch.rolloutReturn(m, 1, NaN, pol, RandStream('mrg32k3a', 'Seed', i), 1, 5, @(s) 0);
            end
            tc.verifyEqual(rand(), expected);
        end

        function ucbTriesUntriedThenMaximises(tc)
            rule = OceanAlgo.StateSearch.UCB1Selection(2);
            rs = RandStream('mrg32k3a', 'Seed', 2);
            tc.verifyEqual(rule.select([5 9 1], [3 0 2], rs), 2);
            Q = [1 0.5 0.9]; N = [10 1 4];
            u = Q + 2 * sqrt(log(15) ./ N); [~, best] = max(u);
            tc.verifyEqual(rule.select(Q, N, rs), best);
        end

        function searchResultValidates(tc)
            r = OceanAlgo.StateSearch.SearchResult('Algorithm', "x", 'Actions', [1 2], 'ActionValues', [0 1], 'Action', 2);
            tc.verifyEqual(r.Action, 2);
            tc.verifyError(@() OceanAlgo.StateSearch.SearchResult('Status', "bogus"), 'OceanAlgo:StateSearch:BadStatus');
            tc.verifyError(@() OceanAlgo.StateSearch.SearchResult('Actions', 1, 'ActionValues', [1 2]), ...
                'OceanAlgo:StateSearch:BadResult');
        end
    end
end
