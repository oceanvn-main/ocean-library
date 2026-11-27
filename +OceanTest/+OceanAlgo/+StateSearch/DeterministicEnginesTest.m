classdef DeterministicEnginesTest < OceanTest.BaseTestCase
    % DETERMINISTICENGINESTEST  A*, weighted A*, IDA* and beam search against independent oracles
    % (OceanMath Dijkstra, brute-force enumeration, one-step greedy lookahead).
    properties (Constant)
        Fx = 'OceanTest.OceanAlgo.StateSearch.Fixtures.'
    end

    methods (Test)
        % ---------------- A* ----------------
        function aStarZeroHeuristicMatchesDijkstra(tc)
            rs = RandStream('mrg32k3a', 'Seed', 1);
            for trial = 1:15
                n = 25; W = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel.randomGraph(n, 0.12, rs);
                m = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel(W, n);
                [~, dist] = OceanMath.Algorithms.Graph.Dijkstra.solve(W, 1, n);
                r = OceanAlgo.StateSearch.AStarSearch().search(m, 1);
                tc.verifyEqual(r.Status, "ok");
                tc.verifyEqual(-r.Value, dist, 'AbsTol', 1e-9);
                tc.verifyEqual(sum(arrayfun(@(k) W(r.Path.States{k}, r.Path.States{k+1}), 1:numel(r.Path.Actions))), dist, ...
                    'AbsTol', 1e-9);                                   % the path really has that cost
                tc.verifyEqual(r.Path.States{end}, n);
            end
        end

        function aStarAdmissibleInconsistentHeuristicStaysOptimal(tc)
            rs = RandStream('mrg32k3a', 'Seed', 2);
            for trial = 1:15
                n = 25; W = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel.randomGraph(n, 0.12, rs);
                m = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel(W, n);
                d = tc.costToGo_(W, n);
                u = rand(rs, n, 1);
                h = @(s) u(s) * d(s);                                   % admissible, generally inconsistent
                [~, dist] = OceanMath.Algorithms.Graph.Dijkstra.solve(W, 1, n);
                r = OceanAlgo.StateSearch.AStarSearch('Heuristic', h).search(m, 1);
                tc.verifyEqual(-r.Value, dist, 'AbsTol', 1e-9);
            end
        end

        function aStarConsistentHeuristicExpandsLessAndNeverReopens(tc)
            rows = 8; cols = 10; goal = rows * cols;
            blocked = [14 24 34 44 54 64 37 47 57 67 77];
            W = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel.unitGrid(rows, cols, blocked);
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel(W, goal);
            h = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel.gridHeuristic(rows, cols, goal);
            r0 = OceanAlgo.StateSearch.AStarSearch().search(m, 1);
            r1 = OceanAlgo.StateSearch.AStarSearch('Heuristic', h, 'RecordTrace', true).search(m, 1);
            [~, dist] = OceanMath.Algorithms.Graph.Dijkstra.solve(W, 1, goal);
            tc.verifyEqual([-r0.Value, -r1.Value], [dist dist], 'AbsTol', 1e-12);
            tc.verifyLessThanOrEqual(r1.Work.Expansions, r0.Work.Expansions);
            tc.verifyEqual(r1.Work.Reopened, 0);
            tc.verifyTrue(all(diff(r1.Trace.F) >= -1e-12));         % consistent h: f never decreases
        end

        function weightedAStarWithinBound(tc)
            rs = RandStream('mrg32k3a', 'Seed', 3);
            for w = [1.5 2 4]
                for trial = 1:10
                    n = 25; W = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel.randomGraph(n, 0.15, rs);
                    m = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel(W, n);
                    d = tc.costToGo_(W, n); u = rand(rs, n, 1);    % random admissible heuristic u(s) * h*(s)
                    [~, dist] = OceanMath.Algorithms.Graph.Dijkstra.solve(W, 1, n);
                    r = OceanAlgo.StateSearch.AStarSearch('Heuristic', @(s) u(s) * d(s), 'Weight', w).search(m, 1);
                    tc.verifyLessThanOrEqual(-r.Value, w * dist + 1e-9);
                    tc.verifyEqual(r.Algorithm, "weighted A*");
                end
            end
        end

        function aStarReopensWithInconsistentHeuristic(tc)
            % S->A (1), S->B (2), A->C (3), B->C (1), C->G (4); optimal S-B-C-G = 7. h(B) = 5 is admissible
            % (true cost-to-go 5) but inconsistent (h(B) > c(B,C) + h(C) = 1): C is first closed with g = 4 via A,
            % then reached again with g = 3 via B and must be re-opened.
            W = zeros(5); W(1,2) = 1; W(1,3) = 2; W(2,4) = 3; W(3,4) = 1; W(4,5) = 4;
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel(W, 5);
            hv = [0 0 5 0 0];
            r = OceanAlgo.StateSearch.AStarSearch('Heuristic', @(s) hv(s)).search(m, 1);
            tc.verifyEqual(r.Value, -7);
            tc.verifyEqual(r.Work.Reopened, 1);
            tc.verifyEqual(cell2mat(r.Path.States), [1 3 4 5]);
            tc.verifyEqual(OceanAlgo.StateSearch.IDAStarSearch('Heuristic', @(s) hv(s)).search(m, 1).Value, -7);
        end

        function zeroProbabilityRowsAreDropped(tc)
            o = @(n, r, t, p) struct('Next', n, 'Reward', r, 'Terminated', t, 'Probability', p);
            T = {[o(1, -1, true, 1), o(1, -50, true, 0)]};
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP(T);
            tc.verifyEqual(OceanAlgo.StateSearch.AStarSearch().search(m, 1).Value, -1);
            tc.verifyEqual(numel(OceanAlgo.StateSearch.checkOutcomes(T{1})), 1);
        end

        function aStarStatusesAndErrors(tc)
            W = [0 1 0; 0 0 0; 0 0 0];                                  % 3 unreachable
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel(W, 3);
            tc.verifyEqual(OceanAlgo.StateSearch.AStarSearch().search(m, 1).Status, "no_path");
            tc.verifyEqual(OceanAlgo.StateSearch.AStarSearch().search(m, 3).Status, "at_goal");
            tc.verifyEqual(OceanAlgo.StateSearch.AStarSearch().search(m, 2).Status, "no_legal_action");
            tiny = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP.tiny();   % stochastic
            tc.verifyError(@() OceanAlgo.StateSearch.AStarSearch().search(tiny, 1), 'OceanAlgo:StateSearch:NotDeterministic');
            o = @(n, r, t) struct('Next', n, 'Reward', r, 'Terminated', t, 'Probability', 1);
            pos = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP({o(1, 2, true)});
            tc.verifyError(@() OceanAlgo.StateSearch.AStarSearch().search(pos, 1), 'OceanAlgo:StateSearch:PositiveReward');
        end

        function aStarBudgetReturnsLegalAction(tc)
            rows = 6; cols = 6; W = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel.unitGrid(rows, cols, []);
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel(W, rows * cols);
            r = OceanAlgo.StateSearch.AStarSearch('MaxExpansions', 3).search(m, 1);
            tc.verifyEqual(r.Status, "budget_exhausted");
            tc.verifyTrue(ismember(r.Action, m.actions(1)));
            tc.verifyEqual(r.Work.Expansions, 3);
            tc.verifyTrue(isnan(r.Value));                                % a partial path is not a plan value
        end

        function terminatedTransitionsAreGoals(tc)
            % Chain 1 -> 2 -> 3 (cost 1 each) then a terminating action from 3 (cost 1); shortcut 1 -> end costs 5.
            o = @(n, r, t) struct('Next', n, 'Reward', r, 'Terminated', t, 'Probability', 1);
            T = cell(3, 2);
            T{1,1} = o(2, -1, false); T{1,2} = o(3, -5, true);
            T{2,1} = o(3, -1, false);
            T{3,1} = o(3, -1, true);
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP(T);
            for engine = {OceanAlgo.StateSearch.AStarSearch(), OceanAlgo.StateSearch.IDAStarSearch()}
                r = engine{1}.search(m, 1);
                tc.verifyEqual(r.Value, -3); tc.verifyEqual(r.Path.Actions, [1 1 1]);
                tc.verifyTrue(r.Path.Terminated);
            end
        end

        % ---------------- IDA* ----------------
        function idaStarMatchesAStarCost(tc)
            rs = RandStream('mrg32k3a', 'Seed', 4);
            for trial = 1:12
                n = 14; W = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel.randomGraph(n, 0.2, rs);
                m = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel(W, n);
                d = tc.costToGo_(W, n); u = rand(rs, n, 1); h = @(s) u(s) * d(s);
                a = OceanAlgo.StateSearch.AStarSearch('Heuristic', h).search(m, 1);
                i0 = OceanAlgo.StateSearch.IDAStarSearch().search(m, 1);
                i1 = OceanAlgo.StateSearch.IDAStarSearch('Heuristic', h).search(m, 1);
                tc.verifyEqual([i0.Value, i1.Value], [a.Value, a.Value], 'AbsTol', 1e-9);
                tc.verifyEqual(i1.Path.States{end}, n);
                tc.verifyTrue(all(diff(i1.Work.Thresholds) > 0));
            end
        end

        function idaStarStatuses(tc)
            W = [0 1 0; 1 0 0; 0 0 0];                                  % cycle 1 <-> 2, goal 3 unreachable
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel(W, 3);
            tc.verifyEqual(OceanAlgo.StateSearch.IDAStarSearch().search(m, 1).Status, "no_path");
            G = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel.unitGrid(6, 6, []);
            g = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel(G, 36);
            r = OceanAlgo.StateSearch.IDAStarSearch('MaxExpansions', 5).search(g, 1);
            tc.verifyEqual(r.Status, "budget_exhausted");
            tc.verifyTrue(ismember(r.Action, g.actions(1)));
        end

        % ---------------- Beam search ----------------
        function beamWidthOneIsGreedyLookahead(tc)
            rs = RandStream('mrg32k3a', 'Seed', 5);
            for trial = 1:10
                n = 12; W = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel.randomGraph(n, 0.3, rs);
                W(n, 1) = randi(rs, 9);                                 % every node has an exit: no dead ends
                m = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel(W, 0);  % no goal
                hv = randn(rs, n, 1); h = @(s) hv(s); gamma = 0.9; D = 6;
                r = OceanAlgo.StateSearch.BeamSearch('BeamWidth', 1, 'MaxDepth', D, 'LeafValue', h, ...
                    'DiscountFactor', gamma).search(m, 1);
                s = 1; expected = zeros(1, D);
                for d = 1:D                                             % one-step lookahead, ties -> first
                    acts = m.actions(s);
                    vals = arrayfun(@(a) -W(s, a) + gamma * h(a), acts);
                    [~, k] = max(vals); expected(d) = acts(k); s = acts(k);
                end
                tc.verifyEqual(r.Path.Actions, expected);
            end
        end

        function wideBeamIsExhaustive(tc)
            rs = RandStream('mrg32k3a', 'Seed', 6);
            for trial = 1:10
                m = tc.randomDeterministicMDP_(7, 3, rs);
                hv = randn(rs, 7, 1); h = @(s) hv(s); gamma = 0.8; D = 4;
                truth = tc.bruteForce_(m, 1, D, h, gamma);
                for merge = [true false]
                    r = OceanAlgo.StateSearch.BeamSearch('BeamWidth', 10000, 'MaxDepth', D, 'LeafValue', h, ...
                        'DiscountFactor', gamma, 'MergeDuplicates', merge).search(m, 1);
                    tc.verifyEqual(r.Value, truth, 'AbsTol', 1e-12);
                end
            end
        end

        function diverseBeamSpreadsFirstActions(tc)
            W = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel.unitGrid(5, 5, []);
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel(W, 25);
            h = OceanTest.OceanAlgo.StateSearch.Fixtures.GraphModel.gridHeuristic(5, 5, 25);
            same = OceanAlgo.StateSearch.BeamSearch('BeamWidth', 2, 'Groups', 2, 'DiversityStrength', 0, ...
                'MaxDepth', 1, 'LeafValue', @(s) -h(s), 'NBest', 2).search(m, 1);
            diverse = OceanAlgo.StateSearch.BeamSearch('BeamWidth', 2, 'Groups', 2, 'DiversityStrength', 100, ...
                'MaxDepth', 1, 'LeafValue', @(s) -h(s), 'NBest', 4).search(m, 1);
            tc.verifyEqual(numel(unique(arrayfun(@(c) c.Actions(1), same.Candidates))), 1);
            tc.verifyGreaterThan(numel(unique(arrayfun(@(c) c.Actions(1), diverse.Candidates))), 1);
            tc.verifyEqual(diverse.Algorithm, "diverse beam search");
            tc.verifyError(@() OceanAlgo.StateSearch.BeamSearch('BeamWidth', 3, 'Groups', 2), ...
                'OceanAlgo:StateSearch:BadGroups');
        end

        function beamFindsTerminatedPlanAndRespectsBudget(tc)
            o = @(n, r, t) struct('Next', n, 'Reward', r, 'Terminated', t, 'Probability', 1);
            T = cell(3, 2);
            T{1,1} = o(2, -1, false); T{1,2} = o(3, -5, true);
            T{2,1} = o(3, -1, false);
            T{3,1} = o(3, -1, true);
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP(T);
            r = OceanAlgo.StateSearch.BeamSearch('BeamWidth', 2, 'MaxDepth', 5).search(m, 1);
            tc.verifyEqual([r.Value, r.Path.Actions], [-3 1 1 1]);
            tc.verifyEqual(r.ActionValues, [-3 -5]);
            b = OceanAlgo.StateSearch.BeamSearch('BeamWidth', 2, 'MaxDepth', 5, 'MaxExpansions', 1).search(m, 1);
            tc.verifyEqual(b.Status, "budget_exhausted");
            tc.verifyTrue(ismember(b.Action, [1 2]));
            d = OceanAlgo.StateSearch.BeamSearch('BeamWidth', 2, 'Groups', 2, 'MaxDepth', 5, 'MaxExpansions', 1).search(m, 1);
            tc.verifyEqual(d.Status, "budget_exhausted");                % group 2 never expanded: no crash
            tc.verifyTrue(ismember(d.Action, [1 2]));
            tiny = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP.tiny();
            tc.verifyError(@() OceanAlgo.StateSearch.BeamSearch().search(tiny, 1), 'OceanAlgo:StateSearch:NotDeterministic');
        end
    end

    methods (Static, Access = private)
        function d = costToGo_(W, goal)
            % Exact cost-to-go to `goal` for every node (Dijkstra on the reversed graph, via the oracle).
            n = size(W, 1); d = inf(n, 1);
            for s = 1:n
                [~, d(s)] = OceanMath.Algorithms.Graph.Dijkstra.solve(W, s, goal);
            end
            d(goal) = 0; d(isinf(d)) = 1e6;
        end
        function m = randomDeterministicMDP_(S, A, rs)
            o = @(n, r, t) struct('Next', n, 'Reward', r, 'Terminated', t, 'Probability', 1);
            T = cell(S, A);
            for s = 1:S
                for a = 1:A
                    if rand(rs) < 0.2, continue; end                       % some illegal actions
                    T{s, a} = o(randi(rs, S), randn(rs), rand(rs) < 0.15);
                end
            end
            m = OceanTest.OceanAlgo.StateSearch.Fixtures.TableMDP(T);
        end
        function v = bruteForce_(m, s, d, h, gamma)
            % Best score over every action sequence of length <= d (independent recursion).
            acts = m.actions(s);
            if isempty(acts) || d == 0, v = h(s); return; end
            v = -Inf;
            for a = acts
                o = m.outcomes(s, a);
                if o.Terminated, c = 0; else, c = OceanTest.OceanAlgo.StateSearch.DeterministicEnginesTest.bruteForce_(m, o.Next, d - 1, h, gamma); end
                v = max(v, o.Reward + gamma * c);
            end
        end
    end
end
