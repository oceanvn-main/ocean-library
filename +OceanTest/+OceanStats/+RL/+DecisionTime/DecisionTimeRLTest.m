classdef DecisionTimeRLTest < OceanTest.BaseTestCase
    % DECISIONTIMERLTEST  OceanAlgo.StateSearch engines applied to RL models through WorldModelSearchModel,
    % checked against the RL library's own exact backups (BellmanOptimality) and value iteration.

    methods (Test)
        function adapterMatchesBellmanBackup(tc)
            model = OceanTest.OceanStats.RL.Dyna.Fixtures.TinyRTDPModel();
            pm = OceanStats.RL.DecisionTime.WorldModelSearchModel(model);
            V0 = [0; 8; 99];
            [v, a, q] = OceanStats.RL.DP.BellmanOptimality.backup(model, 1, V0, 0.5);
            r = OceanAlgo.StateSearch.DepthLimitedSearch('Depth', 1, 'LeafValue', @(s) V0(s), ...
                'DiscountFactor', 0.5).search(pm, 1);
            tc.verifyEqual(r.ActionValues, q(r.Actions), 'AbsTol', 1e-12);
            tc.verifyEqual([r.Action, r.Value], [a, v], 'AbsTol', 1e-12);
            tc.verifyEqual(pm.actions(2), 1);
            tc.verifyEmpty(pm.actions(3));
        end

        function pathEnginesFindOptimalCliffWalk(tc)
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv();
            pm = OceanStats.RL.DecisionTime.WorldModelSearchModel(env);
            V = tc.valueIteration_(pm, 1);
            start = env.idx(4, 1);
            tc.assertEqual(V(start), -13, 'AbsTol', 1e-9);              % up, 11 x right, down
            [gr, gc] = deal(4, 12);
            h = @(s) abs(floor((s - 1) / 12) + 1 - gr) + abs(mod(s - 1, 12) + 1 - gc);   % Manhattan, admissible
            engines = {OceanAlgo.StateSearch.AStarSearch(), OceanAlgo.StateSearch.AStarSearch('Heuristic', h), ...
                OceanAlgo.StateSearch.IDAStarSearch('Heuristic', h), ...
                OceanAlgo.StateSearch.BeamSearch('BeamWidth', 48, 'MaxDepth', 20, 'LeafValue', @(s) -h(s))};
            for k = 1:numel(engines)
                r = engines{k}.search(pm, start);
                tc.verifyEqual(r.Value, V(start), 'AbsTol', 1e-9, r.Algorithm);
                tc.verifyTrue(r.Path.Terminated, r.Algorithm);
            end
        end

        function depthLimitedMatchesValueIterationSweepsOnStochasticWindyGridworld(tc)
            env = OceanStats.RL.Env.Classic.WindyGridworldEnv('KingMoves', true, 'StochasticWind', true);
            pm = OceanStats.RL.DecisionTime.WorldModelSearchModel(env);
            S = pm.numStates(); V = zeros(S, 1);
            for k = 1:3                                                % 2 sweeps give V_2, then the root q_3
                if k < 3, V = tc.sweep_(pm, V, 1); end
            end
            start = env.idx(4, 1);
            [~, ~, q] = OceanStats.RL.DP.BellmanOptimality.backup(pm.WorldModel, start, V, 1);
            r = OceanAlgo.StateSearch.DepthLimitedSearch('Depth', 3).search(pm, start);
            tc.verifyEqual(r.ActionValues, q(r.Actions), 'AbsTol', 1e-12);
        end

        function agentEpisodesReachTheOptimalReturn(tc)
            env = OceanStats.RL.Env.Classic.WindyGridworldEnv();
            pm = OceanStats.RL.DecisionTime.WorldModelSearchModel(env);
            V = tc.valueIteration_(pm, 1);
            start = env.idx(4, 1);
            tc.assertEqual(V(start), -15, 'AbsTol', 1e-9);              % S&B Example 6.5
            % Depth-limited search with h = 0 needs Depth > distance to goal (15): at Depth = 15 a path cut off at
            % the horizon scores -15, the same as reaching the goal, every action ties and the first one ("up")
            % is chosen forever -- the imperfect-leaf-value horizon effect of S&B 8.9.
            stuck = OceanStats.RL.DecisionTime.DecisionTimeAgent(OceanAlgo.StateSearch.DepthLimitedSearch('Depth', 15), pm) ...
                .runEpisode(env, 'Seed', 1, 'MaxSteps', 30);
            tc.verifyEqual(stuck.Ended, "max_steps");
            for engine = {OceanAlgo.StateSearch.AStarSearch(), OceanAlgo.StateSearch.DepthLimitedSearch('Depth', 16)}
                agent = OceanStats.RL.DecisionTime.DecisionTimeAgent(engine{1}, pm);
                ep = agent.runEpisode(env, 'Seed', 1, 'MaxSteps', 100);
                tc.verifyEqual(ep.Return, -15);
                tc.verifyEqual(ep.Ended, "terminated");
                tc.verifyEqual(ep.States(1), start);
            end
        end

        function mctsAndRolloutChooseTheSafeFirstMoveOnASmallCliff(tc)
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv('Rows', 3, 'Cols', 5);
            pm = OceanStats.RL.DecisionTime.WorldModelSearchModel(env);
            gamma = 0.95;
            [V, Q] = tc.valueIteration_(pm, gamma);
            start = env.idx(3, 1);
            [~, best] = max(Q(start, :));
            tc.assertEqual(best, 1);                                    % up: the right move falls off the cliff
            % Rollout with the optimal policy as rollout policy: greedy on q_pi* is optimal.
            piStar = zeros(pm.numStates(), 1);
            for s = 1:pm.numStates()
                acts = pm.actions(s); if isempty(acts), piStar(s) = 1; continue; end
                [~, k] = max(Q(s, acts)); piStar(s) = acts(k);
            end
            ro = OceanAlgo.StateSearch.RolloutSearch('RolloutPolicy', OceanStats.RL.DecisionTime.TableRolloutPolicy(piStar), ...
                'NumRollouts', 1, 'Horizon', 200, 'DiscountFactor', gamma).search(pm, start);
            tc.verifyEqual(ro.Action, 1);
            tc.verifyEqual(ro.Value, V(start), 'AbsTol', 1e-6);
            % MCTS with random rollouts; c matched to the return scale (cliff costs -100).
            mc = OceanAlgo.StateSearch.MonteCarloTreeSearch('NumIterations', 4000, 'ExplorationConstant', 50, ...
                'MaxDepth', 30, 'DiscountFactor', gamma, 'Seed', 3).search(pm, start);
            tc.verifyEqual(mc.Action, 1);
        end

        function tableRolloutPolicyValidates(tc)
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv('Rows', 2, 'Cols', 3);
            pm = OceanStats.RL.DecisionTime.WorldModelSearchModel(env);
            rs = RandStream('mrg32k3a', 'Seed', 1);
            tc.verifyError(@() OceanStats.RL.DecisionTime.TableRolloutPolicy([1.5; 2]), ...
                'OceanStats:RL:DecisionTime:BadPolicyTable');
            tc.verifyError(@() OceanStats.RL.DecisionTime.TableRolloutPolicy([0.5 0.4]), ...
                'OceanStats:RL:DecisionTime:BadPolicyTable');
            goal = env.idx(2, 3);
            bad = OceanStats.RL.DecisionTime.TableRolloutPolicy(ones(6, 1));
            tc.verifyError(@() bad.act(pm, goal, rs), 'OceanStats:RL:DecisionTime:InadmissibleAction');
            p = zeros(6, 4); p(:, 2) = 1;
            tc.verifyEqual(OceanStats.RL.DecisionTime.TableRolloutPolicy(p).act(pm, 1, rs), 2);
        end

        function learnedModelPlansOnlyOnObservedPairs(tc)
            model = OceanStats.RL.Dyna.Models.DeterministicTabularModel(2, 1, true(2, 1));
            model.update(struct('StateIndex', 1, 'ActionIndex', 1, 'Reward', -1, 'NextStateIndex', 2, ...
                'Terminated', false, 'Truncated', false, 'Origin', "real", 'RealStep', 1, 'PlanningStep', 0));
            pm = OceanStats.RL.DecisionTime.WorldModelSearchModel(model);
            % Only (1,1) was observed: state 2 has no known action, so every search treats it as a leaf.
            tc.verifyEmpty(pm.actions(2));
            for d = 1:3
                r = OceanAlgo.StateSearch.DepthLimitedSearch('Depth', d, 'LeafValue', @(s) 10 * s).search(pm, 1);
                tc.verifyEqual(r.Value, -1 + 20);
            end
            m = OceanAlgo.StateSearch.MonteCarloTreeSearch('NumIterations', 20, 'LeafValue', @(s) 10 * s).search(pm, 1);
            tc.verifyEqual(m.ActionValues, 19, 'AbsTol', 1e-12);
        end
    end

    methods (Static, Access = private)
        function V = sweep_(pm, V, gamma)
            Vn = V;
            for s = 1:pm.numStates()
                Vn(s) = OceanStats.RL.DP.BellmanOptimality.backup(pm.WorldModel, s, V, gamma);
            end
            V = Vn;
        end
        function [V, Q] = valueIteration_(pm, gamma)
            S = pm.numStates(); V = zeros(S, 1);
            for it = 1:5000
                Vn = OceanTest.OceanStats.RL.DecisionTime.DecisionTimeRLTest.sweep_(pm, V, gamma);
                if max(abs(Vn - V)) < 1e-12, V = Vn; break; end
                V = Vn;
            end
            Q = -inf(S, pm.WorldModel.NumActions);
            for s = 1:S
                if isempty(pm.actions(s)), continue; end
                [~, ~, q] = OceanStats.RL.DP.BellmanOptimality.backup(pm.WorldModel, s, V, gamma);
                Q(s, :) = q; Q(s, isnan(q)) = -Inf;
            end
        end
    end
end
