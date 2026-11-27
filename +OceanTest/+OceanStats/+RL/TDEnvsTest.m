classdef TDEnvsTest < OceanTest.BaseTestCase
    % TDENVSTEST  The four TD environments: exact models (against DP and hand calculation), sampling agreeing with
    % the model, seeding, termination vs truncation, and compatibility with the existing sampling contract.

    methods (Test)
        function testRandomWalkExactValuesAreKOverNPlusOne(testCase)
            env = OceanStats.RL.Env.Classic.RandomWalkEnv();
            V = env.exactPolicyValues(env.uniformPolicy(), 1);
            testCase.verifyEqual(V(2:6)', (1:5) / 6, 'AbsTol', 1e-12);
            testCase.verifyEqual(V([1 7]), [0; 0]);
            env19 = OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner', 19);
            V19 = env19.exactPolicyValues(env19.uniformPolicy(), 1);
            testCase.verifyEqual(V19(2:20)', (1:19) / 20, 'AbsTol', 1e-12);
        end

        function testModelsAreProperDistributionsAndTerminalsHaveNoActions(testCase)
            envs = {OceanStats.RL.Env.Classic.RandomWalkEnv(), OceanStats.RL.Env.Classic.CliffWalkingEnv(), ...
                    OceanStats.RL.Env.Classic.WindyGridworldEnv('StochasticWind', true, 'KingMoves', true), ...
                    OceanStats.RL.Env.Classic.MaximizationBiasEnv()};
            for i = 1:numel(envs)
                e = envs{i};
                for s = 1:e.NumStates
                    if e.is_terminal(s)
                        testCase.verifyEmpty(e.get_valid_actions(s));
                        continue
                    end
                    for a = e.get_valid_actions(s)
                        [nx, p, r] = e.get_transitions(s, a);
                        testCase.verifyEqual(sum(p), 1, 'AbsTol', 1e-12);
                        testCase.verifyTrue(all(nx >= 1 & nx <= e.NumStates));
                        testCase.verifyEqual(size(r), size(p));
                    end
                end
            end
        end

        function testWindyGridworldOptimalPathLengthsMatchTheBook(testCase)
            % Sutton & Barto Ex 6.5: 15 steps; Ex 6.9: 7 with king moves (and 7 with the stay action too)
            for cfg = {{}, {'KingMoves', true}, {'KingMoves', true, 'StayAction', true}}
                env = OceanStats.RL.Env.Classic.WindyGridworldEnv(cfg{1}{:});
                V = OceanStats.RL.DP.ValueIterationSolver(env, 'DiscountFactor', 1).run();
                expected = -15;
                if numel(cfg{1}) > 0, expected = -7; end
                testCase.verifyEqual(V(env.idx(4, 1)), expected, 'AbsTol', 1e-9);
            end
        end

        function testWindyGridworldWindMovesAgentUpHandWorked(testCase)
            env = OceanStats.RL.Env.Classic.WindyGridworldEnv();
            % from (4,4) (wind 1) moving right: column 5, row 4 - 1 = 3
            [nx, p, r] = env.get_transitions(env.idx(4, 4), 2);
            testCase.verifyEqual(nx, env.idx(3, 5));  testCase.verifyEqual(p, 1);  testCase.verifyEqual(r, -1);
            % from (4,7) (wind 2) moving up: row 4 - 1 - 2 = 1 (clamped at the top)
            testCase.verifyEqual(env.get_transitions(env.idx(4, 7), 1), env.idx(1, 7));
            % walls: moving left from column 1 stays in column 1
            [nx1] = env.get_transitions(env.idx(4, 1), 4);
            testCase.verifyEqual(nx1, env.idx(4, 1));
            % stochastic wind: 3 equiprobable outcomes in windy columns, 1 elsewhere
            sw = OceanStats.RL.Env.Classic.WindyGridworldEnv('StochasticWind', true);
            [~, p3] = sw.get_transitions(sw.idx(4, 7), 2);
            testCase.verifyEqual(p3, ones(3, 1) / 3, 'AbsTol', 1e-14);
            [~, p1] = sw.get_transitions(sw.idx(4, 2), 2);
            testCase.verifyEqual(p1, 1);
        end

        function testCliffWalkingHandWorkedAndOptimum(testCase)
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv();
            start = env.idx(4, 1);  goal = env.idx(4, 12);
            [nx, ~, r] = env.get_transitions(start, 2);                 % right from start = into the cliff
            testCase.verifyEqual([nx r], [start -100]);                  % back to start, -100, NOT terminal
            [nx2, ~, r2] = env.get_transitions(env.idx(3, 5), 3);        % down from row 3 into the cliff row
            testCase.verifyEqual([nx2 r2], [start -100]);
            [nx3, ~, r3] = env.get_transitions(env.idx(3, 12), 3);       % down into the goal
            testCase.verifyEqual([nx3 r3], [goal -1]);
            testCase.verifyTrue(env.is_terminal(goal));
            V = OceanStats.RL.DP.ValueIterationSolver(env, 'DiscountFactor', 1).run();
            testCase.verifyEqual(V(start), -13, 'AbsTol', 1e-9);        % optimal path: 13 steps along the cliff edge
        end

        function testMaximizationBiasModelAndNoise(testCase)
            env = OceanStats.RL.Env.Classic.MaximizationBiasEnv();
            testCase.verifyEqual(env.get_valid_actions(1), [1 2]);
            testCase.verifyEqual(env.get_valid_actions(2), 1:10);
            testCase.verifyEqual(env.get_transitions(1, 2), 3);          % right -> terminal
            [~, ~, r] = env.get_transitions(2, 4);
            testCase.verifyEqual(r, -0.1);                               % DP sees the EXPECTED reward
            env.setSeed(3);
            rs = zeros(20000, 1);
            for i = 1:numel(rs)
                env.reset();  env.step(1);  [~, rs(i)] = env.step(3);
            end
            testCase.verifyEqual(mean(rs), -0.1, 'AbsTol', 0.04);        % sampled rewards ~ N(-0.1, 1)
            testCase.verifyEqual(std(rs), 1, 'AbsTol', 0.04);
            V = OceanStats.RL.DP.ValueIterationSolver(env, 'DiscountFactor', 1).run();
            testCase.verifyEqual(V(1), 0, 'AbsTol', 1e-9);               % optimal: go right immediately
        end

        function testSamplingMatchesTheModelFrequencies(testCase)
            env = OceanStats.RL.Env.Classic.WindyGridworldEnv('StochasticWind', true);
            env.setSeed(11);
            n = 9000;  s0 = env.idx(4, 7);
            draws = zeros(n, 1);
            for i = 1:n
                env.reset();  env.set_state(s0);
                draws(i) = env.step(2);
            end
            nx = env.get_transitions(s0, 2);
            for k = 1:numel(nx)
                testCase.verifyEqual(mean(draws == nx(k)), 1/3, 'AbsTol', 0.03);
            end
        end

        function testSeedingIsReproducible(testCase)
            traj = @(seed) OceanTest.OceanStats.RL.TDEnvsTest.rollout_(seed);
            testCase.verifyEqual(traj(7), traj(7));
            testCase.verifyNotEqual(traj(7), traj(8));
        end

        function testTerminationVersusTruncation(testCase)
            W = @(inner, n) OceanStats.RL.Env.Wrappers.TimeLimit(inner, n);
            env = W(OceanStats.RL.Env.Classic.RandomWalkEnv(), 3);
            env.reset('Seed', 1);
            [~, ~, te, tr] = env.step(2);  testCase.verifyFalse(te || tr);
            [~, ~, te, tr] = env.step(1);  testCase.verifyFalse(te || tr);
            [~, ~, te, tr] = env.step(2);                                % 3rd step: time limit, not at a terminal
            testCase.verifyTrue(~te && tr);
            % reaching a terminal on the last allowed step is TERMINATION, not truncation
            env2 = W(OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner', 1), 1);
            env2.reset('Seed', 1);
            [~, ~, te, tr] = env2.step(2);
            testCase.verifyTrue(te && ~tr);
            % environments without the wrapper never truncate
            e3 = OceanStats.RL.Env.Classic.CliffWalkingEnv();  e3.reset('Seed', 1);
            [~, ~, ~, tr] = e3.step(1);
            testCase.verifyFalse(tr);
        end

        function testInvalidActionIsRejected(testCase)
            env = OceanStats.RL.Env.Classic.MaximizationBiasEnv();  env.setSeed(1);  env.reset();
            testCase.verifyError(@() env.step(5), 'OceanStats:RL:Env:TabularEnvironment:InvalidAction');   % A only has actions 1, 2
        end

        function testExistingSamplingContractStillWorks(testCase)
            % the non-breaking was_truncated() addition: existing environments default to false ...
            bj = OceanStats.RL.MonteCarlo.BlackjackEnv();
            bj.reset();
            testCase.verifyFalse(bj.was_truncated());
            % ... and generate_episode (used by the Monte Carlo solver) works on the new environments through the adapter
            env = OceanStats.RL.Compat.LegacyEnvironment(OceanStats.RL.Env.Classic.CliffWalkingEnv());  env.setSeed(2);
            ep = env.generate_episode(@(s, e) 2 + (rand() < 0.5), 'MaxLen', 30);
            testCase.verifyGreaterThan(ep.Length, 0);
            testCase.verifyEqual(size(ep.Rewards, 1), ep.Length);
        end
    end

    methods (Static, Access = private)
        function t = rollout_(seed)
            env = OceanStats.RL.Env.Classic.WindyGridworldEnv('StochasticWind', true);  env.setSeed(seed);   % stochastic dynamics
            s = env.reset();  t = zeros(1, 12);
            for k = 1:12
                [s, ~, te, tr] = env.step(2);  t(k) = s;
                if te || tr, break; end
            end
        end
    end
end
