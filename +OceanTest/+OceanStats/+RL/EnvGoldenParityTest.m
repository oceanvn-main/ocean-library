classdef EnvGoldenParityTest < OceanTest.BaseTestCase
    % ENVGOLDENPARITYTEST  The refactored OceanStats.RL.Env.Classic.* environments (five-output step, TimeLimit as a wrapper)
    % must behave EXACTLY like the original OceanStats.RL.TD.Envs.* (three-output step, integrated TimeLimit): same
    % transition model, same terminal/admissible sets, and bit-identical seeded trajectories (states, actions, rewards,
    % done and truncated flags) over 15 configurations x 3 seeds x 400 steps. The reference is env_golden.mat, captured
    % from the ORIGINAL code by capture_env_golden.m before it was removed.

    methods (Test)
        function testModelsAndTrajectoriesAreIdenticalToTheOriginalEnvironments(testCase)
            g = load(fullfile(fileparts(mfilename('fullpath')), 'env_golden.mat'));
            builders = OceanTest.OceanStats.RL.EnvGoldenParityTest.builders_();
            testCase.verifyEqual(sort(string({g.golden.name})), sort(string(builders.keys())));
            C = OceanStats.RL.Env.Classic.RandomWalkEnv;  %#ok<NASGU> force package resolution
            for k = 1:numel(g.golden)
                gold = g.golden(k);
                mk = builders(gold.name);
                env = mk();
                base = env.unwrapped();
                S = base.NumStates;  A = base.NumActions;
                % ---- exact model, terminal set, admissible set
                model = zeros(0, 5);
                for s = 1:S
                    for a = base.get_valid_actions(s)
                        [n, p, r] = base.get_transitions(s, a);
                        model = [model; repmat([s a], numel(n), 1), n(:), p(:), r(:)]; %#ok<AGROW>
                    end
                end
                testCase.verifyEqual(model, gold.model, ['model ' gold.name]);
                testCase.verifyEqual(base.TerminalMask, gold.terminal, ['terminal ' gold.name]);
                testCase.verifyEqual(base.ValidMask, gold.validmask, ['valid mask ' gold.name]);
                testCase.verifyEqual([S A], size(gold.validmask), ['sizes ' gold.name]);
                % ---- seeded trajectories, bit-identical
                for j = 1:numel(g.seeds)
                    env = mk();
                    s = env.reset('Seed', g.seeds(j));
                    rows = zeros(400, 6);
                    for t = 1:400
                        v = base.get_valid_actions(s);
                        a = v(1 + mod(t - 1, numel(v)));
                        [s2, r, term, trunc, info] = env.step(a);
                        done = term || trunc;
                        rows(t, 1:5) = [s, a, r, done, trunc];
                        if done, s = env.reset(); else, s = s2; end
                        rows(t, 6) = s;
                        testCase.assertEqual(size(info.action_mask), [1 A]);
                    end
                    testCase.verifyEqual(rows, gold.traj{j}, sprintf('trajectory %s seed %d', gold.name, g.seeds(j)));
                end
            end
        end

        function testClassicEnvironmentsHonourTheContract(testCase)
            builders = OceanTest.OceanStats.RL.EnvGoldenParityTest.builders_();
            for name = string(builders.keys())
                mk = builders(char(name));
                env = mk();
                testCase.verifyTrue(isa(env, 'OceanStats.RL.Env.IEnvironment'), name);
                testCase.verifyTrue(isa(env.unwrapped(), 'OceanStats.RL.Env.IModelEnvironment'), name);
                [o, info] = env.reset('Seed', 1);
                testCase.verifyTrue(env.ObservationSpace.contains(o), name);
                a = find(info.action_mask, 1);
                [o2, ~, ~, ~, info2] = env.step(a);
                testCase.verifyTrue(env.ObservationSpace.contains(o2), name);
                testCase.verifyTrue(islogical(info2.action_mask) && numel(info2.action_mask) == env.ActionSpace.N, name);
                m = env.getMetadata();  p = env.getPipelineSpec();
                testCase.verifyFalse(m.IsSymbolic);  testCase.verifyEqual(p.Kind, "NONVIRTUAL");
            end
        end

        function testTerminalOnTheLastAllowedStepIsATerminationNotATruncation(testCase)
            % walk with NumInner = 1 and limit 1: stepping right from the only inner state ends the walk ON the last allowed step.
            % That must be reported as a termination, never a truncation (the equivalence with the original integrated limit
            % is covered by the golden trajectories of walk5_limit4, walk1_limit1 and cliff_limit20 above).
            env = OceanStats.RL.Env.Wrappers.TimeLimit(OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner', 1), 1);
            for seed = 1:20
                env.reset('Seed', seed);
                [~, ~, term, trunc] = env.step(2);
                testCase.verifyTrue(term && ~trunc);                          % step right from the only inner state ends the walk
            end
        end
    end

    methods (Static, Access = private)
        function m = builders_()
            C = 'OceanStats.RL.Env.Classic.';
            W = @(mk, n) OceanStats.RL.Env.Wrappers.TimeLimit(mk(), n);
            m = containers.Map();
            m('walk5') = @() OceanStats.RL.Env.Classic.RandomWalkEnv();
            m('walk19') = @() OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner', 19, 'LeftReward', -1, 'RightReward', 1);
            m('walk4even') = @() OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner', 4);
            m('walk5_limit4') = @() W(@() OceanStats.RL.Env.Classic.RandomWalkEnv(), 4);
            m('walk1_limit1') = @() W(@() OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner', 1), 1);
            m('cliff') = @() OceanStats.RL.Env.Classic.CliffWalkingEnv();
            m('cliff3x5') = @() OceanStats.RL.Env.Classic.CliffWalkingEnv('Rows', 3, 'Cols', 5);
            m('cliff_limit20') = @() W(@() OceanStats.RL.Env.Classic.CliffWalkingEnv(), 20);
            m('windy') = @() OceanStats.RL.Env.Classic.WindyGridworldEnv();
            m('windy_king') = @() OceanStats.RL.Env.Classic.WindyGridworldEnv('KingMoves', true);
            m('windy_king_stay') = @() OceanStats.RL.Env.Classic.WindyGridworldEnv('KingMoves', true, 'StayAction', true);
            m('windy_stoch') = @() OceanStats.RL.Env.Classic.WindyGridworldEnv('StochasticWind', true);
            m('windy_small') = @() OceanStats.RL.Env.Classic.WindyGridworldEnv('StochasticWind', true, 'Rows', 5, 'Start', [3 1], 'Goal', [3 8]);
            m('maxbias') = @() OceanStats.RL.Env.Classic.MaximizationBiasEnv();
            m('maxbias3') = @() OceanStats.RL.Env.Classic.MaximizationBiasEnv('NumBActions', 3, 'MeanB', 0.5, 'StdB', 2);
            assert(ischar(C));
        end
    end
end
