classdef EnvLegacyGoldenParityTest < OceanTest.BaseTestCase
    % ENVLEGACYGOLDENPARITYTEST  The relocated Blackjack and Racetrack (OceanStats.RL.Env.Classic.*) and their deprecated
    % names (OceanStats.RL.MonteCarlo.*, now thin adapters) must reproduce, draw for draw, what the ORIGINAL classes did under
    % MATLAB's global random stream: rng(seed) followed by the same fixed action rule gives bit-identical states, rewards and
    % done flags (env_golden_legacy.mat, captured from the original code by capture_env_golden_legacy.m).
    % An UNSEEDED environment keeps drawing from the global stream (so existing rng(k) based experiments are unchanged);
    % once seeded (setSeed / reset('Seed')) it owns an independent stream and never touches the global one.

    methods (Test)
        function testClassicEnvironmentsReproduceTheOriginalsUnderTheGlobalStream(testCase)
            g = load(fullfile(fileparts(mfilename('fullpath')), 'env_golden_legacy.mat'));
            mk = OceanTest.OceanStats.RL.EnvLegacyGoldenParityTest.builders_(false);
            for k = 1:numel(g.G)
                for j = 1:numel(g.seeds)
                    rng(g.seeds(j));
                    env = mk(g.G(k).name);
                    s = env.reset();
                    n = size(g.G(k).traj{j}, 1);
                    rows = zeros(n, numel(s) + 1 + 1 + 1 + numel(s));
                    for t = 1:n
                        a = OceanTest.OceanStats.RL.EnvLegacyGoldenParityTest.action_(env, s, t);
                        [s2, r, term, trunc] = env.step(a);
                        rows(t, :) = [s, a, r, term || trunc, s2];
                        if term || trunc, s = env.reset(); else, s = s2; end
                    end
                    testCase.verifyEqual(rows, g.G(k).traj{j}, sprintf('%s seed %d', g.G(k).name, g.seeds(j)));
                end
            end
        end

        function testDeprecatedNamesReproduceTheOriginalsToo(testCase)
            g = load(fullfile(fileparts(mfilename('fullpath')), 'env_golden_legacy.mat'));
            mk = OceanTest.OceanStats.RL.EnvLegacyGoldenParityTest.builders_(true);
            for k = 1:numel(g.G)
                rng(g.seeds(1));
                env = mk(g.G(k).name);
                s = env.reset();
                n = size(g.G(k).traj{1}, 1);
                rows = zeros(n, size(g.G(k).traj{1}, 2));
                for t = 1:n
                    a = OceanTest.OceanStats.RL.EnvLegacyGoldenParityTest.action_(env, s, t);
                    [s2, r, done] = env.step(a);
                    rows(t, :) = [s, a, r, done, s2];
                    if done, s = env.reset(); else, s = s2; end
                end
                testCase.verifyEqual(rows, g.G(k).traj{1}, g.G(k).name);
                testCase.verifyFalse(env.was_truncated());
            end
            % the old surface still works through the adapter
            rt = OceanStats.RL.MonteCarlo.RacetrackEnv("Track1");
            testCase.verifyEqual(rt.Height, 32);  testCase.verifyGreaterThan(rt.TotalStates, 100);
            testCase.verifyEqual(rt.get_num_actions(), 9);
            rt.set_fail_prob(0);  testCase.verifyEqual(rt.FailProb, 0);
            bj = OceanStats.RL.MonteCarlo.BlackjackEnv();
            testCase.verifyEqual([bj.N_STATES, bj.get_num_actions(), bj.state_to_index([22 5 0])], [200 2 0]);
            testCase.verifyEqual(size(bj.get_all_states()), [200 3]);
        end

        function testSeededEnvironmentsAreReproducibleAndLeaveTheGlobalStreamAlone(testCase)
            for name = ["blackjack" "track1"]
                mk = OceanTest.OceanStats.RL.EnvLegacyGoldenParityTest.builders_(false);
                run = @(seed) OceanTest.OceanStats.RL.EnvLegacyGoldenParityTest.rollout_(mk(char(name)), seed, 80);
                testCase.verifyEqual(run(4), run(4), name);
                testCase.verifyNotEqual(run(4), run(5), name);
                rng(9);  before = rand();  rng(9);
                run(4);
                testCase.verifyEqual(rand(), before, name + ": the global stream must not be consumed by a seeded run");
            end
        end

        function testDeprecatedShimsAreIntrospectableAndReadOnly(testCase)
            rt = OceanStats.RL.MonteCarlo.RacetrackEnv("Track1");
            testCase.verifyTrue(isprop(rt, 'Grid') && isprop(rt, 'StartCells') && ismethod(rt, 'set_fail_prob'));
            testCase.verifyEqual(size(rt.StartCells(1, :)), [1 2]);
            testCase.verifyEqual(rt.StartCells(end, :), rt.StartCells(size(rt.StartCells, 1), :));
            testCase.verifyError(@() OceanTest.OceanStats.RL.EnvLegacyGoldenParityTest.assign_(rt), ?MException);   % read-only
            bj = OceanStats.RL.MonteCarlo.BlackjackEnv();
            testCase.verifyTrue(isprop(bj, 'AllStatesCache') && ismethod(bj, 'index_to_action'));
            testCase.verifyEqual(size(bj.AllStatesCache), [200 3]);
            meta = OceanStats.RL.MonteCarlo.RacetrackEnv.getMetadata();
            testCase.verifyEqual(meta.ClassName, "OceanStats.RL.MonteCarlo.RacetrackEnv");
        end

        function testAdaptersFailLoudlyOnTheWrongKindOfEnvironment(testCase)
            id = 'OceanStats:RL:Compat:LegacyEnvironment:';
            testCase.verifyError(@() OceanStats.RL.DP.ValueIterationSolver(OceanStats.RL.Env.Classic.BlackjackEnv()), [id 'NoModel']);
            testCase.verifyError(@() OceanStats.RL.DP.PolicyIterationSolver(OceanStats.RL.Env.Classic.RacetrackEnv("Track1")), [id 'NoModel']);
            testCase.verifyError(@() OceanStats.RL.DP.ValueIterationSolver(5), [id 'BadEnvironment']);
            testCase.verifyError(@() OceanStats.RL.MonteCarlo.MonteCarloSolver(struct()), [id 'BadEnvironment']);
            % a wrapped model environment is NOT silently unwrapped (a time limit changes what an episode is) ...
            w = OceanStats.RL.Env.Wrappers.TimeLimit(OceanStats.RL.Env.Classic.RandomWalkEnv(), 5);
            testCase.verifyError(@() OceanStats.RL.DP.ValueIterationSolver(w), [id 'WrapperChangesModel']);
            % ... but the underlying model can be solved on purpose
            V = OceanStats.RL.DP.ValueIterationSolver(w.unwrapped(), 'DiscountFactor', 1).run();
            testCase.verifyEqual(V(2:6), ones(5, 1), 'AbsTol', 1e-6);         % optimal value: always step right to the +1 end
            % a wrapped sample-only environment is accepted by the sampling adapter and keeps its own time limit
            e = OceanStats.RL.Compat.LegacyEnvironment.adapt(OceanStats.RL.Env.Wrappers.TimeLimit(OceanStats.RL.Env.Classic.RandomWalkEnv(), 1), ...
                'OceanStats.RL.MonteCarlo.IEpisodeEnvironment');
            e.reset();  [~, ~, done] = e.step(2);
            testCase.verifyTrue(done);  testCase.verifyTrue(e.was_truncated());
        end

        function testTheDeprecatedPathIsNotDramaticallySlower(testCase)
            % regression gate for the earlier subsref design (about 10x slower): the adapter may cost a few times a bare call,
            % but not an order of magnitude
            n = 6000;
            direct = OceanStats.RL.Env.Classic.BlackjackEnv();  direct.reset('Seed', 1);
            t0 = tic;
            for k = 1:n
                [s2, ~, te] = direct.step(mod(k, 2));  direct.state_to_index(s2);  direct.get_valid_actions(s2);
                if te, direct.reset(); end
            end
            tDirect = toc(t0);
            shim = OceanStats.RL.MonteCarlo.BlackjackEnv();  shim.setSeed(1);  shim.reset();
            t0 = tic;
            for k = 1:n
                [s2, ~, done] = shim.step(mod(k, 2));  shim.state_to_index(s2);  shim.get_valid_actions(s2);
                if done, shim.reset(); end
            end
            tShim = toc(t0);
            testCase.verifyLessThan(tShim, 5 * tDirect + 0.2, sprintf('shim %.3fs vs direct %.3fs', tShim, tDirect));
        end

        function testEnumerableAndContractPropertiesOfTheSampleOnlyEnvironments(testCase)
            for name = ["blackjack" "track1"]
                mk = OceanTest.OceanStats.RL.EnvLegacyGoldenParityTest.builders_(false);
                env = mk(char(name));
                testCase.verifyTrue(isa(env, 'OceanStats.RL.Env.IEnvironment'));
                testCase.verifyTrue(isa(env, 'OceanStats.RL.Env.IEnumerableEnvironment'));
                testCase.verifyFalse(isa(env, 'OceanStats.RL.Env.IModelEnvironment'), name + " is sample-only");
                [o, info] = env.reset('Seed', 1);
                testCase.verifyTrue(env.ObservationSpace.contains(o));
                testCase.verifyEqual(numel(info.action_mask), env.ActionSpace.N);
                testCase.verifyTrue(any(info.action_mask));
                m = env.getMetadata();  p = env.getPipelineSpec();
                testCase.verifyFalse(m.IsSymbolic);  testCase.verifyEqual(p.Kind, "NONVIRTUAL");
                S = env.get_all_states();
                testCase.verifyEqual(env.state_to_index(S(7, :)), 7);          % row order defines the index
            end
            bj = OceanStats.RL.Env.Classic.BlackjackEnv();
            testCase.verifyEqual(bj.state_to_index([25 3 0]), 0);              % a bust is a terminal outside the table
            testCase.verifyTrue(bj.is_terminal([25 3 0]));
            testCase.verifyError(@() bj.step(1), 'BlackjackEnv:EpisodeEnded');
        end
    end

    methods (Static, Access = private)
        function assign_(env)
            env.Grid = 1;
        end

        function mk = builders_(legacyNames)
            if legacyNames
                b = struct('blackjack', @() OceanStats.RL.MonteCarlo.BlackjackEnv(), ...
                    'track1', @() OceanStats.RL.MonteCarlo.RacetrackEnv("Track1"), ...
                    'track2', @() OceanStats.RL.MonteCarlo.RacetrackEnv("Track2"), ...
                    'track1_nofail', @() OceanStats.RL.MonteCarlo.RacetrackEnv("Track1", 'FailProb', 0));
            else
                b = struct('blackjack', @() OceanStats.RL.Env.Classic.BlackjackEnv(), ...
                    'track1', @() OceanStats.RL.Env.Classic.RacetrackEnv("Track1"), ...
                    'track2', @() OceanStats.RL.Env.Classic.RacetrackEnv("Track2"), ...
                    'track1_nofail', @() OceanStats.RL.Env.Classic.RacetrackEnv("Track1", 'FailProb', 0));
            end
            mk = @(name) b.(name)();
        end

        function a = action_(env, s, t)
            v = env.get_valid_actions(s);
            a = v(1 + mod(t - 1, numel(v)));
        end

        function traj = rollout_(env, seed, n)
            s = env.reset('Seed', seed);
            traj = zeros(n, numel(s) + 2);
            for t = 1:n
                a = OceanTest.OceanStats.RL.EnvLegacyGoldenParityTest.action_(env, s, t);
                [s2, r, term, trunc] = env.step(a);
                traj(t, :) = [s2, r, term || trunc];
                if term || trunc, s = env.reset(); else, s = s2; end
            end
        end
    end
end
