classdef GymEnvTest < OceanTest.BaseTestCase
    % GYMENVTEST  OceanStats.RL.Env.External.GymEnv / GymTabularEnv: a Gymnasium environment behind the IEnvironment
    % contract. Skipped (assumption) when MATLAB has no Python with gymnasium. Checks: identical rows to raw Gymnasium,
    % the exact model equals env.unwrapped.P, strict refusals change nothing, an unseeded run is reproducible via rng.

    properties (TestParameter)
        Name = {'FrozenLake-v1', 'Taxi-v4', 'CartPole-v1'}
    end

    methods (TestClassSetup)
        function needsGymnasium(tc)
            ok = false;
            try
                py.importlib.import_module('gymnasium');
                ok = true;
            catch
            end
            tc.assumeTrue(ok, 'needs Python with the gymnasium package');
        end
    end

    methods (Static)
        function env = make(name, varargin)
            if startsWith(name, "CartPole")
                env = OceanStats.RL.Env.External.GymEnv(name, varargin{:});
            else
                env = OceanStats.RL.Env.External.GymTabularEnv(name, varargin{:});
            end
        end
    end

    methods (Test)
        function rowsEqualRawGymnasium(tc, Name)
            env = tc.make(string(Name));  pick = RandStream('mt19937ar', 'Seed', 1);  n = 300;
            obs = env.reset('Seed', 7);  off = double(env.IsDiscreteObs);
            rows = [obs - off, 0, 0, 0];  acts = zeros(1, n);  resets = false(1, n);  ended = false;
            for t = 1:n
                if ended
                    resets(t) = true;  obs = env.reset();  rows(end + 1, :) = [obs - off, 0, 0, 0]; %#ok<AGROW>
                    ended = false;
                else
                    acts(t) = randi(pick, env.NumActions);
                    [obs, r, term, trunc, info] = env.step(acts(t));
                    rows(end + 1, :) = [obs - off, r, term, trunc]; %#ok<AGROW>
                    tc.verifyTrue(env.ObservationSpace.contains(obs));
                    if term, tc.verifyFalse(any(info.action_mask)); end
                    ended = term || trunc;
                end
            end
            ref = double(py.gym_bridge.reference(Name, 7, py.list(num2cell(acts - 1)), py.list(num2cell(resets))));
            tc.verifyEqual(rows, ref);
        end

        function strictRefusalsChangeNothing(tc, Name)
            env = tc.make(string(Name));
            id = 'OceanStats:RL:Env:External:GymEnv:';
            tc.verifyError(@() env.step(1), [id 'NotReset']);
            a = env.reset('Seed', 3);
            tc.verifyError(@() env.step(0), [id 'InvalidAction']);
            tc.verifyError(@() env.step(env.NumActions + 1), [id 'InvalidAction']);
            tc.verifyError(@() env.step(1.5), [id 'InvalidAction']);
            s1 = env.step(1);
            twin = tc.make(string(Name));  b = twin.reset('Seed', 3);  s2 = twin.step(1);
            tc.verifyEqual(a, b);  tc.verifyEqual(s1, s2);               % the refused calls drew nothing
        end

        function unseededRunIsReproducibleThroughRng(tc, Name)
            g = RandStream.getGlobalStream();  saved = g.State;
            restore = onCleanup(@() set(g, 'State', saved));
            rng(11);  a = tc.make(string(Name));  o1 = a.reset();  x1 = a.step(1);
            rng(11);  b = tc.make(string(Name));  o2 = b.reset();  x2 = b.step(1);
            tc.verifyEqual(o1, o2);  tc.verifyEqual(x1, x2);
            before = g.State;  a.reset('Seed', 5);  a.step(1);
            tc.verifyEqual(g.State, before, 'a seeded run must not touch the global stream');
        end

        function modelEqualsGymnasiumTable(tc)
            env = tc.make("FrozenLake-v1");
            m = cell(py.gym_bridge.Bridge('FrozenLake-v1').model());
            nxt = double(m{1}) + 1;  prob = double(m{2});  rew = double(m{3});
            for s = 1:env.NumStates
                for a = 1:env.NumActions
                    [n, p, r] = env.get_transitions(s, a);
                    tc.verifyEqual(sum(p), 1, 'AbsTol', 1e-12);
                    for k = 1:numel(n)
                        hit = squeeze(nxt(s, a, :)) == n(k) & squeeze(prob(s, a, :)) > 0;
                        tc.verifyEqual(p(k), sum(prob(s, a, hit)), 'AbsTol', 1e-12);
                        tc.verifyEqual(r(k), sum(prob(s, a, hit) .* rew(s, a, hit)) / p(k), 'AbsTol', 1e-12);
                    end
                end
            end
            tc.verifyEqual(find(env.TerminalMask)', [6 8 12 13 16]);      % holes and the goal of the 4x4 map
            tc.verifyEmpty(env.get_valid_actions(16));
            tc.verifyError(@() OceanStats.RL.Env.External.GymEnv("NoSuchEnvironment-v0"), ...
                'OceanStats:RL:Env:External:GymEnv:NoGymnasium');
        end
    end
end
