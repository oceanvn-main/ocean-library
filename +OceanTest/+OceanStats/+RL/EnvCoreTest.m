classdef EnvCoreTest < OceanTest.BaseTestCase
    % ENVCORETEST  The algorithm-free environment core OceanStats.RL.Env: spaces, the IEnvironment contract (through the
    % TabularEnvironment fixture), wrappers, and the dependency rule (no learner package may be referenced).

    methods (Test)
        % ---------------- spaces ----------------
        function testDiscrete(testCase)
            sp = OceanStats.RL.Env.Spaces.Discrete(4);
            testCase.verifyTrue(all(arrayfun(@sp.contains, 1:4)));
            testCase.verifyFalse(sp.contains(0));  testCase.verifyFalse(sp.contains(5));
            testCase.verifyFalse(sp.contains(1.5));  testCase.verifyFalse(sp.contains([1 2]));  testCase.verifyFalse(sp.contains(NaN));
            testCase.verifyFalse(sp.contains("a"));
            st = RandStream('mt19937ar', 'Seed', 1);
            x = arrayfun(@(~) sp.sample(st), 1:400);
            testCase.verifyTrue(all(arrayfun(@sp.contains, x)));
            testCase.verifyEqual(sort(unique(x)), 1:4);                        % every value is reachable
            z = OceanStats.RL.Env.Spaces.Discrete(3, 'Start', 0);
            testCase.verifyTrue(z.contains(0));  testCase.verifyFalse(z.contains(3));
            testCase.verifyError(@() OceanStats.RL.Env.Spaces.Discrete(0), 'MATLAB:validators:mustBePositive');
        end

        function testBox(testCase)
            sp = OceanStats.RL.Env.Spaces.Box([0 -1], [1 1]);
            testCase.verifyTrue(sp.contains([0.5 0]));  testCase.verifyTrue(sp.contains([0 1]));
            testCase.verifyFalse(sp.contains([1.1 0]));  testCase.verifyFalse(sp.contains([0.5 0 0]));  testCase.verifyFalse(sp.contains([NaN 0]));
            st = RandStream('mt19937ar', 'Seed', 2);
            for k = 1:50, testCase.verifyTrue(sp.contains(sp.sample(st))); end
            half = OceanStats.RL.Env.Spaces.Box(0, Inf);
            testCase.verifyTrue(half.contains(1e9));  testCase.verifyTrue(half.sample(st) >= 0);
            testCase.verifyError(@() OceanStats.RL.Env.Spaces.Box([1 1], [0 0]), 'OceanStats:RL:Env:Spaces:Box:BadBounds');
            testCase.verifyError(@() OceanStats.RL.Env.Spaces.Box([0 0], [1 1 1]), 'OceanStats:RL:Env:Spaces:Box:SizeMismatch');
        end

        function testMultiDiscrete(testCase)
            sp = OceanStats.RL.Env.Spaces.MultiDiscrete([3 4]);
            testCase.verifyTrue(sp.contains([3 4]));  testCase.verifyFalse(sp.contains([4 1]));  testCase.verifyFalse(sp.contains([1 0]));
            st = RandStream('mt19937ar', 'Seed', 3);
            for k = 1:50, testCase.verifyTrue(sp.contains(sp.sample(st))); end
        end

        % ---------------- IEnvironment contract (through the tabular fixture) ----------------
        function testResetAndStepFollowTheContract(testCase)
            env = OceanTest.OceanStats.RL.Fixtures.TinyChainEnv();
            [o, info] = env.reset('Seed', 1);
            testCase.verifyEqual(o, 1);
            testCase.verifyTrue(env.ObservationSpace.contains(o));
            testCase.verifyEqual(info.action_mask, [true true]);
            [o, r, term, trunc, info] = env.step(1);
            testCase.verifyEqual([o r term trunc], [2 0 false false]);
            testCase.verifyEqual(info.action_mask, [true false]);            % only action 1 admissible in state 2
            steps = 0;
            while ~term && steps < 100
                [o, r, term, trunc] = env.step(1);  steps = steps + 1;
            end
            testCase.verifyTrue(term);  testCase.verifyFalse(trunc);
            testCase.verifyEqual([o r], [4 1]);
        end

        function testInvalidActionsAreRejected(testCase)
            env = OceanTest.OceanStats.RL.Fixtures.TinyChainEnv();
            env.reset('Seed', 1);  env.step(1);                              % now in state 2, action 2 inadmissible
            id = 'OceanStats:RL:Env:TabularEnvironment:InvalidAction';
            testCase.verifyError(@() env.step(2), id);
            testCase.verifyError(@() env.step(3), id);                        % outside the action space
            testCase.verifyError(@() env.step(1.5), id);
        end

        function testSeedingIsTheOnlyRandomnessControl(testCase)
            roll = @(env, seed) OceanTest.OceanStats.RL.EnvCoreTest.rollout_(env, seed);
            a = roll(OceanTest.OceanStats.RL.Fixtures.TinyChainEnv(), 5);
            b = roll(OceanTest.OceanStats.RL.Fixtures.TinyChainEnv(), 5);
            c = roll(OceanTest.OceanStats.RL.Fixtures.TinyChainEnv(), 6);
            testCase.verifyEqual(a, b);
            testCase.verifyNotEqual(a, c);
            % setSeed then reset == reset('Seed')
            e1 = OceanTest.OceanStats.RL.Fixtures.TinyChainEnv();  e1.setSeed(5);  e1.reset();
            e2 = OceanTest.OceanStats.RL.Fixtures.TinyChainEnv();  e2.reset('Seed', 5);
            testCase.verifyEqual(e1.step(1), e2.step(1));
            % the global rng is not consumed
            rng(7);  before = rand();  rng(7);
            roll(OceanTest.OceanStats.RL.Fixtures.TinyChainEnv(), 5);
            testCase.verifyEqual(rand(), before);
        end

        function testModelIsConsistentWithSampling(testCase)
            env = OceanTest.OceanStats.RL.Fixtures.TinyChainEnv();
            [n, p, r] = env.get_transitions(2, 1);
            testCase.verifyEqual([n p r], [3 0.8 0; 2 0.2 0]);
            testCase.verifyEqual(env.get_valid_actions(1), [1 2]);
            testCase.verifyEqual(env.get_valid_actions(2), 1);
            testCase.verifyEqual(size(env.get_valid_actions(4)), [1 0]);
            testCase.verifyTrue(env.is_terminal(4));  testCase.verifyFalse(env.is_terminal(3));
            % empirical frequency of 2 -> 3
            env.reset('Seed', 11);  hits = 0;  N = 4000;
            for k = 1:N
                env.set_state(2);
                [o] = env.step(1);  hits = hits + (o == 3);
            end
            testCase.verifyEqual(hits / N, 0.8, 'AbsTol', 0.03);            % binomial sd = 0.006 -> 5 sigma
        end

        function testExactPolicyValuesAgainstHandSolution(testCase)
            env = OceanTest.OceanStats.RL.Fixtures.TinyChainEnv();
            P = [1 0; 1 0; 1 0; 0 0];
            testCase.verifyEqual(env.exactPolicyValues(P, 1), [1; 1; 1; 0], 'AbsTol', 1e-12);
            V = env.exactPolicyValues(P, 0.9);
            v3 = 1;  v2 = 0.72 / 0.82;  v1 = 0.9 * v2;
            testCase.verifyEqual(V, [v1; v2; v3; 0], 'AbsTol', 1e-12);
            testCase.verifyEqual(sum(env.uniformPolicy(), 2), [1; 1; 1; 0]);
        end

        function testModelValidation(testCase)
            env = OceanTest.OceanStats.RL.Fixtures.TinyChainEnv();
            id = 'OceanStats:RL:Env:TabularEnvironment:BadModel';
            ok = cell(2, 1);  ok{1, 1} = struct('Next', 2, 'Prob', 1, 'Reward', 0);
            tm = [false; true];  sp = [1; 0];
            testCase.verifyError(@() env.setBadModel(OceanTest.OceanStats.RL.EnvCoreTest.setfield_(ok, 1, 'Prob', 0.5), tm, sp), id);            % does not sum to 1
            testCase.verifyError(@() env.setBadModel(OceanTest.OceanStats.RL.EnvCoreTest.setfield_(ok, 1, 'Next', 3), tm, sp), id);              % state out of range
            testCase.verifyError(@() env.setBadModel(OceanTest.OceanStats.RL.EnvCoreTest.setfield_(ok, 1, 'Reward', [0; 0]), tm, sp), id);       % length mismatch
            testCase.verifyError(@() env.setBadModel(ok, tm, [0; 0]), id);                                     % no start state
            testCase.verifyError(@() env.setBadModel(ok, [true; true], sp), id);                                % terminal with an action
            testCase.verifyError(@() env.setBadModel(ok, true, sp), id);                                        % wrong mask length
        end

        function testPillars(testCase)
            env = OceanTest.OceanStats.RL.Fixtures.TinyChainEnv();
            m = env.getMetadata();  p = env.getPipelineSpec();
            testCase.verifyFalse(m.IsSymbolic);
            testCase.verifyTrue(all(isfield(m, ["SymbolicLatex" "Symbols" "Assumptions" "Reference"])));
            testCase.verifyEqual(p.Kind, "NONVIRTUAL");
            testCase.verifyTrue(all(isfield(p, ["Inputs" "Outputs" "DiagramSymbol" "ComposesFrom"])));
        end

        % ---------------- wrappers ----------------
        function testTimeLimitTruncatesButNeverOverridesTermination(testCase)
            inner = OceanTest.OceanStats.RL.Fixtures.TinyChainEnv();
            env = OceanStats.RL.Env.Wrappers.TimeLimit(inner, 2);
            env.reset('Seed', 1);
            [~, ~, term, trunc] = env.step(1);                               % 1 -> 2
            testCase.verifyEqual([term trunc], [false false]);
            [~, ~, term, trunc] = env.step(1);                               % 2 -> 3 or 2 (2 steps taken)
            testCase.verifyEqual([term trunc], [false true]);
            % counter restarts on reset, and a terminal reached exactly at the limit is a termination
            env.reset();  inner.set_state(3);
            [o, ~, term, trunc] = env.step(1);
            testCase.verifyEqual([o term trunc], [4 true false]);
            env2 = OceanStats.RL.Env.Wrappers.TimeLimit(OceanTest.OceanStats.RL.Fixtures.TinyChainEnv(), 1);
            env2.reset('Seed', 1);  env2.unwrapped().set_state(3);
            [~, ~, term, trunc] = env2.step(1);                              % terminal on the very last allowed step
            testCase.verifyEqual([term trunc], [true false]);
            env.reset();
            [~, ~, ~, trunc] = env.step(2);  testCase.verifyFalse(trunc);      % 1 step after reset
        end

        function testWrapperForwardsSeedSpacesAndUnwrapped(testCase)
            inner = OceanTest.OceanStats.RL.Fixtures.TinyChainEnv();
            env = OceanStats.RL.Env.Wrappers.TimeLimit(inner, 50);
            testCase.verifyEqual(env.ObservationSpace.N, 4);
            testCase.verifyEqual(env.ActionSpace.N, 2);
            testCase.verifyTrue(env.unwrapped() == inner);
            testCase.verifyTrue(isa(env.unwrapped(), 'OceanStats.RL.Env.IModelEnvironment'));   % model reachable through unwrapped()
            a = OceanTest.OceanStats.RL.EnvCoreTest.rollout_(env, 9);
            b = OceanTest.OceanStats.RL.EnvCoreTest.rollout_(OceanStats.RL.Env.Wrappers.TimeLimit(OceanTest.OceanStats.RL.Fixtures.TinyChainEnv(), 50), 9);
            testCase.verifyEqual(a, b);
            m = env.getMetadata();  p = env.getPipelineSpec();
            testCase.verifyTrue(any(contains(string(m.Assumptions), "50")));
            testCase.verifyEqual(p.ComposesFrom{1}, 'OceanTest.OceanStats.RL.Fixtures.TinyChainEnv');
            testCase.verifyError(@() OceanStats.RL.Env.Wrappers.TimeLimit(inner, 0), 'MATLAB:validators:mustBePositive');
        end

        % ---------------- critic round 1: contract details ----------------
        function testSpacesSampleWithoutAStream(testCase)
            testCase.verifyTrue(OceanStats.RL.Env.Spaces.Discrete(3).contains(OceanStats.RL.Env.Spaces.Discrete(3).sample()));
            testCase.verifyTrue(OceanStats.RL.Env.Spaces.Box([0 0], [1 1]).contains(OceanStats.RL.Env.Spaces.Box([0 0], [1 1]).sample()));
            testCase.verifyTrue(OceanStats.RL.Env.Spaces.MultiDiscrete([2 3]).contains(OceanStats.RL.Env.Spaces.MultiDiscrete([2 3]).sample()));
            testCase.verifyError(@() OceanStats.RL.Env.Spaces.Box([NaN 0], [1 1]), 'OceanStats:RL:Env:Spaces:Box:BadBounds');
        end

        function testStepBeforeResetGivesAClearError(testCase)
            env = OceanTest.OceanStats.RL.Fixtures.TinyChainEnv();
            testCase.verifyError(@() env.step(1), 'OceanStats:RL:Env:TabularEnvironment:NotReset');
            testCase.verifyError(@() env.get_transitions(2, 2), 'OceanStats:RL:Env:TabularEnvironment:InvalidAction');
        end

        function testUnseededEnvironmentFollowsTheGlobalStream(testCase)
            roll = @(env) OceanTest.OceanStats.RL.EnvCoreTest.rollout_(env, []);
            rng(3);  a = roll(OceanTest.OceanStats.RL.Fixtures.TinyChainEnv());
            rng(3);  b = roll(OceanTest.OceanStats.RL.Fixtures.TinyChainEnv());
            rng(4);  c = roll(OceanTest.OceanStats.RL.Fixtures.TinyChainEnv());
            testCase.verifyEqual(a, b);                                       % rng(k) controls an unseeded run
            testCase.verifyNotEqual(a, c);
        end

        function testMoreModelValidation(testCase)
            env = OceanTest.OceanStats.RL.Fixtures.TinyChainEnv();
            id = 'OceanStats:RL:Env:TabularEnvironment:BadModel';
            good = cell(2, 1);  good{1, 1} = struct('Next', 2, 'Prob', 1, 'Reward', 0);
            tm = [false; true];  sp = [1; 0];
            F = @OceanTest.OceanStats.RL.EnvCoreTest.setfield_;
            testCase.verifyError(@() env.setBadModel(F(good, 1, 'Prob', NaN), tm, sp), id);              % NaN probability
            testCase.verifyError(@() env.setBadModel(F(good, 1, 'Reward', Inf), tm, sp), id);            % non-finite reward
            testCase.verifyError(@() env.setBadModel(good, tm, [NaN; 0]), id);                            % NaN start distribution
            testCase.verifyError(@() env.setBadModel(good, tm, [0; 1]), id);                              % start mass on a terminal
            testCase.verifyError(@() env.setBadModel(cell(2, 1), tm, sp), id);                            % non-terminal state without actions
            testCase.verifyError(@() env.setBadModel(cat(3, good, good), tm, sp), id);                    % not 2-D
        end

        function testTerminalObservationsHaveAnAllFalseActionMask(testCase)
            env = OceanTest.OceanStats.RL.Fixtures.TinyChainEnv();
            env.reset('Seed', 1);  env.set_state(3);
            [~, ~, term, ~, info] = env.step(1);
            testCase.verifyTrue(term);  testCase.verifyEqual(info.action_mask, [false false]);
            bj = OceanStats.RL.Env.Classic.BlackjackEnv();  bj.reset('Seed', 2);
            [~, ~, term, ~, info] = bj.step(0);                                % stick ends the episode
            testCase.verifyTrue(term);  testCase.verifyEqual(info.action_mask, [false false]);
            rt = OceanStats.RL.Env.Classic.RacetrackEnv("Track1", 'FailProb', 0);  rt.reset('Seed', 1);
            finished = false;
            for k = 1:2000
                [~, ~, term, ~, info] = rt.step(9);                            % accelerate up-right until the line or a crash resets
                if term, finished = true; break; end
                if rt.unwrapped().is_terminal(rt.unwrapped().CurrentState), break; end
            end
            if finished, testCase.verifyEqual(info.action_mask, false(1, 9)); end
        end

        % ---------------- the dependency rule ----------------
        function testDependencyRuleScannerCatchesViolations(testCase)
            % positive controls: the scanner must FLAG each of these (a scanner that cannot fail proves nothing)
            bad = ["x = OceanStats.RL.MonteCarlo.BlackjackEnv();", "% used by the DP solver", "% Sarsa and Q-learning", ...
                   "% a Monte Carlo estimate", "% temporal-difference methods", "y = OceanStats.RL.TD.Factory", ...
                   "obj = OceanStats.RL.Compat.LegacyEnvironment(env);", "% dynamic programming", "% IEpisodeEnvironment", ...
                   "% the learner asks", "% a ValueIterationSolver", "% TD(0) target"];
            for k = 1:numel(bad)
                testCase.verifyNotEmpty(OceanTest.OceanStats.RL.EnvCoreTest.violations_(bad(k)), "must flag: " + bad(k));
            end
            % ... and must NOT flag ordinary environment text
            ok = ["% state index, action mask and truncation", "obj.Steps = obj.Steps + 1;", "% Sutton & Barto (2018) Example 5.1", ...
                  "% the ADP-like word 'adapter' and words like update, add, tdata"];
            for k = 1:numel(ok)
                testCase.verifyEmpty(OceanTest.OceanStats.RL.EnvCoreTest.violations_(ok(k)), "must not flag: " + ok(k));
            end
        end

        function testEnvPackageDoesNotDependOnAnyLearnerPackage(testCase)
            root = fileparts(which('OceanStats.RL.Env.IEnvironment'));
            files = dir(fullfile(root, '**', '*.m'));
            testCase.verifyGreaterThan(numel(files), 15, 'the scan must actually cover the package');
            for k = 1:numel(files)
                txt = string(fileread(fullfile(files(k).folder, files(k).name)));
                testCase.verifyEmpty(OceanTest.OceanStats.RL.EnvCoreTest.violations_(txt), ...
                    sprintf('%s refers to a learning method or learner package', files(k).name));
            end
        end
    end

    methods (Static, Access = private)
        function v = violations_(txt)
            % names of learning methods / learner packages that must not appear anywhere in +Env (code, comments, strings)
            pats = {'OceanStats\.RL\.(MonteCarlo|DP|TD|Compat)\>', '\<Compat\>', 'Monte\s?Carlo', '\<Sarsa\>', 'Q-learning', ...
                    'dynamic programming', 'temporal[- ]difference', 'IEpisode', '[Ss]olver', '[Ll]earner'};
            v = {};
            for i = 1:numel(pats)
                if ~isempty(regexp(char(txt), pats{i}, 'once', 'ignorecase')), v{end+1} = pats{i}; end %#ok<AGROW>
            end
            for pat = {'\<DP\>', '\<TD\>'}                                 % case-sensitive abbreviations
                if ~isempty(regexp(char(txt), pat{1}, 'once')), v{end+1} = pat{1}; end %#ok<AGROW>
            end
        end
        function m = setfield_(model, k, f, v)
            m = model;  st = m{k, 1};  st.(f) = v;  m{k, 1} = st;
        end
        function traj = rollout_(env, seed)
            % flattened [obs r terminated truncated] rows of a short rollout under action 1
            if isempty(seed), o = env.reset(); else, o = env.reset('Seed', seed); end
            rows = zeros(0, 4);
            for k = 1:30
                [o, r, term, trunc] = env.step(1);
                rows(end+1, :) = [o r term trunc]; %#ok<AGROW>
                if term || trunc, break; end
            end
            traj = [o(1) * 0 + rows(:)'];
        end
    end
end
