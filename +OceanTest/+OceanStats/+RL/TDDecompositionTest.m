classdef TDDecompositionTest < OceanTest.BaseTestCase
    % TDDECOMPOSITIONTEST  The parts the monolithic TDLearner was split into (Sampler, value functions, behaviour policies,
    % backups, observers), each tested on its own, plus the extension and dependency rules of the decomposition. Bit-level
    % equivalence with the old learner is TDRefactorGoldenTest's job.

    methods (Test)
        % ---------------- Sampler ----------------
        function testSamplerEncodesTheEnvironmentAndNothingElse(testCase)
            sm = OceanStats.RL.TD.Sampler(OceanStats.RL.Compat.LegacyEnvironment(OceanStats.RL.Env.Classic.RandomWalkEnv()), 100);
            testCase.verifyEqual([sm.NumS sm.NumA], [7 2]);
            testCase.verifyEqual(sm.ValidMask(:, 1)', [false true(1, 5) false]);        % terminals have no admissible action
            testCase.verifyEqual(sm.ActionValues, [1 2]);
            sm.seed(1);
            si = sm.start();
            testCase.verifyEqual(si, 4);
            [s2, r, term, trunc, done] = sm.step(2);
            testCase.verifyEqual([s2 r term trunc done], [5 0 false false false]);
            testCase.verifyEqual(sm.Steps, 1);
        end

        function testSamplerTruncatesAtMaxStepsAndNeverAtATerminal(testCase)
            sm = OceanStats.RL.TD.Sampler(OceanStats.RL.Compat.LegacyEnvironment(OceanStats.RL.Env.Classic.RandomWalkEnv()), 3);
            testCase.verifyFalse(sm.capped(false, 2));
            testCase.verifyTrue(sm.capped(false, 3));
            testCase.verifyFalse(sm.capped(true, 3));                                    % an episode that has ended is not capped
        end

        function testSamplerHandlesATerminalOutsideTheTable(testCase)
            sm = OceanStats.RL.TD.Sampler(OceanStats.RL.Compat.LegacyEnvironment(OceanStats.RL.Env.Classic.BlackjackEnv()), Inf);
            sm.seed(3);
            seen = false;
            for k = 1:200
                sm.start();  done = false;
                while ~done
                    [s2i, ~, term, ~, done] = sm.step(2);                                 % hit until the episode ends
                    if sm.OutsideLast
                        seen = true;
                        testCase.verifyTrue(term && s2i == 1, 'a bust is a terminal, indexed by the placeholder');
                    else
                        testCase.verifyFalse(sm.OutsideLast);
                    end
                end
            end
            testCase.verifyTrue(seen, 'hitting until the end must bust at least once in 200 hands');
        end

        function testSamplerWarnsWhenTheEnvironmentCannotBeSeeded(testCase)
            sm = OceanStats.RL.TD.Sampler(OceanTest.OceanStats.RL.Fixtures.UnseedableEpisodeEnv(), Inf);
            testCase.verifyWarning(@() sm.seed(1), 'OceanStats:RL:TD:TDLearner:EnvNotSeedable');
        end

        % ---------------- value functions ----------------
        function testTabularQAndVSemantics(testCase)
            valid = logical([1 1; 1 0; 0 0]);
            q = OceanStats.RL.TD.Values.TabularQ(valid, 2);
            testCase.verifyEqual(q.estimates(1), [2 2]);
            testCase.verifyEqual(q.estimates(2), [2 0]);                                 % invalid stays at 0
            testCase.verifyEqual(q.touch(1, 2), 1);  testCase.verifyEqual(q.touch(1, 2), 2);
            q.update(1, 2, 0.5, 4);
            testCase.verifyEqual(q.get(1, 2), 3);                                        % 2 + 0.5 (4 - 2)
            testCase.verifyEqual(q.zerosLike(), zeros(3, 2));
            q.applyIncrement([1 0; 0 0; 0 0]);
            testCase.verifyEqual(q.get(1, 1), 3);
            testCase.verifyEqual(q.kind(), "action");
            v = OceanStats.RL.TD.Values.TabularV(3, 1);
            testCase.verifyEqual(v.kind(), "state");
            v.update(2, 1, 0.25, 5);
            testCase.verifyEqual(v.stateValues([1 2]), [1; 2]);                          % 1 + 0.25 (5 - 1)
            testCase.verifyError(@() q.stateValues(1), 'OceanStats:RL:TD:IValueFunction:NotStateValues');
        end

        function testDoubleTabularQUsesBothTablesAndTheirOwnCounts(testCase)
            valid = true(3, 2);
            d = OceanStats.RL.TD.Values.DoubleTabularQ(valid, 0);
            testCase.verifyTrue(d.updatesBeforeNextAction());
            stream = RandStream('mt19937ar', 'Seed', 5);
            ss = OceanStats.RL.TD.StepSize.ConstantStepSize(1);
            expectedFlips = rand(RandStream('mt19937ar', 'Seed', 5), 1, 40) < 0.5;
            flips = false(1, 40);
            for k = 1:40
                flips(k) = d.oneStep([], ss, 1, 1, 1, 2, [], false, 0.9, [], valid(2, :), stream);
            end
            testCase.verifyEqual(flips, expectedFlips);                                  % the coin is the only draw
            testCase.verifyEqual(d.Counts(1, 1), nnz(expectedFlips));
            testCase.verifyEqual(d.Counts2(1, 1), nnz(~expectedFlips));                   % separate visit counts
            testCase.verifyEqual(d.finalQ(), (d.Q + d.Q2) / 2);
            testCase.verifyEqual(d.behaviourEstimates(1), (d.Q(1, :) + d.Q2(1, :)) / 2);
        end

        % ---------------- behaviour policies ----------------
        function testPoliciesAndTheirValidation(testCase)
            valid = logical([1 1; 1 1; 0 0]);
            f = OceanStats.RL.TD.Policies.FixedPolicy([], valid);
            testCase.verifyEqual(f.P, [0.5 0.5; 0.5 0.5; 0 0]);                           % default: uniform over admissible actions
            id = 'OceanStats:RL:TD:TDLearner:BadPolicy';
            testCase.verifyError(@() OceanStats.RL.TD.Policies.FixedPolicy(ones(2, 2), valid), id);           % wrong size
            testCase.verifyError(@() OceanStats.RL.TD.Policies.FixedPolicy([1 1; 1 1; 0 0], valid), id);      % does not sum to 1
            partial = logical([1 0; 1 1; 0 0]);
            testCase.verifyError(@() OceanStats.RL.TD.Policies.FixedPolicy([0.5 0.5; 0.5 0.5; 0 0], partial), id);   % mass on an inadmissible action
            testCase.verifyEqual(OceanStats.RL.TD.Policies.FixedPolicy([1 0; 0.5 0.5; 0.3 0.7], partial).P(3, :), [0.3 0.7]);   % a terminal row is ignored
            e = OceanStats.RL.TD.Policies.ExplorationPolicy(OceanStats.RL.TD.Exploration.EpsilonGreedyExploration(0));
            q = OceanStats.RL.TD.Values.TabularQ(valid, 0);
            q.update(1, 2, 1, 3);
            testCase.verifyEqual(e.probabilities(q, 1, valid(1, :), 0), [0 1]);          % epsilon 0: greedy
            stream = RandStream('mt19937ar', 'Seed', 1);
            testCase.verifyEqual(e.sample([0 1], stream), 2);
        end

        % ---------------- observers ----------------
        function testRecorderKeepsEpisodesInTheReplayFormat(testCase)
            r = OceanStats.RL.TD.Observers.Recorder();
            r.begin(3, 1);  r.step(-1, 4, 2, false);  r.step(-1, 9, 0, true);  r.coin(true);  r.finish(true, false);
            ep = r.Episodes{1};
            testCase.verifyEqual(ep.S, [3; 4; 0]);                                       % a terminal outside the table is recorded as 0
            testCase.verifyEqual(ep.A, [1; 2]);
            testCase.verifyEqual(ep.R, [-1; -1]);
            testCase.verifyEqual(ep.Coin, 1);                                            % stored as double, as the replay format expects
            testCase.verifyEqual([ep.Terminated ep.Truncated], [true false]);
            testCase.verifyEqual(fieldnames(ep)', {'S', 'A', 'R', 'Coin', 'Terminated', 'Truncated'});
            n = OceanStats.RL.TD.Observers.NullObserver();                                % every hook exists and does nothing
            n.begin(1, 1);  n.step(0, 1, 1, false);  n.coin(true);  n.action(1);  n.finish(true, false);  n.finishArrays(1, 1, 0, true, false, false);
        end

        % ---------------- extension rule ----------------
        function testANewBackupIsOneFileAndNeedsNoEditToTheLearner(testCase)
            OceanNLP.Text.StrategyResolver.unregisterAll();
            testCase.addTeardown(@() OceanNLP.Text.StrategyResolver.unregisterAll());
            OceanNLP.Text.StrategyResolver.register("OceanStats.RL.TD.Backups", "mcreturn", @OceanTest.OceanStats.RL.Fixtures.MonteCarloReturnBackup);
            env = OceanStats.RL.Env.Classic.RandomWalkEnv();
            mc = OceanStats.RL.TD.TDLearner('Env', env, 'Task', 'prediction', 'Mode', 'mcreturn', 'StepSize', 0.05, 'Episodes', 300, 'Seed', 4).run();
            ns = OceanStats.RL.TD.TDLearner('Env', env, 'Task', 'prediction', 'Mode', 'nstep', 'N', 1000, 'StepSize', 0.05, 'Episodes', 300, 'Seed', 4).run();
            testCase.verifyEqual(mc, ns, 'AbsTol', 1e-13);                                % the same estimator, written independently
            testCase.verifyError(@() OceanStats.RL.TD.TDLearner('Env', env, 'Mode', 'mcreturn'), 'OceanStats:RL:TD:TDLearner:BadMode');   % control not supported: declared, not hard-coded
            testCase.verifyTrue(any(strcmp(OceanStats.RL.TD.Factory.validNames("backup"), 'mcreturn')));
        end

        function testValueFunctionAndPolicyHooksReplaceTheDefaults(testCase)
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv('Rows', 3, 'Cols', 5);
            base = {'Env', env, 'Target', 'sarsa', 'StepSize', 0.3, 'Gamma', 0.9, 'Exploration', 0.2, 'Episodes', 25, 'Seed', 2};
            [~, Q0] = OceanStats.RL.TD.TDLearner(base{:}).run();
            hookV = @(sm, iv) OceanStats.RL.TD.Values.TabularQ(sm.ValidMask, iv);
            hookP = @(sm) OceanStats.RL.TD.Policies.ExplorationPolicy(OceanStats.RL.TD.Exploration.EpsilonGreedyExploration(0.2));
            [~, Q1] = OceanStats.RL.TD.TDLearner(base{:}, 'ValueFunction', hookV, 'Policy', hookP).run();
            testCase.verifyEqual(Q1, Q0);
            hookGreedy = @(sm) OceanStats.RL.TD.Policies.ExplorationPolicy(OceanStats.RL.TD.Exploration.EpsilonGreedyExploration(0));
            [~, Q2] = OceanStats.RL.TD.TDLearner(base{:}, 'Policy', hookGreedy).run();
            testCase.verifyNotEqual(Q2, Q0);                                              % a different behaviour policy really is used
        end

        function testCompatibilityIsDecidedFromDeclaredCapabilities(testCase)
            B = @(n) OceanStats.RL.TD.Factory.create("backup", n);
            testCase.verifyEqual(OceanStats.RL.TD.Factory.validNames("backup"), {'lambdareturn', 'nstep', 'onestep', 'trace'});
            testCase.verifyTrue(B("onestep").supportsUpdateFirstValues());
            testCase.verifyFalse(B("trace").supportsUpdateFirstValues());
            testCase.verifyTrue(B("trace").supportsOffline("state") && ~B("trace").supportsOffline("action"));
            testCase.verifyFalse(B("nstep").allowsOffPolicyTarget("action"));
            testCase.verifyTrue(B("nstep").allowsOffPolicyTarget("state"));
            testCase.verifyFalse(B("lambdareturn").supportsValueKind("action"));
        end

        % ---------------- critic round 1: pluggable observers, foreign parts run end to end ----------------
        function testAnyObserverCanBePluggedInAndSeesEveryStep(testCase)
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv('Rows', 3, 'Cols', 5);
            base = {'Env', env, 'Target', 'sarsa', 'StepSize', 0.3, 'Gamma', 0.9, 'Exploration', 0.2, 'Episodes', 30, 'Seed', 6};
            log = OceanStats.RL.TD.Observers.ReturnLogger();
            cnt = OceanTest.OceanStats.RL.Fixtures.CountingObserver();
            [~, Q0, ~, h0] = OceanStats.RL.TD.TDLearner(base{:}).run();
            [~, Q1, ~, h1] = OceanStats.RL.TD.TDLearner(base{:}, 'Observer', {log, cnt}, 'Record', true).run();
            testCase.verifyEqual(Q1, Q0);                                                % observing never changes learning
            testCase.verifyEqual(log.Returns, h1.EpisodeReturns);
            testCase.verifyEqual(log.Lengths, h1.EpisodeLengths);
            testCase.verifyEqual(cnt.Steps, h1.TotalSteps);
            testCase.verifyEqual([cnt.Episodes cnt.Starts], [30 30]);
            testCase.verifyEqual(numel(h1.Episodes), 30);                                % Record still works next to a custom observer
            testCase.verifyFalse(isfield(h0, 'Episodes'));
            % a run restarts the observer's state (startRun), and a single observer (not in a cell) works too
            OceanStats.RL.TD.TDLearner(base{:}, 'Observer', cnt).run();
            testCase.verifyEqual(cnt.Steps, h1.TotalSteps);
            testCase.verifyError(@() OceanStats.RL.TD.Observers.CompositeObserver({1}), 'OceanStats:RL:TD:Observers:BadObserver');
            % n-step and lambda-return report through finishArrays
            cnt2 = OceanTest.OceanStats.RL.Fixtures.CountingObserver();
            [~, ~, ~, h2] = OceanStats.RL.TD.TDLearner('Env', OceanStats.RL.Env.Classic.RandomWalkEnv(), 'Task', 'prediction', 'Mode', 'lambdareturn', ...
                'StepSize', 0.1, 'Episodes', 20, 'Seed', 1, 'Observer', cnt2).run();
            testCase.verifyEqual([cnt2.Steps cnt2.Episodes], [h2.TotalSteps 20]);
        end

        function testAForeignValueFunctionRunsEndToEnd(testCase)
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv('Rows', 3, 'Cols', 5);
            base = {'Env', env, 'Target', 'qlearning', 'StepSize', 0.4, 'Gamma', 0.95, 'Exploration', 0.2, 'Episodes', 40, 'Seed', 3};
            [~, Q0] = OceanStats.RL.TD.TDLearner(base{:}).run();
            big = @(sm, iv) OceanTest.OceanStats.RL.Fixtures.ClippedQ(sm.ValidMask, iv, 1e9);
            [~, Q1] = OceanStats.RL.TD.TDLearner(base{:}, 'ValueFunction', big).run();
            testCase.verifyEqual(Q1, Q0);                                                % a huge clip is the identity
            small = @(sm, iv) OceanTest.OceanStats.RL.Fixtures.ClippedQ(sm.ValidMask, iv, 3);
            [~, Q2] = OceanStats.RL.TD.TDLearner(base{:}, 'ValueFunction', small).run();
            testCase.verifyLessThanOrEqual(max(abs(Q2(:))), 3);
            testCase.verifyNotEqual(Q2, Q0);
            [~, Q3] = OceanStats.RL.TD.TDLearner(base{:}, 'Target', 'sarsa', 'Mode', 'nstep', 'N', 3, 'ValueFunction', small).run();   % works in another backup too
            testCase.verifyLessThanOrEqual(max(abs(Q3(:))), 3);
        end

        function testAForeignBehaviourPolicyRunsEndToEnd(testCase)
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv('Rows', 3, 'Cols', 5);
            base = {'Env', env, 'Target', 'sarsa', 'StepSize', 0.3, 'Gamma', 0.9, 'Episodes', 30, 'Seed', 8};
            hook = @(sm) OceanTest.OceanStats.RL.Fixtures.UniformPolicy();
            [~, Qa] = OceanStats.RL.TD.TDLearner(base{:}, 'Policy', hook).run();
            [~, Qb] = OceanStats.RL.TD.TDLearner(base{:}, 'Exploration', 1).run();       % epsilon = 1 is exactly uniform
            testCase.verifyEqual(Qa, Qb);
            % prediction with a foreign policy: the reported pi comes from the policy interface, not from a field
            walk = OceanStats.RL.Env.Classic.RandomWalkEnv();
            [Va, ~, pa] = OceanStats.RL.TD.TDLearner('Env', walk, 'Task', 'prediction', 'StepSize', 0.1, 'Episodes', 50, 'Seed', 2, 'Policy', hook).run();
            [Vb, ~, pb] = OceanStats.RL.TD.TDLearner('Env', walk, 'Task', 'prediction', 'StepSize', 0.1, 'Episodes', 50, 'Seed', 2).run();
            testCase.verifyEqual(Va, Vb);  testCase.verifyEqual(pa, pb);
        end

        function testHooksAreValidatedAgainstThePartsActuallyUsed(testCase)
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv('Rows', 3, 'Cols', 5);
            id = 'OceanStats:RL:TD:TDLearner:';
            dbl = @(sm, iv) OceanStats.RL.TD.Values.DoubleTabularQ(sm.ValidMask, iv);
            plain = @(sm, iv) OceanStats.RL.TD.Values.TabularQ(sm.ValidMask, iv);
            L = @(varargin) OceanStats.RL.TD.TDLearner('Env', env, 'Target', 'qlearning', varargin{:});
            testCase.verifyError(@() L('Double_learning', true, 'ValueFunction', plain), [id 'BadCombination']);   % the flag would be silently ignored
            testCase.verifyError(@() L('Target', 'sarsa', 'ValueFunction', dbl).run(), [id 'BadCombination']);      % it would silently train Q-learning under a Sarsa label
            testCase.verifyError(@() L('Target', 'expectedsarsa', 'ValueFunction', dbl).run(), [id 'BadCombination']);
            testCase.verifyError(@() L('ValueFunction', @(sm, iv) 5).run(), [id 'BadValueFunction']);
            testCase.verifyError(@() L('Policy', @(sm) 5).run(), [id 'BadPolicy']);
            testCase.verifyError(@() L('Observer', 5).run(), [id 'BadObserver']);
            testCase.verifyError(@() L('Observer', {5}).run(), [id 'BadObserver']);
            testCase.verifyError(@() L('Extra', [struct('a', 1) struct('a', 2)]), 'MATLAB:validation:IncompatibleSize');
            testCase.verifyError(@() L('Mode', 'trace', 'ValueFunction', dbl).run(), [id 'BadCombination']);       % a double estimator only works one-step
            testCase.verifyError(@() L('Target', 'sarsa', 'Mode', 'nstep', 'ValueFunction', dbl).run(), [id 'BadCombination']);
            [~, Q] = L('ValueFunction', dbl, 'Episodes', 10, 'Seed', 1).run();                                      % one-step is fine
            testCase.verifyTrue(all(isfinite(Q(:))));
            wrongKind = @(sm, iv) OceanStats.RL.TD.Values.TabularV(sm.NumS, iv);
            testCase.verifyError(@() L('ValueFunction', wrongKind).run(), [id 'BadValueFunction']);
            testCase.verifyError(@() L('PolicyProbs', ones(15, 4) / 4), [id 'BadCombination']);                     % a policy to evaluate in control
            v = OceanStats.RL.TD.Values.TabularV(3, 0);
            testCase.verifyError(@() v.finalQ(), 'OceanStats:RL:TD:IValueFunction:NotActionValues');
            testCase.verifyEqual(v.finalV(), zeros(3, 1));
        end

        function testModeValidation(testCase)
            env = OceanStats.RL.Env.Classic.RandomWalkEnv();
            id = 'OceanStats:RL:TD:TDLearner:';
            testCase.verifyError(@() OceanStats.RL.TD.TDLearner('Env', env, 'Mode', 'nope'), [id 'UnknownMode']);
            testCase.verifyError(@() OceanStats.RL.TD.TDLearner('Env', env, 'Mode', ["a" "b"]), [id 'BadMode']);
            testCase.verifyError(@() OceanStats.RL.TD.TDLearner('Env', env, 'Mode', 3), [id 'BadMode']);
            testCase.verifyError(@() OceanStats.RL.TD.TDLearner('Env', env, 'Mode', ''), [id 'BadMode']);
            try
                OceanStats.RL.TD.TDLearner('Env', env, 'Mode', 'nope');
            catch ME
                testCase.verifyTrue(contains(ME.message, 'onestep') && contains(ME.message, 'lambdareturn'), 'the message lists the valid modes');
            end
            b = OceanStats.RL.TD.Backups.NStepBackup();                                   % an IBackup object is accepted directly
            L = OceanStats.RL.TD.TDLearner('Env', env, 'Task', 'prediction', 'Mode', b, 'N', 2);
            testCase.verifyEqual(L.Mode, "nstep");
        end

        function testExtraSettingsReachTheBackup(testCase)
            % a backup that needs a setting the learner does not know reads ctx.extra (no edit to the learner or the Context)
            testCase.verifyEqual(OceanStats.RL.TD.Context().extra, struct());
            L = OceanStats.RL.TD.TDLearner('Env', OceanStats.RL.Env.Classic.RandomWalkEnv(), 'Extra', struct('temperature', 2));
            testCase.verifyEqual(L.Extra.temperature, 2);
        end

        function testNewComponentsAreHonestAboutTheirPillars(testCase)
            p = OceanStats.RL.TD.Policies.ExplorationPolicy(OceanStats.RL.TD.Exploration.EpsilonGreedyExploration(0.1));
            testCase.verifyFalse(p.getMetadata().IsSymbolic);                              % no closed form -> no faked formula
            testCase.verifyTrue(OceanStats.RL.TD.Policies.FixedPolicy([], true(2, 2)).getMetadata().IsSymbolic);
            for c = {OceanStats.RL.TD.Backups.OneStepBackup(), OceanStats.RL.TD.Values.TabularQ(true(2, 2), 0), p}
                m = c{1}.getMetadata();
                testCase.verifyFalse(contains(string(m.Assumptions{1}), "one scalar per (state, action)"), class(c{1}));
            end
            spec = OceanStats.RL.TD.Backups.NStepBackup().getPipelineSpec();
            testCase.verifyEqual(spec.Inputs.Name, "context");
        end

        % ---------------- dependency rules of the decomposition ----------------
        function testDependencyScannerHasPower(testCase)
            bad = ["env = ctx.sampler.Env; env.step(a);", "x = obj.Env.reset();", "isa(t, 'OceanStats.RL.TD.Targets.SarsaTarget')", ...
                   "y = OceanStats.RL.TD.Targets.QLearningTarget();", "fprintf('%d', 1); v = Values.TabularQ(a);"];
            for k = 1:numel(bad)
                testCase.verifyNotEmpty(OceanTest.OceanStats.RL.TDDecompositionTest.violations_(bad(k), true), "must flag: " + bad(k));
            end
            ok = ["% obj.Env.step is only mentioned in a comment", "x = 1;  % trailing note about Env.step(a)", "ob.step(r, s2i, a2, sm.OutsideLast);", ...
                  "fprintf('%d steps', n);"];
            for k = 1:numel(ok)
                testCase.verifyEmpty(OceanTest.OceanStats.RL.TDDecompositionTest.violations_(ok(k), true), "must not flag: " + ok(k));
            end
            sm = OceanStats.RL.TD.Sampler(OceanStats.RL.Compat.LegacyEnvironment(OceanStats.RL.Env.Classic.RandomWalkEnv()), 10);
            testCase.verifyError(@() sm.Env, ?MException);                                 % the environment is not reachable through the Sampler
        end

        function testOnlyTheSamplerTouchesTheEnvironmentAndNoPartNamesAConcreteAlgorithm(testCase)
            root = fileparts(which('OceanStats.RL.TD.TDLearner'));
            files = [dir(fullfile(root, '+Backups', '*.m')); dir(fullfile(root, '+Values', '*.m')); ...
                     dir(fullfile(root, '+Policies', '*.m')); dir(fullfile(root, '+Observers', '*.m')); dir(fullfile(root, 'Context.m'))];
            testCase.verifyGreaterThan(numel(files), 15);
            for k = 1:numel(files)
                txt = fileread(fullfile(files(k).folder, files(k).name));
                v = OceanTest.OceanStats.RL.TDDecompositionTest.violations_(txt, endsWith(files(k).folder, '+Backups') || strcmp(files(k).name, 'Context.m'));
                testCase.verifyEmpty(v, [files(k).name ' breaks a dependency rule: ' strjoin(string(v), ', ')]);
            end
            % TDLearner itself may name the DEFAULT parts (in buildValues_/buildPolicy_) but never an environment call or a target class
            txt = fileread(fullfile(root, 'TDLearner.m'));
            testCase.verifyEmpty(regexp(OceanTest.OceanStats.RL.TDDecompositionTest.code_(txt), '\.Env\.(step|reset|state_to_index|get_valid_actions|get_all_states|is_terminal|was_truncated)', 'once'));
            testCase.verifyEmpty(regexp(OceanTest.OceanStats.RL.TDDecompositionTest.code_(txt), 'Targets\.(Sarsa|QLearning|ExpectedSarsa|TD0)Target', 'once'));
        end
    end

    methods (Static, Access = private)
        function c = code_(txt)
            % strip real comments only: whole-line comments and trailing comments introduced by whitespace + % + whitespace
            c = regexprep(char(txt), '(?m)^[ \t]*%[^\n]*', '');
            c = regexprep(c, '(?m)[ \t]%[ \t][^\n]*', '');
        end
        function v = violations_(txt, checkValueNames)
            c = OceanTest.OceanStats.RL.TDDecompositionTest.code_(txt);
            pats = {'\.Env\.(step|reset|state_to_index|get_valid_actions|get_all_states|is_terminal|was_truncated)', ...
                    '\<env\.(step|reset)\(', 'Targets\.(Sarsa|QLearning|ExpectedSarsa|TD0)Target', 'isa\([^)]*Target'};
            if checkValueNames
                pats = [pats, {'Values\.(Tabular|DoubleTabular)[A-Za-z]*', 'Policies\.(Exploration|Fixed)Policy', 'Observers\.(Recorder|NullObserver)'}];
            end
            v = {};
            for i = 1:numel(pats)
                if ~isempty(regexp(c, pats{i}, 'once')), v{end+1} = pats{i}; end %#ok<AGROW>
            end
        end
    end
end
