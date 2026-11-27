classdef TDLearnerTest < OceanTest.BaseTestCase
    % TDLEARNERTEST  The loop, tables and modes of OceanStats.RL.TD.TDLearner. Checks are of four kinds:
    %   * exact algebraic EQUIVALENCES between modes (same seed => identical tables),
    %   * an independent MATLAB replay of the recorded transitions (the Python replay is the external check),
    %   * convergence to exact ground truth (DP / closed form),
    %   * termination-vs-truncation semantics, validation errors, reproducibility, 3 pillars.

    methods (Test)
        % ---------------- reproducibility ----------------
        function testSameSeedGivesIdenticalTablesDifferentSeedDoesNot(testCase)
            mk = @(seed) OceanStats.RL.TD.TDLearner('Env', OceanStats.RL.Env.Classic.CliffWalkingEnv(), 'Target', 'sarsa', ...
                'StepSize', 0.5, 'Episodes', 30, 'Seed', seed);
            [~, Q1] = mk(3).run();  [~, Q2] = mk(3).run();  [~, Q3] = mk(4).run();
            testCase.verifyEqual(Q1, Q2);
            testCase.verifyNotEqual(Q1, Q3);
        end

        % ---------------- exact equivalences between modes ----------------
        function testSarsaLambdaZeroEqualsOneStepSarsa(testCase)
            [~, Qa] = OceanTest.OceanStats.RL.TDLearnerTest.control_('sarsa', 'onestep');
            [~, Qb] = OceanTest.OceanStats.RL.TDLearnerTest.control_('sarsa', 'trace', 'Lambda', 0);
            testCase.verifyEqual(Qb, Qa, 'AbsTol', 1e-13);
        end

        function testQLambdaZeroEqualsQLearning(testCase)
            [~, Qa] = OceanTest.OceanStats.RL.TDLearnerTest.control_('qlearning', 'onestep');
            [~, Qb] = OceanTest.OceanStats.RL.TDLearnerTest.control_('qlearning', 'trace', 'Lambda', 0);
            testCase.verifyEqual(Qb, Qa, 'AbsTol', 1e-13);
        end

        function testNStepSarsaWithNOneEqualsOneStepSarsa(testCase)
            [~, Qa] = OceanTest.OceanStats.RL.TDLearnerTest.control_('sarsa', 'onestep');
            [~, Qb] = OceanTest.OceanStats.RL.TDLearnerTest.control_('sarsa', 'nstep', 'N', 1);
            testCase.verifyEqual(Qb, Qa, 'AbsTol', 1e-13);
        end

        function testTraceModeWithTraceStrategiesAgreeWhenNoStateIsRevisited(testCase)
            % on a chain where every state is visited at most once per episode, replacing and accumulating
            % traces are the same thing (a state's trace is bumped once, from 0)
            env = OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner', 4);
            pol = zeros(6, 2);  pol(2:5, 2) = 1;                               % always go right: no revisits
            f = @(tr) OceanStats.RL.TD.TDLearner('Env', env, 'Task', 'prediction', 'Mode', 'trace', 'Lambda', 0.7, ...
                'Trace', tr, 'PolicyProbs', pol, 'StepSize', 0.2, 'Episodes', 50, 'Seed', 1).run();
            testCase.verifyEqual(f('replacing'), f('accumulating'), 'AbsTol', 1e-13);
        end

        function testOfflineBackwardViewEqualsForwardLambdaReturn(testCase)
            % Sutton & Barto ch. 12: offline TD(lambda) with accumulating traces == the offline lambda-return
            % algorithm, exactly, for any lambda (including states revisited within an episode)
            env = OceanStats.RL.Env.Classic.RandomWalkEnv();
            for lam = [0 0.4 0.9 1]
                back = OceanStats.RL.TD.TDLearner('Env', env, 'Task', 'prediction', 'Mode', 'trace', 'Offline', true, ...
                    'Lambda', lam, 'StepSize', 0.1, 'Gamma', 0.95, 'Episodes', 200, 'Seed', 5).run();
                fwd = OceanStats.RL.TD.TDLearner('Env', env, 'Task', 'prediction', 'Mode', 'lambdareturn', ...
                    'Lambda', lam, 'StepSize', 0.1, 'Gamma', 0.95, 'Episodes', 200, 'Seed', 5).run();
                testCase.verifyEqual(back, fwd, 'AbsTol', 1e-11, sprintf('lambda = %g', lam));
            end
        end

        function testNStepPredictionWithNOneEqualsTD0(testCase)
            env = OceanStats.RL.Env.Classic.RandomWalkEnv();
            a = OceanStats.RL.TD.TDLearner('Env', env, 'Task', 'prediction', 'StepSize', 0.1, 'Episodes', 200, 'Seed', 2).run();
            b = OceanStats.RL.TD.TDLearner('Env', env, 'Task', 'prediction', 'Mode', 'nstep', 'N', 1, ...
                'StepSize', 0.1, 'Episodes', 200, 'Seed', 2).run();
            testCase.verifyEqual(b, a, 'AbsTol', 1e-13);
        end

        function testLargeNNStepPredictionEqualsMonteCarloReturnUpdate(testCase)
            % n >= episode length => every update uses the full return (no bootstrap): constant-alpha every-visit MC
            env = OceanStats.RL.Env.Classic.RandomWalkEnv();
            [V, ~, ~, h] = OceanStats.RL.TD.TDLearner('Env', env, 'Task', 'prediction', 'Mode', 'nstep', 'N', 1000, ...
                'StepSize', 0.1, 'Episodes', 100, 'Seed', 6, 'Record', true).run();
            Vmc = zeros(7, 1);  cnt = zeros(7, 1);
            for e = 1:numel(h.Episodes)
                ep = h.Episodes{e};  T = numel(ep.R);
                for t = 1:T
                    G = sum(ep.R(t:T));
                    Vmc(ep.S(t)) = Vmc(ep.S(t)) + 0.1 * (G - Vmc(ep.S(t)));
                    cnt(ep.S(t)) = cnt(ep.S(t)) + 1;
                end
            end
            testCase.verifyEqual(V, Vmc, 'AbsTol', 1e-12);
        end

        % ---------------- independent replay of recorded transitions ----------------
        function testQLearningMatchesAnIndependentReplay(testCase)
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv('Rows', 3, 'Cols', 5);
            [~, Q, ~, h] = OceanStats.RL.TD.TDLearner('Env', env, 'Target', 'qlearning', 'StepSize', 0.3, 'Gamma', 0.9, ...
                'Exploration', 0.2, 'Episodes', 40, 'Seed', 9, 'Record', true).run();
            Qr = zeros(env.NumStates, env.NumActions);
            valid = false(size(Qr));
            for s = 1:size(Qr, 1), for a = env.get_valid_actions(s), valid(s, a) = true; end, end
            for e = 1:numel(h.Episodes)
                ep = h.Episodes{e};
                for t = 1:numel(ep.R)
                    s = ep.S(t);  a = ep.A(t);  s2 = ep.S(t + 1);
                    term = ep.Terminated && t == numel(ep.R);
                    q2 = Qr(s2, :);  q2(~valid(s2, :)) = -Inf;
                    y = ep.R(t) + 0.9 * ~term * max(q2);
                    if term, y = ep.R(t); end
                    Qr(s, a) = Qr(s, a) + 0.3 * (y - Qr(s, a));
                end
            end
            testCase.verifyEqual(Q, Qr, 'AbsTol', 1e-12);
        end

        function testTD0MatchesAnIndependentReplayWithHarmonicStepSize(testCase)
            env = OceanStats.RL.Env.Classic.RandomWalkEnv();
            [V, ~, ~, h] = OceanStats.RL.TD.TDLearner('Env', env, 'Task', 'prediction', ...
                'StepSize', OceanStats.RL.TD.StepSize.HarmonicStepSize('Power', 0.8), 'Gamma', 1, 'Episodes', 60, 'Seed', 4, 'Record', true).run();
            Vr = zeros(7, 1);  n = zeros(7, 1);
            for e = 1:numel(h.Episodes)
                ep = h.Episodes{e};
                for t = 1:numel(ep.R)
                    s = ep.S(t);  s2 = ep.S(t + 1);
                    n(s) = n(s) + 1;
                    term = ep.Terminated && t == numel(ep.R);
                    y = ep.R(t) + ~term * Vr(s2);
                    Vr(s) = Vr(s) + n(s)^-0.8 * (y - Vr(s));
                end
            end
            testCase.verifyEqual(V, Vr, 'AbsTol', 1e-12);
        end

        % ---------------- convergence to ground truth ----------------
        function testTD0ConvergesToTheExactValues(testCase)
            env = OceanStats.RL.Env.Classic.RandomWalkEnv();
            V = OceanStats.RL.TD.TDLearner('Env', env, 'Task', 'prediction', ...
                'StepSize', OceanStats.RL.TD.StepSize.HarmonicStepSize('Power', 0.6), 'Episodes', 5000, 'Seed', 3).run();
            testCase.verifyEqual(V(2:6)', (1:5) / 6, 'AbsTol', 0.03);
        end

        function testTD0ConvergesToExactValuesForANonUniformPolicyAndDiscount(testCase)
            env = OceanStats.RL.Env.Classic.RandomWalkEnv();
            pol = zeros(7, 2);  pol(2:6, 1) = 0.3;  pol(2:6, 2) = 0.7;
            exact = env.exactPolicyValues(pol, 0.9);
            V = OceanStats.RL.TD.TDLearner('Env', env, 'Task', 'prediction', 'PolicyProbs', pol, 'Gamma', 0.9, ...
                'StepSize', OceanStats.RL.TD.StepSize.HarmonicStepSize('Power', 0.6), 'Episodes', 6000, 'Seed', 8).run();
            testCase.verifyEqual(V(2:6), exact(2:6), 'AbsTol', 0.03);
        end

        function testNStepAndLambdaPredictionAlsoConverge(testCase)
            env = OceanStats.RL.Env.Classic.RandomWalkEnv();
            h = OceanStats.RL.TD.StepSize.HarmonicStepSize('Power', 0.6);
            V3 = OceanStats.RL.TD.TDLearner('Env', env, 'Task', 'prediction', 'Mode', 'nstep', 'N', 3, 'StepSize', h, 'Episodes', 4000, 'Seed', 1).run();
            Vl = OceanStats.RL.TD.TDLearner('Env', env, 'Task', 'prediction', 'Mode', 'trace', 'Lambda', 0.8, 'StepSize', h, 'Episodes', 4000, 'Seed', 1).run();
            testCase.verifyEqual(V3(2:6)', (1:5) / 6, 'AbsTol', 0.04);
            testCase.verifyEqual(Vl(2:6)', (1:5) / 6, 'AbsTol', 0.04);
        end

        function testQLearningConvergesToDPOptimalValues(testCase)
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv('Rows', 3, 'Cols', 6);
            Vstar = OceanStats.RL.DP.ValueIterationSolver(env, 'DiscountFactor', 1).run();
            V = OceanStats.RL.TD.TDLearner('Env', env, 'Target', 'qlearning', 'Exploration', 0.3, 'Episodes', 3000, 'Seed', 2, ...
                'StepSize', OceanStats.RL.TD.StepSize.HarmonicStepSize('Power', 0.7)).run();
            live = ~env.TerminalMask;
            testCase.verifyLessThan(max(abs(V(live) - Vstar(live))), 0.05);
        end

        function testWindyGridworldGreedyPolicyFindsTheOptimalPath(testCase)
            env = OceanStats.RL.Env.Classic.WindyGridworldEnv();
            for t = ["sarsa" "qlearning" "expectedsarsa"]
                [~, Q] = OceanStats.RL.TD.TDLearner('Env', env, 'Target', char(t), 'StepSize', 0.5, 'Exploration', 0.1, ...
                    'Episodes', 500, 'Seed', 5).run();
                testCase.verifyEqual(OceanTest.OceanStats.RL.TDLearnerTest.greedySteps_(env, Q), 15, char(t));
            end
        end

        function testDoubleQLearningReducesMaximizationBias(testCase)
            % Sutton & Barto Fig 6.5. Stated expectation before running: plain Q-learning chooses the bad 'left'
            % action a large fraction of early episodes, Double Q-learning much less.
            env = OceanStats.RL.Env.Classic.MaximizationBiasEnv();
            frac = @(dbl) mean(arrayfun(@(sd) OceanTest.OceanStats.RL.TDLearnerTest.leftRate_(env, dbl, sd), 1:40));
            ql = frac(false);  dq = frac(true);
            testCase.verifyGreaterThan(ql, 0.25);
            testCase.verifyLessThan(dq, ql - 0.1);
        end

        function testControlNeverSelectsInvalidActionsAndLeavesThemUntouched(testCase)
            env = OceanStats.RL.Env.Classic.MaximizationBiasEnv();          % state 1 admits only actions 1, 2 of 10
            [~, Q] = OceanStats.RL.TD.TDLearner('Env', env, 'Episodes', 200, 'Seed', 1, 'StepSize', 0.1, 'Exploration', 0.5).run();
            testCase.verifyEqual(Q(1, 3:10), zeros(1, 8));
        end

        % ---------------- termination vs truncation ----------------
        function testTruncationKeepsTheBootstrapButTerminationDropsIt(testCase)
            % every episode is cut after ONE step: with a correct bootstrap V(4) <- 0 + V(neighbour) stays 0.5
            env = OceanStats.RL.Env.Wrappers.TimeLimit(OceanStats.RL.Env.Classic.RandomWalkEnv(), 1);
            V = OceanStats.RL.TD.TDLearner('Env', env, 'Task', 'prediction', 'InitialValue', 0.5, ...
                'StepSize', 0.3, 'Episodes', 300, 'Seed', 1).run();
            testCase.verifyEqual(V(4), 0.5, 'AbsTol', 1e-12);
            % ... the same through the n-step buffer and the learner's own MaxSteps cut-off
            V2 = OceanStats.RL.TD.TDLearner('Env', OceanStats.RL.Env.Classic.RandomWalkEnv(), 'Task', 'prediction', 'Mode', 'nstep', ...
                'N', 2, 'MaxSteps', 1, 'InitialValue', 0.5, 'StepSize', 0.3, 'Episodes', 300, 'Seed', 1).run();
            testCase.verifyEqual(V2(4), 0.5, 'AbsTol', 1e-12);
            V3 = OceanStats.RL.TD.TDLearner('Env', OceanStats.RL.Env.Classic.RandomWalkEnv(), 'Task', 'prediction', 'MaxSteps', 1, ...
                'InitialValue', 0.5, 'StepSize', 0.3, 'Episodes', 300, 'Seed', 1).run();
            testCase.verifyEqual(V3(4), 0.5, 'AbsTol', 1e-12);
            % TRUE termination drops the bootstrap: a wildly wrong initial value is washed out
            e1 = OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner', 1);
            V4 = OceanStats.RL.TD.TDLearner('Env', e1, 'Task', 'prediction', 'InitialValue', 10, ...
                'StepSize', OceanStats.RL.TD.StepSize.HarmonicStepSize(), 'Episodes', 4000, 'Seed', 2).run();
            testCase.verifyEqual(V4(2), 0.5, 'AbsTol', 0.03);
        end

        % ---------------- validation, history, 3 pillars ----------------
        function testInvalidCombinationsAreRejected(testCase)
            env = OceanStats.RL.Env.Classic.RandomWalkEnv();
            T = @(varargin) OceanStats.RL.TD.TDLearner('Env', env, varargin{:});
            id = 'OceanStats:RL:TD:TDLearner:';
            testCase.verifyError(@() T('Task', 'prediction', 'Target', 'qlearning'), [id 'BadTarget']);
            testCase.verifyError(@() T('Task', 'control', 'Target', 'td0'), [id 'BadTarget']);
            testCase.verifyError(@() T('Mode', 'lambdareturn'), [id 'BadMode']);
            testCase.verifyError(@() T('Mode', 'nstep', 'Target', 'qlearning'), [id 'BadTarget']);
            testCase.verifyError(@() T('Double', true, 'Target', 'sarsa'), [id 'BadCombination']);
            testCase.verifyError(@() T('Offline', true), [id 'BadCombination']);
            testCase.verifyError(@() T('Target', 'nope'), 'OceanNLP:Text:StrategyResolver:Unknown');
        end

        function testHistoryAndReturnedShapes(testCase)
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv();
            [V, Q, pi, h] = OceanStats.RL.TD.TDLearner('Env', env, 'Episodes', 25, 'Seed', 1, 'StepSize', 0.5).run();
            testCase.verifyEqual(size(V), [48 1]);  testCase.verifyEqual(size(Q), [48 4]);  testCase.verifyEqual(size(pi), [48 1]);
            testCase.verifyEqual(numel(h.EpisodeReturns), 25);
            testCase.verifyEqual(numel(h.EpisodeLengths), 25);
            testCase.verifyEqual(h.TotalSteps, sum(h.EpisodeLengths));
            testCase.verifyEqual(V(env.idx(4, 12)), 0);                 % terminal
            testCase.verifyTrue(all(pi(~env.TerminalMask) >= 1 & pi(~env.TerminalMask) <= 4));
        end

        function testDecayingExplorationScheduleIsUsed(testCase)
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv('Rows', 3, 'Cols', 5);
            [~, ~, ~, h] = OceanStats.RL.TD.TDLearner('Env', env, 'Target', 'qlearning', 'StepSize', 0.5, 'Episodes', 300, 'Seed', 1, ...
                'Exploration', OceanStats.RL.TD.Exploration.DecayingEpsilonGreedyExploration('Initial', 1, 'Final', 0)).run();
            testCase.verifyLessThan(mean(h.EpisodeLengths(end-49:end)), mean(h.EpisodeLengths(1:50)));   % exploration fades, episodes shorten
        end

        function testLearnerRunsOnTheExistingBlackjackEnvironment(testCase)
            env = OceanStats.RL.Env.Classic.BlackjackEnv();
            [V, ~, ~, h] = OceanStats.RL.TD.TDLearner('Env', env, 'Task', 'prediction', 'StepSize', 0.05, 'Episodes', 200, 'Seed', 1).run();
            testCase.verifyTrue(all(isfinite(V)));
            testCase.verifyEqual(numel(h.EpisodeReturns), 200);
        end

        % ---------------- critic round 1: extension rule, reproducibility, validation ----------------
        function testANewTargetFileWorksInEveryModeWithoutEditingTheLearner(testCase)
            OceanNLP.Text.StrategyResolver.unregisterAll();
            testCase.addTeardown(@() OceanNLP.Text.StrategyResolver.unregisterAll());
            OceanNLP.Text.StrategyResolver.register("OceanStats.RL.TD.Targets", "mix", @OceanTest.OceanStats.RL.Fixtures.MixTarget);
            C = @OceanTest.OceanStats.RL.TDLearnerTest.control_;
            [~, Qm1] = C('mix', 'onestep');
            [~, Qm2] = C('mix', 'nstep', 'N', 1);                        % the tail of the n-step return comes from Target.bootstrap
            [~, Qm3] = C('mix', 'trace', 'Lambda', 0);
            [~, Qs] = C('sarsa', 'onestep');
            testCase.verifyEqual(Qm2, Qm1, 'AbsTol', 1e-13);
            testCase.verifyEqual(Qm3, Qm1, 'AbsTol', 1e-13);
            testCase.verifyGreaterThan(max(abs(Qm1(:) - Qs(:))), 1e-3);  % it is a different rule, not a renamed Sarsa
            [~, Qm4] = C('mix', 'nstep', 'N', 3);
            testCase.verifyTrue(all(isfinite(Qm4(:))));
        end

        function testRunTwiceAndSharedEnvironmentAreReproducible(testCase)
            env = OceanStats.RL.Env.Classic.WindyGridworldEnv('StochasticWind', true, 'Rows', 5, 'Start', [3 1], 'Goal', [3 8]);
            L = OceanStats.RL.TD.TDLearner('Env', env, 'Target', 'sarsa', 'StepSize', 0.4, 'Episodes', 20, 'Seed', 5);
            [~, Qa] = L.run();  [~, Qb] = L.run();
            testCase.verifyEqual(Qb, Qa);
            env.reset();  env.step(1);                                    % the environment used in between must not matter
            [~, Qc] = OceanStats.RL.TD.TDLearner('Env', env, 'Target', 'sarsa', 'StepSize', 0.4, 'Episodes', 20, 'Seed', 5).run();
            testCase.verifyEqual(Qc, Qa);
        end

        function testEnvironmentWithoutSetSeedWarns(testCase)
            env = OceanTest.OceanStats.RL.Fixtures.UnseedableEpisodeEnv();
            L = OceanStats.RL.TD.TDLearner('Env', env, 'Task', 'prediction', 'Episodes', 2, 'Seed', 1);
            testCase.verifyWarning(@() L.run(), 'OceanStats:RL:TD:TDLearner:EnvNotSeedable');
            % a seedable environment (Blackjack now owns a seedable stream) does not warn and is reproducible
            mk = @() OceanStats.RL.TD.TDLearner('Env', OceanStats.RL.Env.Classic.BlackjackEnv(), 'Task', 'prediction', ...
                'StepSize', 0.05, 'Episodes', 100, 'Seed', 3);
            testCase.verifyWarningFree(@() mk().run());
            testCase.verifyEqual(mk().run(), mk().run());
        end

        function testPolicyProbsAreValidated(testCase)
            env = OceanStats.RL.Env.Classic.RandomWalkEnv();
            S = env.NumStates;  A = env.NumActions;
            id = 'OceanStats:RL:TD:TDLearner:BadPolicy';
            mk = @(P) OceanStats.RL.TD.TDLearner('Env', env, 'Task', 'prediction', 'PolicyProbs', P, 'Episodes', 1);
            testCase.verifyError(@() mk(ones(2, 2)).run(), id);                         % wrong size
            testCase.verifyError(@() mk(zeros(S, A)).run(), id);                        % zero rows for non-terminal states
            P = 0.5 * ones(S, A);  P(2, 1) = -0.5;  P(2, 2) = 1.5;
            testCase.verifyError(@() mk(P).run(), id);                                  % negative entry
            ok = 0.5 * ones(S, A);
            [~] = mk(ok).run();                                                         % a valid one still runs
        end

        function testPredictionRejectsActionValueTargets(testCase)
            env = OceanStats.RL.Env.Classic.RandomWalkEnv();
            for m = ["onestep" "nstep" "trace"]
                testCase.verifyError(@() OceanStats.RL.TD.TDLearner('Env', env, 'Task', 'prediction', 'Mode', m, 'Target', 'expectedsarsa'), ...
                    'OceanStats:RL:TD:TDLearner:BadTarget');
            end
        end

        function testWatkinsCutMakesQLambdaDifferFromSarsaLambda(testCase)
            C = @OceanTest.OceanStats.RL.TDLearnerTest.control_;
            [~, Qw] = C('qlearning', 'trace', 'Lambda', 0.9);
            [~, Qs] = C('sarsa', 'trace', 'Lambda', 0.9);
            testCase.verifyGreaterThan(max(abs(Qw(:) - Qs(:))), 1e-3);
        end

        function testGammaZeroLearnsTheImmediateRewardOnly(testCase)
            env = OceanStats.RL.Env.Classic.RandomWalkEnv();                                % reward 1 only on stepping off the right end
            [V, ~] = OceanStats.RL.TD.TDLearner('Env', env, 'Task', 'prediction', 'Gamma', 0, ...
                'StepSize', OceanStats.RL.TD.StepSize.HarmonicStepSize(), 'Episodes', 3000, 'Seed', 2).run();
            testCase.verifyEqual(V(2:5), zeros(4, 1), 'AbsTol', 1e-12);                % no bootstrapping: nothing leaks backward
            testCase.verifyEqual(V(6), 0.5, 'AbsTol', 0.06);                            % E[R] = P(right) * 1
        end

        function testSoftmaxExplorationRunsInTheLearner(testCase)
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv('Rows', 3, 'Cols', 5);
            for dbl = [false true]
                [~, Q, ~, h] = OceanStats.RL.TD.TDLearner('Env', env, 'Target', 'qlearning', 'Double', dbl, 'Exploration', 'softmax', ...
                    'StepSize', 0.3, 'Gamma', 0.95, 'Episodes', 30, 'Seed', 3).run();
                testCase.verifyTrue(all(isfinite(Q(:))));
                testCase.verifyEqual(numel(h.EpisodeReturns), 30);
            end
        end

        function testMaxStepsOneStillUpdatesAndTruncates(testCase)
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv('Rows', 3, 'Cols', 5);
            [~, Q, ~, h] = OceanStats.RL.TD.TDLearner('Env', env, 'Target', 'sarsa', 'StepSize', 0.5, 'MaxSteps', 1, 'InitialValue', 2, ...
                'Episodes', 5, 'Seed', 1, 'Record', true).run();
            testCase.verifyTrue(all(h.EpisodeLengths == 1));
            testCase.verifyTrue(all(cellfun(@(e) e.Truncated && ~e.Terminated, h.Episodes)));
            testCase.verifyTrue(all(isfinite(Q(:))));
        end

        function testBlackjackTerminalOutsideTheTableIsNotRecordedAsARealState(testCase)
            env = OceanStats.RL.Env.Classic.BlackjackEnv();
            L = OceanStats.RL.TD.TDLearner('Env', env, 'Task', 'prediction', 'StepSize', 0.05, 'Episodes', 200, 'Seed', 1, 'Record', true);
            [~, ~, ~, h] = L.run();
            ends = cellfun(@(e) e.S(end), h.Episodes);
            testCase.verifyTrue(any(ends == 0));                          % busts are marked 0, never the placeholder state 1 ...
            busts = cellfun(@(e) e.S(end) == 0, h.Episodes);
            testCase.verifyTrue(all(cellfun(@(e) numel(e.S) >= 2, h.Episodes(busts))));
        end

        function testEnvironmentsHonourThePillars(testCase)
            for env = {OceanStats.RL.Env.Classic.CliffWalkingEnv(), OceanStats.RL.Env.Classic.RandomWalkEnv(), ...
                       OceanStats.RL.Env.Classic.WindyGridworldEnv(), OceanStats.RL.Env.Classic.MaximizationBiasEnv()}
                m = env{1}.getMetadata();  p = env{1}.getPipelineSpec();
                testCase.verifyFalse(m.IsSymbolic);
                testCase.verifyEqual(p.Kind, "NONVIRTUAL");
                testCase.verifyTrue(all(isfield(p, ["Inputs" "Outputs" "DiagramSymbol" "ComposesFrom"])));
            end
        end

        function testThreePillars(testCase)
            L = OceanStats.RL.TD.TDLearner('Env', OceanStats.RL.Env.Classic.CliffWalkingEnv(), 'Target', 'expectedsarsa');
            m = L.getMetadata();
            testCase.verifyFalse(m.IsSymbolic);
            testCase.verifyTrue(contains(m.ComposedPartReference.TargetSymbolicLatex, "\sum_a"));       % pointer resolves to the real formula
            p = L.getPipelineSpec();
            testCase.verifyEqual(p.Kind, "VIRTUAL");
            testCase.verifyTrue(all(cellfun(@(c) exist(c, 'class') == 8, p.ComposesFrom)));
            testCase.verifyEqual(p.ComposesFrom{1}, 'OceanStats.RL.TD.Targets.ExpectedSarsaTarget');
        end
    end

    methods (Static, Access = private)
        function [V, Q] = control_(target, mode, varargin)
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv('Rows', 3, 'Cols', 5);
            [V, Q] = OceanStats.RL.TD.TDLearner('Env', env, 'Target', target, 'Mode', mode, 'StepSize', 0.3, 'Gamma', 0.95, ...
                'Exploration', 0.2, 'Episodes', 60, 'Seed', 12, varargin{:}).run();
        end

        function n = greedySteps_(env, Q)
            s = env.reset();  n = 0;
            while true
                q = Q(s, :);  q(~ismember(1:size(Q, 2), env.get_valid_actions(s))) = -Inf;
                [~, a] = max(q);
                [s, ~, d] = env.step(a);  n = n + 1;
                if d || n > 500, break; end
            end
        end

        function r = leftRate_(env, dbl, seed)
            [~, ~, ~, h] = OceanStats.RL.TD.TDLearner('Env', env, 'Target', 'qlearning', 'Double', dbl, 'StepSize', 0.1, ...
                'Exploration', 0.1, 'Episodes', 300, 'Seed', seed, 'Record', true).run();
            r = mean(cellfun(@(e) e.A(1) == 1, h.Episodes));
        end
    end
end
