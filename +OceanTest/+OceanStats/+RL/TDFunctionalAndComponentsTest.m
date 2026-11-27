classdef TDFunctionalAndComponentsTest < OceanTest.BaseTestCase
    % TDFUNCTIONALANDCOMPONENTSTEST  Pure functions, targets, step sizes, exploration rules, traces and the
    % convention-based Factory. Every expected value is worked out by hand in this file from the definitions.

    methods (Test)
        % ---------------- pure functions ----------------
        function testOneStepTargetHandWorked(testCase)
            F = @OceanStats.RL.TD.Functional.oneStepTarget;
            testCase.verifyEqual(F(1, 4, false, 0.5), 1 + 0.5 * 4, 'AbsTol', 1e-14);      % 3
            testCase.verifyEqual(F(1, 4, true, 0.5), 1, 'AbsTol', 1e-14);                  % terminal: bootstrap dropped
            testCase.verifyEqual(F([1 2], [4 6], [false true], 1), [5 2], 'AbsTol', 1e-14);   % vectorised
        end

        function testNStepReturnHandWorked(testCase)
            G = @OceanStats.RL.TD.Functional.nStepReturn;
            % rewards 1,2,3 gamma .5 bootstrap 10:  1 + .5*2 + .25*3 + .125*10 = 4
            testCase.verifyEqual(G([1; 2; 3], 10, 0.5), 1 + 1 + 0.75 + 1.25, 'AbsTol', 1e-14);
            testCase.verifyEqual(G([1; 2; 3], 0, 0.5), 2.75, 'AbsTol', 1e-14);             % terminated within the window
            testCase.verifyEqual(G(5, 7, 0.9), 5 + 0.9 * 7, 'AbsTol', 1e-14);             % n = 1 is the one-step target
        end

        function testLambdaReturnsEndpointsAndHandWorked(testCase)
            LR = @OceanStats.RL.TD.Functional.lambdaReturns;
            r = [1; 2; 3];  nv = [10; 20; 30];  g = 0.5;
            % lambda = 0 -> one-step targets  r_t + g*v(S_{t+1}) (last one drops v if terminated)
            testCase.verifyEqual(LR(r, nv, true, g, 0), [1 + 5; 2 + 10; 3], 'AbsTol', 1e-13);
            testCase.verifyEqual(LR(r, nv, false, g, 0), [1 + 5; 2 + 10; 3 + 15], 'AbsTol', 1e-13);
            % lambda = 1, terminated -> Monte Carlo returns: G3 = 3, G2 = 2 + .5*3, G1 = 1 + .5*G2
            testCase.verifyEqual(LR(r, nv, true, g, 1), [1 + 0.5 * 3.5; 3.5; 3], 'AbsTol', 1e-13);
            % lambda = .5 by hand (v_next = 10,20,30): G3 = 3 ; G2 = 2 + .5*(.5*20 + .5*3) = 7.75 ;
            %                      G1 = 1 + .5*(.5*10 + .5*7.75) = 5.4375
            testCase.verifyEqual(LR(r, nv, true, g, 0.5), [5.4375; 7.75; 3], 'AbsTol', 1e-13);
        end

        function testEpsilonGreedyProbsHandWorkedAndTies(testCase)
            P = @OceanStats.RL.TD.Functional.epsilonGreedyProbs;
            p = P([1 5 3], true(1, 3), 0.3);            % greedy = action 2:  .1 each + .7 on the greedy one
            testCase.verifyEqual(p, [0.1 0.8 0.1], 'AbsTol', 1e-14);
            p2 = P([2 2 1], true(1, 3), 0.3);           % tie for the max between 1 and 2: .7/2 each on top of .1
            testCase.verifyEqual(p2, [0.45 0.45 0.1], 'AbsTol', 1e-14);
            p3 = P([9 5 3], [false true true], 0.2);    % action 1 invalid (even though its q is largest)
            testCase.verifyEqual(p3, [0 0.9 0.1], 'AbsTol', 1e-14);
            testCase.verifyEqual(sum(p3), 1, 'AbsTol', 1e-14);
        end

        function testSoftmaxProbsHandWorked(testCase)
            p = OceanStats.RL.TD.Functional.softmaxProbs([0 log(3) 100], [true true false], 1);
            testCase.verifyEqual(p, [0.25 0.75 0], 'AbsTol', 1e-14);                       % e^0 : e^ln3 = 1 : 3
            big = OceanStats.RL.TD.Functional.softmaxProbs([1000 1001], true(1, 2), 1);   % no overflow
            testCase.verifyEqual(sum(big), 1, 'AbsTol', 1e-14);
        end

        function testExpectedAndGreedyValuesIgnoreInvalidEntries(testCase)
            q = [NaN 4 2];  valid = [false true true];
            testCase.verifyEqual(OceanStats.RL.TD.Functional.greedyValue(q, valid), 4);
            testCase.verifyEqual(OceanStats.RL.TD.Functional.expectedValue(q, [0.9 0.25 0.75], valid), 0.25 * 4 + 0.75 * 2, 'AbsTol', 1e-14);
        end

        % ---------------- targets ----------------
        function testTargetsHandWorked(testCase)
            T = OceanStats.RL.TD.Targets.QLearningTarget();
            S = OceanStats.RL.TD.Targets.SarsaTarget();
            E = OceanStats.RL.TD.Targets.ExpectedSarsaTarget();
            Z = OceanStats.RL.TD.Targets.TD0Target();
            q = [1 5 3];  valid = true(1, 3);  p = [0.2 0.5 0.3];  g = 0.9;  r = 2;
            testCase.verifyEqual(T.compute(r, q, 1, false, g, p, valid), 2 + 0.9 * 5, 'AbsTol', 1e-14);          % max
            testCase.verifyEqual(S.compute(r, q, 3, false, g, p, valid), 2 + 0.9 * 3, 'AbsTol', 1e-14);          % Q(s', a'=3)
            testCase.verifyEqual(E.compute(r, q, 3, false, g, p, valid), 2 + 0.9 * (0.2 + 2.5 + 0.9), 'AbsTol', 1e-14);
            testCase.verifyEqual(Z.compute(r, 4, 1, false, g, [], true), 2 + 0.9 * 4, 'AbsTol', 1e-14);
            for t = {T, S, E, Z}
                testCase.verifyEqual(t{1}.compute(r, q, 1, true, g, p, valid), r, 'AbsTol', 1e-14, ...
                    "terminal must drop the bootstrap: " + t{1}.name());
            end
            testCase.verifyTrue(T.IsOffPolicy);
            testCase.verifyFalse(S.IsOffPolicy);
        end

        function testQLearningTargetRespectsValidMask(testCase)
            T = OceanStats.RL.TD.Targets.QLearningTarget();
            y = T.compute(0, [100 1 2], 1, false, 1, [0 0.5 0.5], [false true true]);   % action 1 is illegal
            testCase.verifyEqual(y, 2);
        end

        % ---------------- step sizes, exploration, traces ----------------
        function testStepSizes(testCase)
            c = OceanStats.RL.TD.StepSize.ConstantStepSize(0.25);
            testCase.verifyEqual(c.alpha(1), 0.25);  testCase.verifyEqual(c.alpha(1000), 0.25);
            h = OceanStats.RL.TD.StepSize.HarmonicStepSize();
            testCase.verifyEqual([h.alpha(1) h.alpha(2) h.alpha(4)], [1 0.5 0.25], 'AbsTol', 1e-14);
            h2 = OceanStats.RL.TD.StepSize.HarmonicStepSize('Power', 0.5, 'Scale', 0.5);
            testCase.verifyEqual(h2.alpha(4), 0.5 / 2, 'AbsTol', 1e-14);
            testCase.verifyLessThanOrEqual(h2.alpha(1), 1);
            testCase.verifyError(@() OceanStats.RL.TD.StepSize.ConstantStepSize(0), 'MATLAB:validators:mustBePositive');
        end

        function testExplorationRules(testCase)
            valid = true(1, 3);  q = [0 1 0];
            e = OceanStats.RL.TD.Exploration.EpsilonGreedyExploration(0.3);
            testCase.verifyEqual(e.probabilities(q, valid, 0), [0.1 0.8 0.1], 'AbsTol', 1e-14);
            d = OceanStats.RL.TD.Exploration.DecayingEpsilonGreedyExploration('Initial', 1, 'Final', 0.1);
            testCase.verifyEqual(sum(d.probabilities(q, valid, 0.5)), 1, 'AbsTol', 1e-14);
            testCase.verifyEqual(d.probabilities(q, valid, 0), [1/3 1/3 1/3], 'AbsTol', 1e-14);          % eps = 1
            testCase.verifyEqual(d.probabilities(q, valid, 1), [0.1/3 0.9 + 0.1/3 0.1/3], 'AbsTol', 1e-14);   % eps = final
            s = OceanStats.RL.TD.Exploration.SoftmaxExploration(1);
            testCase.verifyEqual(s.probabilities([0 log(3) 0], [true true false], 0), [0.25 0.75 0], 'AbsTol', 1e-14);
        end

        function testSelectSamplesFromTheDistribution(testCase)
            e = OceanStats.RL.TD.Exploration.EpsilonGreedyExploration(0.5);
            rs = RandStream('mt19937ar', 'Seed', 1);
            n = 20000;  counts = zeros(1, 3);
            for i = 1:n, a = e.select([0 1 0], true(1, 3), 0, rs); counts(a) = counts(a) + 1; end
            testCase.verifyEqual(counts / n, [1/6 2/3 1/6], 'AbsTol', 0.02);
            % invalid actions are never selected
            for i = 1:200
                testCase.verifyNotEqual(e.select([5 1 0], [false true true], 0, rs), 1);
            end
        end

        function testTraces(testCase)
            E = [0 2; 0 0];                                   % column-major linear index: 3 is entry (1,2)
            acc = OceanStats.RL.TD.Traces.AccumulatingTrace();  rep = OceanStats.RL.TD.Traces.ReplacingTrace();
            testCase.verifyEqual(acc.visit(E, 3), [0 3; 0 0]);          % accumulate: 2 + 1
            testCase.verifyEqual(acc.visit(E, 1), [1 2; 0 0]);
            testCase.verifyEqual(rep.visit(E, 3), [0 1; 0 0]);          % replace: 2 -> 1
            testCase.verifyEqual(rep.visit(E, 1), [1 2; 0 0]);
        end

        % ---------------- factory / extension rule / 3 pillars ----------------
        function testFactoryDiscoversComponentsByConvention(testCase)
            F = @OceanStats.RL.TD.Factory.validNames;
            % the SHIPPED names must be discoverable; extra files are allowed (extension rule: add a file, edit nothing)
            testCase.verifyTrue(all(ismember({'expectedsarsa', 'qlearning', 'sarsa', 'td0'}, F("target"))));
            testCase.verifyTrue(all(ismember({'constant', 'harmonic'}, F("stepsize"))));
            testCase.verifyTrue(all(ismember({'decayingepsilongreedy', 'epsilongreedy', 'softmax'}, F("exploration"))));
            testCase.verifyTrue(all(ismember({'accumulating', 'replacing'}, F("trace"))));
            testCase.verifyClass(OceanStats.RL.TD.Factory.create("target", "QLearning"), 'OceanStats.RL.TD.Targets.QLearningTarget');
            testCase.verifyClass(OceanStats.RL.TD.Factory.create("stepsize", "constant", 0.3), 'OceanStats.RL.TD.StepSize.ConstantStepSize');
            testCase.verifyError(@() OceanStats.RL.TD.Factory.create("target", "nope"), 'OceanNLP:Text:StrategyResolver:Unknown');
            testCase.verifyError(@() OceanStats.RL.TD.Factory.create("kind", "x"), 'OceanStats:RL:TD:Factory:UnknownKind');
        end

        function testEveryDiscoveredComponentHonoursThreePillars(testCase)
            for kind = OceanStats.RL.TD.Factory.kinds()
                for n = OceanStats.RL.TD.Factory.validNames(kind)
                    c = OceanStats.RL.TD.Factory.create(kind, n{1});
                    label = kind + "/" + n{1};
                    testCase.verifyTrue(isa(c, 'OceanStats.RL.TD.ITDComponent'), label);
                    m = c.getMetadata();
                    if m.IsSymbolic       % an honest-opacity component may say false; then it must not claim a formula
                        testCase.verifyGreaterThan(strlength(m.SymbolicLatex), 0, label);
                    end
                    testCase.verifyGreaterThan(strlength(m.Reference), 0, label);
                    p = c.getPipelineSpec();
                    testCase.verifyEqual(p.Kind, "NONVIRTUAL", label);
                    testCase.verifyTrue(all(isfield(p, ["Inputs" "Outputs" "DiagramSymbol" "ComposesFrom"])), label);
                end
            end
        end

        function testNewTargetIsOneFileViaRegisterWithoutEditingAnything(testCase)
            OceanNLP.Text.StrategyResolver.unregisterAll();
            testCase.addTeardown(@() OceanNLP.Text.StrategyResolver.unregisterAll());
            % a foreign class registered under the target kind is available with no edit to any TD file
            OceanNLP.Text.StrategyResolver.register("OceanStats.RL.TD.Targets", "myrule", @OceanStats.RL.TD.Targets.SarsaTarget);
            testCase.verifyTrue(any(strcmp(OceanStats.RL.TD.Factory.validNames("target"), 'myrule')));
            testCase.verifyClass(OceanStats.RL.TD.Factory.create("target", "myrule"), 'OceanStats.RL.TD.Targets.SarsaTarget');
        end
    end
end
