classdef ExperimentToolsTest < OceanTest.BaseTestCase
    % EXPERIMENTTOOLSTEST  Tools behind the generative-model comparisons:
    %   * ComplexShapeImages: shape, range, balanced classes, reproducible
    %     seeds, every class draws ink, and the classes are learnable;
    %   * SampleEvaluator metrics: Frechet distance == its closed form for
    %     Gaussians, nearest-neighbour distance on a hand example;
    %   * Training.Checkpoint reuses finished work; Training.fit trains.

    methods (Test)
        function testComplexShapeImages(testCase)
            G = OceanData.Generators.Shapes2D.ComplexShapeImages;
            [X, y, F] = G.generate(60, 'Size', 24, 'Seed', 3);
            testCase.verifySize(X, [24 24 1 60]);
            testCase.verifyTrue(all(X >= 0 & X <= 1, 'all'));
            testCase.verifyEqual(histcounts(y, 0.5:6.5), 10 * ones(1, 6));            % balanced
            testCase.verifyEqual(height(F), 60);
            testCase.verifyEqual(G.generate(60, 'Size', 24, 'Seed', 3), X);                   % same n + seed -> same images
            ink = squeeze(sum(X, [1 2 3]));
            for c = 1:6
                testCase.verifyGreaterThan(min(ink(y == c)), 5, G.ClassNames(c));         % never empty
                testCase.verifyLessThan(max(ink(y == c)), 0.8 * 24 ^ 2, G.ClassNames(c)); % never fills the frame
            end
            frac = mean(X > 0 & X < 1, 'all');
            testCase.verifyGreaterThan(frac, 0.01);                                      % anti-aliased edges exist
        end

        function testShapeClassesAreLearnable(testCase)
            G = OceanData.Generators.Shapes2D.ComplexShapeImages;
            [X, y] = G.generate(1200, 'Size', 16, 'Seed', 1);
            [Xt, yt] = G.generate(300, 'Size', 16, 'Seed', 2);
            ev = OceanDL.Generative.SampleEvaluator(X, y, 'NumClasses', 6, 'Steps', 400, 'Seed', 1);
            testCase.verifyGreaterThan(ev.testAccuracy(Xt, yt), 0.8);
            r = ev.evaluate(Xt(:, :, :, 1:150), Xt(:, :, :, 151:300), X);
            testCase.verifyEqual(r.ClassesHit, 6);                                       % real data covers all classes
            testCase.verifyEqual(r.Sharpness, 1, 'RelTol', 0.15);                       % real vs real
        end

        function testFrechetAndNearestNeighbourDistances(testCase)
            s = RandStream('mt19937ar', 'Seed', 4);
            A = 1 + 2 * randn(s, 1, 2e5); B = -0.5 + 0.5 * randn(s, 1, 2e5);
            d = OceanDL.Generative.SampleEvaluator.frechetDistance(A, B);
            testCase.verifyEqual(d, (1 + 0.5) ^ 2 + (2 - 0.5) ^ 2, 'RelTol', 0.02);      % 1-D closed form
            C = randn(s, 3, 5000);
            testCase.verifyEqual(OceanDL.Generative.SampleEvaluator.frechetDistance(C, C), 0, 'AbsTol', 1e-8);
            % Multivariate closed form with full covariances.
            L1 = [1 0 0; 0.5 1 0; 0.2 -0.3 0.8]; L2 = [0.7 0 0; -0.2 1.2 0; 0.1 0.4 0.5];
            S1 = L1 * L1.'; S2 = L2 * L2.'; m1 = [1; 0; -1]; m2 = [0; 2; 0];
            P = m1 + L1 * randn(s, 3, 4e5); Q = m2 + L2 * randn(s, 3, 4e5);
            exact = sum((m1 - m2) .^ 2) + trace(S1 + S2 - 2 * sqrtm(S1 * S2));
            testCase.verifyEqual(OceanDL.Generative.SampleEvaluator.frechetDistance(P, Q), exact, 'RelTol', 0.02);
            % Singular covariances (fewer samples than features, dead feature):
            % finite, non-negative, no warnings.
            Fa = [randn(s, 20, 8); zeros(4, 8)]; Fb = [randn(s, 20, 8) + 1; zeros(4, 8)];
            d = testCase.verifyWarningFree(@() OceanDL.Generative.SampleEvaluator.frechetDistance(Fa, Fb));
            testCase.verifyTrue(isfinite(d) && d >= 0);
            Xtr = cat(4, zeros(2), ones(2));
            Xg = cat(4, 0.1 * ones(2), 0.8 * ones(2));
            testCase.verifyEqual(OceanDL.Generative.SampleEvaluator.nnDistance(Xg, Xtr), mean([0.2 0.4]), 'AbsTol', 1e-12);
        end

        function testCheckpointAndFit(testCase)
            dir_ = fullfile(tempname, 'ckpt');
            cleanup = onCleanup(@() rmdir(fileparts(dir_), 's'));
            calls = 0;
            function v = work(), calls = calls + 1; v = struct('A', 42); end
            a = OceanDL.Training.Checkpoint.run(dir_, "unit k=1", @work);
            b = OceanDL.Training.Checkpoint.run(dir_, "unit k=1", @work);
            testCase.verifyEqual([a.A b.A calls], [42 42 1]);                           % second call reused
            net = OceanDL.nn.Linear(3, 1, 'Seed', 1);
            s = RandStream('mt19937ar', 'Seed', 5); X = randn(s, 3, 200); Y = [1 -2 0.5] * X + 0.3;
            r = OceanDL.Training.fit(net, @(k) OceanDL.Autograd.F.mseLoss(net.forward(X), Y), 300, 'LR', 5e-2, ...
                'Record', @() net.Weight.Data(1));
            testCase.verifyLessThan(r.Curve(end), 1e-3 * r.Curve(1));
            testCase.verifySize(r.Records, [300 1]);
            testCase.verifyGreaterThan(r.StepSeconds, 0);
        end
    end
end
