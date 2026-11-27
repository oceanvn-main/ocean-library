classdef DiffusionInspectTest < OceanTest.BaseTestCase
    % DIFFUSIONINSPECTTEST  Tools for looking inside the diffusion process:
    %   * EmpiricalDenoiser == Analytic.gmm on the same components (with and
    %     without smoothing); limits: near t = 0 it returns the nearest data
    %     image, at pure noise (flow, t = 1) the data mean with all K images
    %     "in the running"; image-shaped inputs work;
    %   * TrajectoryRecorder records every network call without changing the
    %     sampler's result;
    %   * samplers' 'TStart': starting part-way from the exact intermediate
    %     state reproduces the full run.

    methods (Test)
        function testEmpiricalDenoiserEqualsGaussianMixture(testCase)
            import OceanDL.Generative.Diffusion.*
            s = RandStream('mt19937ar', 'Seed', 1);
            mu = randn(s, 5, 7); x = randn(s, 5, 3);
            for sch = {Schedules.RectifiedFlow(), Schedules.CosineVP()}
                for sm = [0 0.3]
                    den = EmpiricalDenoiser(mu, sch{1}, 'SmoothStd', sm);
                    ref = Analytic.gmm(mu, max(sm, 1e-12) * ones(1, 7), ones(1, 7) / 7, sch{1});
                    [a0, ae] = den.predict(x, 0.4);
                    [b0, be] = ref(x, 0.4);
                    testCase.verifyEqual(a0, b0, 'AbsTol', 1e-9);
                    testCase.verifyEqual(ae, be, 'AbsTol', 1e-9);
                end
            end
        end

        function testEmpiricalDenoiserLimits(testCase)
            import OceanDL.Generative.Diffusion.*
            s = RandStream('mt19937ar', 'Seed', 2);
            mu = randn(s, 6, 20);
            den = EmpiricalDenoiser(mu, Schedules.RectifiedFlow());
            x = mu(:, 7) + 0.01 * randn(s, 6, 1);
            x0 = den.predict(x, 1e-3);                                   % almost clean input
            testCase.verifyEqual(x0, mu(:, 7), 'AbsTol', 1e-6);
            testCase.verifyEqual(den.LastTopIndex, 7);
            testCase.verifyEqual(den.LastEffectiveN, 1, 'AbsTol', 1e-6);
            x0 = den.predict(randn(s, 6, 1), 1);                         % pure noise: alpha = 0
            testCase.verifyEqual(x0, mean(mu, 2), 'AbsTol', 1e-12);
            testCase.verifyEqual(den.LastEffectiveN, 20, 'AbsTol', 1e-9);
            imgs = reshape(mu, 2, 3, 1, 20);                              % image layout in, image layout out
            deni = EmpiricalDenoiser(imgs, Schedules.RectifiedFlow());
            testCase.verifySize(deni.predict(randn(s, 2, 3, 1, 4), 0.5), [2 3 1 4]);
        end

        function testRecorderAndPartialStart(testCase)
            import OceanDL.Generative.Diffusion.*
            sch = Schedules.RectifiedFlow();
            pred = Analytic.gaussian([1; -1], 0.5, sch);
            X1 = randn(RandStream('mt19937ar', 'Seed', 3), 2, 5);
            smp = Samplers.EulerODE('Steps', 10);
            plain = smp.sample(pred, sch, [2 5], 'X1', X1);
            rec = TrajectoryRecorder(pred);
            viaRec = smp.sample(rec.handle(), sch, [2 5], 'X1', X1);
            testCase.verifyEqual(viaRec, plain);                          % recording changes nothing
            testCase.verifyEqual(rec.numCalls(), 10);
            testCase.verifyEqual(rec.T, linspace(1, 0.1, 10), 'AbsTol', 1e-12);
            testCase.verifyEqual(rec.X{1}, X1);
            % Start at t = 0.5 from the recorded state of the full run: same end point.
            k = find(abs(rec.T - 0.5) < 1e-12);
            half = Samplers.EulerODE('Steps', 5).sample(pred, sch, [2 5], 'X1', rec.X{k}, 'TStart', 0.5);
            testCase.verifyEqual(half, plain, 'AbsTol', 1e-12);
        end
    end
end
