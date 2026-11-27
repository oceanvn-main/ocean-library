classdef DiffusionCoreTest < OceanTest.BaseTestCase
    % DIFFUSIONCORETEST  The diffusion family without any trained network:
    %   * forward process: LinearVP's closed form == iterating the DDPM chain;
    %     endpoints and monotone logSNR for all four schedules; analytic
    %     derivatives == finite differences;
    %   * target conversions eps / x0 / v / velocity are exact inverses;
    %   * samplers driven by the EXACT denoiser of Gaussian(-mixture) data:
    %     - probability-flow ODE (DDIM eta 0, Euler, Heun) reaches the closed-
    %       form transport map; Heun converges at 2nd order, Euler at 1st;
    %     - DDIM eta 1 == DDPM ancestral (two independent code paths);
    %     - stochastic and ODE samplers reproduce mixture weights and variances.

    methods (Test)
        function testLinearVPClosedFormEqualsTheDDPMChain(testCase)
            % Book notation (Lai et al., Sec. 2.2.1): x_i = alpha_i x_{i-1} + beta_i eps_i,
            % alpha_i = sqrt(1 - beta_i^2); closed form x_i = alphaBar_i x_0 + sqrt(1 - alphaBar_i^2) eps (Eq. 2.2.1).
            sch = OceanDL.Generative.Diffusion.Schedules.LinearVP('L', 1000);
            beta = sch.beta();                                     % beta_i: step noise STD (book symbol)
            s = RandStream('mt19937ar', 'Seed', 1);
            x0 = randn(s, 1, 20000) * 2 + 3;
            x = x0; alphaBar = 1;
            for i = 1:400                                          % iterate the transition kernel
                alpha_i = sqrt(1 - beta(i) ^ 2);
                x = alpha_i * x + beta(i) * randn(s, size(x));
                alphaBar = alphaBar * alpha_i;
            end
            testCase.verifyEqual(sch.alphaBar(400), alphaBar, 'RelTol', 1e-12);
            testCase.verifyEqual(sch.alpha(400 / 1000), alphaBar, 'RelTol', 1e-12);          % continuous t = i / L
            testCase.verifyEqual(sch.alphaStep(400), sqrt(1 - sch.betaSq(400)), 'RelTol', 1e-15);
            testCase.verifyEqual(sch.betaSq([1 1000]), [1e-4 0.02], 'AbsTol', 1e-15);      % Ho's linear variances
            r = x - sch.alpha(0.4) * x0;                            % remaining noise
            testCase.verifyEqual(std(r), sch.sigma(0.4), 'RelTol', 0.02);
            testCase.verifyEqual(sch.alpha(0.4) ^ 2 + sch.sigma(0.4) ^ 2, 1, 'AbsTol', 1e-12);
        end

        function testDdpmReverseStepIsBookEquation224(testCase)
            % One DDPMAncestral step (i -> i-1) on LinearVP equals Lemma 2.2.2 /
            % Eq. (2.2.4) of Lai et al., written with the schedule's DISCRETE
            % book-notation accessors (independent of the sampler's code path):
            %   mu = alphaBar_{i-1} beta_i^2 / (1 - alphaBar_i^2) x_phi
            %      + (1 - alphaBar_{i-1}^2) alpha_i / (1 - alphaBar_i^2) x_i
            %   sigma^2(i) = (1 - alphaBar_{i-1}^2) / (1 - alphaBar_i^2) beta_i^2
            import OceanDL.Generative.Diffusion.*
            sch = Schedules.LinearVP();
            L = sch.L;
            s = RandStream('mt19937ar', 'Seed', 14);
            x_i = randn(s, 3, 4); x_phi = randn(s, 3, 4);
            predict = @(x, t) OceanTest.OceanDL.Generative.DiffusionCoreTest.fixedEstimate(x_phi, x);   % fixed clean estimate
            smp = Samplers.DDPMAncestral('Steps', L);
            for i = [2 50 500 1000]
                ab_i = sch.alphaBar(i); ab_im1 = sch.alphaBar(i - 1);
                a_i = sch.alphaStep(i); b2_i = sch.betaSq(i);
                mu = ab_im1 * b2_i / (1 - ab_i ^ 2) * x_phi + (1 - ab_im1 ^ 2) * a_i / (1 - ab_i ^ 2) * x_i;
                var_i = (1 - ab_im1 ^ 2) / (1 - ab_i ^ 2) * b2_i;
                eps_i = randn(RandStream('mt19937ar', 'Seed', i), size(x_i));
                x_im1 = smp.step(predict, sch, x_i, i / L, (i - 1) / L, RandStream('mt19937ar', 'Seed', i));
                testCase.verifyEqual(x_im1, mu + sqrt(var_i) * eps_i, 'AbsTol', 1e-12, sprintf('step %d', i));
                testCase.verifyEqual(a_i ^ 2 + b2_i, 1, 'AbsTol', 1e-15);                   % alpha_i = sqrt(1 - beta_i^2)
            end
            % i = 1: alphaBar_0 = 1, sigma^2(1) = 0 -> the clean estimate, no noise.
            testCase.verifyEqual(smp.step(predict, sch, x_i, 1 / L, 0, s), x_phi);
        end

        function testScheduleEndpointsMonotonicityAndDerivatives(testCase)
            import OceanDL.Generative.Diffusion.Schedules.*
            schedules = {LinearVP(), CosineVP(), EDM(), RectifiedFlow(), RectifiedFlow('Shift', 3), ...
                CosineVP('Shift', 0.5)};
            t = linspace(0.02, 0.98, 49) + 3.7e-4;          % off LinearVP's grid kinks (piecewise-linear log abar)
            h = 1e-6;
            for k = 1:numel(schedules)
                sc = schedules{k}; name = class(sc);
                testCase.verifyTrue(all(diff(sc.logSNR(t)) < 0), name);
                testCase.verifyEqual(sc.dalpha(t), (sc.alpha(t + h) - sc.alpha(t - h)) / (2 * h), 'RelTol', 1e-4, 'AbsTol', 1e-7, name);
                testCase.verifyEqual(sc.dsigma(t), (sc.sigma(t + h) - sc.sigma(t - h)) / (2 * h), 'RelTol', 1e-4, 'AbsTol', 1e-7, name);
            end
            flow = RectifiedFlow();
            testCase.verifyEqual([flow.alpha(0) flow.sigma(0) flow.alpha(1) flow.sigma(1)], [1 0 0 1]);
            edm = EDM();
            testCase.verifyEqual([edm.sigma(0) edm.sigma(1)], [0.002 80], 'RelTol', 1e-12);
            testCase.verifyEqual(edm.tFromSigma(edm.sigma(0.37)), 0.37, 'AbsTol', 1e-12);
            cos = CosineVP();
            testCase.verifyEqual([cos.logSNR(0) cos.logSNR(1)], [15 -15], 'AbsTol', 1e-9);
        end

        function testTargetConversionsAreExactInverses(testCase)
            T = OceanDL.Generative.Diffusion.Targets;
            import OceanDL.Generative.Diffusion.Schedules.*
            s = RandStream('mt19937ar', 'Seed', 2);
            x0 = randn(s, 3, 5); e = randn(s, 3, 5);
            for sc = {LinearVP(), CosineVP(), EDM(), RectifiedFlow('Shift', 2)}
                for t = [0.1 0.5 0.9]
                    a = sc{1}.alpha(t); sg = sc{1}.sigma(t); da = sc{1}.dalpha(t); ds = sc{1}.dsigma(t);
                    x = a * x0 + sg * e;
                    for kind = ["eps", "x0", "v", "velocity"]
                        p = T.fromX0Eps(kind, x0, e, a, sg, da, ds);
                        [x0r, er] = T.toX0Eps(kind, p, x, a, sg, da, ds);
                        testCase.verifyEqual(x0r, x0, 'AbsTol', 1e-9, kind);
                        testCase.verifyEqual(er, e, 'AbsTol', 1e-9, kind);
                    end
                    testCase.verifyEqual(T.fromX0Eps("velocity", x0, e, a, sg, da, ds), da * x0 + ds * e);  % = dx/dt
                end
            end
        end

        function testOdeSamplersReachTheExactTransportMap(testCase)
            % Gaussian data N(mu, s0^2): every deterministic sampler must land
            % on the closed-form flow map of its own starting noise.
            import OceanDL.Generative.Diffusion.Schedules.* OceanDL.Generative.Diffusion.Samplers.*
            A = OceanDL.Generative.Diffusion.Analytic;
            mu = [2; -1]; s0 = 0.5;
            for sc = {CosineVP(), RectifiedFlow(), EDM('SigmaData', 0.5)}
                sch = sc{1};
                pred = A.gaussian(mu, s0, sch);
                X1 = sch.priorScale() * randn(RandStream('mt19937ar', 'Seed', 3), 2, 50);
                exact = A.gaussianFlowMap(X1, mu, s0, sch);
                heun = HeunODE('Steps', 200).sample(pred, sch, [2 50], 'X1', X1);
                testCase.verifyEqual(heun, exact, 'AbsTol', 2e-3, class(sch));          % measured 4e-5 .. 1.4e-3
                % DDIM is a first-order solver: the error halves when the steps double.
                e200 = max(abs(DDIM('Steps', 200).sample(pred, sch, [2 50], 'X1', X1) - exact), [], 'all');
                e400 = max(abs(DDIM('Steps', 400).sample(pred, sch, [2 50], 'X1', X1) - exact), [], 'all');
                testCase.verifyLessThan(e400, 1e-2, class(sch));
                testCase.verifyEqual(e200 / e400, 2, 'RelTol', 0.1, class(sch));
            end
        end

        function testHeunIsSecondOrderEulerFirstOrder(testCase)
            import OceanDL.Generative.Diffusion.Schedules.* OceanDL.Generative.Diffusion.Samplers.*
            A = OceanDL.Generative.Diffusion.Analytic;
            sch = RectifiedFlow();
            % Two separated components: the flow bends, so 1-step Euler is not exact.
            pred = A.gmm([-2 2], [0.3 0.3], [0.5 0.5], sch);
            X1 = randn(RandStream('mt19937ar', 'Seed', 4), 1, 200);
            ref = HeunODE('Steps', 2000).sample(pred, sch, [1 200], 'X1', X1);
            err = @(smp) max(abs(smp.sample(pred, sch, [1 200], 'X1', X1) - ref));
            rE = err(EulerODE('Steps', 40)) / err(EulerODE('Steps', 80));
            rH = err(HeunODE('Steps', 40)) / err(HeunODE('Steps', 80));
            testCase.verifyEqual(rE, 2, 'RelTol', 0.25);            % halving h halves the error
            testCase.verifyEqual(rH, 4, 'RelTol', 0.35);            % ... quarters it
        end

        function testDdimEtaOneEqualsDdpmAncestral(testCase)
            import OceanDL.Generative.Diffusion.Schedules.* OceanDL.Generative.Diffusion.Samplers.*
            sch = CosineVP();
            pred = OceanDL.Generative.Diffusion.Analytic.gmm([-1 3], [0.4 0.2], [0.3 0.7], sch);
            a = DDPMAncestral('Steps', 30).sample(pred, sch, [1 500], 'Stream', RandStream('mt19937ar', 'Seed', 5));
            b = DDIM('Steps', 30, 'Eta', 1).sample(pred, sch, [1 500], 'Stream', RandStream('mt19937ar', 'Seed', 5));
            testCase.verifyEqual(a, b, 'AbsTol', 1e-10);
        end

        function testSamplersReproduceAMixture(testCase)
            import OceanDL.Generative.Diffusion.Schedules.* OceanDL.Generative.Diffusion.Samplers.*
            A = OceanDL.Generative.Diffusion.Analytic;
            means = [-3 0 3; 0 3 0]; stds = [0.3 0.5 0.3]; w = [0.2 0.3 0.5];
            cases = {LinearVP(),       DDPMAncestral('Steps', 1000)
                     CosineVP(),       DDIM('Steps', 50)
                     EDM(),            HeunODE('Steps', 50)         % 18 (Karras) is too few for these narrow modes: std 0.37
                     RectifiedFlow(),  EulerODE('Steps', 50)};
            M = 4000;
            for k = 1:size(cases, 1)
                [sch, smp] = cases{k, :};
                [X, info] = smp.sample(A.gmm(means, stds, w, sch), sch, [2 M], 'Stream', RandStream('mt19937ar', 'Seed', 6));
                [~, c] = min(sum(X .^ 2, 1) - 2 * means.' * X + sum(means .^ 2, 1).', [], 1);
                name = sprintf('%s + %s', class(sch), class(smp));
                testCase.verifyEqual(histcounts(c, 0.5:3.5) / M, w, 'AbsTol', 0.03, name);
                for j = 1:3
                    testCase.verifyEqual(std(X(:, c == j), 0, 2), stds(j) * [1; 1], 'RelTol', 0.12, name);
                end
                testCase.verifyGreaterThan(info.NFE, 0);
            end
        end
    end
    methods (Static)
        function [x0, eps] = fixedEstimate(x_phi, x)
            % A predictor that always returns the same clean estimate (1 or 2 outputs).
            x0 = x_phi;
            eps = zeros(size(x));
        end
    end
end
