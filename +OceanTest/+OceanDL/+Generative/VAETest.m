classdef VAETest < OceanTest.BaseTestCase
    % VAETEST  VAE correctness against exact results:
    %   * closed-form KL(q || N(0,I)) == Monte-Carlo E_q[log q - log p];
    %   * linear-Gaussian model (probabilistic PCA): with the TRUE posterior
    %     as encoder the negative ELBO equals -log p(x) exactly (Gaussian
    %     marginal), and a wrong encoder is worse by exactly KL(q || posterior);
    %   * free bits: channels below lambda contribute no KL gradient;
    %   * Bernoulli likelihood formula, sampling / reconstruction shapes;
    %   * training on USPS digits improves the ELBO and uses several latents;
    %   * a conv (SSCB) VAE runs in single precision after Module.to.

    methods (Test)
        function testKLClosedFormMatchesMonteCarlo(testCase)
            s = RandStream('mt19937ar', 'Seed', 1);
            mu = randn(s, 3, 2); lv = 0.5 * randn(s, 3, 2);
            q = OceanDL.Generative.DiagonalGaussian(mu, lv);
            kl = q.klStandardNormal().Data;
            S = 4e5;
            z = mu + exp(lv / 2) .* randn(s, 3, 2, S);
            logq = q.logProb(z).Data;
            logp = -0.5 * (log(2 * pi) + z .^ 2);
            testCase.verifyEqual(mean(logq - logp, 3), kl, 'AbsTol', 0.01);
            testCase.verifyTrue(all(kl >= 0, 'all'));
        end

        function testLinearGaussianElboIsExactWithTruePosterior(testCase)
            [vae, W, b, sigma] = OceanTest.OceanDL.Generative.VAETest.ppca(0);
            s = RandStream('mt19937ar', 'Seed', 2);
            xs = randn(s, 4, 3) * 2;
            C = W * W.' + sigma^2 * eye(4);
            for k = 1:3
                x = xs(:, k);
                logp = -0.5 * (4 * log(2 * pi) + log(det(C)) + (x - b).' / C * (x - b));
                L = vae.forward(repmat(x, 1, 20000));
                testCase.verifyEqual(L.Data, -logp, 'RelTol', 5e-3);                   % ELBO == log p(x)
            end
            % A wrong encoder (posterior mean shifted by delta) loses exactly
            % KL(q || posterior) = 0.5 * sum(delta.^2 ./ posteriorVariance).
            delta = 0.3;
            [vae2, ~, ~, ~, postVar] = OceanTest.OceanDL.Generative.VAETest.ppca(delta);
            x = xs(:, 1);
            logp = -0.5 * (4 * log(2 * pi) + log(det(C)) + (x - b).' / C * (x - b));
            L2 = vae2.forward(repmat(x, 1, 20000));
            testCase.verifyEqual(L2.Data + logp, 0.5 * sum(delta ^ 2 ./ postVar), 'RelTol', 0.05);
        end

        function testFreeBitsRemoveKlGradientBelowLambda(testCase)
            mk = @(fb, beta) OceanDL.Generative.VAE(OceanDL.nn.Linear(5, 4, 'Seed', 3), OceanDL.nn.Linear(2, 5, 'Seed', 4), ...
                'Layout', "CB", 'FreeBits', fb, 'Beta', beta, 'Seed', 5);
            X = randn(RandStream('mt19937ar', 'Seed', 6), 5, 7);
            a = mk(100, 1); c = mk(0, 0);
            La = a.forward(X); Lc = c.forward(X);
            testCase.verifyEqual(La.Data - Lc.Data, 2 * 100, 'AbsTol', 1e-9);        % sum_c max(KL_c, 100) = 2 * 100
            La.backward(); Lc.backward();
            testCase.verifyEqual(a.Encoder.Weight.Grad, c.Encoder.Weight.Grad, 'AbsTol', 1e-12);
            st = a.LastStats;
            testCase.verifyEqual(st.NegELBO, st.Recon + st.KL, 'AbsTol', 1e-12);
            testCase.verifyEqual(sum(st.KLPerChannel), st.KL, 'AbsTol', 1e-12);
        end

        function testBernoulliLikelihoodSamplingAndShapes(testCase)
            vae = OceanDL.Generative.VAE(OceanDL.nn.Linear(6, 4, 'Seed', 1), OceanDL.nn.Linear(2, 6, 'Seed', 2), ...
                'Layout', "CB", 'Likelihood', "bernoulli", 'Seed', 3);
            X = double(rand(RandStream('mt19937ar', 'Seed', 4), 6, 5) > 0.5);
            logits = randn(6, 5);
            nll = vae.reconstructionNll(logits, X).Data;
            p = 1 ./ (1 + exp(-logits));
            testCase.verifyEqual(nll, -sum(X .* log(p) + (1 - X) .* log(1 - p), 'all'), 'RelTol', 1e-12);
            S = vae.sample(9, 2);
            testCase.verifySize(S, [6 9]);
            testCase.verifyTrue(all(S >= 0 & S <= 1, 'all'));
            testCase.verifySize(vae.reconstruct(X), [6 5]);
        end

        function testTrainingOnDigitsImprovesElboAndUsesLatents(testCase)
            import OceanDL.Layout.DimFormat
            [rows, ~, fmt] = OceanData.Adapters.UspsDigitsReader.read("test");
            X4 = DimFormat.toCanonical(DimFormat.fromRowMajor(reshape(rows.', 1, []), fmt.Shape), fmt.Format);
            X = (reshape(X4, 256, []) + 1) / 2;                                     % CB, pixel intensities in [0, 1]
            enc = OceanDL.nn.Sequential(OceanDL.nn.Linear(256, 128, 'Seed', 1), OceanDL.nn.Activation("relu"), ...
                OceanDL.nn.Linear(128, 16, 'Seed', 2));
            dec = OceanDL.nn.Sequential(OceanDL.nn.Linear(8, 128, 'Seed', 3), OceanDL.nn.Activation("relu"), ...
                OceanDL.nn.Linear(128, 256, 'Seed', 4));
            vae = OceanDL.Generative.VAE(enc, dec, 'Layout', "CB", 'Likelihood', "bernoulli", 'Seed', 5);
            opt = OceanDL.optim.ModuleOptimizer(vae.parameters(), @() OceanStats.Optim.AdamWOptimizer('LR', 2e-3, 'WeightDecay', 0));
            rs = RandStream('mt19937ar', 'Seed', 6);
            neg = zeros(1, 400);
            for it = 1:400
                idx = randi(rs, size(X, 2), 1, 64);
                opt.zeroGrad(); L = vae.forward(X(:, idx)); L.backward(); opt.step();
                neg(it) = vae.LastStats.NegELBO;
            end
            testCase.verifyLessThan(mean(neg(end-19:end)), 0.75 * mean(neg(1:20)));
            testCase.verifyGreaterThanOrEqual(OceanDL.Generative.VAE.activeUnits(vae, X), 3);
        end

        function testConvVaeInSinglePrecision(testCase)
            enc = OceanDL.Vision.ConvEncoder('InChannels', 1, 'Widths', [8 16], 'OutChannels', 4, 'Groups', 4, 'Seed', 1);
            dec = OceanDL.Vision.ConvDecoder('LatentChannels', 2, 'Widths', [8 16], 'OutChannels', 1, 'Groups', 4, 'Seed', 2);
            vae = OceanDL.Generative.VAE(enc, dec, 'LearnSigma', true, 'Seed', 3).to('DataType', "single");
            L = vae.forward(rand(8, 8, 1, 3, 'single'));
            testCase.verifyClass(L.Data, 'single');
            L.backward();
            testCase.verifyClass(vae.LogSigma.Grad, 'single');
            testCase.verifySize(vae.sample(2, [4 4 2]), [8 8 1 2]);
            testCase.verifyTrue(isfinite(vae.LastStats.BitsPerDim));
        end
    end

    methods (Static)
        function [vae, W, b, sigma, postVar] = ppca(delta)
            % Linear-Gaussian VAE: decoder x = W z + b, noise sigma; encoder set to
            % the exact posterior N(M^-1 W'(x - b), sigma^2 M^-1), M = W'W + sigma^2 I
            % (diagonal because W has orthogonal columns), mean shifted by delta.
            W = [1 0; 0 2; 1 0; 0 -1] .* [1.5 0.5];                                % orthogonal columns
            b = [0.5; -1; 2; 0]; sigma = 0.7;
            M = W.' * W + sigma^2 * eye(2);
            postVar = diag(sigma^2 * inv(M));
            enc = OceanDL.nn.Linear(4, 4); dec = OceanDL.nn.Linear(2, 4);
            A = M \ W.';
            enc.Weight.Data = [A; zeros(2, 4)];
            enc.Bias.Data = [-A * b + delta; log(postVar)];
            dec.Weight.Data = W; dec.Bias.Data = b;
            vae = OceanDL.Generative.VAE(enc, dec, 'Layout', "CB", 'Sigma', sigma, 'Seed', 7);
        end
    end
end
