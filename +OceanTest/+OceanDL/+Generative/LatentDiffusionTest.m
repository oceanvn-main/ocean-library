classdef LatentDiffusionTest < OceanTest.BaseTestCase
    % LATENTDIFFUSIONTEST  Diffusion in autoencoder latent space:
    %   calibration gives unit-variance latents; training updates ONLY the
    %   denoiser (the autoencoder's weights are untouched and never exposed
    %   to the optimiser); generation returns images of the pixel shape;
    %   decode(encode(x)) == the autoencoder's own reconstruction; works with
    %   a VAE (continuous) and a VQ-VAE (re-quantised) autoencoder.

    methods (Test)
        function testLatentDiffusionWithAVae(testCase)
            import OceanDL.Generative.Diffusion.*
            X = OceanTest.OceanDL.Generative.VQVAETest.digits();
            X = X(:, :, :, 1:256);
            enc = OceanDL.Vision.ConvEncoder('InChannels', 1, 'Widths', [8 16], 'OutChannels', 4, 'Groups', 4, ...
                'MidAttention', false, 'Seed', 1);
            dec = OceanDL.Vision.ConvDecoder('LatentChannels', 2, 'Widths', [8 16], 'OutChannels', 1, 'Groups', 4, ...
                'MidAttention', false, 'Seed', 2);
            vae = OceanDL.Generative.VAE(enc, dec, 'Seed', 3);
            before = vae.stateDict();
            net = OceanDL.Vision.UNet2D('InChannels', 2, 'Widths', [16 16], 'Groups', 8, 'Seed', 4);
            ldm = LatentDiffusion(vae, DiffusionModel(net, Schedules.RectifiedFlow(), 'Target', "velocity", 'Seed', 5));
            s = ldm.calibrate(X);
            testCase.verifyEqual(ldm.LatentShape, [8 8 2]);
            Z = ldm.encode(X);
            testCase.verifyEqual(std(Z(:)), 1, 'RelTol', 1e-10);                          % unit variance after scaling
            testCase.verifyEqual(Z, s * vae.encode(X).mode().Data, 'AbsTol', 1e-12);
            testCase.verifyEqual(numel(ldm.parameters()), numel(net.parameters()));        % autoencoder hidden from optimisers
            opt = OceanDL.optim.ModuleOptimizer(ldm.parameters(), @() OceanStats.Optim.AdamWOptimizer('LR', 2e-3, 'WeightDecay', 0));
            rs = RandStream('mt19937ar', 'Seed', 6); losses = zeros(1, 60);
            for it = 1:60
                idx = randi(rs, 256, 1, 32);
                opt.zeroGrad(); L = ldm.forward(X(:, :, :, idx)); L.backward(); opt.step();
                losses(it) = L.Data;
            end
            testCase.verifyLessThan(mean(losses(end-9:end)), mean(losses(1:10)));
            after = vae.stateDict();
            for k = keys(before)
                testCase.verifyEqual(after(k{1}), before(k{1}), k{1});                    % frozen
            end
            [G, info] = ldm.generate(3, OceanDL.Generative.Diffusion.Samplers.EulerODE('Steps', 4), ...
                'Stream', RandStream('mt19937ar', 'Seed', 7));
            testCase.verifySize(G, [16 16 1 3]);
            testCase.verifyEqual(info.NFE, 4);
            testCase.verifyEqual(ldm.decode(ldm.encode(X(:, :, :, 1:4))), vae.reconstruct(X(:, :, :, 1:4)), 'AbsTol', 1e-10);
        end

        function testLatentDiffusionWithAVqVae(testCase)
            import OceanDL.Generative.Diffusion.*
            X = OceanTest.OceanDL.Generative.VQVAETest.digits();
            X = X(:, :, :, 1:128);
            enc = OceanDL.Vision.ConvEncoder('InChannels', 1, 'Widths', [8 16], 'OutChannels', 3, 'Groups', 4, ...
                'MidAttention', false, 'Seed', 1);
            dec = OceanDL.Vision.ConvDecoder('LatentChannels', 3, 'Widths', [8 16], 'OutChannels', 1, 'Groups', 4, ...
                'MidAttention', false, 'Seed', 2);
            vq = OceanDL.Generative.VQVAE(enc, OceanDL.Generative.Quantizers.FSQ([5 5 5]), dec);
            net = OceanDL.Vision.DiT('InChannels', 3, 'PatchSize', 2, 'Dim', 16, 'Depth', 1, 'Heads', 2, 'Seed', 3);
            ldm = LatentDiffusion(vq, DiffusionModel(net, Schedules.CosineVP(), 'Target', "v", 'Seed', 4));
            ldm.calibrate(X);
            testCase.verifyEqual(ldm.LatentShape, [8 8 3]);
            L = ldm.forward(X(:, :, :, 1:16)); L.backward();
            testCase.verifyTrue(isfinite(L.Data));
            testCase.verifyEqual(ldm.decode(ldm.encode(X(:, :, :, 1:4))), vq.reconstruct(X(:, :, :, 1:4)), 'AbsTol', 1e-10);
            G = ldm.generate(2, OceanDL.Generative.Diffusion.Samplers.DDIM('Steps', 3));
            testCase.verifySize(G, [16 16 1 2]);
            testCase.verifyError(@() LatentDiffusion(net, DiffusionModel(net, Schedules.CosineVP())), ...
                'OceanDL:Generative:Diffusion:LatentDiffusion:Autoencoder');
        end
    end
end
