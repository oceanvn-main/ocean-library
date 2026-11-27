classdef LatentDiffusion < OceanDL.nn.Module
    % LATENTDIFFUSION  Diffusion in an autoencoder's latent space (Rombach et
    %   al. 2022 -- Stable Diffusion; also SD3, FLUX, DiT, Sora, Veo):
    %
    %     z  = ScaleFactor * Enc(x)          frozen, pre-trained autoencoder
    %     loss = DiffusionModel(z, y)        only the denoiser learns
    %     generate: z ~ sampler, x = Dec(z / ScaleFactor)
    %
    %   The autoencoder removes imperceptible detail (e.g. 16x16x1 pixels ->
    %   8x8xC latents), so the expensive denoiser works on far fewer values.
    %   ScaleFactor (calibrate()) = 1 / std(latents) makes the latents unit
    %   variance -- what the noise schedules assume (SD's 0.18215 was measured
    %   the same way).
    %   Autoencoders: OceanDL.Generative.VAE (posterior mean, or a sample with
    %   'Sample' true) or OceanDL.Generative.VQVAE (quantised latents; decoded
    %   latents are re-quantised first, as in LDM-VQ). The autoencoder is held
    %   in a private property, so parameters() / optimisers never touch it.

    properties (SetAccess = protected)
        Diffusion
        ScaleFactor (1,1) double = 1
        LatentShape (1,:) double = []      % per-sample latent size, e.g. [8 8 4]
        Sample (1,1) logical = false
    end

    properties (Access = private)
        Autoencoder
    end

    methods
        function obj = LatentDiffusion(autoencoder, diffusion, opts)
            arguments
                autoencoder OceanDL.nn.Module
                diffusion OceanDL.Generative.Diffusion.DiffusionModel
                opts.ScaleFactor (1,1) double = NaN
                opts.Sample (1,1) logical = false
            end
            if ~isa(autoencoder, 'OceanDL.Generative.VAE') && ~isa(autoencoder, 'OceanDL.Generative.VQVAE')
                error('OceanDL:Generative:Diffusion:LatentDiffusion:Autoencoder', 'Expected a VAE or a VQVAE.');
            end
            obj.Autoencoder = autoencoder;
            obj.Diffusion = diffusion;
            obj.Sample = opts.Sample;
            if ~isnan(opts.ScaleFactor), obj.ScaleFactor = opts.ScaleFactor; end
            autoencoder.eval();
        end

        function s = calibrate(obj, X)
            % Set ScaleFactor = 1 / std of the latents of X (and LatentShape).
            Z = obj.encode(X, 1);
            s = 1 / std(double(gather(Z(:))));
            obj.ScaleFactor = s;
        end

        function Z = encode(obj, X, scale)
            % Scaled latents of images X (no graph).
            if nargin < 3, scale = obj.ScaleFactor; end
            ae = obj.Autoencoder;
            if isa(ae, 'OceanDL.Generative.VAE')
                q = OceanDL.Autograd.GradMode.noGrad(@() ae.encode(X));
                if obj.Sample, Z = q.rsample(); else, Z = q.mode(); end
            else
                Z = ae.Quantizer.lookup(ae.encodeIndices(X));
            end
            Z = scale * OceanDL.Autograd.Tensor.raw(Z);
            sz = size(Z, 1:4);
            obj.LatentShape = sz(1:3);
        end

        function X = decode(obj, Z)
            ae = obj.Autoencoder;
            Z = Z / obj.ScaleFactor;
            if isa(ae, 'OceanDL.Generative.VAE')
                X = ae.mean(OceanDL.Autograd.GradMode.noGrad(@() ae.decode(Z)));
            else
                was = ae.Quantizer.Training; ae.Quantizer.eval();
                cleanup = onCleanup(@() ae.Quantizer.train(was));
                X = OceanDL.Autograd.GradMode.noGrad(@() ae.Decoder.forward(ae.Quantizer.forward(Z)));
            end
            X = OceanDL.Autograd.Tensor.raw(X);
        end

        function L = forward(obj, X, labels)
            if nargin < 3, labels = []; end
            L = obj.Diffusion.forward(obj.encode(X), labels);
        end

        function [X, info] = generate(obj, n, sampler, opts)
            arguments
                obj
                n (1,1) double
                sampler OceanDL.Generative.Diffusion.Samplers.ISampler
                opts.Labels = []
                opts.Guidance (1,1) double = 1
                opts.Stream = RandStream.getGlobalStream()
            end
            if isempty(obj.LatentShape)
                error('OceanDL:Generative:Diffusion:LatentDiffusion:Shape', 'Call calibrate() or forward() first (unknown latent size).');
            end
            [Z, info] = obj.Diffusion.generate(n, obj.LatentShape, sampler, 'Labels', opts.Labels, ...
                'Guidance', opts.Guidance, 'Stream', opts.Stream);
            X = obj.decode(Z);
        end
    end
end
