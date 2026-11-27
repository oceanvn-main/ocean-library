classdef VQVAE < OceanDL.nn.Module
    % VQVAE  Encoder -> discrete bottleneck (any IQuantizer) -> decoder
    %   (van den Oord et al. 2017; FSQ variant: Mentzer et al. 2023).
    %
    %   LOSS = mean((decoder(z_q) - x).^2) + quantizer auxiliary loss
    %   (the reconstruction term is the Gaussian log-likelihood up to
    %   constants; the prior over codes is uniform while training the
    %   autoencoder, so the KL term is the constant log K and is dropped).
    %
    %   Generation is two-stage: train the VQ-VAE, then fit a prior over the
    %   code grids (CodePrior) and decode its samples with decodeIndices().

    properties (SetAccess = protected)
        Encoder
        Quantizer
        Decoder
    end

    properties (SetAccess = private)
        LastStats = struct()
    end

    methods
        function obj = VQVAE(encoder, quantizer, decoder)
            arguments
                encoder OceanDL.nn.Module
                quantizer OceanDL.Generative.Quantizers.IQuantizer
                decoder OceanDL.nn.Module
            end
            obj.Encoder = encoder;
            obj.Quantizer = quantizer;
            obj.Decoder = decoder;
        end

        function L = forward(obj, X)
            [zq, info] = obj.Quantizer.forward(obj.Encoder.forward(X));
            recon = OceanDL.Autograd.F.mseLoss(obj.Decoder.forward(zq), X);
            L = recon + info.Loss;
            obj.LastStats = struct('Recon', double(gather(recon.Data)), ...
                'QuantLoss', double(gather(OceanDL.Autograd.Tensor.raw(info.Loss))), ...
                'Perplexity', info.Perplexity, 'Usage', info.Usage, 'Restarted', info.Restarted);
        end

        function idx = encodeIndices(obj, X)
            % Code grid of each image (no codebook updates, no graph).
            was = obj.Quantizer.Training;
            obj.Quantizer.eval();
            cleanup = onCleanup(@() obj.Quantizer.train(was));
            [~, info] = OceanDL.Autograd.GradMode.noGrad(@() obj.Quantizer.forward(obj.Encoder.forward(X)));
            idx = info.Indices;
        end

        function X = decodeIndices(obj, idx)
            X = OceanDL.Autograd.GradMode.noGrad(@() obj.Decoder.forward(obj.Quantizer.lookup(idx)));
            X = OceanDL.Autograd.Tensor.raw(X);
        end

        function Xr = reconstruct(obj, X)
            Xr = obj.decodeIndices(obj.encodeIndices(X));
        end
    end
end
