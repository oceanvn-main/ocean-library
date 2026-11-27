classdef VAE < OceanDL.nn.Module
    % VAE  Variational autoencoder (Kingma & Welling 2014; Goodfellow Ch. 20;
    %   Szeliski 2nd ed. Ch. 5) around ANY encoder / decoder module pair.
    %
    %   Encoder: x -> 2C channels = [Mu, LogVar] of q(z|x) (DiagonalGaussian)
    %   Decoder: z -> parameters of p(x|z)
    %   Prior:   p(z) = N(0, I)
    %
    %   LOSS = negative ELBO per sample (nats), minimised:
    %     L = E_q[ -log p(x|z) ]  +  Beta * KL( q(z|x) || p(z) )
    %         (1 reparameterised sample)    (closed form, DiagonalGaussian)
    %   Beta = 1 is the true ELBO (<= log p(x)); Beta > 1 is beta-VAE (Higgins
    %   2017: more compression / disentanglement, blurrier); Beta is public so
    %   a caller can warm it up from 0 (KL annealing).
    %
    %   LIKELIHOOD ('Likelihood'):
    %     "gaussian"   -log p = 0.5 ((x - m)/sigma)^2 + log sigma + 0.5 log(2 pi)
    %                  per element. 'Sigma' fixed (sigma^2 plays the role of
    %                  1/Beta), or 'LearnSigma' true: one learned log sigma --
    %                  the calibrated decoder of sigma-VAE (Rybkin et al. 2021),
    %                  which removes the need to tune Beta by hand.
    %     "bernoulli"  decoder outputs logits, x in [0, 1]:
    %                  -log p = max(z,0) - z x + log(1 + exp(-|z|))
    %   FREE BITS ('FreeBits' lambda > 0, Kingma et al. 2016): the KL of each
    %   latent channel (batch mean, nats) counts as max(KL_c, lambda) -- the
    %   optimiser gains nothing by pushing a channel below lambda, which
    %   prevents POSTERIOR COLLAPSE (decoder ignoring z, KL_c -> 0).
    %
    %   After forward(), LastStats holds Recon, KL (nats/sample), KLPerChannel,
    %   NegELBO (= Recon + KL, the Beta = 1 bound) and BitsPerDim.

    properties
        Beta (1,1) double {mustBeNonnegative} = 1
    end

    properties (SetAccess = protected)
        Encoder
        Decoder
        LogSigma            % Parameter (learned) or double scalar (fixed)
        Likelihood (1,1) string
        FreeBits (1,1) double
        Layout (1,1) string
    end

    properties (SetAccess = private)
        Stream
        LastStats = struct()
    end

    methods
        function obj = VAE(encoder, decoder, opts)
            arguments
                encoder OceanDL.nn.Module
                decoder OceanDL.nn.Module
                opts.Likelihood (1,1) string {mustBeMember(opts.Likelihood, ["gaussian", "bernoulli"])} = "gaussian"
                opts.Sigma (1,1) double {mustBePositive} = 1
                opts.LearnSigma (1,1) logical = false
                opts.Beta (1,1) double {mustBeNonnegative} = 1
                opts.FreeBits (1,1) double {mustBeNonnegative} = 0
                opts.Layout (1,1) string {mustBeMember(opts.Layout, ["SSCB", "CB"])} = "SSCB"
                opts.Seed = []
            end
            obj.Encoder = encoder;
            obj.Decoder = decoder;
            obj.Likelihood = opts.Likelihood;
            if opts.LearnSigma
                obj.LogSigma = OceanDL.nn.Parameter(log(opts.Sigma));
            else
                obj.LogSigma = log(opts.Sigma);
            end
            obj.Beta = opts.Beta;
            obj.FreeBits = opts.FreeBits;
            obj.Layout = opts.Layout;
            if isempty(opts.Seed), obj.Stream = RandStream.getGlobalStream();
            else, obj.Stream = RandStream('mt19937ar', 'Seed', opts.Seed); end
        end

        function L = forward(obj, X)
            N = OceanDL.Generative.Latent.batchSize(X, obj.Layout);
            q = obj.encode(X);
            out = obj.Decoder.forward(q.rsample(obj.Stream));
            nll = obj.reconstructionNll(out, X) / N;                          % nats per sample
            klElem = q.klStandardNormal();
            d = OceanDL.Generative.Latent.channelDim(obj.Layout);
            others = setdiff(1:ndims(OceanDL.Autograd.Tensor.raw(klElem)) + 1, d);
            klc = reshape(sum(klElem, others), [], 1) / N;                    % per latent channel, nats per sample
            if obj.FreeBits > 0
                klTerm = sum(relu(klc - obj.FreeBits) + obj.FreeBits, 'all');  % sum_c max(KL_c, lambda)
            else
                klTerm = sum(klc, 'all');
            end
            L = nll + obj.Beta * klTerm;
            recon = double(gather(nll.Data)); kl = double(gather(sum(klc.Data)));
            dims = numel(OceanDL.Autograd.Tensor.raw(X)) / N;
            obj.LastStats = struct('Recon', recon, 'KL', kl, 'KLPerChannel', double(gather(klc.Data)), ...
                'NegELBO', recon + kl, 'BitsPerDim', (recon + kl) / (dims * log(2)));
        end

        function q = encode(obj, X)
            q = OceanDL.Generative.DiagonalGaussian.fromChannels(obj.Encoder.forward(X), obj.Layout);
        end

        function out = decode(obj, z)
            out = obj.Decoder.forward(z);
        end

        function m = mean(obj, out)
            % Mean of p(x|z) from decoder outputs (Gaussian means or Bernoulli logits).
            if obj.Likelihood == "bernoulli", m = sigmoid(OceanDL.Autograd.Tensor.wrap(out)); else, m = out; end
        end

        function Xr = reconstruct(obj, X)
            % Decoder mean at the posterior mean (no sampling), as a plain array.
            Xr = OceanDL.Autograd.GradMode.noGrad(@() obj.mean(obj.decode(obj.encode(X).mode())));
            Xr = OceanDL.Autograd.Tensor.raw(Xr);
        end

        function X = sample(obj, n, latentSize, stream)
            % Draw z ~ N(0, I) of size [latentSize n] and decode (mean of p(x|z)).
            if nargin < 4 || isempty(stream), stream = obj.Stream; end
            P = obj.parameters();
            z = cast(randn(stream, [latentSize(:).' n]), 'like', P{1}.Data);
            X = OceanDL.Autograd.GradMode.noGrad(@() obj.mean(obj.decode(z)));
            X = OceanDL.Autograd.Tensor.raw(X);
        end

        function nll = reconstructionNll(obj, out, X)
            % -log p(x|z) summed over all elements (Tensor).
            nll = OceanDL.Generative.VAE.nll(out, X, obj.Likelihood, obj.LogSigma);
        end
    end

    methods (Static)
        function nll = nll(out, X, likelihood, logSigma)
            % -log p(x | decoder output) summed over all elements (shared with
            % HierarchicalVAE): "bernoulli" logits or "gaussian" means with log sigma.
            out = OceanDL.Autograd.Tensor.wrap(out);
            if likelihood == "bernoulli"
                nll = sum(relu(out) - out .* X + log(1 + exp(-abs(out))), 'all');
                return;
            end
            n = numel(OceanDL.Autograd.Tensor.raw(X));
            nll = 0.5 * sum((out - X) .^ 2, 'all') .* exp(-2 * logSigma) + n * (logSigma + 0.5 * log(2 * pi));
        end

        function [n, variance] = activeUnits(vae, X, threshold)
            % ACTIVE UNITS (Burda et al. 2016): latent dimension u is active if
            % Var_x( E_q[z_u | x] ) > threshold (default 0.01) -- a collapsed
            % dimension has the same posterior mean for every input.
            if nargin < 3, threshold = 0.01; end
            mu = OceanDL.Autograd.Tensor.raw(OceanDL.Autograd.GradMode.noGrad(@() vae.encode(X).mode()));
            d = ndims(mu); if vae.Layout == "SSCB", d = 4; else, d = 2; end
            variance = var(double(gather(mu)), 0, d);
            n = nnz(variance > threshold);
        end
    end

    methods (Access = protected)
        function applyBuffers(obj, fn)
            if ~isa(obj.LogSigma, 'OceanDL.nn.Parameter'), obj.LogSigma = fn(obj.LogSigma); end
        end
    end
end
