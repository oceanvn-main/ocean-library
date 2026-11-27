classdef DiagonalGaussian
    % DIAGONALGAUSSIAN  q(z|x) = N(Mu, diag(exp(LogVar))) on Tensors -- the
    %   VAE posterior (Kingma & Welling 2014; Goodfellow Ch. 20).
    %
    %   rsample   z = Mu + exp(LogVar/2) .* eps,  eps ~ N(0, I)
    %             REPARAMETERISATION: the noise is an input, so dz/dMu = 1 and
    %             dz/dsigma = eps -- gradients flow through the sample.
    %   klStandardNormal  KL(q || N(0, I)) per element, closed form:
    %             0.5 * (Mu.^2 + exp(LogVar) - 1 - LogVar)      (nats)
    %   logProb   log q(z) per element:
    %             -0.5 * (log(2 pi) + LogVar + (z - Mu).^2 ./ exp(LogVar))
    %   klTo(p)   KL(q || p) to another diagonal Gaussian, per element (nats):
    %             0.5 * (lv_p - lv_q + (exp(lv_q) + (mu_q - mu_p).^2) ./ exp(lv_p) - 1)
    %             -- the per-layer term of hierarchical VAEs, whose priors
    %             p(z_l | z_>l) are themselves predicted Gaussians.

    properties (SetAccess = private)
        Mu
        LogVar
    end

    methods
        function obj = DiagonalGaussian(mu, logVar)
            obj.Mu = OceanDL.Autograd.Tensor.wrap(mu);
            obj.LogVar = OceanDL.Autograd.Tensor.wrap(logVar);
        end

        function z = rsample(obj, stream)
            if nargin < 2 || isempty(stream), stream = RandStream.getGlobalStream(); end
            m = OceanDL.Autograd.Tensor.raw(obj.Mu);
            e = cast(randn(stream, size(m)), 'like', m);
            z = obj.Mu + exp(0.5 * obj.LogVar) .* e;
        end

        function z = mode(obj)
            z = obj.Mu;
        end

        function kl = klStandardNormal(obj)
            kl = 0.5 * (obj.Mu .^ 2 + exp(obj.LogVar) - 1 - obj.LogVar);
        end

        function kl = klTo(obj, p)
            kl = 0.5 * (p.LogVar - obj.LogVar + (exp(obj.LogVar) + (obj.Mu - p.Mu) .^ 2) ./ exp(p.LogVar) - 1);
        end

        function lp = logProb(obj, z)
            lp = -0.5 * (log(2 * pi) + obj.LogVar + (z - obj.Mu) .^ 2 ./ exp(obj.LogVar));
        end
    end

    methods (Static)
        function q = fromChannels(h, layout)
            % Split the encoder output along the channel axis: first half Mu,
            % second half LogVar (AutoencoderKL "double_z" convention).
            d = OceanDL.Generative.Latent.channelDim(layout);
            C = size(h, d);
            if mod(C, 2)
                error('OceanDL:Generative:DiagonalGaussian:Channels', ...
                    'Encoder must output 2 x latent channels (mean and log-variance); got %d.', C);
            end
            c = C / 2;
            if layout == "SSCB"
                q = OceanDL.Generative.DiagonalGaussian(h(:, :, 1:c, :), h(:, :, c+1:end, :));
            else
                q = OceanDL.Generative.DiagonalGaussian(h(1:c, :), h(c+1:end, :));
            end
        end
    end
end
