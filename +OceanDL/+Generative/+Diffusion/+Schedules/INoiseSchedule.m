classdef (Abstract) INoiseSchedule
    % INOISESCHEDULE  One formulation for the whole diffusion family:
    %
    %     x_t = alpha(t) * x0 + sigma(t) * eps,    eps ~ N(0, I),   t in [0, 1]
    %
    %   t = 0 is (almost) clean data, t = 1 is (almost) pure noise. A schedule
    %   is fully described by alpha(t), sigma(t) and their time derivatives:
    %     VP (DDPM, cosine)      alpha^2 + sigma^2 = 1
    %     VE / EDM               alpha = 1, sigma from sigma_min to sigma_max
    %     rectified flow         alpha = 1 - t, sigma = t      (straight paths)
    %   logSNR(t) = log(alpha^2 / sigma^2) decreases in t for every schedule.
    %
    %   The derivatives define the probability-flow ODE used by the ODE
    %   samplers:  dx/dt = alpha'(t) E[x0 | x_t] + sigma'(t) E[eps | x_t].
    %
    %   Sampling contract (used by every ISampler):
    %     grid(N)      decreasing times from 1; steps run between consecutive
    %                  entries; if sigma(grid(end)) > 0 the sampler finishes with
    %                  one denoising step x <- E[x0 | x].
    %     priorScale() the first sample is priorScale() * eps.
    %     netTime(t)   the time input given to the network (EDM: log(sigma)/4).

    methods (Abstract)
        a = alpha(obj, t)
        s = sigma(obj, t)
        da = dalpha(obj, t)
        ds = dsigma(obj, t)
    end

    methods
        function l = logSNR(obj, t)
            l = 2 * (log(obj.alpha(t)) - log(obj.sigma(t)));
        end

        function r = snr(obj, t)
            r = (obj.alpha(t) ./ obj.sigma(t)) .^ 2;
        end

        function t = grid(~, N)
            t = linspace(1, 0, N + 1);
        end

        function c = priorScale(obj)
            c = obj.sigma(1);
        end

        function c = netTime(~, t)
            c = t;
        end
    end
end
