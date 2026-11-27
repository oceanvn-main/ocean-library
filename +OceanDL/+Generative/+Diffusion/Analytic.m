classdef Analytic
    % ANALYTIC  Exact (Bayes-optimal) denoisers for data whose distribution is
    %   known in closed form -- reference predictors for testing samplers and
    %   for measuring how far a trained network is from optimal.
    %
    %   Data: Gaussian mixture  p(x0) = sum_k w_k N(mu_k, s_k^2 I)  in R^D.
    %   Under x_t = a x0 + s eps each component stays Gaussian in x_t:
    %     x_t | k ~ N(a mu_k, (a^2 s_k^2 + s^2) I)
    %   so the posterior responsibilities r_k(x_t) are Gaussian likelihood
    %   ratios and (Tweedie / linear-Gaussian regression per component)
    %     E[x0 | x_t] = sum_k r_k ( mu_k + a s_k^2 / (a^2 s_k^2 + s^2) (x_t - a mu_k) )
    %     E[eps | x_t] = (x_t - a E[x0 | x_t]) / s
    %   The returned predictor is  [x0hat, epshat] = predict(x, t)  with x as
    %   (D x M) columns -- the interface every ISampler consumes.

    methods (Static)
        function predict = gmm(means, stds, weights, schedule)
            arguments
                means double                     % D x K
                stds (1,:) double                % 1 x K
                weights (1,:) double             % 1 x K, sums to 1
                schedule
            end
            predict = @(x, t) OceanDL.Generative.Diffusion.Analytic.gmmPosterior(x, t, means, stds, weights, schedule);
        end

        function predict = gaussian(mu, s0, schedule)
            predict = OceanDL.Generative.Diffusion.Analytic.gmm(mu, s0, 1, schedule);
        end

        function [x0, eps] = gmmPosterior(x, t, means, stds, weights, schedule)
            a = schedule.alpha(t); s = schedule.sigma(t);
            [D, K] = size(means);
            M = size(x, 2);
            logr = zeros(K, M); mk = zeros(D, M, K);
            for k = 1:K
                v = a ^ 2 * stds(k) ^ 2 + s ^ 2;
                d = x - a * means(:, k);
                logr(k, :) = log(weights(k)) - 0.5 * D * log(v) - sum(d .^ 2, 1) / (2 * v);
                mk(:, :, k) = means(:, k) + (a * stds(k) ^ 2 / v) * d;
            end
            r = exp(logr - max(logr, [], 1)); r = r ./ sum(r, 1);
            x0 = sum(mk .* reshape(r.', 1, M, K), 3);
            eps = (x - a * x0) / s;
        end

        function x0 = gaussianFlowMap(x1, mu, s0, schedule)
            % Exact endpoint of the probability-flow ODE for Gaussian data
            % N(mu, s0^2 I) started from x1 at t = 1: the ODE is the monotone
            % transport between the Gaussians N(a1 mu, a1^2 s0^2 + s1^2) and
            % N(mu, s0^2), i.e. the affine quantile map per coordinate.
            a1 = schedule.alpha(1); s1 = schedule.sigma(1);
            x0 = mu + s0 * (x1 - a1 * mu) / sqrt(a1 ^ 2 * s0 ^ 2 + s1 ^ 2);
        end
    end
end
