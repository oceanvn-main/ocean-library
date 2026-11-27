classdef DDIM < OceanDL.Generative.Diffusion.Samplers.ISampler
    % DDIM  (Song, Meng & Ermon 2021) in the notation of Lai et al., "The
    %   Principles of Diffusion Models" (Eq. 2.2.15; Sec. 9.2):
    %
    %     x_{i-1} = alphaBar_{i-1} x_phi
    %             + sqrt(1 - alphaBar_{i-1}^2 - Eta^2 tildeBeta_i^2) eps_phi
    %             + Eta tildeBeta_i eps_i                                          (Eq. 2.2.15)
    %
    %   x_phi, eps_phi: the model's clean and noise estimates at x_i.
    %   tildeBeta_i^2 = (1 - alphaBar_{i-1}^2) / (1 - alphaBar_i^2) * beta_i^2 is the
    %   DDPM posterior variance (sigma^2(i) of Eq. 2.2.4).
    %     Eta = 1: Eq. (2.2.15) exactly -- the DDPM sampler rewritten (same
    %              samples as DDPMAncestral, tested).
    %     Eta = 0: deterministic -- the fresh noise is removed and the remaining
    %              noise coefficient becomes sqrt(1 - alphaBar_{i-1}^2) (Sec. 9.2);
    %              exact for the probability-flow ODE when the prediction is
    %              constant over the step, so few steps work.
    %   General schedules / skipped steps: as in DDPMAncestral, with
    %   1 - alphaBar^2 read as sigma^2 = schedule.sigma(t)^2.

    properties (SetAccess = private)
        Eta (1,1) double
    end

    methods
        function obj = DDIM(opts)
            arguments
                opts.Steps (1,1) double = 50
                opts.Eta (1,1) double {mustBeNonnegative} = 0
            end
            obj.Steps = opts.Steps;
            obj.Eta = opts.Eta;
        end

        function [x_im1, nfe] = step(obj, predict, schedule, x_i, t_i, t_im1, stream)
            [x_phi, eps_phi] = predict(x_i, t_i); nfe = 1;  % model's clean and noise estimates at x_i

            alphaBar_im1 = schedule.alpha(t_im1);           % alphaBar_{i-1}
            sigmaSq_im1  = schedule.sigma(t_im1) ^ 2;       % 1 - alphaBar_{i-1}^2
            if sigmaSq_im1 == 0                             % reached clean data
                x_im1 = x_phi;
                return;
            end

            tildeBetaSq_i = 0;                              % tildeBeta_i^2 (only needed when Eta > 0)
            if obj.Eta > 0
                alphaBar_i = schedule.alpha(t_i);
                sigmaSq_i  = schedule.sigma(t_i) ^ 2;       % 1 - alphaBar_i^2
                alpha_i    = alphaBar_i / alphaBar_im1;     % alpha_i
                betaSq_i   = sigmaSq_i - alpha_i ^ 2 * sigmaSq_im1;   % beta_i^2
                tildeBetaSq_i = sigmaSq_im1 / sigmaSq_i * betaSq_i;  % posterior variance
            end
            freshStd = obj.Eta * sqrt(max(0, tildeBetaSq_i));        % Eta * tildeBeta_i

            x_im1 = alphaBar_im1 * x_phi ...
                  + sqrt(max(0, sigmaSq_im1 - freshStd ^ 2)) * eps_phi;   % Eq. (2.2.15), first two terms
            if freshStd > 0
                eps_i = cast(randn(stream, size(x_i)), 'like', x_i);
                x_im1 = x_im1 + freshStd * eps_i;                        % Eq. (2.2.15), fresh-noise term
            end
        end
    end
end
