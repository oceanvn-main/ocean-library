classdef DDPMAncestral < OceanDL.Generative.Diffusion.Samplers.ISampler
    % DDPMANCESTRAL  DDPM reverse (denoising) process, in the notation of Lai et
    %   al., "The Principles of Diffusion Models", Sec. 2.2.2-2.2.6:
    %   start from x_L ~ p_prior = N(0, I) and, for i = L .. 1, sample
    %
    %     x_{i-1} ~ p_phi(x_{i-1} | x_i) = N(x_{i-1}; mu_phi(x_i, i), sigma^2(i) I)   (Eq. 2.2.5)
    %
    %   whose mean is the reverse conditional mean of Lemma 2.2.2 (Eq. 2.2.4)
    %   with the clean sample x replaced by the model's estimate x_phi(x_i, i):
    %
    %     mu    = alphaBar_{i-1} beta_i^2 / (1 - alphaBar_i^2) * x_phi
    %           + (1 - alphaBar_{i-1}^2) alpha_i / (1 - alphaBar_i^2) * x_i          (Eq. 2.2.4)
    %     sigma^2(i) = (1 - alphaBar_{i-1}^2) / (1 - alphaBar_i^2) * beta_i^2       (Eq. 2.2.4)
    %                = tildeBeta_i^2, the DDPM posterior variance                 (Eq. 2.2.15)
    %   and the last step (i = 1, alphaBar_0 = 1) is noise-free: x_0 = x_phi.
    %
    %   Book symbols (Sec. 2.2.1): alpha_i = sqrt(1 - beta_i^2) is the signal kept
    %   by ONE step, beta_i its noise std; alphaBar_i = prod_k alpha_k.
    %
    %   GENERAL SCHEDULES / SKIPPED STEPS: the code reads only alphaBar and the
    %   noise std sigma = sqrt(1 - alphaBar^2) at the two levels (schedule.alpha,
    %   schedule.sigma), and forms the one-jump quantities from them:
    %     alpha_i  = alphaBar_i / alphaBar_{i-1}
    %     beta_i^2 = sigma_i^2 - alpha_i^2 sigma_{i-1}^2    (= 1 - alpha_i^2 for DDPM)
    %   so the same step is exact for any schedule and for a jump over several
    %   steps (the general reverse kernel of Lemma 4.5.2). With LinearVP and
    %   'Steps' = L it is exactly the book's (Ho et al.'s) DDPM sampler.
    %   Equivalent forms: Eq. (2.2.14) (eps-prediction) and Eq. (2.2.15) = DDIM
    %   with Eta = 1 (tested: DiffusionCoreTest/testDdimEtaOneEqualsDdpmAncestral).

    methods
        function obj = DDPMAncestral(opts)
            arguments
                opts.Steps (1,1) double = 1000
            end
            obj.Steps = opts.Steps;
        end

        function [x_im1, nfe] = step(~, predict, schedule, x_i, t_i, t_im1, stream)
            % One reverse step x_i (level t_i) -> x_{i-1} (level t_im1 < t_i).
            x_phi = predict(x_i, t_i); nfe = 1;             % model's clean estimate x_phi(x_i, i)  (Eq. 2.2.11-2.2.12)

            alphaBar_i   = schedule.alpha(t_i);             % alphaBar_i
            alphaBar_im1 = schedule.alpha(t_im1);           % alphaBar_{i-1}
            sigmaSq_i    = schedule.sigma(t_i) ^ 2;         % 1 - alphaBar_i^2      (noise variance at level i)
            sigmaSq_im1  = schedule.sigma(t_im1) ^ 2;       % 1 - alphaBar_{i-1}^2

            if sigmaSq_im1 == 0                             % reached clean data (alphaBar_0 = 1): no noise added
                x_im1 = x_phi;
                return;
            end

            alpha_i  = alphaBar_i / alphaBar_im1;           % alpha_i = sqrt(1 - beta_i^2)
            betaSq_i = sigmaSq_i - alpha_i ^ 2 * sigmaSq_im1; % beta_i^2 (step variance)

            mu = (alphaBar_im1 * betaSq_i / sigmaSq_i) * x_phi ...   % Eq. (2.2.4), coefficient of x
               + (sigmaSq_im1 * alpha_i / sigmaSq_i) * x_i;         % Eq. (2.2.4), coefficient of x_i
            tildeBetaSq_i = sigmaSq_im1 / sigmaSq_i * betaSq_i;     % sigma^2(i) of Eq. (2.2.4) = tildeBeta_i^2

            eps_i = cast(randn(stream, size(x_i)), 'like', x_i);    % fresh noise
            x_im1 = mu + sqrt(max(0, tildeBetaSq_i)) * eps_i;       % x_{i-1} ~ p_phi(x_{i-1} | x_i), Eq. (2.2.5)
        end
    end
end
