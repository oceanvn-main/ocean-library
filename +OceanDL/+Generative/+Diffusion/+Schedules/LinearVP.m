classdef LinearVP < OceanDL.Generative.Diffusion.Schedules.INoiseSchedule
    % LINEARVP  The DDPM forward process (Ho et al. 2020) in the notation of
    %   Lai et al., "The Principles of Diffusion Models", Sec. 2.2.1 (p. 50):
    %
    %     transition kernel   p(x_i | x_{i-1}) = N(x_i; alpha_i x_{i-1}, beta_i^2 I)
    %                         x_i = alpha_i x_{i-1} + beta_i eps_i,   alpha_i := sqrt(1 - beta_i^2)
    %     perturbation kernel p_i(x_i | x_0) = N(x_i; alphaBar_i x_0, (1 - alphaBar_i^2) I)
    %                         alphaBar_i := prod_{k=1..i} alpha_k
    %                         x_i = alphaBar_i x_0 + sqrt(1 - alphaBar_i^2) eps        (Eq. 2.2.1)
    %     prior               p_prior = N(0, I)
    %
    %   NOTATION WARNING: in the book beta_i is the noise STANDARD DEVIATION of
    %   step i. Ho et al. (and most code) call the step VARIANCE "beta_t". Ho's
    %   linear schedule "beta_t from 1e-4 to 0.02" therefore means
    %     beta_i^2 linear from 'BetaSqStart' = 1e-4 to 'BetaSqEnd' = 0.02,  i = 1..L
    %   (L = 1000 steps; Ho's T). Every variable below is named after the BOOK symbol.
    %
    %   Continuous time (the INoiseSchedule interface used by the samplers):
    %     t = i / L  hits step i EXACTLY:  alpha(i/L) = alphaBar_i,
    %                                      sigma(i/L) = sqrt(1 - alphaBar_i^2);
    %   between steps log(alphaBar^2) is interpolated linearly.

    properties (SetAccess = private)
        L (1,1) double                  % number of steps (book L; Ho's T)
        BetaSqStart (1,1) double        % beta_1^2   (Ho: beta_1 = 1e-4)
        BetaSqEnd (1,1) double          % beta_L^2   (Ho: beta_T = 0.02)
        LogAlphaBarSq (1,:) double      % log(alphaBar_i^2), i = 0..L  (alphaBar_0 = 1)
    end

    methods
        function obj = LinearVP(opts)
            arguments
                opts.L (1,1) double {mustBePositive, mustBeInteger} = 1000
                opts.BetaSqStart (1,1) double {mustBeInRange(opts.BetaSqStart, 0, 1, 'exclusive')} = 1e-4
                opts.BetaSqEnd (1,1) double {mustBeInRange(opts.BetaSqEnd, 0, 1, 'exclusive')} = 0.02
            end
            obj.L = opts.L; obj.BetaSqStart = opts.BetaSqStart; obj.BetaSqEnd = opts.BetaSqEnd;
            alphaSq = 1 - obj.betaSq();                        % alpha_i^2 = 1 - beta_i^2
            obj.LogAlphaBarSq = [0, cumsum(log(alphaSq))];     % alphaBar_i^2 = prod alpha_k^2
        end

        % ---------------- discrete steps, book notation ----------------
        function b2 = betaSq(obj, i)
            % beta_i^2 -- the step VARIANCE (Ho's beta_i), linear in i.
            b2 = linspace(obj.BetaSqStart, obj.BetaSqEnd, obj.L);
            if nargin > 1, b2 = b2(i); end
        end

        function b = beta(obj, i)
            % beta_i -- the step noise STANDARD DEVIATION (book symbol).
            if nargin > 1, b = sqrt(obj.betaSq(i)); else, b = sqrt(obj.betaSq()); end
        end

        function a = alphaStep(obj, i)
            % alpha_i = sqrt(1 - beta_i^2)  (signal kept by ONE step).
            a = sqrt(1 - obj.betaSq(i));
        end

        function ab = alphaBar(obj, i)
            % alphaBar_i = prod_{k<=i} alpha_k  (signal kept after i steps), i = 0..L.
            ab = exp(0.5 * obj.LogAlphaBarSq(i + 1));
        end

        % ---------------- continuous-time interface ----------------
        function a = alpha(obj, t)
            % alphaBar at continuous t (= alphaBar_i at t = i/L).
            a = exp(0.5 * interp1(0:obj.L, obj.LogAlphaBarSq, t * obj.L, 'linear'));
        end

        function s = sigma(obj, t)
            % sqrt(1 - alphaBar^2): noise standard deviation of p_i(x_i | x_0).
            s = sqrt(max(0, 1 - obj.alpha(t) .^ 2));
        end

        function da = dalpha(obj, t)
            % d alphaBar / dt. alphaBar = exp(0.5 log alphaBar^2) with log alphaBar^2
            % piecewise linear in t, so d alphaBar/dt = 0.5 alphaBar * (slope of the segment).
            i = min(obj.L, max(1, floor(t * obj.L) + 1));
            slope = (obj.LogAlphaBarSq(i + 1) - obj.LogAlphaBarSq(i)) * obj.L;
            da = 0.5 * obj.alpha(t) .* slope;
        end

        function ds = dsigma(obj, t)
            % d/dt sqrt(1 - alphaBar^2) = -alphaBar * dalphaBar/dt / sqrt(1 - alphaBar^2).
            ds = -obj.alpha(t) .* obj.dalpha(t) ./ obj.sigma(t);
        end
    end
end
