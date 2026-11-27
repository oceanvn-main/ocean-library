classdef CosineVP < OceanDL.Generative.Diffusion.Schedules.INoiseSchedule
    % COSINEVP  Cosine schedule (Nichol & Dhariwal 2021) in its continuous
    %   logSNR form (Kingma et al. 2021; Hoogeboom et al. 2023):
    %     logSNR(t) = -2 log tan(u(t)),   u = u_min + t (u_max - u_min)
    %   i.e. abar(t) = cos^2(u), clipped so logSNR runs from LogSNRMax (t = 0)
    %   to LogSNRMin (t = 1) -- never exactly clean or exactly pure noise, so
    %   every target conversion stays finite. Variance preserving:
    %     alpha^2 = sigmoid(logSNR),  sigma^2 = sigmoid(-logSNR).
    %   'Shift' s > 0 adds 2 log(s) to logSNR (lower SNR = more noise for s < 1;
    %   the resolution shift of simple diffusion for larger images).

    properties (SetAccess = private)
        LogSNRMin (1,1) double
        LogSNRMax (1,1) double
        Shift (1,1) double
        UMin (1,1) double
        UMax (1,1) double
    end

    methods
        function obj = CosineVP(opts)
            arguments
                opts.LogSNRMin (1,1) double = -15
                opts.LogSNRMax (1,1) double = 15
                opts.Shift (1,1) double {mustBePositive} = 1
            end
            obj.LogSNRMin = opts.LogSNRMin; obj.LogSNRMax = opts.LogSNRMax; obj.Shift = opts.Shift;
            obj.UMin = atan(exp(-0.5 * opts.LogSNRMax));
            obj.UMax = atan(exp(-0.5 * opts.LogSNRMin));
        end

        function l = logSNR(obj, t)
            l = -2 * log(tan(obj.UMin + t * (obj.UMax - obj.UMin))) + 2 * log(obj.Shift);
        end

        function a = alpha(obj, t)
            a = sqrt(1 ./ (1 + exp(-obj.logSNR(t))));
        end

        function s = sigma(obj, t)
            s = sqrt(1 ./ (1 + exp(obj.logSNR(t))));
        end

        function da = dalpha(obj, t)
            % alpha^2 = sigmoid(l):  d alpha / dt = 0.5 alpha sigma^2 l'
            da = 0.5 * obj.alpha(t) .* obj.sigma(t) .^ 2 .* obj.dlogSNR(t);
        end

        function ds = dsigma(obj, t)
            ds = -0.5 * obj.sigma(t) .* obj.alpha(t) .^ 2 .* obj.dlogSNR(t);
        end

        function dl = dlogSNR(obj, t)
            u = obj.UMin + t * (obj.UMax - obj.UMin);
            dl = -2 * (obj.UMax - obj.UMin) ./ (sin(u) .* cos(u));
        end
    end
end
