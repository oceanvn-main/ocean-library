classdef EDM < OceanDL.Generative.Diffusion.Schedules.INoiseSchedule
    % EDM  Karras et al. 2022 ("Elucidating the Design Space"): variance
    %   exploding, alpha = 1, sigma is the time variable. Time maps to the
    %   Karras sampling spacing
    %     sigma(t) = ( sigma_min^(1/rho) + t (sigma_max^(1/rho) - sigma_min^(1/rho)) )^rho
    %   so a uniform grid in t IS the paper's rho = 7 grid. grid(N) gives N
    %   noise levels sigma_max .. sigma_min; the sampler's final denoising
    %   step x <- D(x) is the paper's last step to sigma = 0.
    %   SigmaData is the data standard deviation used by the preconditioning
    %   (DiffusionModel 'Target' = "edm"); netTime(t) = log(sigma) / 4 (c_noise).

    properties (SetAccess = private)
        SigmaMin (1,1) double
        SigmaMax (1,1) double
        Rho (1,1) double
        SigmaData (1,1) double
    end

    methods
        function obj = EDM(opts)
            arguments
                opts.SigmaMin (1,1) double {mustBePositive} = 0.002
                opts.SigmaMax (1,1) double {mustBePositive} = 80
                opts.Rho (1,1) double {mustBePositive} = 7
                opts.SigmaData (1,1) double {mustBePositive} = 0.5
            end
            obj.SigmaMin = opts.SigmaMin; obj.SigmaMax = opts.SigmaMax;
            obj.Rho = opts.Rho; obj.SigmaData = opts.SigmaData;
        end

        function a = alpha(~, t)
            a = ones(size(t));
        end

        function s = sigma(obj, t)
            lo = obj.SigmaMin ^ (1 / obj.Rho); hi = obj.SigmaMax ^ (1 / obj.Rho);
            s = (lo + t * (hi - lo)) .^ obj.Rho;
        end

        function da = dalpha(~, t)
            da = zeros(size(t));
        end

        function ds = dsigma(obj, t)
            lo = obj.SigmaMin ^ (1 / obj.Rho); hi = obj.SigmaMax ^ (1 / obj.Rho);
            ds = obj.Rho * (lo + t * (hi - lo)) .^ (obj.Rho - 1) * (hi - lo);
        end

        function t = tFromSigma(obj, s)
            lo = obj.SigmaMin ^ (1 / obj.Rho); hi = obj.SigmaMax ^ (1 / obj.Rho);
            t = (s .^ (1 / obj.Rho) - lo) / (hi - lo);
        end

        function t = grid(~, N)
            t = linspace(1, 0, N);                 % N levels sigma_max..sigma_min, then denoise
        end

        function c = netTime(obj, t)
            c = log(obj.sigma(t)) / 4;
        end
    end
end
