classdef RectifiedFlow < OceanDL.Generative.Diffusion.Schedules.INoiseSchedule
    % RECTIFIEDFLOW  Flow matching with straight paths (Lipman et al. 2023;
    %   Liu et al. 2023; SD3, FLUX, Movie Gen):
    %     x_t = (1 - tau) x0 + tau eps,   velocity dx/dt = tau' (eps - x0)
    %   tau(t) = t by default. 'Shift' s applies the SD3 timestep shift
    %     tau(t) = s t / (1 + (s - 1) t)
    %   (s > 1 spends more of the trajectory at high noise -- used for larger
    %   images, whose many pixels keep more signal at a given noise level).
    %   Neither variance preserving nor exploding: alpha + sigma = 1.

    properties (SetAccess = private)
        Shift (1,1) double
    end

    methods
        function obj = RectifiedFlow(opts)
            arguments
                opts.Shift (1,1) double {mustBePositive} = 1
            end
            obj.Shift = opts.Shift;
        end

        function tau = tau(obj, t)
            s = obj.Shift;
            tau = s * t ./ (1 + (s - 1) * t);
        end

        function d = dtau(obj, t)
            s = obj.Shift;
            d = s ./ (1 + (s - 1) * t) .^ 2;
        end

        function a = alpha(obj, t)
            a = 1 - obj.tau(t);
        end

        function s = sigma(obj, t)
            s = obj.tau(t);
        end

        function da = dalpha(obj, t)
            da = -obj.dtau(t);
        end

        function ds = dsigma(obj, t)
            ds = obj.dtau(t);
        end
    end
end
