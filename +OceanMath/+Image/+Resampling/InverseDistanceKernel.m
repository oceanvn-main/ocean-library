classdef InverseDistanceKernel < OceanMath.Image.Resampling.IRBFKernel
    % INVERSEDISTANCEKERNEL  Inverse Distance Weighting (IDW) kernel: 1 / (r^p + eps).
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-30

    properties
        Power (1,1) double = 2.0  % Decay power parameter
    end

    methods
        function obj = InverseDistanceKernel(options)
            arguments
                options.Power (1,1) double = 2.0
            end
            obj.Power = options.Power;
        end

        function val = evaluate(obj, r)
            val = 1 ./ (r.^obj.Power + eps);
        end
    end
end
