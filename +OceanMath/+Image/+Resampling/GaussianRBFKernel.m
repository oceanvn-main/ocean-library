classdef GaussianRBFKernel < OceanMath.Image.Resampling.IRBFKernel
    % GAUSSIANRBFKERNEL  Gaussian RBF kernel: exp(-beta * r^2).
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-30

    properties
        Beta (1,1) double = 1.0  % Precision/scale parameter
    end

    methods
        function obj = GaussianRBFKernel(options)
            arguments
                options.Beta (1,1) double = 1.0
            end
            obj.Beta = options.Beta;
        end

        function val = evaluate(obj, r)
            val = exp(-obj.Beta * (r.^2));
        end
    end
end
