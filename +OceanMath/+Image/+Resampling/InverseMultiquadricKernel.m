classdef InverseMultiquadricKernel < OceanMath.Image.Resampling.IRBFKernel
    % INVERSEMULTIQUADRICKERNEL  Inverse Multiquadric RBF kernel: 1 / sqrt(r^2 + c^2).
    %
    %   Implements textbook eq. 4.6 (Szeliski §4.1.1).
    %
    %   Author: Antigravity Code Assistant
    %   Date:   2026-08-18

    properties
        C (1,1) double {mustBePositive} = 1.0  % Scale / smoothness parameter c
    end

    methods
        function obj = InverseMultiquadricKernel(options)
            arguments
                options.C (1,1) double {mustBePositive} = 1.0
            end
            obj.C = options.C;
        end

        function val = evaluate(obj, r)
            % Evaluate phi(r) = 1 / sqrt(r^2 + c^2)
            val = 1.0 ./ sqrt(r.^2 + obj.C^2);
        end
    end
end
