classdef MultiquadricKernel < OceanMath.Image.Resampling.IRBFKernel
    % MULTIQUADRICKERNEL  Multiquadric RBF kernel: sqrt(r^2 + c^2).
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-30

    properties
        C (1,1) double = 1.0  % Smoothness constant parameter
    end

    methods
        function obj = MultiquadricKernel(options)
            arguments
                options.C (1,1) double = 1.0
            end
            obj.C = options.C;
        end

        function val = evaluate(obj, r)
            val = sqrt(r.^2 + obj.C^2);
        end
    end
end
