classdef ThinPlateKernel < OceanMath.Image.Resampling.IRBFKernel
    % THINPLATEKERNEL  Thin Plate Spline RBF kernel: r^2 * log(r).
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-30

    methods
        function val = evaluate(obj, r)
            % Evaluate U(r) = r^2 * log(r)
            % Clamp r near 0 to avoid log(0) singularity
            r_clamped = max(r, 1e-12);
            val = (r.^2) .* log(r_clamped);
        end
    end
end
