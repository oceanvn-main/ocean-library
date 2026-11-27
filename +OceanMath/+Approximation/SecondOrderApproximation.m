classdef SecondOrderApproximation < OceanMath.Approximation.IApproximationMethod
    % SECONDORDERAPPROXIMATION Quadratic (Score Test) gain formula.
    %
    % Computes the second-order approximation of objective gain:
    %
    %   gain = sum(gradient .^ 2) / curvature
    %
    % This is equivalent to the Score Test statistic (Rao, 1948): the squared
    % directional derivative normalised by its variance under the current model.
    % When gradient ~ (h'Ws) and curvature ~ sigma_hh (Schur complement),
    % this recovers the PolyMARS Score Test exactly.
    %
    % References:
    %   Rao, C.R. (1948) "Large sample tests of statistical hypotheses..."
    %   Stone et al. (1997) "Polynomial Splines ... Extended Linear Modeling"
    %
    % See also: OceanMath.Approximation.IApproximationMethod
    
    methods
        function gain = compute(~, gradient, curvature)
            % COMPUTE gain = sum(gradient.^2) / curvature
            %
            % INPUTS:
            %   gradient  - (1 x K) adjusted score projection
            %   curvature - scalar Schur complement σ_hh
            if curvature <= 1e-12
                gain = 0;
                return;
            end
            gain = sum(gradient .^ 2) / curvature;
        end
    end
end
