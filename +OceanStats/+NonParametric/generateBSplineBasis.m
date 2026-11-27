function [H, knots] = generateBSplineBasis(x, range, knots_input, degree)
% GENERATE_BSPLINE_BASIS Generates B-Spline Basis Matrix
% 
% Inputs:
%   x           - Input data vector (Nx1)
%   range       - [min, max] range of the domain
%   knots_input - Scalar (Number of internal knots) OR Vector (Specific locations)
%   degree      - Degree of spline (default 3 for cubic)
%
% Outputs:
%   H     - Basis Matrix (NxM)
%   knots - Full Knot vector (Augmented with boundary repeats)
%
% Author: Ocean Library System

    if nargin < 4, degree = 3; end

    % --- 1. Handle Knots (Scalar vs Vector) ---
    if isscalar(knots_input)
        % Case A: User gave a number (e.g., 10). Generate uniform knots.
        % linspace generates N+2 points (Start, ... N internal ..., End)
        raw_knots = linspace(range(1), range(2), knots_input + 2);
        knots_internal = raw_knots(2:end-1); % Strip start/end
    else
        % Case B: User gave specific locations (e.g., Quantiles).
        knots_internal = knots_input(:)'; % Ensure row vector
        
        % Safety: Clip to be strictly inside domain
        knots_internal = knots_internal(knots_internal > range(1) & knots_internal < range(2));
    end
    
    % --- 2. Augment Knots (Clamped B-Spline) ---
    % Repeat boundaries 'degree' times + the actual boundary itself
    % Standard convention: degree+1 repeats at ends
    knots = [repmat(range(1), 1, degree+1), knots_internal, repmat(range(2), 1, degree+1)];
    
    % --- 3. Generate Basis Matrix ---
    % Delegates to OceanMath.Geometry.BSplineBasis (iterative Cox-de Boor).
    % This replaces the old point-by-point recursive loop with a single
    % vectorized call: O(n*p) per point instead of O(2^p).
    H = OceanMath.Geometry.BSplineBasis.evaluate(knots, degree, x(:));
end