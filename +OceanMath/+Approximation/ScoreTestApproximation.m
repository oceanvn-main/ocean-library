classdef ScoreTestApproximation < OceanStats.Base.IObjectiveApproximation
    % SCORETESTAPPROXIMATION WLS-state extractor for the PolyMARS Score Test.
    %
    % Implements extractComponents() for the Weighted Least Squares context
    % of PolyMARS: extracts the adjusted score projection (gradient) and the
    % Schur complement (curvature) for a candidate basis column h.
    %
    % The actual gain formula is delegated to obj.Method (default:
    % SecondOrderApproximation, yielding the classic Score Test statistic):
    %
    %   gradient   = h'W s_adj        (adjusted score in candidate direction)
    %   curvature  = sigma_hh          (Schur complement under W)
    %   gain       = Method.compute(gradient, curvature)
    %              = sum(gradient.^2) / curvature    [for SecondOrderApproximation]
    %
    % Because the gain formula is injected, this class can also be used with
    % a FirstOrderApproximation or any other IApproximationMethod without change.
    %
    % References:
    %   Stone et al. (1997) "Polynomial Splines ... Extended Linear Modeling"
    %
    % See also: OceanStats.Base.IObjectiveApproximation,
    %           OceanMath.Approximation.SecondOrderApproximation
    
    methods
        function [gradient, curvature] = extractComponents(~, wls, h_cand, W, s)
            % EXTRACTCOMPONENTS Extract gradient and curvature from WLS state.
            %
            % INPUTS:
            %   wls    - OceanMath.LinearAlgebra.IncrementalWLS (current state)
            %   h_cand - (N x 1) candidate basis evaluation
            %   W      - (N x 1) diagonal Fisher weights
            %   s      - (N x K) score matrix (Y_indicator - P_current)
            %
            % OUTPUTS:
            %   gradient  - (1 x K) adjusted score projection h'W s_adj
            %   curvature - scalar Schur complement σ_hh
            
            hw  = h_cand .* W;          % (N x 1) weighted candidate
            
            % h'WB — projection of h onto current basis (1 x M)
            hWB = hw' * wls.B;
            
            % Schur complement: σ_hh = h'Wh - h'WB (B'WB)^{-1} B'Wh
            hWh = hw' * h_cand;
            if rcond(wls.BtWB) < 1e-12
                BtWB_inv = pinv(wls.BtWB);
            else
                BtWB_inv = inv(wls.BtWB); %small M, fine here
            end
            curvature = hWh - hWB * (BtWB_inv * hWB');  % scalar
            
            % Adjusted score projection: h'Ws - h'WB (B'WB)^{-1} B'Ws
            hWs      = hw' * s;                          % (1 x K)
            gradient = hWs - hWB * (BtWB_inv * wls.BtWs); % (1 x K)
        end
    end
end
