function pacf = parcorr(y, maxLag)
    % PARCORR Compute empirical partial autocorrelation function (PACF) of a series.
    %
    % Computes the sample partial autocorrelation values for a 1-D vector y
    % up to a maximum lag specified by maxLag. Uses the Durbin-Levinson algorithm.
    %
    % INPUTS:
    %   y      - (N x 1) or (1 x N) double, the time series data
    %   maxLag - (1,1) double, maximum lag to compute (default: 20)
    %
    % OUTPUTS:
    %   pacf   - (1 x maxLag) double, partial autocorrelation values for lags 1 to maxLag
    %
    % See also: OceanMath.TimeSeries.Diagnostics.autocorr
    
    arguments
        y (:, :) double
        maxLag (1, 1) double {mustBePositive, mustBeInteger} = 20
    end
    
    if ~isvector(y)
        error('parcorr:InvalidInput', 'y must be a vector.');
    end
    
    % Force column vector
    y = y(:);
    N = length(y);
    
    if maxLag >= N
        error('parcorr:MaxLagTooLarge', 'maxLag must be less than the length of the time series.');
    end
    
    % Compute empirical ACF using the new sub-package path
    acf = OceanMath.TimeSeries.Diagnostics.autocorr(y, maxLag);
    
    % Solve Durbin-Levinson recursion
    phiMatrix = zeros(maxLag, maxLag);
    phiMatrix(1, 1) = acf(2);
    
    for k = 2:maxLag
        num = acf(k+1) - sum(phiMatrix(k-1, 1:k-1) .* acf(k:-1:2));
        den = 1 - sum(phiMatrix(k-1, 1:k-1) .* acf(2:k));
        
        if abs(den) < 1e-12
            % Handle numeric singularity if perfectly predictable
            phiMatrix(k, k) = 0.0;
        else
            phiMatrix(k, k) = num / den;
        end
        
        for j = 1:k-1
            phiMatrix(k, j) = phiMatrix(k-1, j) - phiMatrix(k, k) * phiMatrix(k-1, k-j);
        end
    end
    
    % PACF values for lags 1 to maxLag are on the diagonal
    pacf = diag(phiMatrix)';
end
