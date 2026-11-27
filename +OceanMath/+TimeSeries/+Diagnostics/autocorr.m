function acf = autocorr(y, maxLag)
    % AUTOCORR Compute empirical autocorrelation function (ACF) of a series.
    %
    % Computes the sample autocorrelation values for a 1-D vector y up to
    % a maximum lag specified by maxLag.
    %
    % INPUTS:
    %   y      - (N x 1) or (1 x N) double, the time series data
    %   maxLag - (1,1) double, maximum lag to compute (default: 20)
    %
    % OUTPUTS:
    %   acf    - (1 x maxLag+1) double, autocorrelation values for lags 0 to maxLag
    %
    % See also: OceanMath.TimeSeries.Diagnostics.parcorr
    
    arguments
        y (:, :) double
        maxLag (1, 1) double {mustBePositive, mustBeInteger} = 20
    end
    
    if ~isvector(y)
        error('autocorr:InvalidInput', 'y must be a vector.');
    end
    
    % Force column vector
    y = y(:);
    N = length(y);
    
    if maxLag >= N
        error('autocorr:MaxLagTooLarge', 'maxLag must be less than the length of the time series.');
    end
    
    % Center the series
    yMean = mean(y);
    yDev = y - yMean;
    
    % Lag 0 variance
    c0 = sum(yDev.^2) / N;
    
    if c0 < 1e-12
        acf = [1, zeros(1, maxLag)];
        return;
    end
    
    acf = zeros(1, maxLag + 1);
    acf(1) = 1.0;
    
    % Compute sample autocovariance and normalize
    for k = 1:maxLag
        ck = sum(yDev(k+1:end) .* yDev(1:end-k)) / N;
        acf(k+1) = ck / c0;
    end
end
