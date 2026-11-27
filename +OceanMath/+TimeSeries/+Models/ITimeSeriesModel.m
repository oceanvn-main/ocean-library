classdef ITimeSeriesModel < handle
    % ITIMESERIESMODEL Abstract interface for time series process models.
    %
    % This class defines the interface contract for time series models (e.g. AR,
    % MA, ARMA). It requires implementing simulation, theoretical ACF/PACF,
    % stationarity/invertibility checks, and equation representation.
    %
    % See also: OceanMath.TimeSeries.Models.ARMA
    
    methods (Abstract)
        % SIMULATE Generate a realization of the time series model.
        %
        % INPUTS:
        %   N     - (1,1) double, number of time steps (length of series)
        %   sigma - (1,1) double, standard deviation of white noise
        %
        % OUTPUTS:
        %   y     - (N x 1) double, the simulated time series sequence
        y = simulate(obj, N, sigma)
        
        % THEORETICALACF Compute the theoretical autocorrelation function.
        %
        % INPUTS:
        %   lags - (1,1) double, maximum lag to compute
        %
        % OUTPUTS:
        %   rho  - (1 x lags+1) double, ACF values for lags 0 to lags
        rho = theoreticalACF(obj, lags)
        
        % THEORETICALPACF Compute the theoretical partial autocorrelation function.
        %
        % INPUTS:
        %   lags - (1,1) double, maximum lag to compute
        %
        % OUTPUTS:
        %   phi  - (1 x lags) double, PACF values for lags 1 to lags
        phi = theoreticalPACF(obj, lags)
        
        % ISSTATIONARY Check if the process is stationary.
        %
        % OUTPUTS:
        %   tf - (1,1) logical, true if the process is stationary
        tf = isStationary(obj)
        
        % ISINVERTIBLE Check if the process is invertible.
        %
        % OUTPUTS:
        %   tf - (1,1) logical, true if the process is invertible
        tf = isInvertible(obj)
        
        % GETEQUATION Return a string representation of the model equation.
        %
        % OUTPUTS:
        %   str - (1,1) string, the equation representation
        str = getEquation(obj)
    end
end
