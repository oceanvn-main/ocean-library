function [pdf, contributions] = calculateKDE(data, grid, lambda)
%CALCULATE_KDE Computes Gaussian Kernel Density Estimate
%Inputs:   
%   data: Vector of observed data points
%   grid: vector of points where density is evaluated at
%   lambda: Smoothing parameter (bandwidth)
%Ouputs:
%   pdf: The estimaetd density function values
%   contributions: Matrix where row i is the kernel for data point i

    N = length(data);
    
    %Prepare matrices for vectorized calculation
    %X_grid matrix: rows = grid points, cols = duplicate
    %Data matrix: rows = duplicate, cols = data points
    [X, D] = meshgrid(grid, data);
    
    %Gaussian Kernel Formula: 1/ sqrt(2*pi)*lambda *
    %exp(-1/2((x-xi)/lambda^2) (1D formula)
    % Calculate argument for exp
    u = (X' - D') ./ lambda;
    %Compute Standard normal density
    K = (1 / sqrt(2*pi)).* exp(-0.5 *u.^2);
    %Scale by bandwidth
    contributions = K ./ lambda;

    %Sum contributions over data points (columns) and normalize by N
    pdf = sum(contributions, 2) ./ N; %For each grid point i, adds up the contributions from all data cols
    if isrow(grid), pdf = pdf'; end
end