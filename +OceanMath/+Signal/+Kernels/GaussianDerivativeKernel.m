classdef GaussianDerivativeKernel < OceanMath.Signal.Kernels.IFilterKernel
    % GAUSSIANDERIVATIVEKERNEL  2D Gaussian derivative kernel supporting arbitrary steering.
    %
    %   Generates 2D kernels of first or second-order Gaussian derivatives, steered
    %   to a specific angle.
    %
    %   Constructor:
    %       GaussianDerivativeKernel('Size', [H, W], 'Sigma', sigma, 'Order', order, 'Angle', angle_deg)
    %
    %   Inputs:
    %       Size  - [1 x 2] kernel grid dimensions (default: [15, 15]).
    %       Sigma - Standard deviation of the Gaussian envelope (default: 2.0).
    %       Order - Derivative order: 1 or 2 (default: 1).
    %       Angle - Steering angle in degrees (default: 0).
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-07-13
    
    properties (Access = private)
        Size   (1,2) double = [15, 15]
        Sigma  (1,1) double = 2.0
        Order  (1,1) double = 1
        Angle  (1,1) double = 0
    end
    
    properties (SetAccess = private)
        Weights double
    end
    
    methods
        function obj = GaussianDerivativeKernel(options)
            % GAUSSIANDERIVATIVEKERNEL Construct a Gaussian derivative kernel.
            arguments
                options.Size   (1,2) double = [15, 15]
                options.Sigma  (1,1) double = 2.0
                options.Order  (1,1) double = 1
                options.Angle  (1,1) double = 0
            end
            obj.Size = options.Size;
            obj.Sigma = options.Sigma;
            obj.Order = options.Order;
            obj.Angle = options.Angle;
            
            if obj.Order ~= 1 && obj.Order ~= 2
                error('Derivative Order must be either 1 or 2.');
            end
            if obj.Sigma <= 0
                error('Sigma must be positive.');
            end
            
            % Compute the weights
            obj.computeWeights();
        end
        
        function K = getMatrix(obj)
            % GETMATRIX Returns the [H x W] double matrix of the steered kernel.
            K = obj.Weights;
        end
        
        function [h, v] = get1DDecomposition(obj)
            % GET1DDECOMPOSITION Returns horizontal and vertical 1D kernels if separable.
            if ~obj.isSeparable()
                h = [];
                v = [];
                return;
            end
            
            h_sz = obj.Size(2);
            v_sz = obj.Size(1);
            px = floor(h_sz / 2);
            py = floor(v_sz / 2);
            x = -px:px;
            y = -py:py;
            
            % Base 1D Gaussian components
            g_x = exp(-x.^2 / (2 * obj.Sigma^2)) / (sqrt(2*pi) * obj.Sigma);
            g_y = exp(-y.^2 / (2 * obj.Sigma^2)) / (sqrt(2*pi) * obj.Sigma);
            
            theta = deg2rad(obj.Angle);
            
            % Determine separable 1D factors based on aligned angles
            if obj.Order == 1
                gd_x = -(x / (obj.Sigma^2)) .* g_x;
                gd_y = -(y / (obj.Sigma^2)) .* g_y;
                
                % Check if aligned to horizontal axis
                if abs(cos(theta)) > 0.999
                    h = gd_x * cos(theta);
                    v = g_y';
                else
                    h = g_x;
                    v = gd_y' * sin(theta);
                end
            else % Order == 2
                gdd_x = ((x.^2 / (obj.Sigma^4)) - (1 / (obj.Sigma^2))) .* g_x;
                gdd_y = ((y.^2 / (obj.Sigma^4)) - (1 / (obj.Sigma^2))) .* g_y;
                
                % Check if aligned to horizontal axis
                if abs(cos(theta)) > 0.999
                    h = gdd_x * cos(theta)^2;
                    v = g_y';
                else
                    h = g_x;
                    v = gdd_y' * sin(theta)^2;
                end
            end
        end
        
        function tf = isSeparable(obj)
            % ISSEPARABLE Returns true if the steering angle aligns with the grid axes.
            % Oriented derivative filters at arbitrary angles are rank-2 and not separable.
            theta = deg2rad(obj.Angle);
            tf = (abs(sin(theta)) < 1e-4) || (abs(cos(theta)) < 1e-4);
        end
    end
    
    methods (Access = private)
        function computeWeights(obj)
            % Computes the 2D steered weight matrix from analytical derivatives.
            h_sz = obj.Size(2);
            v_sz = obj.Size(1);
            px = floor(h_sz / 2);
            py = floor(v_sz / 2);
            
            [X, Y] = meshgrid(-px:px, -py:py);
            
            % 2D Gaussian envelope
            G = exp(-(X.^2 + Y.^2) / (2 * obj.Sigma^2)) / (2 * pi * obj.Sigma^2);
            
            theta = deg2rad(obj.Angle);
            
            if obj.Order == 1
                % First-order partial derivatives
                Gx = -(X / (obj.Sigma^2)) .* G;
                Gy = -(Y / (obj.Sigma^2)) .* G;
                
                % Steerable combination
                obj.Weights = cos(theta) * Gx + sin(theta) * Gy;
            else
                % Second-order partial derivatives
                Gxx = ((X.^2 / (obj.Sigma^4)) - (1 / (obj.Sigma^2))) .* G;
                Gyy = ((Y.^2 / (obj.Sigma^4)) - (1 / (obj.Sigma^2))) .* G;
                Gxy = (X .* Y / (obj.Sigma^4)) .* G;
                
                % Steerable combination
                obj.Weights = cos(theta)^2 * Gxx + 2*cos(theta)*sin(theta) * Gxy + sin(theta)^2 * Gyy;
            end
        end
    end
end
