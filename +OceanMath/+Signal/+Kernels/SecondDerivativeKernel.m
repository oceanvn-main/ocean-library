classdef SecondDerivativeKernel < OceanMath.Signal.Kernels.IFilterKernel
    % SECONDDERIVATIVEKERNEL  Represents a 2D second-order cross-derivative (corner) kernel.
    %
    %   Signal Processing Intuition:
    %     This is a second-order derivative High-Pass Filter (HPF). Taking a second-order
    %     derivative in the spatial domain corresponds to multiplying by (j * w)^2 = -w^2
    %     in the frequency domain. This completely blocks uniform (DC) regions and strongly
    %     amplifies high-frequency variations.
    %     
    %     Because it applies a second difference [1, -2, 1] in both horizontal and vertical
    %     directions, it is highly sensitive to features where the intensity gradient changes
    %     rapidly in both axes simultaneously. This makes it a classic operator for detecting:
    %       1. Corners and intersections.
    %       2. Fine spot textures and high-frequency noise.
    %     
    %     Since the sum of the kernel coefficients is exactly 0, uniform areas are mapped
    %     to a value of 0, while high-frequency features yield strong positive/negative peaks.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-15
    
    properties (Access = private)
        Size (1,1) double = 3
    end
    
    methods
        function obj = SecondDerivativeKernel(size_val)
            % SECONDDERIVATIVEKERNEL Construct a SecondDerivativeKernel.
            %   SecondDerivativeKernel(N) creates a symmetric N x N corner/cross-derivative kernel.
            arguments
                size_val (1,1) double {mustBeInteger, mustBePositive} = 3
            end
            if mod(size_val, 2) == 0
                error('SecondDerivativeKernel:SizeMustBeOdd', 'Kernel size must be odd.');
            end
            obj.Size = size_val;
        end
        
        function K = getMatrix(obj)
            [h, v] = obj.get1DDecomposition();
            K = v * h;
        end
        
        function [h, v] = get1DDecomposition(obj)
            N = obj.Size;
            if N == 3
                k1d = 1/2 * [1, -2, 1];
            else
                % Convolve [1, -2, 1] with binomial smoother of order N-3
                b = zeros(1, N - 2);
                for k = 0:(N-3)
                    b(k+1) = nchoosek(N-3, k);
                end
                k1d = conv([1, -2, 1], b);
                k1d = k1d / (2^(N-3) * 2);
            end
            h = k1d;
            v = k1d';
        end
        
        function tf = isSeparable(~)
            tf = true;
        end
        
        function s = getSize(obj)
            s = obj.Size;
        end
    end
end
