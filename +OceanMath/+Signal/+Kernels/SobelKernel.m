classdef SobelKernel < OceanMath.Signal.Kernels.IFilterKernel
    % SOBELKERNEL  Represents a discrete Sobel derivative kernel.
    %
    %   Signal Processing Intuition:
    %     This is a directional Band-Pass / Derivative Filter. Taking a derivative in the
    %     spatial domain corresponds to multiplying by j * w in the frequency domain. This
    %     acts as a High-Pass Filter (HPF) that amplifies edges, details, and noise.
    %     
    %     To make the derivative operator robust against noise, the Sobel filter combines:
    %       1. A derivative filter (HPF) in the direction of interest (e.g. horizontal).
    %       2. A binomial smoothing filter (LPF) in the orthogonal direction (e.g. vertical).
    %     
    %     By combining LPF and HPF properties, the Sobel kernel attenuates high-frequency noise
    %     while extracting clean, directional image gradients (edges).
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-15
    
    properties (Access = private)
        Direction (1,1) string = "horizontal"
        Size      (1,1) double = 3
    end
    
    methods
        function obj = SobelKernel(options)
            % SOBELKERNEL Construct a SobelKernel.
            %   SobelKernel('Direction', 'horizontal'|'vertical', 'Size', N)
            arguments
                options.Direction (1,1) string {mustBeMember(options.Direction, {'horizontal', 'vertical'})} = "horizontal"
                options.Size      (1,1) double {mustBeInteger, mustBePositive} = 3
            end
            if mod(options.Size, 2) == 0
                error('SobelKernel:SizeMustBeOdd', 'Kernel size must be odd.');
            end
            obj.Direction = options.Direction;
            obj.Size = options.Size;
        end
        
        function K = getMatrix(obj)
            [h, v] = obj.get1DDecomposition();
            K = v * h;
        end
        
        function [h, v] = get1DDecomposition(obj)
            N = obj.Size;
            % 1. Smoothing vector of length N (binomial coefficients of order N-1)
            s = zeros(1, N);
            for k = 0:(N-1)
                s(k+1) = nchoosek(N-1, k);
            end
            s = s / sum(s);
            
            % 2. Derivative vector of length N (central difference convolved with binomial smoother of order N-3)
            if N == 3
                d = 1/2 * [-1, 0, 1];
            else
                % Binomial of order N-3
                b = zeros(1, N - 2);
                for k = 0:(N-3)
                    b(k+1) = nchoosek(N-3, k);
                end
                d = conv([-1, 0, 1], b);
                % Normalize derivative scale
                d = d / (2^(N-3) * 2);
            end
            
            if strcmp(obj.Direction, "horizontal")
                h = d;
                v = s';
            else
                h = s;
                v = d';
            end
        end
        
        function tf = isSeparable(~)
            tf = true;
        end
        
        function d = getDirection(obj)
            d = obj.Direction;
        end
        
        function s = getSize(obj)
            s = obj.Size;
        end
    end
end
