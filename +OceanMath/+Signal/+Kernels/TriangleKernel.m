classdef TriangleKernel < OceanMath.Signal.Kernels.IFilterKernel
    % TRIANGLEKERNEL  Represents a discrete bilinear / triangle filter kernel.
    %
    %   Signal Processing Intuition:
    %     This is a bilinear Low-Pass Filter (LPF). A triangle (tent) function in the
    %     spatial domain is equivalent to convolving a box filter with itself.
    %     Consequently, in the frequency domain, its frequency response is the square
    %     of a sinc function: H(w) = [sin(w * W / 4) / (w * W / 4)]^2.
    %     
    %     Squaring the sinc function yields two major improvements over a simple box filter:
    %       1. Steeper roll-off: Frequencies are attenuated twice as fast.
    %       2. Side-lobe suppression: The peak of the secondary lobes drops from roughly
    %          -13dB down to -26dB (from 13% of the main lobe height down to 1.7%).
    %          This drastically reduces side-lobe noise leakage and mitigates spatial
    %          ringing artifacts compared to a box filter.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-15
    
    properties (Access = private)
        Size (1,1) double = 3
    end
    
    methods
        function obj = TriangleKernel(size_val)
            % TRIANGLEKERNEL Construct a TriangleKernel.
            %   TriangleKernel(N) creates a symmetric N x N bilinear/triangle kernel.
            arguments
                size_val (1,1) double {mustBeInteger, mustBePositive} = 3
            end
            if mod(size_val, 2) == 0
                error('TriangleKernel:SizeMustBeOdd', 'Kernel size must be odd.');
            end
            obj.Size = size_val;
        end
        
        function K = getMatrix(obj)
            [h, v] = obj.get1DDecomposition();
            K = v * h;
        end
        
        function [h, v] = get1DDecomposition(obj)
            N = obj.Size;
            k1d = floor((N + 1) / 2) - abs(-(floor(N / 2)) : floor(N / 2));
            k1d = k1d / sum(k1d);
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
