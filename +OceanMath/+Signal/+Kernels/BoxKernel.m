classdef BoxKernel < OceanMath.Signal.Kernels.IFilterKernel
    % BOXKERNEL  Represents a discrete average box blur kernel.
    %
    %   Signal Processing Intuition:
    %     This is a simple local averaging Low-Pass Filter (LPF). In the frequency
    %     domain, a rectangular box in the spatial domain corresponds to a sinc function
    %     frequency response: H(w) = sin(w * W / 2) / (w * W / 2).
    %     
    %     Because the sinc function decays slowly (proportionally to 1/w) and has
    %     significant secondary lobes (oscillations), the box filter suffers from:
    %       1. Side-lobe leakage: High frequencies corresponding to the peaks of the
    %          side lobes are passed through rather than blocked.
    %       2. Gibbs phenomenon: Sharp spatial cut-offs lead to ringing and blocky
    %          halo artifacts around edges in the output image.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-15
    
    properties (Access = private)
        Size (1,2) double {mustBeInteger, mustBePositive} = [5, 5]
    end
    
    methods
        function obj = BoxKernel(size_val)
            % BOXKERNEL Construct a BoxKernel.
            %   BoxKernel(N) creates an N x N box kernel.
            %   BoxKernel([H, W]) creates an H x W box kernel.
            arguments
                size_val (1,:) double {mustBeInteger, mustBePositive} = 5
            end
            if isscalar(size_val)
                obj.Size = [size_val, size_val];
            else
                obj.Size = size_val(1:2);
            end
        end
        
        function K = getMatrix(obj)
            H = obj.Size(1);
            W = obj.Size(2);
            K = 1 / (H * W) * ones(H, W);
        end
        
        function [h, v] = get1DDecomposition(obj)
            H = obj.Size(1);
            W = obj.Size(2);
            h = 1 / W * ones(1, W); 
            v = 1 / H * ones(H, 1);
        end
        
        function tf = isSeparable(~)
            tf = true;
        end
        
        function s = getSize(obj)
            s = obj.Size;
        end
    end
end
