classdef CustomKernel < OceanMath.Signal.Kernels.IFilterKernel
    % CUSTOMKERNEL  Wraps any raw numeric matrix as an IFilterKernel.
    %
    %   This class acts as an adapter, translating raw numeric matrices/vectors
    %   into the polymorphic IFilterKernel interface. It computes the 1D
    %   separable decomposition dynamically using Singular Value Decomposition (SVD).
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-19
    
    properties (Access = private)
        Matrix (:,:) double
    end
    
    methods
        function obj = CustomKernel(matrix_val)
            % CUSTOMKERNEL Construct a CustomKernel object.
            arguments
                matrix_val (:,:) double
            end
            obj.Matrix = matrix_val;
        end
        
        function K = getMatrix(obj)
            % GETMATRIX Returns the raw filter coefficients.
            K = obj.Matrix;
        end
        
        function [h, v] = get1DDecomposition(obj)
            % GET1DDECOMPOSITION Performs SVD to find horizontal and vertical 1D filters.
            [H, W] = size(obj.Matrix);
            if H == 1 || W == 1
                % 1D vector kernels are trivially separable
                h = obj.Matrix(:)';
                v = 1.0;
            else
                [U, S, V] = svd(obj.Matrix);
                s = diag(S);
                v = U(:, 1) * sqrt(s(1));
                h = V(:, 1)' * sqrt(s(1));
            end
        end
        
        function tf = isSeparable(obj)
            % ISSEPARABLE Checks if the singular values indicate a rank-1 matrix.
            [H, W] = size(obj.Matrix);
            if H == 1 || W == 1
                tf = true;
            else
                [~, S, ~] = svd(obj.Matrix);
                s = diag(S);
                tf = (length(s) <= 1) || (s(2) < 1e-12 * s(1));
            end
        end
    end
end
