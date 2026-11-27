classdef (Abstract) IFilterKernel < handle
    % IFILTERKERNEL  Abstract interface for 1D and 2D discrete filter kernels.
    %
    %   All discrete filter kernels used in signal and image processing (e.g. Box,
    %   Gaussian, Sobel) implement this uniform interface. It exposes the 2D matrix
    %   representation and encapsulates the pre-decomposed 1D horizontal and vertical
    %   kernels for separable filtering.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-15
    
    methods (Abstract)
        % GETMATRIX Returns the [H x W] double matrix representation of the kernel.
        K = getMatrix(obj)
        
        % GET1DDECOMPOSITION Returns the analytical 1D decompositions:
        %   h - [1 x W] double horizontal kernel vector
        %   v - [H x 1] double vertical kernel vector
        % Such that getMatrix() is numerically equal to v * h.
        [h, v] = get1DDecomposition(obj)
        
        % ISSEPARABLE Returns true if the kernel is separable (can be decomposed).
        tf = isSeparable(obj)
    end
end
