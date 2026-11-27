classdef (Abstract) IRBFKernel < handle
    % IRBFKERNEL  Abstract interface for swappable RBF kernels.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-30

    methods (Abstract)
        val = evaluate(obj, r)
    end
end
