classdef (Abstract) ISVMKernel < handle
    % ISVMKERNEL  Abstract kernel interface for Support Vector Machines.
    %
    %   All SVM kernels implement this interface so that SVMClassifier and
    %   SVR remain completely decoupled from the kernel choice.
    %
    %   Convention
    %   ----------
    %   X1 : (n1 x d) double  — first set of row-vectors
    %   X2 : (n2 x d) double  — second set of row-vectors
    %   K  : (n1 x n2) double — kernel (Gram) matrix; K(i,j) = k(x1_i, x2_j)
    %
    %   ESL Reference: §12.3 — Kernel SVMs
    %   Author: Ocean Library | 2026-03-12

    methods (Abstract)
        % COMPUTE  Evaluate the kernel matrix between two data matrices.
        K = compute(obj, X1, X2)

        % NAME  Human-readable kernel name (used for display only).
        s = name(obj)
    end

    methods
        function describe(obj)
            % DESCRIBE  Print a one-line summary.
            fprintf('[%s]\n', obj.name());
        end
    end
end
