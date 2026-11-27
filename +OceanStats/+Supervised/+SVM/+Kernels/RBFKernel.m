classdef RBFKernel < OceanStats.Supervised.SVM.Base.ISVMKernel
    % RBFKERNEL  Radial Basis Function (Gaussian) kernel.
    %
    %   K(x, z) = exp( -Gamma * ||x - z||^2 )
    %
    %   Gamma controls the width of the Gaussian. Large Gamma → narrow
    %   bell → complex (high-variance) boundary. Small Gamma → wide bell
    %   → smooth boundary (high bias).
    %
    %   Rule of thumb: Gamma = 1 / (2 * sigma^2).
    %
    %   ESL Reference: §12.3.2
    %   Author: Ocean Library | 2026-03-12

    properties
        Gamma (1,1) double {mustBePositive} = 1.0
    end

    methods
        function obj = RBFKernel(options)
            arguments
                options.Gamma (1,1) double {mustBePositive} = 1.0
            end
            obj.Gamma = options.Gamma;
        end

        function K = compute(obj, X1, X2)
            % COMPUTE  K(i,j) = exp(-Gamma * ||X1(i,:) - X2(j,:)||^2)
            %
            %   Uses the identity:
            %   ||x-z||^2 = ||x||^2 + ||z||^2 - 2*x.z'
            %   for O(n1*n2*d) computation without explicit difference loops.
            arguments
                obj
                X1  (:,:) double
                X2  (:,:) double
            end
            sq1 = sum(X1.^2, 2);          % (n1 x 1)
            sq2 = sum(X2.^2, 2);          % (n2 x 1)
            D2  = sq1 + sq2' - 2*(X1*X2'); % (n1 x n2) squared distances
            D2  = max(D2, 0);              % numerical guard against tiny negatives
            K   = exp(-obj.Gamma .* D2);
        end

        function s = name(obj)
            s = sprintf('RBF(gamma=%.4g)', obj.Gamma);
        end
    end
end
