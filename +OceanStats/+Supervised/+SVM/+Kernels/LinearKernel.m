classdef LinearKernel < OceanStats.Supervised.SVM.Base.ISVMKernel
    % LINEARKERNEL  Linear (dot-product) kernel: K(x,z) = x . z'
    %
    %   Equivalent to no kernel transform. Produces a linear decision
    %   boundary in the original feature space. Fastest kernel; use when
    %   data is linearly separable or when d >> N.
    %
    %   ESL Reference: §12.3.1
    %   Author: Ocean Library | 2026-03-12

    methods
        function K = compute(~, X1, X2)
            % COMPUTE  K = X1 * X2'   (n1 x n2)
            arguments
                ~
                X1  (:,:) double
                X2  (:,:) double
            end
            K = X1 * X2';
        end

        function s = name(~)
            s = 'Linear';
        end
    end
end
