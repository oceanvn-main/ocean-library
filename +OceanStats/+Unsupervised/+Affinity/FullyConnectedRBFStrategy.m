classdef FullyConnectedRBFStrategy < OceanStats.Base.IAffinityStrategy
    % FULLYCONNECTEDRBFSTRATEGY Radial Basis Function affinity graph.
    %
    % Connects all points to all other points (fully connected), weighted by 
    % an RBF (Gaussian) kernel: W(i,j) = exp(-dist(i,j)^2 / (2*sigma^2)).
    
    properties
        Sigma (1,1) double {mustBePositive} = 1.0
        DistanceFcn = [] % (optional) function_handle: @(A,B) -> (N_A x N_B) squared distances
    end
    
    methods
        function obj = FullyConnectedRBFStrategy(opts)
            arguments
                opts.Sigma (1,1) double {mustBePositive} = 1.0
                opts.DistanceFcn = [] % function_handle
            end
            
            obj.Sigma = opts.Sigma;
            obj.DistanceFcn = opts.DistanceFcn;
        end
        
        function W = computeAffinity(obj, X)
            % 1. Compute pairwise squared distances
            if ~isempty(obj.DistanceFcn)
                D2 = obj.DistanceFcn(X, X);
            else
                % Highly optimized vectorized squared Euclidean distance
                D2 = sum(X.^2, 2) + sum(X.^2, 2)' - 2 * (X * X');
            end
            D2(D2 < 0) = 0; % Numerical stability
            
            % 2. Apply RBF Kernel
            W = exp(-D2 / (2 * obj.Sigma^2));
            
            % Zero out the diagonal (no self-loops in standard spectral clustering graph)
            N = size(X, 1);
            W(1:N+1:end) = 0;
            
            % Ensure perfect symmetry
            W = (W + W') / 2;
        end
    end
end
