classdef MutualKNNStrategy < OceanStats.Base.IAffinityStrategy
    % MUTUALKNNSTRATEGY Mutual K-Nearest Neighbors affinity graph.
    %
    % Connects point i to j if i is among the K-nearest neighbors of j, 
    % OR vice-versa. Edge weights are unweighted (1) for connections and 0 otherwise.
    
    properties
        NumNeighbors (1,1) double {mustBePositive, mustBeInteger} = 10
        DistanceFcn = [] % (optional) function_handle: @(A,B) -> (N_A x N_B) squared distances
    end
    
    methods
        function obj = MutualKNNStrategy(opts)
            arguments
                opts.NumNeighbors (1,1) double {mustBePositive, mustBeInteger} = 10
                opts.DistanceFcn = [] % function_handle
            end
            
            obj.NumNeighbors = opts.NumNeighbors;
            obj.DistanceFcn = opts.DistanceFcn;
        end
        
        function W = computeAffinity(obj, X)
            N = size(X, 1);
            
            % 1. Compute pairwise squared distances
            if ~isempty(obj.DistanceFcn)
                D2 = obj.DistanceFcn(X, X);
            else
                % Highly optimized vectorized squared Euclidean distance
                D2 = sum(X.^2, 2) + sum(X.^2, 2)' - 2 * (X * X');
            end
            D2(D2 < 0) = 0; % Numerical stability
            
            % 2. Construct mutual k-NN similarity graph
            [~, sorted_indices] = sort(D2, 2);
            % Ignore self (index 1 is always the point itself since distance=0)
            k = min(obj.NumNeighbors, N - 1);
            k_indices = sorted_indices(:, 2:(k+1)); 
            
            W = zeros(N, N);
            for i = 1:N
                W(i, k_indices(i, :)) = 1; %Set 1 to the K nearest neighbors 
            end
            
            % Symmetric union to satisfy "or vice-versa"
            W = max(W, W'); 
        end
    end
end
