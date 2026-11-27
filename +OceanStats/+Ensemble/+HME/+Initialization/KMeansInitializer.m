classdef KMeansInitializer < OceanStats.Ensemble.HME.IInitializer
    % KMEANSINITIALIZER Strategy for territorial expert/gate alignment using K-Means.
    
    properties
        NumRestarts = 5
        MaxKMeansIter = 50
    end
    
    methods
        function initialize(obj, model, X, Y)
            leaves = model.getLeaves();
            num_experts = length(leaves);
            
            fprintf('  Initialization Strategy: Centroid-Ordered K-Means (%d segments)...\n', num_experts);
            
            % 1. Extract features (skip intercept for clustering)
            featX = X;
            if size(X, 2) > 1 && all(X(:, 1) == 1)
                featX = X(:, 2:end);
            end
            
            % 2. Cluster the data in feature space
            kmeans = OceanStats.Unsupervised.KMeans();
            % Note: In the future, we could pass KMeans options here if needed.
            labels_raw = kmeans.fit(featX, num_experts);
            centroids = kmeans.Centroids;
            
            % 3. SORT CLUSTERS BY FEATURE CENTROID (Important for Hierarchical Order)
            % This ensures Leaf 1 gets Cluster 1 (the one furthest left in X-space).
            [~, order] = sort(centroids(:, 1)); 
            
            labels = zeros(size(labels_raw));
            for i = 1:num_experts
                labels(labels_raw == order(i)) = i;
            end
            
            % 4. Recursive Warmup (Experts + Gates)
            model.warmup(X, Y, labels);
        end
    end
end
