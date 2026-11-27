classdef SpectralClustering < handle
    % SPECTRALCLUSTERING Unnormalized spectral clustering using k-nearest neighbors graph.
    % 
    % Implements the algorithm described in ESL Section 14.5.3.
    %
    % USAGE:
    %   sc = OceanStats.Unsupervised.SpectralClustering();
    %   labels = sc.fit(X, K, 'NumNeighbors', 10);
    
    properties
        Labels
        Eigenvectors      % The extracted eigenvectors used for clustering (N x m)
        Eigenvalues       % All computed eigenvalues (sorted)
        AllEigenvectors   % All computed eigenvectors
        AdjacencyMatrix   % The similarity graph W
        Laplacian         % Unnormalized graph Laplacian L = G - W
    end
    
    methods
        function labels = fit(obj, X, K, opts)
            % FIT Cluster the data X into K groups.
            arguments
                obj
                X                   double
                K                   (1,1) double {mustBePositive, mustBeInteger}
                opts.NumEigenvectors double = [] % Default is K (usually number of clusters minus 1, or just K)
                opts.AffinityStrategy = [] % Default will be MutualKNNStrategy('NumNeighbors', 10)
                opts.LaplacianMode  (1,1) string {mustBeMember(opts.LaplacianMode, ["Unnormalized", "Symmetric", "RandomWalk"])} = "Unnormalized"
            end
            
            % According to standard unnormalized spectral clustering (e.g. von Luxburg tutorial),
            % we often use K eigenvectors. The ESL figure uses 2 eigenvectors for K=3. 
            % So m = K-1 is also common. We default to K.
            %Indicator vectors
            if isempty(opts.NumEigenvectors)
                m = K; 
            else
                m = opts.NumEigenvectors;
            end
            
            N = size(X, 1);
            
            % 1. Construct affinity graph using the injected strategy
            if isempty(opts.AffinityStrategy)
                affinity = OceanStats.Unsupervised.Affinity.MutualKNNStrategy('NumNeighbors', 10);
            else
                affinity = opts.AffinityStrategy;
            end
            
            W = affinity.computeAffinity(X);
            obj.AdjacencyMatrix = W;
            
            % 3. Graph Laplacian
            d = sum(W, 2); % Degree vector
            G = diag(d);   % Degree matrix
            
            switch opts.LaplacianMode
                case "Unnormalized"
                    L = G - W;
                case "Symmetric" %Fix problem of Randomwalk
                    % L_sym = I - D^{-1/2} W D^{-1/2}
                    d_inv_sqrt = 1 ./ sqrt(d);
                    d_inv_sqrt(isinf(d_inv_sqrt)) = 0; % Handle isolated vertices
                    D_inv_sqrt = diag(d_inv_sqrt);
                    L = eye(N) - D_inv_sqrt * W * D_inv_sqrt;

                case "RandomWalk" % This break symmetric due to matrix multiplication is not commutative
                    % L_rw = I - D^{-1} W #Notice D^-1 W is not symmetric
                    d_inv = 1 ./ d;
                    d_inv(isinf(d_inv)) = 0;
                    D_inv = diag(d_inv);
                    L = eye(N) - D_inv * W;
            end
            
            obj.Laplacian = L;
            
            % 4. Eigenvalue Decomposition
            % Using eig() for full decomposition as N is typically small in these experiments.
            % It avoids 'smallestabs' singular warnings in eigs.
            [V, D] = eig(L);
            eig_vals = diag(D);
            
            % Sort eigenvalues and eigenvectors
            [eig_vals, sort_idx] = sort(eig_vals);
            
            Z_all = V(:, sort_idx); %Sort eigen vectors
            
            obj.Eigenvalues = eig_vals;
            obj.AllEigenvectors = Z_all;
            
            % Extract the m smallest eigenvectors (ignoring the first trivial constant one)
            % ESL: "Spectral clustering finds the m eigenvectors Z_{N x m} corresponding 
            % to the m smallest eigenvalues of L (ignoring the trivial constant eigenvector)."
            Z = Z_all(:, 2:(m+1));
            obj.Eigenvectors = Z;
            
            % 5. Cluster the rows of Z using standard K-means
            km = OceanStats.Unsupervised.KMeans();
            obj.Labels = km.fit(Z, K, 'NumRestarts', 10);
            labels = obj.Labels;
        end
    end
end
