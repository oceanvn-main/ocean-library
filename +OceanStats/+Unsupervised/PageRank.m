classdef PageRank < handle
    % PAGERANK Compute PageRank solutions by the power method.
    % 
    % Implements the PageRank algorithm as described in ESL Section 14.5.3, 
    % formulations (14.107) - (14.110).
    %
    % The core recurrence is:
    %   p_i = (1 - d) + d * sum_{j=1}^N (L_{ij} / c_j) * p_j
    %
    % In matrix notation (14.108 / 14.109):
    %   p = [(1 - d) * e * e^T / N + d * L * D_c^{-1}] * p
    %   p = A * p
    % 
    % where L_{ij} = 1 if page j points to page i.
    % c_j = sum_i L_{ij} is the number of out-links of page j.
    %
    % USAGE:
    %   pr = OceanStats.Unsupervised.PageRank();
    %   p = pr.fit(L, 'DampingFactor', 0.85);
    %
    % Note: The sum of the PageRanks p_i is N (the number of web pages).
    
    properties (SetAccess = private)
        PageRanks       % (N x 1) vector of final PageRank scores
        AdjacencyMatrix % L matrix used for computation
        DampingFactor   % The used d parameter
        NumIterations   % Actual iterations taken to converge
    end
    
    methods
        function p = fit(obj, L, opts)
            % FIT Compute the PageRank given link matrix L.
            %
            % INPUTS:
            %   L - (N x N) double matrix where L(i,j) = 1 if j -> i
            %       (This is the transpose of the standard adjacency matrix A)
            %
            % Name-Value Options:
            %   DampingFactor - (scalar) constant d. Default: 0.85
            %   MaxIter       - (scalar) maximum power iterations. Default: 1000
            %   Tol           - (scalar) convergence tolerance. Default: 1e-6
            %
            % OUTPUTS:
            %   p - (N x 1) PageRank vector, summing to N.
            
            arguments
                obj
                L               (:,:) double
                opts.DampingFactor (1,1) double {mustBeGreaterThan(opts.DampingFactor, 0), mustBeLessThan(opts.DampingFactor, 1)} = 0.85
                opts.MaxIter       (1,1) double {mustBePositive, mustBeInteger} = 1000
                opts.Tol           (1,1) double {mustBePositive} = 1e-6
            end
            
            [N, cols] = size(L);
            if N ~= cols
                error('OceanStats:Unsupervised:PageRank:NotSquare', 'Link matrix L must be square.');
            end
            
            d = opts.DampingFactor;
            obj.AdjacencyMatrix = L;
            obj.DampingFactor = d;
            
            % Compute out-links c_j
            c = sum(L, 1)'; % Sum over rows (i), resulting in N x 1 vector of column sums
            
            % Handle dangling nodes (c_j = 0). 
            % In ESL 14.22 Figure 14.47 there are no dangling nodes, but robust code should handle it.
            % A common convention: dangling nodes teleport uniformly to all nodes.
            danglingMask = (c == 0);
            if any(danglingMask)
                % Effectively replace column j with 1/N
                % This makes c_j = 1 and L(:, j) = 1/N
                L(:, danglingMask) = 1/N;
                c(danglingMask) = 1; % Normalization will be handled by D_c^{-1}
            end
            
            % Construct D_c^{-1}
            D_c_inv = diag(1 ./ c);
            
            % Construct the Markov matrix A (Formulation 14.109)
            % A = (1 - d) * (e * e^T) / N + d * L * D_c^{-1}
            e = ones(N, 1);
            A = ((1 - d) / N) * (e * e') + d * L * D_c_inv;
            
            % Power method (Formulation 14.110)
            % Initialize p_0
            p_k = ones(N, 1); % sum is N
            
            for iter = 1:opts.MaxIter
                p_next = A * p_k;
                
                % Normalize so that e^T * p_next = N
                sum_p = sum(p_next);
                p_next = N * (p_next / sum_p);
                
                % Check convergence
                if norm(p_next - p_k, inf) < opts.Tol
                    obj.NumIterations = iter;
                    p_k = p_next;
                    break;
                end
                
                p_k = p_next;
            end
            
            if iter == opts.MaxIter
                warning('OceanStats:Unsupervised:PageRank:NotConverged', ...
                    'Power iteration did not converge within %d iterations.', opts.MaxIter);
                obj.NumIterations = iter;
            end
            
            obj.PageRanks = p_k;
            p = p_k;
        end
    end
end
