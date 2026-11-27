classdef ModifiedGramSchmidt < OceanMath.LinearAlgebra.Orthogonalization.IOrthogonalizer
    % MODIFIEDGRAMSCHMIDT Implementation of MGS with Re-orthogonalization.
    % 
    % Excellent for fast projections (adding single vectors) while maintaining
    % numerical stability via a second re-orthogonalization pass.
    
    methods
        function [Q, R] = decompose(obj, A)
            [N, M] = size(A);
            Q = zeros(N, M);
            R = zeros(M, M);
            
            for j = 1:M
                v = A(:, j);
                
                % First pass MGS
                for i = 1:j-1
                    R(i, j) = Q(:, i)' * v;
                    v = v - R(i, j) * Q(:, i);
                end
                
                % Second pass Re-orthogonalization
                for i = 1:j-1
                    correction = Q(:, i)' * v;
                    R(i, j) = R(i, j) + correction;
                    v = v - correction * Q(:, i);
                end
                
                R(j, j) = norm(v);
                if R(j, j) > 1e-12
                    Q(:, j) = v / R(j, j);
                else
                    Q(:, j) = zeros(N, 1);
                end
            end
        end
        
        function [Q_new, R_new] = addColumns(obj, Q_old, R_old, H)
            [N, M] = size(Q_old);
            K = size(H, 2);
            
            R12 = zeros(M, K);
            H_perp = H;
            
            % First Pass MGS
            for j = 1:K
                for i = 1:M
                    r_ij = Q_old(:, i)' * H_perp(:, j);
                    R12(i, j) = R12(i, j) + r_ij;
                    H_perp(:, j) = H_perp(:, j) - r_ij * Q_old(:, i);
                end
            end
            
            % Second Pass Re-orthogonalization
            for j = 1:K
                for i = 1:M
                    r_ij = Q_old(:, i)' * H_perp(:, j);
                    R12(i, j) = R12(i, j) + r_ij;
                    H_perp(:, j) = H_perp(:, j) - r_ij * Q_old(:, i);
                end
            end
            
            % Decompose remainder
            [Q2, R22] = obj.decompose(H_perp);
            
            Q_new = [Q_old, Q2];
            zeros_block = zeros(K, M);
            R_new = [R_old, R12; zeros_block, R22];
        end
        
        function [rss, q_new] = fastProject(obj, Q_old, h, res_old)
            q_new = h;
            M = size(Q_old, 2);
            
            % Pass 1
            for k = 1:M
                proj = Q_old(:, k)' * q_new;
                q_new = q_new - proj * Q_old(:, k);
            end
            
            % Pass 2 (Re-ortho)
            for k = 1:M
                proj = Q_old(:, k)' * q_new;
                q_new = q_new - proj * Q_old(:, k);
            end
            
            norm_q = norm(q_new);
            if norm_q < 1e-8
                rss = inf;
                q_new = zeros(size(h));
                return;
            end
            
            q_new = q_new / norm_q;
            explained = (q_new' * res_old)^2;
            rss = sum(res_old.^2) - explained;
            if rss < 0; rss = 0; end
        end
    end
end
