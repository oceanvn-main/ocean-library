classdef HouseholderReflection < OceanMath.LinearAlgebra.Orthogonalization.IOrthogonalizer
    % HOUSEHOLDERREFLECTION Manual implementation of Householder QR.
    % 
    % The "gold standard" for numerical stability, guaranteeing machine
    % precision orthogonality through consecutive reflection matrices,
    % bypassing MATLAB's built-in qr() functionality natively.
    
    methods
        function [Q, R] = decompose(obj, A)
            [N, M] = size(A);
            Q = eye(N);
            R = A;
            
            % Thin QR decomposition
            num_reflections = min(N-1, M);
            
            for k = 1:num_reflections
                x = R(k:N, k);
                
                % Standard Householder vector calculation
                e1 = zeros(length(x), 1);
                e1(1) = 1;
                
                % Compute norm
                nrm_x = norm(x);
                if nrm_x < 1e-12
                    continue; % Already zeroed out
                end
                
                % Ensure numerical stability with sign
                sgn = sign(x(1));
                if sgn == 0; sgn = 1; end
                
                % Core reflection vector
                v = x + sgn * nrm_x * e1;
                v = v / norm(v);
                
                % Apply reflection to R
                R(k:N, k:M) = R(k:N, k:M) - 2 * v * (v' * R(k:N, k:M));
                
                % Accumulate Q (Q = Q1 * Q2 * ... * Q_m)
                Q(:, k:N) = Q(:, k:N) - 2 * (Q(:, k:N) * v) * v';
                
                % Explicitly zero out sub-diagonal for numeric stability
                if k < N
                    R(k+1:N, k) = 0;
                end
            end
            
            % Extract Thin QR matrices
            Q = Q(:, 1:M);
            R = R(1:M, :);
        end
        
        function [Q_new, R_new] = addColumns(obj, Q_old, R_old, H)
            % Householder decomposes remainders robustly, but we must
            % project H onto existing Q rigidly to avoid losing strict
            % orthogonality during incremental updating.
            [N, M] = size(Q_old);
            K = size(H, 2);
            
            R12 = zeros(M, K);
            H_perp = H;
            
            % 1st pass: Modified Gram-Schmidt
            for j = 1:K
                for i = 1:M
                    q_i = Q_old(:, i);
                    r_ij = q_i' * H_perp(:, j);
                    R12(i, j) = R12(i, j) + r_ij;
                    H_perp(:, j) = H_perp(:, j) - r_ij * q_i;
                end
            end
            
            % 2nd pass: Re-orthogonalization (Critical for stability)
            for j = 1:K
                for i = 1:M
                    q_i = Q_old(:, i);
                    r_ij = q_i' * H_perp(:, j);
                    R12(i, j) = R12(i, j) + r_ij;
                    H_perp(:, j) = H_perp(:, j) - r_ij * q_i;
                end
            end
            
            % Utilize our manual Householder decomposition on the orthogonal complement
            [Q2, R22] = obj.decompose(H_perp);
            
            Q_new = [Q_old, Q2];
            zeros_block = zeros(K, M);
            R_new = [R_old, R12; zeros_block, R22];
        end
        
        function [rss, q_new] = fastProject(obj, Q_old, h, res_old)
            % fastProject requires projecting a single vector instantly into space,
            % which is fundamentally a Gram-Schmidt problem (orthogonal projections),
            % as Householder builds full system reflectors.
            % We will use a highly stringent Re-orthogonalization MGS here
            % because it is mathematically equivalent for single vector additions,
            % but dramatically faster inside knot search loops.
            
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
            % For multi-response (N x K), q_new' * res_old is (1 x K). We want the sum of squares across K.
            explained_vec = q_new' * res_old;
            explained = sum(explained_vec.^2, 'all');
            
            rss = sum(res_old.^2, 'all') - explained;
            if rss < 0; rss = 0; end
        end
    end
end
