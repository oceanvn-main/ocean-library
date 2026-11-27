classdef MulticlassDevianceLoss < OceanStats.Loss.ITreeLoss
    % MULTICLASSDEVIANCELOSS  Multinomial Deviance (Softmax Cross-Entropy) for MART.
    %
    %   This is the loss function used by Friedman's MART algorithm for
    %   K-class classification (ESL §10.10, Algorithm 10.4).
    %
    %   CONVENTION:
    %     Y_true — N×1 integer class labels, 1-indexed (1..K)
    %     Y_pred — N×K matrix of raw additive scores F_k(x)
    %              (NOT probabilities). Softmax applied internally.
    %
    %   GRADIENT (ILoss convention = positive gradient):
    %     G_{i,k} = p_{i,k} - 1[y_i == k]   (softmax minus one-hot)
    %     GBM negates this to get MART pseudo-residuals: r = y - p.
    %
    %   LEAF CONSTANT (Friedman 2001, multi-class Newton approximation):
    %     gamma_k = ((K-1)/K) * sum(r_k) / sum(|r_k|*(1-|r_k|))
    %
    %   See also: OceanStats.Loss.ILoss

    methods
        function L = forward(~, Y_true, Y_pred)
            % FORWARD  Mean multinomial deviance = 2 × cross-entropy.
            Z      = Y_pred - max(Y_pred, [], 2);
            E      = exp(Z);
            P      = E ./ sum(E, 2);             % N×K softmax
            N      = size(Y_pred, 1);
            idx    = sub2ind(size(P), (1:N)', Y_true(:));
            p_true = max(P(idx), eps);
            L      = 2 * (-mean(log(p_true)));
        end

        function G = gradient(~, Y_true, Y_pred)
            % GRADIENT  Positive gradient of deviance: G_{i,k} = p_k - 1[y==k].
            % GBM negates this to get pseudo-residuals (y - p).
            Z      = Y_pred - max(Y_pred, [], 2);
            E      = exp(Z);
            P      = E ./ sum(E, 2);             % N×K softmax
            [N, K] = size(Y_pred);
            oneHot = zeros(N, K);
            idx    = sub2ind([N, K], (1:N)', Y_true(:));
            oneHot(idx) = 1;
            G = P - oneHot;                      % N×K positive gradient
        end

        function H = hessian(~, ~, Y_pred)
            % HESSIAN  Diagonal: H_{i,k} = p_k*(1 - p_k).
            Z = Y_pred - max(Y_pred, [], 2);
            E = exp(Z);
            P = E ./ sum(E, 2);
            H = P .* (1 - P);                    % N×K
        end

        function n = name(~)
            n = "Multinomial Deviance";
        end

        function gamma = initPrediction(~, Y_true)
            % F_0(k) = log(N_k / N)  [log prior class probability]. Returns K×1.
            Y_true = Y_true(:);
            K      = max(Y_true);
            counts = histc(Y_true, 1:K);  %#ok<HISTC>
            p      = max(counts, 1) / numel(Y_true);
            gamma  = log(p(:));                  % K×1
        end

        function gamma = computeLeafConstant(~, Y_true, F_prev, leafMask)
            % Friedman (2001) multi-class Newton step for leaf constants.
            %
            % EXACT FORMULA (ESL Eq 10.58):
            %   gamma_k = (K-1)/K * ( gamma_raw_k - (1/K) * sum_l(gamma_raw_l) )
            %
            % where gamma_raw_k = sum(r_ik) / sum(|r_ik|*(1-|r_ik|))
            %
            % The mean-centering term -(1/K)*sum_l enforces zero-sum symmetry
            % across the K leaf constants, keeping softmax probabilities calibrated.
            % Omitting it (the previous bug) introduced systematic drift in the
            % scores between classes.
            K      = size(F_prev, 2);
            F_leaf = F_prev(leafMask, :);

            % Softmax for leaf samples (log-sum-exp stabilised)
            Z_leaf  = F_leaf - max(F_leaf, [], 2);
            P_leaf  = exp(Z_leaf) ./ sum(exp(Z_leaf), 2);    % n_leaf × K

            % One-hot for leaf samples
            y_leaf  = Y_true(leafMask);
            n_leaf  = numel(y_leaf);
            oneHot  = zeros(n_leaf, K);
            idx     = sub2ind([n_leaf K], (1:n_leaf)', y_leaf(:));
            oneHot(idx) = 1;

            % Pseudo-residuals in leaf:  r = y - p    (n_leaf × K)
            r    = oneHot - P_leaf;
            rAbs = abs(r);

            % Step 1: uncentered Newton step  (1×K)
            num       = sum(r,    1);
            denom     = sum(rAbs .* (1 - rAbs), 1) + eps;
            gamma_raw = num ./ denom;                         % 1×K (no scale yet)

            % Step 2: mean-center across classes  (ESL Eq 10.58)
            %   gamma_k_centered = gamma_raw_k - (1/K) * sum_l(gamma_raw_l)
            gamma_centered = gamma_raw - mean(gamma_raw);     % 1×K, sums to ~0

            % Step 3: apply (K-1)/K shrinkage factor
            scale = (K - 1) / K;
            gamma = (scale * gamma_centered)';                % K×1
        end
    end
end
