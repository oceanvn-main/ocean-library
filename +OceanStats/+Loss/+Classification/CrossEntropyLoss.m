classdef CrossEntropyLoss < OceanStats.Loss.ITreeLoss
    % CROSSENTROPYLOSS  Softmax Cross-Entropy loss for multi-class classification.
    %
    %   GENERALITY:
    %     This is the general K-class formulation. Binary classification is the
    %     special case K=2, which is handled automatically — no separate class needed.
    %
    %   CONVENTION:
    %     Y_true — N×1 integer class labels, 1-indexed (e.g. 1, 2, 3 for K=3)
    %     Y_pred — N×K matrix of raw logit scores (one column per class).
    %              NOT probabilities. Softmax is applied internally.
    %
    %   FORMULA (per sample i):
    %     p_k     = exp(z_k) / sum_j(exp(z_j))          [softmax probability, class k]
    %     L_i     = -log(p_{y_i})                        [cross-entropy for true class]
    %
    %   GRADIENT (N×K matrix):
    %     g_{i,k} = p_{i,k} - 1[k == y_i]               [prob minus one-hot indicator]
    %     For the true class: gradient = p - 1
    %     For all other classes: gradient = p
    %
    %   HESSIAN (N×K matrix, diagonal entries of the full K×K Hessian per sample):
    %     h_{i,k} = p_{i,k} * (1 - p_{i,k})             [Bernoulli variance per class]
    %     Note: Full Hessian per sample is diag(p) - p*p' but diagonal suffices
    %     for most second-order tree methods (XGBoost-style).
    %
    %   NUMERICAL STABILITY:
    %     Uses the log-sum-exp trick to prevent overflow in exp():
    %       log Z = max(z) + log(sum(exp(z - max(z))))
    %
    %   BINARY CASE (K=2):
    %     Automatically reduces to sigmoid cross-entropy when K=2.
    %     Y_true should still be {1, 2}, not {0, 1}.
    %
    %   USAGE:
    %     loss = OceanStats.Loss.Classification.CrossEntropyLoss();
    %
    %     % Multi-class (K=3)
    %     Y_true = [1; 3; 2; 1];           % N×1, class labels 1..K
    %     Y_pred = randn(4, 3);             % N×K, raw logits
    %     L = loss.forward(Y_true, Y_pred);
    %     G = loss.gradient(Y_true, Y_pred);    % N×K
    %     H = loss.hessian(Y_true, Y_pred);     % N×K
    %
    %     % Binary case (K=2)
    %     Y_true = [1; 2; 1; 2];
    %     Y_pred = randn(4, 2);
    %
    %   See also: OceanStats.Loss.Classification.ExponentialLoss
    %             OceanStats.Loss.Classification.HingeLoss
    %             OceanStats.Loss.ILoss

    methods
        function L = forward(obj, Y_true, Y_pred)
            % FORWARD  Compute mean softmax cross-entropy loss.
            %
            %   Inputs:
            %     Y_true — N×1 integer labels in 1..K
            %     Y_pred — N×K raw logits
            %
            %   Output:
            %     L — scalar mean loss
            obj.checkInputs_(Y_true, Y_pred);
            P  = obj.softmax_(Y_pred);    % N×K probabilities
            N  = size(Y_pred, 1);

            % Gather the probability assigned to the true class for each sample
            idx    = sub2ind(size(P), (1:N)', Y_true); % N×1 linear indices
            p_true = P(idx);                            % N×1
            p_true = max(p_true, eps);                  % guard log(0)
            L      = -mean(log(p_true));
        end

        function G = gradient(obj, Y_true, Y_pred)
            % GRADIENT  First derivative of loss w.r.t. each logit.
            %
            %   Output:
            %     G — N×K matrix  (p_k - 1[k == y_i])
            obj.checkInputs_(Y_true, Y_pred);
            [N, K] = size(Y_pred);
            P      = obj.softmax_(Y_pred);   % N×K

            % One-hot encode Y_true: indicator(i, k) = 1 iff Y_true(i) == k
            oneHot    = zeros(N, K);
            idx       = sub2ind([N, K], (1:N)', Y_true);
            oneHot(idx) = 1;

            G = P - oneHot;    % N×K — the elegant softmax gradient result
        end

        function H = hessian(obj, Y_true, Y_pred)
            % HESSIAN  Diagonal second derivative w.r.t. each logit.
            %
            %   Output:
            %     H — N×K matrix  (p_k * (1 - p_k) per class)
            %
            %   Note: The full per-sample Hessian is diag(p) - p*p' (K×K).
            %   We return only the diagonal, which is sufficient for
            %   XGBoost-style independent leaf updates per class.
            obj.checkInputs_(Y_true, Y_pred);
            P = obj.softmax_(Y_pred);    % N×K
            H = P .* (1 - P);                        % N×K diagonal Hessian
        end

        function n = name(obj, Y_pred)
            if nargin > 1
                K = size(Y_pred, 2);
                n = sprintf("Cross-Entropy (%d-class)", K);
            else
                n = "Cross-Entropy (multi-class)";
            end
        end

        function gamma = initPrediction(~, Y_true)
            % F_0 = log-odds of prior class probability.
            % For multi-class: F_0(k) = log(p_k / (1 - p_k)) where p_k = N_k / N.
            % Returns K×1 vector (one logit per class).
            Y_true  = Y_true(:);
            K       = max(Y_true);
            counts  = histc(Y_true, 1:K);  %#ok<HISTC>
            p       = max(counts, 1) / numel(Y_true);
            gamma   = log(p ./ (1 - p + eps));   % K×1 log-odds
            gamma   = gamma(:);
        end

        function gamma = computeLeafConstant(obj, Y_true, F_prev, leafMask)
            % Newton-step leaf constant for multi-class cross-entropy.
            %
            % Formula: gamma = sum(y - p) / sum(p * (1 - p))
            % where p = softmax(F_prev).
            
            if isvector(F_prev)
                % Binary (Single column logit) -> Sigmoid
                p     = 1 ./ (1 + exp(-F_prev(leafMask)));
                y     = Y_true(leafMask) == 2; % Assume class 2 is positive
                r     = y - p;
                gamma = sum(r) / (sum(p .* (1 - p)) + eps);
            else
                % Multi-class (N x K logits) -> Softmax
                [N_full, K] = size(F_prev);
                P_leaf = obj.softmax_(F_prev(leafMask, :)); % N_leaf x K
                
                % Build one-hot for leaf samples
                y_leaf = Y_true(leafMask);
                N_leaf = numel(y_leaf);
                oneHot = zeros(N_leaf, K);
                idx    = sub2ind([N_leaf, K], (1:N_leaf)', y_leaf(:));
                oneHot(idx) = 1;
                
                % Newton Step: Sum(Grad) / Sum(Hess)
                % Gradient = y - p, Hessian = p*(1-p)
                r_leaf = oneHot - P_leaf;
                h_leaf = P_leaf .* (1 - P_leaf);
                
                gamma  = sum(r_leaf, 1) ./ (sum(h_leaf, 1) + eps);
                gamma  = gamma(:);  % K×1
            end
        end
    end

    methods (Access = private)
        function checkInputs_(~, Y_true, Y_pred)
            % Validate dimensions: Y_true N×1, Y_pred N×K
            N = size(Y_pred, 1);
            K = size(Y_pred, 2);
            if ~isvector(Y_true) || numel(Y_true) ~= N
                error('OceanStats:Loss:CrossEntropyLoss:DimMismatch', ...
                    'Y_true must be N×1 with N matching rows of Y_pred (N=%d).', N);
            end
            Y_true = Y_true(:);
            if any(Y_true < 1) || any(Y_true > K) || any(Y_true ~= floor(Y_true))
                error('OceanStats:Loss:CrossEntropyLoss:InvalidLabels', ...
                    'Y_true must be integer class labels in 1..K (K=%d).', K);
            end
        end
    end

    methods (Static, Access = private)
        function P = softmax_(Z)
            % SOFTMAX_  Numerically stable softmax over rows.
            %
            %   Uses log-sum-exp trick: subtract row max before exp()
            %   to prevent overflow without changing the result.
            %
            %   Input:  Z — N×K raw logits
            %   Output: P — N×K probabilities (rows sum to 1)
            Z_shift = Z - max(Z, [], 2);       % N×K, row-wise max subtracted
            E       = exp(Z_shift);            % N×K
            P       = E ./ sum(E, 2);          % N×K, each row sums to 1
        end
    end
end
