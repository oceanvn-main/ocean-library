classdef CTCLoss < OceanDL.Autograd.Function
    % CTCLOSS  Connectionist Temporal Classification (Graves, Fernandez, Gomez
    %   & Schmidhuber 2006): train a per-step classifier on SEQUENCE labels
    %   without knowing which input step produces which label.
    %
    %   INPUT: logits (K x N x T) -- one score vector per sample and time step
    %   ("CBT" layout); class Blank (default 1) means "no label here".
    %   LABELS: targets{n} = [l_1 ... l_U] (classes other than Blank), plus the
    %   number of valid input steps T_n per sample (default T).
    %
    %   p(l | x) = sum over every ALIGNMENT pi (one class per step) that
    %   collapses to l -- "merge repeats, then drop blanks": e.g. for l = "1 1 2"
    %   the path  (-, 1, 1, -, 1, 2, 2)  -> 1 - 1 2  -> "1 1 2".
    %   L = -log p(l | x).  There are exponentially many paths; dynamic
    %   programming over the EXTENDED label l' = (b, l_1, b, l_2, ..., b, l_U, b)
    %   (S = 2U + 1 states) makes it O(T S):
    %
    %   FORWARD (alpha_t(s) = mass of path prefixes ending in state s at t):
    %     alpha_1(1) = y_b(1),  alpha_1(2) = y_{l_1}(1)
    %     alpha_t(s) = [alpha_{t-1}(s) + alpha_{t-1}(s-1) + skip(s) alpha_{t-1}(s-2)] y_{l'_s}(t)
    %     skip(s) = 1 iff l'_s is a label and l'_s ~= l'_{s-2}  (a blank may be
    %     jumped only between DIFFERENT labels -- repeats need a blank between)
    %     p = alpha_T(S) + alpha_T(S-1)
    %   BACKWARD (beta, the mirror recursion from t = T):
    %     beta_T(S) = y_b(T), beta_T(S-1) = y_{l_U}(T)
    %     beta_t(s) = [beta_{t+1}(s) + beta_{t+1}(s+1) + skip'(s) beta_{t+1}(s+2)] y_{l'_s}(t)
    %   GRADIENT w.r.t. the logits u (softmax inside the loss), for every t:
    %     dL/du_k(t) = y_k(t) - (1/p) sum_{s : l'_s = k} alpha_t(s) beta_t(s) / y_k(t)
    %   i.e. softmax minus the posterior probability that the alignment uses
    %   class k at step t.
    %   Everything runs in LOG space (log-sum-exp), so long sequences cannot
    %   underflow.
    %
    %   REDUCTION: "mean" = mean over the batch of L_n / U_n (PyTorch's default),
    %              "sum"  = sum of L_n.
    %   An impossible target (T_n too short for the labels + needed blanks)
    %   has p = 0: error, or loss 0 / zero gradient with ZeroInfinity = true
    %   (PyTorch zero_infinity).

    properties (SetAccess = private)
        Targets cell
        InputLengths (1,:) double
        Blank (1,1) double
        Reduction (1,1) string
        ZeroInfinity (1,1) logical
    end

    methods
        function obj = CTCLoss(targets, opts)
            arguments
                targets cell
                opts.InputLengths (1,:) double = []
                opts.Blank (1,1) double {mustBePositive, mustBeInteger} = 1
                opts.Reduction (1,1) string {mustBeMember(opts.Reduction, ["mean", "sum"])} = "mean"
                opts.ZeroInfinity (1,1) logical = false
            end
            obj.Targets = cellfun(@(t) double(t(:).'), targets, 'UniformOutput', false);
            obj.InputLengths = opts.InputLengths;
            obj.Blank = opts.Blank;
            obj.Reduction = opts.Reduction;
            obj.ZeroInfinity = opts.ZeroInfinity;
        end

        function L = forward(obj, U)
            [K, N, T] = size(U, 1:3);
            if numel(obj.Targets) ~= N
                error('OceanDL:Autograd:CTCLoss:BatchMismatch', ...
                    '%d target sequences for a batch of %d.', numel(obj.Targets), N);
            end
            Tn = obj.InputLengths;
            if isempty(Tn), Tn = T * ones(1, N); end
            if numel(Tn) ~= N || any(Tn < 0 | Tn > T | Tn ~= round(Tn))
                error('OceanDL:Autograd:CTCLoss:BadInputLengths', ...
                    'InputLengths must be N = %d integers in [0, %d].', N, T);
            end
            % log-softmax over classes, stable
            m = max(U, [], 1);
            logY = U - (m + log(sum(exp(U - m), 1)));             % K x N x T
            losses = zeros(1, N);
            G = zeros(K, N, T, 'like', U);                         % dL_n/du (before reduction)
            for n = 1:N
                lab = obj.Targets{n};
                if any(lab == obj.Blank) || any(lab < 1 | lab > K)
                    error('OceanDL:Autograd:CTCLoss:BadLabel', ...
                        'Target %d contains the blank class or an out-of-range class.', n);
                end
                ly = reshape(logY(:, n, 1:Tn(n)), K, Tn(n));       % K x Tn
                [losses(n), g] = OceanDL.Autograd.Functions.CTCLoss.single( ...
                    ly, lab, obj.Blank);
                if isinf(losses(n))
                    if ~obj.ZeroInfinity
                        error('OceanDL:Autograd:CTCLoss:Infeasible', ...
                            'Sample %d: %d steps cannot emit %d labels (plus required blanks).', n, Tn(n), numel(lab));
                    end
                    losses(n) = 0;
                    g(:) = 0;
                end
                G(:, n, 1:Tn(n)) = reshape(g, K, 1, Tn(n));
            end
            U_n = cellfun(@numel, obj.Targets);
            if obj.Reduction == "mean"
                w = 1 ./ (max(U_n, 1) * N);                        % d(mean_n L_n/U_n)/dL_n
            else
                w = ones(1, N);
            end
            L = cast(sum(w .* losses), 'like', U);
            obj.Saved.G = G .* reshape(cast(w, 'like', U), 1, N);
            obj.Saved.PerSample = losses;
        end

        function grads = backward(obj, g)
            grads = {g .* obj.Saved.G};
        end

        function l = perSampleLoss(obj)
            l = obj.Saved.PerSample;
        end
    end

    methods (Static)
        function [loss, grad] = single(logY, lab, blank)
            % One sample: logY (K x T) log-probabilities, lab (1 x U) labels.
            [K, T] = size(logY);
            U = numel(lab);
            S = 2 * U + 1;
            if T == 0
                % no input steps: only the empty labelling is possible (p = 1)
                grad = zeros(K, 0, 'like', logY);
                if U == 0, loss = 0; else, loss = inf; end
                return;
            end
            ext = blank * ones(1, S);
            ext(2:2:end) = lab;                                   % l' = (b, l1, b, ..., lU, b)
            skip = false(1, S);
            skip(3:S) = ext(3:S) ~= blank & ext(3:S) ~= ext(1:S-2);
            lp = logY(ext, :);                                    % S x T: log y_{l'_s}(t)

            logA = -inf(S, T);
            logA(1, 1) = lp(1, 1);
            if S > 1, logA(2, 1) = lp(2, 1); end
            for t = 2:T
                prev = logA(:, t-1);
                a1 = [-inf; prev(1:end-1)];                       % from s-1
                a2 = -inf(S, 1);                                  % from s-2 (stays S x 1 when S = 1)
                a2(3:S) = prev(1:S-2);
                a2(~skip) = -inf;
                logA(:, t) = OceanDL.Autograd.Functions.CTCLoss.lse3(prev, a1, a2) + lp(:, t);
            end
            logB = -inf(S, T);
            logB(S, T) = lp(S, T);
            if S > 1, logB(S-1, T) = lp(S-1, T); end
            skipB = false(1, S);                                  % may jump s -> s+2 ?
            skipB(1:S-2) = ext(1:S-2) ~= blank & ext(1:S-2) ~= ext(3:S);
            for t = T-1:-1:1
                nxt = logB(:, t+1);
                b1 = [nxt(2:end); -inf];                          % from s+1
                b2 = -inf(S, 1);                                  % from s+2 (stays S x 1 when S = 1)
                b2(1:S-2) = nxt(3:S);
                b2(~skipB) = -inf;
                logB(:, t) = OceanDL.Autograd.Functions.CTCLoss.lse3(nxt, b1, b2) + lp(:, t);
            end
            if S > 1
                logP = OceanDL.Autograd.Functions.CTCLoss.lse3(logA(S, T), logA(S-1, T), -inf);
            else
                logP = logA(S, T);
            end
            loss = -logP;
            if ~isfinite(logP)
                grad = zeros(K, T);
                loss = inf;
                return;
            end
            % posterior of state s at t: alpha beta / (y p)  (<= 1, safe to exponentiate)
            post = exp(logA + logB - lp - logP);                  % S x T
            occupancy = zeros(K, T);
            for s = 1:S
                occupancy(ext(s), :) = occupancy(ext(s), :) + post(s, :);
            end
            grad = exp(logY) - occupancy;                         % y_k(t) - posterior_k(t)
        end

        function r = lse3(a, b, c)
            % log(exp(a) + exp(b) + exp(c)), elementwise, -inf safe.
            m = max(max(a, b), c);
            finite = isfinite(m);
            r = -inf(size(m));
            r(finite) = m(finite) + log(exp(a(finite) - m(finite)) + exp(b(finite) - m(finite)) ...
                + exp(c(finite) - m(finite)));
        end
    end
end
