classdef F
    % F  Functional API over Tensors (PyTorch torch.nn.functional).
    %
    %   Stateless functions. Most are COMPOSITIONS of primitive ops, so their
    %   gradients come for free from the primitives' VJPs -- e.g.
    %   crossEntropy = -mean(pick(logSoftmax(logits))) needs no hand-written
    %   derivative. Layout convention (repo-wide): FEATURES x BATCH [x TIME],
    %   i.e. samples are columns, classes/features are rows.
    %
    % USAGE:
    %   F = OceanDL.Autograd.F;
    %   h = F.relu(F.linear(x, W, b));
    %   L = F.crossEntropy(logits, labels);             % logits (K x N), labels (1..K)
    %   L = F.crossEntropy(logits, labels, 'Mask', m);  % padded sequences (K x N x T)

    methods (Static)
        % ---------------- activations ----------------
        function y = relu(x),    y = OceanDL.Autograd.Functions.ReLU().apply(x); end
        function y = sigmoid(x), y = OceanDL.Autograd.Functions.Sigmoid().apply(x); end
        function y = tanh(x),    y = OceanDL.Autograd.Functions.Tanh().apply(x); end

        function y = gelu(x, approximate)
            if nargin < 2, approximate = "none"; end
            y = OceanDL.Autograd.Functions.GELU(approximate).apply(x);
        end

        function y = layerNorm(x, gamma, beta, epsilon)
            % LAYERNORM  Normalise every token over the feature axis (dim 1).
            if nargin < 4, epsilon = 1e-5; end
            y = OceanDL.Autograd.Functions.LayerNorm(epsilon).apply(x, gamma, beta);
        end

        function y = silu(x)
            % SILU  x .* sigmoid(x)  (a.k.a. swish; the activation of modern conv autoencoders).
            %   Under OceanDL.Autograd.BackpropRule "deconvnet" / "guided" the backward
            %   pass uses the projection rule instead of the true gradient (same forward).
            T = OceanDL.Autograd.Tensor.wrap(x);
            rule = OceanDL.Autograd.BackpropRule.current();
            if rule == "standard"
                y = T .* sigmoid(T);
            else
                y = OceanDL.Autograd.Functions.RuledSiLU(rule).apply(T);
            end
        end

        function y = groupNorm(x, groups, gamma, beta, epsilon)
            % GROUPNORM  Group normalisation of (H x W x C x N) + per-channel affine;
            %   gamma, beta: (1 x 1 x C) (or [] for no affine).
            if nargin < 5, epsilon = 1e-5; end
            y = OceanDL.Autograd.Functions.GroupNorm(groups, epsilon).apply(x);
            if ~isempty(gamma), y = y .* gamma + beta; end
        end

        function y = upsample(x, scale)
            % UPSAMPLE  Nearest-neighbour upsampling of (H x W x C x N) by an integer factor.
            if nargin < 2, scale = 2; end
            y = OceanDL.Autograd.Functions.Upsample(scale).apply(x);
        end

        function y = dropPath(x, p, training)
            % DROPPATH  Stochastic depth (Huang et al. 2016): drop a residual
            %   branch for a WHOLE sample (dim 2 = batch in "CB"/"CBT"), scaled by
            %   1/(1-p) so the expectation is unchanged. Identity in eval.
            if ~training || p == 0
                y = OceanDL.Autograd.Tensor.wrap(x);
                return;
            end
            data = OceanDL.Autograd.Tensor.raw(x);
            keep = cast(rand(1, size(data, 2)) >= p, 'like', data) / (1 - p);
            y = OceanDL.Autograd.Functions.Mul().apply(x, keep);
        end

        function y = softmax(x, dim)
            if nargin < 2, dim = 1; end
            y = OceanDL.Autograd.Functions.Softmax(dim).apply(x);
        end

        function y = logSoftmax(x, dim)
            if nargin < 2, dim = 1; end
            y = OceanDL.Autograd.Functions.LogSoftmax(dim).apply(x);
        end

        % ---------------- layers ----------------
        function y = linear(x, W, b)
            % LINEAR  y = W x + b,  W (out x in), x (in x N), b (out x 1).
            y = OceanDL.Autograd.Functions.MatMul().apply(W, x);
            if nargin > 2 && ~isempty(b)
                y = OceanDL.Autograd.Functions.Add().apply(y, b);
            end
        end

        function y = conv2d(x, W, b, opts)
            arguments
                x
                W
                b = []
                opts.Stride  (1,2) double = [1 1]
                opts.Padding (1,2) double = [0 0]
            end
            op = OceanDL.Autograd.Functions.Conv2d('Stride', opts.Stride, 'Padding', opts.Padding);
            if isempty(b)
                y = op.apply(x, W);
            else
                y = op.apply(x, W, b);
            end
        end

        function y = conv1d(x, W, b, opts)
            % CONV1D  x (Cin x N x T), W (Cout x Cin/Groups x K), b (Cout x 1, optional)
            %   -> (Cout x N x To). See Functions.Conv1d (torch.nn.functional.conv1d).
            arguments
                x
                W
                b = []
                opts.Stride   (1,1) double = 1
                opts.Padding  (1,1) double = 0
                opts.Dilation (1,1) double = 1
                opts.Groups   (1,1) double = 1
            end
            op = OceanDL.Autograd.Functions.Conv1d('Stride', opts.Stride, 'Padding', opts.Padding, ...
                'Dilation', opts.Dilation, 'Groups', opts.Groups);
            if isempty(b)
                y = op.apply(x, W);
            else
                y = op.apply(x, W, b);
            end
        end

        function y = embedding(W, ids)
            % EMBEDDING  Column lookup: W (D x V), ids (any shape, 1..V)
            %   -> (D x size(ids)). Repeated ids accumulate gradient (Index VJP).
            D = size(W, 1);
            cols = OceanDL.Autograd.Functions.Index({':', ids(:)}).apply(W);
            y = OceanDL.Autograd.Functions.Reshape([D, size(ids)]).apply(cols);
        end

        function y = dropout(x, p, training)
            % DROPOUT  Inverted dropout: keep with prob 1-p, scale by 1/(1-p)
            %   so E[y] = x and evaluation needs no rescaling.
            if ~training || p == 0
                y = OceanDL.Autograd.Tensor.wrap(x);
                return;
            end
            data = OceanDL.Autograd.Tensor.raw(x);
            mask = cast(rand(size(data)) >= p, 'like', data) / (1 - p);
            y = OceanDL.Autograd.Functions.Mul().apply(x, mask);
        end

        % ---------------- losses ----------------
        function L = crossEntropy(logits, targets, opts)
            % CROSSENTROPY  Mean negative log-likelihood of the target class.
            %   logits (K x N) or (K x N x T); targets: integer class per
            %   column (N or N x T). Optional Mask (same size as targets):
            %   only masked-in positions count (padded sequences), and the
            %   mean is over those positions only.
            %     L = -(1/|M|) sum_{(n,t) in M} log softmax(z_{:,n,t})_{y_{n,t}}
            %   LabelSmoothing eps (Szegedy et al. 2016): the target distribution
            %   is (1 - eps) on the true class + eps spread uniformly, so
            %     L = (1 - eps) * NLL  +  eps * mean_k(-log p_k)
            %   discouraging over-confident logits (standard for ViT training).
            arguments
                logits
                targets
                opts.Mask = []
                opts.LabelSmoothing (1,1) double {mustBeGreaterThanOrEqual(opts.LabelSmoothing, 0), mustBeLessThan(opts.LabelSmoothing, 1)} = 0
            end
            K = size(logits, 1);
            M = numel(targets);
            z = OceanDL.Autograd.Functions.Reshape([K, M]).apply(logits);
            logp = OceanDL.Autograd.Functions.LogSoftmax(1).apply(z);
            lin = sub2ind([K, M], double(targets(:)).', 1:M);        % the target entry of each column
            picked = OceanDL.Autograd.Functions.Index({lin}).apply(logp);
            if opts.LabelSmoothing > 0
                e = opts.LabelSmoothing;
                uniform = OceanDL.Autograd.Functions.Mean(1).apply(logp);          % 1 x M: mean_k log p_k
                picked = picked * (1 - e) + uniform * e;
            end
            if isempty(opts.Mask)
                L = -OceanDL.Autograd.Functions.Mean('all').apply(picked);
            else
                w = cast(opts.Mask(:).', 'like', OceanDL.Autograd.Tensor.raw(logits));
                total = OceanDL.Autograd.Functions.Sum('all').apply( ...
                    OceanDL.Autograd.Functions.Mul().apply(picked, w));
                L = -total / max(sum(w), 1);
            end
        end

        function L = ctcLoss(logits, targets, opts)
            % CTCLOSS  Connectionist Temporal Classification on logits (K x N x T);
            %   targets: cell of label rows. See Functions.CTCLoss.
            arguments
                logits
                targets cell
                opts.InputLengths (1,:) double = []
                opts.Blank (1,1) double = 1
                opts.Reduction (1,1) string = "mean"
                opts.ZeroInfinity (1,1) logical = false
            end
            L = OceanDL.Autograd.Functions.CTCLoss(targets, 'InputLengths', opts.InputLengths, ...
                'Blank', opts.Blank, 'Reduction', opts.Reduction, 'ZeroInfinity', opts.ZeroInfinity).apply(logits);
        end

        function L = mseLoss(pred, target)
            % MSELOSS  mean((pred - target).^2) over all elements.
            d = OceanDL.Autograd.Functions.Sub().apply(pred, target);
            L = OceanDL.Autograd.Functions.Mean('all').apply( ...
                OceanDL.Autograd.Functions.Pow(2).apply(d));
        end

        function L = bceWithLogits(z, y)
            % BCEWITHLOGITS  Binary cross-entropy on logits, in the stable form
            %   l = max(z,0) - z y + log(1 + exp(-|z|))   (never exp of a large positive number)
            T = OceanDL.Autograd.Tensor.wrap(z);
            l = relu(T) - T .* y + log(1 + exp(-abs(T)));
            L = mean(l, 'all');
        end
    end
end
