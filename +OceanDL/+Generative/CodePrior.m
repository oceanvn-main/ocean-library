classdef CodePrior < OceanDL.nn.Module
    % CODEPRIOR  Autoregressive prior over VQ-VAE code grids: a causal
    %   (GPT-style) Transformer on the codes in raster order -- the second
    %   stage of VQ-VAE / VQGAN (van den Oord 2017 used a PixelCNN; Esser et
    %   al. 2021 a Transformer). Reuses the shared OceanDL.Transformer core.
    %
    %   p(c) = prod_t p(c_t | c_<t),   c = grid(:) (column-major: row index fastest)
    %   input  [BOS, c_1 .. c_{T-1}]  (BOS = extra token K+1),  target c_1 .. c_T
    %   causal AttnMask: position t attends to positions <= t only.
    %   forward(idx) returns the mean negative log-likelihood per code (nats).
    %
    %   sample(N) draws codes one position at a time (T full forward passes,
    %   no key/value cache -- clarity over speed); 'Temperature' and 'TopK'
    %   shape the sampling distribution.

    properties (SetAccess = protected)
        TokenEmbed
        PosEmbed
        Encoder
        Head
        NumCodes (1,1) double
        GridSize (1,2) double
    end

    methods
        function obj = CodePrior(numCodes, gridSize, opts)
            arguments
                numCodes (1,1) double {mustBePositive, mustBeInteger}
                gridSize (1,2) double {mustBePositive, mustBeInteger}
                opts.Dim (1,1) double = 64
                opts.Depth (1,1) double = 2
                opts.Heads (1,1) double = 4
                opts.Dropout (1,1) double = 0
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            s = @(k) OceanDL.Generative.CodePrior.offset(opts.Seed, k);
            T = prod(gridSize);
            obj.NumCodes = numCodes;
            obj.GridSize = gridSize;
            obj.TokenEmbed = OceanDL.nn.Embedding(numCodes + 1, opts.Dim, 'Seed', s(1), 'DataType', opts.DataType);
            obj.PosEmbed = OceanDL.Transformer.LearnedPositionalEmbedding(T, opts.Dim, 'Seed', s(2), 'DataType', opts.DataType);
            obj.Encoder = OceanDL.Transformer.TransformerEncoder(opts.Depth, opts.Dim, opts.Heads, 4 * opts.Dim, ...
                'Dropout', opts.Dropout, 'Seed', s(3), 'DataType', opts.DataType);
            obj.Head = OceanDL.nn.Linear(opts.Dim, numCodes, 'Seed', s(4), 'DataType', opts.DataType);
        end

        function L = forward(obj, idx)
            codes = obj.toSequence(idx);                                    % N x T
            logits = obj.logits([repmat(obj.NumCodes + 1, size(codes, 1), 1), codes(:, 1:end-1)]);
            L = OceanDL.Autograd.F.crossEntropy(logits, codes);
        end

        function Z = logits(obj, inputs)
            % inputs (N x T) token ids -> logits (K x N x T), causal.
            T = size(inputs, 2);
            x = obj.PosEmbed.forward(obj.TokenEmbed.forward(inputs));      % D x N x T
            causal = tril(true(T));                                         % (query t, key s): s <= t
            Z = obj.Head.forward(obj.Encoder.forward(x, 'AttnMask', causal));
        end

        function idx = sample(obj, n, opts)
            arguments
                obj
                n (1,1) double {mustBePositive, mustBeInteger}
                opts.Temperature (1,1) double {mustBePositive} = 1
                opts.TopK (1,1) double {mustBeNonnegative, mustBeInteger} = 0
                opts.Stream = RandStream.getGlobalStream()
            end
            T = prod(obj.GridSize);
            K = obj.NumCodes;
            inputs = repmat(K + 1, n, T);                                  % future positions: BOS (masked anyway)
            codes = zeros(n, T);
            was = obj.Training; obj.eval();
            cleanup = onCleanup(@() obj.train(was));
            for t = 1:T
                Z = OceanDL.Autograd.Tensor.raw(OceanDL.Autograd.GradMode.noGrad(@() obj.logits(inputs)));
                z = double(gather(Z(:, :, t))) / opts.Temperature;         % K x n
                if opts.TopK > 0 && opts.TopK < K
                    srt = sort(z, 1, 'descend');
                    z(z < srt(opts.TopK, :)) = -Inf;
                end
                p = exp(z - max(z, [], 1)); p = p ./ sum(p, 1);
                u = rand(opts.Stream, 1, n);
                codes(:, t) = sum(cumsum(p, 1) < u, 1).' + 1;              % inverse-CDF draw
                codes(:, t) = min(codes(:, t), K);
                if t < T, inputs(:, t + 1) = codes(:, t); end
            end
            idx = reshape(codes.', [obj.GridSize n]);
        end
    end

    methods (Access = private)
        function codes = toSequence(obj, idx)
            T = prod(obj.GridSize);
            codes = reshape(double(idx), T, []).';                          % N x T, raster (column-major) order
        end
    end

    methods (Static, Access = private)
        function s = offset(seed, k)
            if isempty(seed), s = []; else, s = seed + k; end
        end
    end
end
