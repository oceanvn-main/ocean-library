classdef VectorQuantizer < OceanDL.Generative.Quantizers.IQuantizer
    % VECTORQUANTIZER  VQ-VAE bottleneck (van den Oord et al. 2017) with the
    %   modern fixes for codebook collapse, all options:
    %
    %   QUANTISE  each latent vector z_e (a column, after the optional
    %     projection) is replaced by its nearest code e_k, k = argmin ||z_e - e_k||^2
    %     (computed as ||z||^2 - 2 e'z + ||e||^2, one matrix product).
    %   STRAIGHT-THROUGH  argmin has no gradient, so the decoder's gradient is
    %     copied to z_e:  z_q = z_e + stopgrad(e_k - z_e).
    %   LOSS ('Update' = "grad"):  ||stopgrad(z_e) - e||^2  +  Beta ||z_e - stopgrad(e)||^2
    %     (codebook term moves codes to encoder outputs; COMMITMENT term, Beta =
    %     0.25, keeps the encoder near its codes). Means over elements.
    %   EMA CODEBOOK ('Update' = "ema", Razavi et al. 2019): no codebook term;
    %     per batch, n_k = #vectors assigned to k, s_k = their sum, and
    %       N_k <- d N_k + (1 - d) n_k,   S_k <- d S_k + (1 - d) s_k
    %       e_k  = S_k / N~_k,  N~_k = (N_k + eps) / (sum N + K eps) * sum N   (Laplace)
    %     = online k-means; with Decay d = 0 it is exactly one Lloyd step.
    %   DEAD-CODE RESTART ('RestartDead', Dhariwal et al. 2020): a code whose
    %     EMA usage falls below 'DeadThreshold' is re-seeded with a random
    %     encoder output of the current batch.
    %   DATA INIT ('Init' = "data"): codes start as random encoder outputs of
    %     the first training batch (k-means style) instead of uniform noise.
    %   FACTORISED + L2 CODES ('CodeDim', 'Normalize'; Yu et al. 2022,
    %     ViT-VQGAN): look up in a low-dimensional space (Linear projections
    %     in/out) with L2-normalised vectors (cosine similarity) -- improves
    %     codebook usage.

    properties (SetAccess = protected)
        Codebook            % Parameter (CodeDim x K) for Update "grad"; [] for "ema"
        ProjIn              % Linear Dim -> CodeDim, or []
        ProjOut             % Linear CodeDim -> Dim, or []
        K (1,1) double
        Dim (1,1) double
        CodeDim (1,1) double
        Beta (1,1) double
        Update (1,1) string
        Decay (1,1) double
        Epsilon (1,1) double
        Normalize (1,1) logical
        RestartDead (1,1) logical
        DeadThreshold (1,1) double
        Init (1,1) string
    end

    properties (SetAccess = private)
        EmaCodebook = []    % buffers (moved by Module.to via applyBuffers)
        EmaCount = []
        EmaSum = []
        UsageEma = []
        Initialized (1,1) logical = false
        Stream
    end

    methods
        function obj = VectorQuantizer(numCodes, dim, opts)
            arguments
                numCodes (1,1) double {mustBePositive, mustBeInteger}
                dim (1,1) double {mustBePositive, mustBeInteger}
                opts.CodeDim (1,1) double = 0                   % 0 = no projection (CodeDim = dim)
                opts.Beta (1,1) double = 0.25
                opts.Update (1,1) string {mustBeMember(opts.Update, ["grad", "ema"])} = "ema"
                opts.Decay (1,1) double {mustBeInRange(opts.Decay, 0, 1)} = 0.99
                opts.Epsilon (1,1) double = 1e-5
                opts.Normalize (1,1) logical = false
                opts.RestartDead (1,1) logical = true
                opts.DeadThreshold (1,1) double = 1
                opts.Init (1,1) string {mustBeMember(opts.Init, ["data", "uniform"])} = "data"
                opts.Layout (1,1) string {mustBeMember(opts.Layout, ["SSCB", "CB"])} = "SSCB"
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            obj.K = numCodes; obj.Dim = dim; obj.Beta = opts.Beta; obj.Update = opts.Update;
            obj.Decay = opts.Decay; obj.Epsilon = opts.Epsilon; obj.Normalize = opts.Normalize;
            obj.RestartDead = opts.RestartDead; obj.DeadThreshold = opts.DeadThreshold;
            obj.Init = opts.Init; obj.Layout = opts.Layout;
            s = @(k) OceanDL.Generative.Quantizers.VectorQuantizer.offset(opts.Seed, k);
            if opts.CodeDim > 0 && opts.CodeDim ~= dim
                obj.CodeDim = opts.CodeDim;
                obj.ProjIn = OceanDL.nn.Linear(dim, opts.CodeDim, 'Seed', s(1), 'DataType', opts.DataType);
                obj.ProjOut = OceanDL.nn.Linear(opts.CodeDim, dim, 'Seed', s(2), 'DataType', opts.DataType);
            else
                obj.CodeDim = dim;
            end
            if isempty(opts.Seed), obj.Stream = RandStream.getGlobalStream();
            else, obj.Stream = RandStream('mt19937ar', 'Seed', s(3)); end
            E = cast((2 * rand(obj.Stream, obj.CodeDim, numCodes) - 1) / numCodes, opts.DataType);   % van den Oord init
            if obj.Update == "grad"
                obj.Codebook = OceanDL.nn.Parameter(E);
            else
                obj.EmaCodebook = E; obj.EmaSum = E; obj.EmaCount = ones(1, numCodes, 'like', E);
            end
            obj.UsageEma = ones(1, numCodes, 'like', E) * max(1, opts.DeadThreshold);
            obj.Initialized = opts.Init == "uniform";
        end

        function K = numCodes(obj)
            K = obj.K;
        end

        function E = codes(obj)
            % Current codebook as an array (CodeDim x K), normalised if 'Normalize'.
            if obj.Update == "grad", E = obj.Codebook.Data; else, E = obj.EmaCodebook; end
            if obj.Normalize, E = E ./ sqrt(sum(E .^ 2, 1) + 1e-12); end
        end

        function [zq, info] = forward(obj, z)
            L = OceanDL.Generative.Latent;
            F = OceanDL.Autograd.F;
            [Zc, shape] = L.toColumns(OceanDL.Autograd.Tensor.wrap(z), obj.Layout);   % Dim x M
            ze = Zc;
            if ~isempty(obj.ProjIn), ze = obj.ProjIn.forward(ze); end
            if obj.Normalize, ze = ze ./ sqrt(sum(ze .^ 2, 1) + 1e-12); end
            zeRaw = ze.Data;
            if obj.Training && ~obj.Initialized
                obj.seedCodes(1:obj.K, zeRaw);
                obj.Initialized = true;
            end

            E = obj.codes();
            dist = sum(E .^ 2, 1).' - 2 * (E.' * zeRaw) + sum(zeRaw .^ 2, 1);    % K x M
            [~, idx] = min(dist, [], 1);

            if obj.Update == "grad"
                Et = obj.Codebook;
                if obj.Normalize, Et = Et ./ sqrt(sum(Et .^ 2, 1) + 1e-12); end
                zqT = OceanDL.Autograd.Functions.Index({':', idx}).apply(Et);             % gradient reaches the codes
                loss = F.mseLoss(zqT, zeRaw) + obj.Beta * F.mseLoss(ze, zqT.Data);
                zqRaw = zqT.Data;
            else
                zqRaw = E(:, idx);
                loss = obj.Beta * F.mseLoss(ze, zqRaw);
            end
            out = ze + (zqRaw - zeRaw);                   % straight-through: value e_k, gradient to z_e only
                                                          % (codes learn from the codebook term / EMA, never the decoder)
            if ~isempty(obj.ProjOut), out = obj.ProjOut.forward(out); end
            zq = L.fromColumns(out, shape, obj.Layout);

            [ppl, usage, counts] = OceanDL.Generative.Quantizers.IQuantizer.usageStats(idx, obj.K);
            dead = 0;
            if obj.Training
                if obj.Update == "ema", obj.emaUpdate(zeRaw, idx, counts); end
                obj.UsageEma = obj.Decay * obj.UsageEma + (1 - obj.Decay) * cast(counts, 'like', obj.UsageEma);
                if obj.RestartDead
                    deadIdx = find(obj.UsageEma < obj.DeadThreshold);
                    dead = numel(deadIdx);
                    if dead > 0, obj.seedCodes(deadIdx, zeRaw); end
                end
            end
            info = struct('Indices', reshape(idx, L.indexShape(shape, obj.Layout)), 'Loss', loss, ...
                'Perplexity', ppl, 'Usage', usage, 'Restarted', dead);
        end

        function Z = lookup(obj, indices)
            E = obj.codes();
            cols = E(:, indices(:).');
            Z = OceanDL.Autograd.GradMode.noGrad(@() obj.project(cols));
            Z = OceanDL.Autograd.Tensor.raw(Z);
            shape = OceanDL.Generative.Latent.shapeFromIndices(indices, obj.Dim, obj.Layout);
            Z = OceanDL.Generative.Latent.fromColumns(Z, shape, obj.Layout);
        end
    end

    methods (Access = private)
        function y = project(obj, cols)
            if isempty(obj.ProjOut), y = cols; else, y = obj.ProjOut.forward(cols); end
        end

        function emaUpdate(obj, zeRaw, idx, counts)
            d = obj.Decay;
            M = size(zeRaw, 2);
            onehot = zeros(obj.K, M, 'like', zeRaw);
            onehot(sub2ind([obj.K M], idx, 1:M)) = 1;
            sums = zeRaw * onehot.';                                               % CodeDim x K
            obj.EmaCount = d * obj.EmaCount + (1 - d) * cast(counts, 'like', obj.EmaCount);
            obj.EmaSum = d * obj.EmaSum + (1 - d) * sums;
            n = sum(obj.EmaCount);
            smoothed = (obj.EmaCount + obj.Epsilon) / (n + obj.K * obj.Epsilon) * n;
            obj.EmaCodebook = obj.EmaSum ./ smoothed;
        end

        function seedCodes(obj, which, zeRaw)
            % Replace codes `which` by random encoder outputs of this batch
            % (with replacement; tiny noise separates duplicates).
            M = size(zeRaw, 2);
            pick = randi(obj.Stream, M, 1, numel(which));
            v = zeRaw(:, pick) + 1e-4 * cast(randn(obj.Stream, size(zeRaw, 1), numel(which)), 'like', zeRaw);
            if obj.Update == "grad"
                obj.Codebook.Data(:, which) = v;
            else
                obj.EmaCodebook(:, which) = v;
                obj.EmaSum(:, which) = v;
                obj.EmaCount(which) = 1;
            end
            obj.UsageEma(which) = max(1, obj.DeadThreshold);
        end
    end

    methods (Access = protected)
        function applyBuffers(obj, fn)
            obj.EmaCodebook = fn(obj.EmaCodebook); obj.EmaCount = fn(obj.EmaCount);
            obj.EmaSum = fn(obj.EmaSum); obj.UsageEma = fn(obj.UsageEma);
        end
    end

    methods (Static, Access = private)
        function s = offset(seed, k)
            if isempty(seed), s = []; else, s = seed + k; end
        end
    end
end
