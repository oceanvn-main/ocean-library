classdef SparseDictionary < handle
    % SPARSEDICTIONARY  Sparse autoencoder (dictionary learning) on a layer's
    %   activations: finds "combined features" -- directions that mix several
    %   channels -- which single channels (polysemantic units) can hide.
    %   NOT in the texts: the recipe of Bricken et al. 2023 ("Towards
    %   Monosemanticity"), used on vision models in 2024-25 (e.g. SAEs on CLIP /
    %   DINOv2 / diffusion U-Nets).
    %
    %   Each spatial position of the layer is one sample a (1 x C channels):
    %     f     = ReLU( (a - b_d) W_e + b_e )          sparse code, M >= C features
    %     a_hat = f W_d + b_d                           W_d rows (the atoms) have unit norm
    %     loss  = mean ||a - a_hat||^2 / C + lambda * mean ||f||_1
    %   Activations are first scaled so that E ||a||^2 = C (Scale), which makes
    %   lambda comparable across layers. Adam; atoms re-normalised every step.
    %
    %   D = SparseDictionary(C, 'Features', 4*C, 'L1', 0.05, 'Seed', 0)
    %   D.train(A, 'Steps', 2000, 'Batch', 1024)       A: (n x C) samples
    %   F = D.encode(A)             (n x M) codes;  Ah = D.decode(F)
    %   S = D.metrics(A)            ExplainedVariance, L0 (active features per sample),
    %                               Alive (features that fire on A), Frequency (1 x M)
    %   Fm = D.featureMaps(a)       (h x w x C x N) activations -> (h x w x M x N) feature maps
    %   A = SparseDictionary.samples(a, nMax, seed)   (h x w x C x N) -> (n x C) position samples

    properties (SetAccess = private)
        We                         % C x M encoder
        be                         % 1 x M
        Wd                         % M x C atoms (unit-norm rows)
        bd                         % 1 x C
        Scale (1,1) double = 1
        L1 (1,1) double
        History double = []
    end

    methods
        function obj = SparseDictionary(C, opts)
            arguments
                C (1,1) double {mustBePositive, mustBeInteger}
                opts.Features (1,1) double = 0
                opts.L1 (1,1) double = 0.05
                opts.Seed (1,1) double = 0
            end
            M = opts.Features; if M == 0, M = 4 * C; end
            rs = RandStream('mt19937ar', 'Seed', opts.Seed);
            W = randn(rs, M, C); W = W ./ vecnorm(W, 2, 2);
            obj.Wd = W; obj.We = W.'; obj.be = zeros(1, M); obj.bd = zeros(1, C);
            obj.L1 = opts.L1;
        end

        function train(obj, A, opts)
            arguments
                obj
                A double
                opts.Steps (1,1) double = 2000
                opts.Batch (1,1) double = 1024
                opts.LR (1,1) double = 1e-3
                opts.Seed (1,1) double = 1
            end
            T = OceanDL.Autograd.Tensor;
            C = size(A, 2);
            obj.Scale = sqrt(mean(sum(A .^ 2, 2)) / C);
            A = A / obj.Scale;
            obj.bd = mean(A, 1);
            rs = RandStream('mt19937ar', 'Seed', opts.Seed);
            P = {obj.We, obj.be, obj.Wd, obj.bd};
            m = cellfun(@(p) zeros(size(p)), P, 'UniformOutput', false); v = m;
            obj.History = zeros(1, opts.Steps);
            for t = 1:opts.Steps
                idx = randi(rs, size(A, 1), min(opts.Batch, size(A, 1)), 1);
                a = A(idx, :);
                Tp = cellfun(@(p) OceanDL.Autograd.Tensor(p, 'RequiresGrad', true), P, 'UniformOutput', false);
                f = relu((a - Tp{4}) * Tp{1} + Tp{2});
                ah = f * Tp{3} + Tp{4};
                n = size(a, 1);
                loss = sum((ah - a) .^ 2, 'all') / (n * C) + obj.L1 * sum(f, 'all') / n;   % f >= 0: ||f||_1 = sum f
                loss.backward();
                for k = 1:4                                                % Adam
                    g = Tp{k}.Grad;
                    m{k} = 0.9 * m{k} + 0.1 * g; v{k} = 0.999 * v{k} + 0.001 * g .^ 2;
                    P{k} = P{k} - opts.LR * (m{k} / (1 - 0.9 ^ t)) ./ (sqrt(v{k} / (1 - 0.999 ^ t)) + 1e-8);
                end
                P{3} = P{3} ./ max(vecnorm(P{3}, 2, 2), 1e-8);             % unit-norm atoms
                obj.History(t) = double(T.raw(loss));
            end
            [obj.We, obj.be, obj.Wd, obj.bd] = P{:};
        end

        function F = encode(obj, A)
            F = max(0, (double(A) / obj.Scale - obj.bd) * obj.We + obj.be);
        end

        function Ah = decode(obj, F)
            Ah = (F * obj.Wd + obj.bd) * obj.Scale;
        end

        function S = metrics(obj, A)
            A = double(A);
            F = obj.encode(A);
            R = A - obj.decode(F);
            S.ExplainedVariance = 1 - sum(R .^ 2, 'all') / sum((A - mean(A, 1)) .^ 2, 'all');
            S.L0 = mean(sum(F > 0, 2));
            S.Frequency = mean(F > 0, 1);
            S.Alive = nnz(S.Frequency > 0);
        end

        function Fm = featureMaps(obj, a)
            [h, w, C, N] = size(a, 1:4);
            A = reshape(permute(double(a), [1 2 4 3]), h * w * N, C);
            F = obj.encode(A);
            Fm = permute(reshape(F, h, w, N, []), [1 2 4 3]);
        end
    end

    methods (Static)
        function A = samples(a, nMax, seed)
            [h, w, C, N] = size(a, 1:4);
            A = reshape(permute(double(a), [1 2 4 3]), h * w * N, C);
            if size(A, 1) > nMax
                A = A(randperm(RandStream('mt19937ar', 'Seed', seed), size(A, 1), nMax), :);
            end
        end
    end
end
