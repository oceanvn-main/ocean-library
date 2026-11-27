classdef SampleEvaluator < handle
    % SAMPLEEVALUATOR  Measures generated images against real ones with a
    %   small CNN classifier trained on the REAL training set (a stand-in for
    %   the Inception network behind FID / IS, which this library does not
    %   have -- every number is labelled "classifier-feature" accordingly).
    %
    %   ev = SampleEvaluator(Xtrain, ytrain, 'NumClasses', K);    % trains the CNN
    %   r  = ev.evaluate(Xgen, Xref, Xtrain);
    %     Confidence     mean max_k p(k | x)  -- is each sample recognisably SOME class?
    %     ClassEntropy   entropy of the sample class histogram / log K  (1 = all classes equally)
    %     ClassesHit     classes predicted for >= 1% of samples     (mode coverage)
    %     InceptionLike  exp(E_x KL(p(k|x) || p(k)))  (IS formula, K = max)
    %     FrechetDist    ||mu_g - mu_r||^2 + Tr(S_g + S_r - 2 (S_g S_r)^(1/2)) on
    %                    the classifier's penultimate features (lower = closer)
    %     NNDistance     mean L2 distance sample -> nearest TRAINING image (novelty;
    %                    compare RefNNDistance: real test -> train)
    %     Sharpness      mean |image gradient| of samples / of real images
    %   The classifier's own test accuracy (ev.Accuracy) says how much the
    %   class-based metrics can be trusted.

    properties (SetAccess = private)
        Net
        NumClasses (1,1) double
        Accuracy (1,1) double = NaN
    end

    methods
        function obj = SampleEvaluator(X, y, opts)
            arguments
                X
                y
                opts.NumClasses (1,1) double
                opts.Steps (1,1) double = 1500
                opts.Batch (1,1) double = 64
                opts.Seed (1,1) double = 0
                opts.Net = []                   % pre-trained classifier (skips training)
            end
            obj.NumClasses = opts.NumClasses;
            [H, ~, C] = size(X, 1:3);
            if ~isempty(opts.Net)
                obj.Net = opts.Net;
                return;
            end
            obj.Net = OceanDL.Generative.SampleEvaluator.classifier(C, H, opts.NumClasses, opts.Seed);
            opt = OceanDL.optim.ModuleOptimizer(obj.Net.parameters(), @() OceanStats.Optim.AdamWOptimizer('LR', 2e-3, 'WeightDecay', 1e-4));
            rs = RandStream('mt19937ar', 'Seed', opts.Seed + 1);
            obj.Net.train();
            for s = 1:opts.Steps
                idx = randi(rs, size(X, 4), 1, opts.Batch);
                opt.zeroGrad();
                L = OceanDL.Autograd.F.crossEntropy(obj.Net.forward(X(:, :, :, idx)), y(idx));
                L.backward(); opt.step();
            end
            obj.Net.eval();
        end

        function acc = testAccuracy(obj, X, y)
            [~, pred] = max(obj.probabilities(X), [], 1);
            acc = mean(pred(:) == y(:));
            obj.Accuracy = acc;
        end

        function [P, Fz] = probabilities(obj, X)
            % Class probabilities (K x N) and penultimate features (F x N).
            N = size(X, 4); P = []; Fz = [];
            obj.Net.eval();
            for b = 1:256:N
                k = b:min(N, b + 255);
                [p, f] = OceanDL.Autograd.GradMode.noGrad(@() obj.forwardBoth(X(:, :, :, k)));
                P = [P, p]; Fz = [Fz, f]; %#ok<AGROW>
            end
        end

        function r = evaluate(obj, Xgen, Xref, Xtrain)
            Xgen = double(gather(Xgen)); Xref = double(gather(Xref));
            [Pg, Fg] = obj.probabilities(Xgen);
            [~, Fr] = obj.probabilities(Xref);
            K = obj.NumClasses;
            [~, cls] = max(Pg, [], 1);
            hist = histcounts(cls, 0.5:1:K + 0.5) / numel(cls);
            nz = hist(hist > 0);
            py = mean(Pg, 2);
            kl = sum(Pg .* (log(Pg + 1e-12) - log(py + 1e-12)), 1);
            r = struct( ...
                'Confidence', mean(max(Pg, [], 1)), ...
                'ClassHistogram', hist, ...
                'ClassEntropy', -sum(nz .* log(nz)) / log(K), ...
                'ClassesHit', nnz(hist >= 0.01), ...
                'InceptionLike', exp(mean(kl)), ...
                'FrechetDist', OceanDL.Generative.SampleEvaluator.frechetDistance(Fg, Fr), ...
                'NNDistance', OceanDL.Generative.SampleEvaluator.nnDistance(Xgen, Xtrain), ...
                'RefNNDistance', OceanDL.Generative.SampleEvaluator.nnDistance(Xref, Xtrain), ...
                'Sharpness', OceanDL.Generative.SampleEvaluator.gradMagnitude(Xgen) / ...
                             OceanDL.Generative.SampleEvaluator.gradMagnitude(Xref));
        end
    end

    methods (Access = private)
        function [p, f] = forwardBoth(obj, X)
            layers = obj.Net.Layers;
            h = X;
            for i = 1:numel(layers) - 1
                h = layers{i}.forward(h);
            end
            f = OceanDL.Autograd.Tensor.raw(h);
            p = OceanDL.Autograd.Tensor.raw(softmax(layers{end}.forward(h), 1));
        end
    end

    methods (Static)
        function net = classifier(C, H, K, seed)
            % conv3x3(16)-ReLU-pool, conv3x3(32)-ReLU-pool, flatten, Linear(64)-ReLU [features], Linear(K)
            P = @() OceanDL.nn.Legacy(OceanDL.Pooling.MaxPool2D([2 2]));
            side = H / 4;
            net = OceanDL.nn.Sequential( ...
                OceanDL.nn.Conv2d(C, 16, 3, 'Padding', 1, 'Seed', seed + 1), OceanDL.nn.Activation("relu"), P(), ...
                OceanDL.nn.Conv2d(16, 32, 3, 'Padding', 1, 'Seed', seed + 2), OceanDL.nn.Activation("relu"), P(), ...
                OceanDL.nn.Legacy(OceanDL.Conv.Flatten()), ...
                OceanDL.nn.Linear(side ^ 2 * 32, 64, 'Seed', seed + 3), OceanDL.nn.Activation("relu"), ...
                OceanDL.nn.Linear(64, K, 'Seed', seed + 4));
        end

        function d = frechetDistance(F1, F2)
            % Frechet distance between Gaussians fitted to feature columns.
            % Tr((S1 S2)^(1/2)) = sum sqrt(eig(A)), A = S1^(1/2) S2 S1^(1/2) (symmetric
            % PSD): stable when a covariance is singular (few samples, dead
            % features), where sqrtm(S1 * S2) returns NaN / complex values.
            m1 = mean(F1, 2); m2 = mean(F2, 2);
            S1 = cov(F1.'); S2 = cov(F2.');
            r1 = OceanDL.Generative.SampleEvaluator.psdSqrt(S1);
            A = r1 * S2 * r1; A = (A + A.') / 2;
            trSqrt = sum(sqrt(max(0, eig(A))));
            d = max(0, sum((m1 - m2) .^ 2) + trace(S1) + trace(S2) - 2 * trSqrt);
        end

        function R = psdSqrt(S)
            % Square root of a symmetric positive semi-definite matrix (negative
            % round-off eigenvalues clamped to 0).
            S = (S + S.') / 2;
            [V, E] = eig(S);
            R = V * diag(sqrt(max(0, diag(E)))) * V.';
        end

        function d = nnDistance(X, Xtrain)
            % Mean Euclidean distance from each image to its nearest training image.
            A = reshape(double(X), [], size(X, 4)); B = reshape(double(Xtrain), [], size(Xtrain, 4));
            nb = sum(B .^ 2, 1);
            d = zeros(1, size(A, 2));
            for k = 1:500:size(A, 2)
                kk = k:min(size(A, 2), k + 499);
                D = sum(A(:, kk) .^ 2, 1).' - 2 * A(:, kk).' * B + nb;
                d(kk) = sqrt(max(0, min(D, [], 2))).';
            end
            d = mean(d);
        end

        function g = gradMagnitude(X)
            X = double(X);
            gx = diff(X, 1, 2); gy = diff(X, 1, 1);
            g = (mean(abs(gx), 'all') + mean(abs(gy), 'all')) / 2;
        end
    end
end
