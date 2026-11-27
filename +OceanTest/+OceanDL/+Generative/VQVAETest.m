classdef VQVAETest < OceanTest.BaseTestCase
    % VQVAETEST  End-to-end discrete autoencoders and the code prior:
    %   VQ-VAE (EMA + restarts) and FSQ-VAE on USPS digits reduce the
    %   reconstruction error and keep most codes in use; reconstruct() is
    %   decode(encode()); the prior is causal (future codes never change a
    %   prediction) and learns a deterministic code sequence exactly.

    methods (Test)
        function testDiscreteAutoencodersTrainOnDigits(testCase)
            X = OceanTest.OceanDL.Generative.VQVAETest.digits();
            mk = @(out) OceanDL.Vision.ConvEncoder('InChannels', 1, 'Widths', [16 32 32], 'OutChannels', out, ...
                'Groups', 4, 'MidAttention', false, 'Seed', 1);
            dk = @(lat) OceanDL.Vision.ConvDecoder('LatentChannels', lat, 'Widths', [16 32 32], 'OutChannels', 1, ...
                'Groups', 4, 'MidAttention', false, 'Seed', 2);
            models = {"VQ-EMA", OceanDL.Generative.VQVAE(mk(8), ...
                          OceanDL.Generative.Quantizers.VectorQuantizer(32, 8, 'Update', "ema", 'Seed', 3), dk(8))
                      "FSQ",    OceanDL.Generative.VQVAE(mk(3), OceanDL.Generative.Quantizers.FSQ([5 5 5]), dk(3))};
            for m = 1:size(models, 1)
                [name, net] = models{m, :};
                opt = OceanDL.optim.ModuleOptimizer(net.parameters(), @() OceanStats.Optim.AdamWOptimizer('LR', 2e-3, 'WeightDecay', 0));
                rs = RandStream('mt19937ar', 'Seed', 4);
                rec = zeros(1, 150); use = zeros(1, 150);
                for it = 1:150
                    idx = randi(rs, size(X, 4), 1, 32);
                    opt.zeroGrad(); L = net.forward(X(:, :, :, idx)); L.backward(); opt.step();
                    rec(it) = net.LastStats.Recon; use(it) = net.LastStats.Usage;
                end
                testCase.verifyLessThan(mean(rec(end-9:end)), 0.5 * mean(rec(1:10)), name);
                testCase.verifyGreaterThan(mean(use(end-9:end)), 0.3, name);             % codebook not collapsed
                idx = net.encodeIndices(X(:, :, :, 1:5));
                testCase.verifySize(idx, [4 4 5], name);
                testCase.verifyEqual(net.reconstruct(X(:, :, :, 1:5)), net.decodeIndices(idx), 'AbsTol', 1e-12, name);
            end
        end

        function testCodePriorIsCausal(testCase)
            prior = OceanDL.Generative.CodePrior(5, [2 3], 'Dim', 16, 'Depth', 2, 'Heads', 2, 'Seed', 1);
            a = [6 1 2 3 4 5; 6 5 4 3 2 1];                                   % BOS = 6
            b = a; b(:, 4:end) = [1 1 1; 2 2 2];                             % change the future of position 3
            Za = prior.logits(a).Data; Zb = prior.logits(b).Data;
            testCase.verifyEqual(Za(:, :, 1:3), Zb(:, :, 1:3), 'AbsTol', 1e-12);
            testCase.verifyGreaterThan(max(abs(Za(:, :, 4:end) - Zb(:, :, 4:end)), [], 'all'), 1e-6);
        end

        function testCodePriorLearnsASequenceRule(testCase)
            % Codes c_t = mod(c_1 + t - 2, K) + 1 with c_1 uniform: the best
            % possible loss is log(K) / T (only the first code is uncertain).
            K = 6; grid = [2 3]; T = prod(grid);
            prior = OceanDL.Generative.CodePrior(K, grid, 'Dim', 32, 'Depth', 2, 'Heads', 4, 'Seed', 2);
            opt = OceanDL.optim.ModuleOptimizer(prior.parameters(), @() OceanStats.Optim.AdamWOptimizer('LR', 3e-3, 'WeightDecay', 0));
            rs = RandStream('mt19937ar', 'Seed', 3);
            rule = @(c1) mod(c1 + (0:T-1) - 1, K) + 1;
            for it = 1:300
                c1 = randi(rs, K, 32, 1);
                seqs = cell2mat(arrayfun(rule, c1, 'UniformOutput', false));     % N x T
                idx = reshape(seqs.', [grid 32]);
                opt.zeroGrad(); L = prior.forward(idx); L.backward(); opt.step();
            end
            testCase.verifyLessThan(L.Data, log(K) / T + 0.05);
            S = prior.sample(200, 'Stream', RandStream('mt19937ar', 'Seed', 4));
            testCase.verifySize(S, [grid 200]);
            seq = reshape(S, T, []).';
            ok = all(seq == cell2mat(arrayfun(rule, seq(:, 1), 'UniformOutput', false)), 2);
            testCase.verifyGreaterThan(mean(ok), 0.95);
            testCase.verifyGreaterThan(numel(unique(seq(:, 1))), K - 1);          % first code still varies
        end
    end

    methods (Static)
        function X = digits()
            import OceanDL.Layout.DimFormat
            [rows, ~, fmt] = OceanData.Adapters.UspsDigitsReader.read("test");
            X = DimFormat.toCanonical(DimFormat.fromRowMajor(reshape(rows.', 1, []), fmt.Shape), fmt.Format);
        end
    end
end
