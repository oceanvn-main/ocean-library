classdef CTCTest < OceanTest.BaseTestCase
    % CTCTEST  CTC loss validated three independent ways:
    %   1. PyTorch nn.CTCLoss (loss AND gradient, sum and mean reductions),
    %      fixtures imported through NpyReader + DimFormat;
    %   2. brute force: explicit sum over EVERY alignment path (K^T paths);
    %   3. finite differences (GradCheck) with repeats and ragged lengths;
    %   plus infeasible targets and the greedy decoder / edit distance.

    methods (Test)
        function testMatchesPyTorchCtcLoss(testCase)
            import OceanDL.Layout.DimFormat
            fx = @(n) OceanData.Adapters.NpyReader.read(fullfile(fileparts(mfilename('fullpath')), 'fixtures', [n '.npy']));
            U = DimFormat.toCanonical(fx('ctc_logits_TNC'), "TBC");         % (T,N,C) -> C x N x T
            padded = double(fx('ctc_targets_padded'));
            lens = double(fx('ctc_target_lengths'));
            inLen = double(fx('ctc_input_lengths')).';
            targets = arrayfun(@(n) padded(n, 1:lens(n)) + 1, 1:numel(lens), 'UniformOutput', false);  % class c -> c+1
            for red = ["sum", "mean"]
                u = OceanDL.Autograd.Tensor(U, 'RequiresGrad', true);
                L = OceanDL.Autograd.F.ctcLoss(u, targets, 'InputLengths', inLen, 'Blank', 1, 'Reduction', red);
                L.backward();
                testCase.verifyEqual(L.Data, double(fx(char("ctc_loss_" + red))), 'AbsTol', 1e-10, char(red));
                gRef = DimFormat.toCanonical(fx(char("ctc_grad_" + red + "_TNC")), "TBC");
                testCase.verifyEqual(u.Grad, gRef, 'AbsTol', 1e-10, char(red));
            end
        end

        function testMatchesBruteForceSumOverAlignments(testCase)
            % K = 3 classes (1 = blank), T = 5 steps: enumerate all 3^5 paths,
            % collapse each (merge repeats, drop blanks), sum the path
            % probabilities that produce the target.
            s = RandStream('mt19937ar', 'Seed', 1);
            K = 3; T = 5;
            U = randn(s, K, 1, T);
            Y = exp(U - max(U, [], 1)); Y = Y ./ sum(Y, 1); Y = reshape(Y, K, T);
            for target = {[2 3], [2 2], 3, [3 2 3]}
                p = 0;
                for code = 0:K^T - 1
                    path = mod(floor(code ./ K.^(0:T-1)), K) + 1;
                    collapsed = path([true, path(2:end) ~= path(1:end-1)]);
                    collapsed = collapsed(collapsed ~= 1);
                    if isequal(collapsed, target{1})
                        p = p + prod(Y(sub2ind([K T], path, 1:T)));
                    end
                end
                L = OceanDL.Autograd.F.ctcLoss(U, target, 'Reduction', "sum");
                testCase.verifyEqual(L.Data, -log(p), 'AbsTol', 1e-12, mat2str(target{1}));
            end
        end

        function testGradcheckWithRepeatsAndRaggedLengths(testCase)
            targets = {[2 2 3], [4], [3 2]};
            f = @(u) OceanDL.Autograd.F.ctcLoss(u, targets, 'InputLengths', [7 4 6]);
            r = OceanDL.Autograd.GradCheck.check(f, {randn(4, 3, 7)});
            testCase.verifyTrue(r.Passed, sprintf('max err %.3g', r.MaxAbsError));
            testCase.verifyEqual(r.Analytic{1}(:, 2, 5:7), zeros(4, 1, 3));    % steps past T_n get no gradient
        end

        function testInfeasibleTarget(testCase)
            % "2 2" needs a blank between the repeats: 3 steps minimum.
            U = randn(3, 1, 2);
            testCase.verifyError(@() OceanDL.Autograd.F.ctcLoss(U, {[2 2]}), ...
                'OceanDL:Autograd:CTCLoss:Infeasible');
            L = OceanDL.Autograd.F.ctcLoss(U, {[2 2]}, 'ZeroInfinity', true);
            testCase.verifyEqual(L.Data, 0);
        end

        function testDigitStringsPlacesSourceImagesInOrder(testCase)
            s = RandStream('mt19937ar', 'Seed', 2);
            imgs = reshape(1:16*16*5, 16, 16, 1, 5);              % distinguishable "digits"
            [X, labels, info] = OceanData.Generators.Sequences.DigitStrings.compose(imgs, (0:4).', 20, ...
                'MinLength', 2, 'MaxLength', 5, 'Width', 96, 'Stream', s);
            testCase.verifySize(X, [16 96 1 20]);
            for m = 1:20
                src = info.Sources{m};
                testCase.verifyEqual(labels{m}, src - 1);                   % label = source digit, in order
                cols = find(any(X(:, :, 1, m) ~= -1, 1));
                firstBlock = X(:, cols(1):cols(1)+15, 1, m);
                testCase.verifyEqual(firstBlock, imgs(:, :, 1, src(1)));   % leftmost digit comes first
            end
        end

        function testCrnnShapesAndCtcTrainingStep(testCase)
            net = OceanDL.Recurrent.CRNN(16, 11, 'Channels', [4 8], 'Hidden', 8, 'Seed', 1);
            X = randn(16, 32, 1, 3);
            logits = net.forward(X);
            testCase.verifySize(logits.Data, [11 3 8]);                     % T = 32/4 columns
            L = OceanDL.Autograd.F.ctcLoss(logits, {[2 3], [4 4], 5});
            L.backward();
            P = net.parameters();
            testCase.verifyTrue(all(cellfun(@(p) ~isempty(p.Grad) && any(p.Grad(:) ~= 0), P)));   % gradient reaches the CNN
        end

        function testGreedyDecodingAndEditDistance(testCase)
            import OceanDL.Recurrent.CTCDecoder
            path = [1 4 4 1 1 4 8 8];                                  % blank = 1
            Z = zeros(9, 1, numel(path));
            Z(sub2ind(size(Z), path, ones(size(path)), 1:numel(path))) = 5;
            testCase.verifyEqual(CTCDecoder.greedy(Z, 1), {[4 4 8]});
            testCase.verifyEqual(CTCDecoder.editDistance(double('kitten'), double('sitting')), 3);
            r = CTCDecoder.errorRates({[1 2 3], [5 6]}, {[1 2 3], [5 7 8]});
            testCase.verifyEqual(r.CER, 2 / 6, 'AbsTol', 1e-15);
            testCase.verifyEqual(r.SequenceAccuracy, 0.5);
        end
    end
end
