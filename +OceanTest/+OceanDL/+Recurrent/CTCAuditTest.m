classdef CTCAuditTest < OceanTest.BaseTestCase
    % CTCAUDITTEST  Regression tests for the speech-ASR audit of the existing CTC code:
    %   CTCDecoder.greedy  -- T = 0 / InputLengths 0 used to error (path(keep) on
    %                         an empty path); per-example lengths; length validation.
    %   CTCLoss            -- InputLengths 0 used to error (alpha init indexed t = 1);
    %                         an EMPTY target with T >= 2 used to error (the s-2 / s+2
    %                         shifted vectors grew to 2 x 1 when S = 1);
    %                         now matched to PyTorch nn.CTCLoss on an edge-case batch
    %                         (fixtures/gen_torch_ctc_edge_reference.py), brute force,
    %                         GradCheck, single precision, padded-gradient zero.

    methods (Test)
        function testGreedyHandlesEmptyInputs(testCase)
            import OceanDL.Recurrent.CTCDecoder
            testCase.verifyEqual(CTCDecoder.greedy(zeros(5, 2, 0)), {zeros(1, 0), zeros(1, 0)});
            Z = randn(5, 2, 4);
            out = CTCDecoder.greedy(Z, 1, [0 4]);
            testCase.verifyEqual(out{1}, zeros(1, 0));
            full = CTCDecoder.greedy(Z(:, 2, :), 1);
            testCase.verifyEqual(out{2}, full{1});
        end

        function testGreedyPerExampleLengths(testCase)
            % Example n decoded with T_n equals decoding its truncation alone.
            import OceanDL.Recurrent.CTCDecoder
            s = RandStream('mt19937ar', 'Seed', 4);
            Z = 3 * randn(s, 4, 3, 9);
            lens = [9 2 5];
            out = CTCDecoder.greedy(OceanDL.Autograd.Tensor(Z), 1, lens);
            for n = 1:3
                one = CTCDecoder.greedy(Z(:, n, 1:lens(n)), 1);
                testCase.verifyEqual(out{n}, one{1});
                [~, path] = max(reshape(Z(:, n, 1:lens(n)), 4, lens(n)), [], 1);       % independent collapse
                c = path([true, path(2:end) ~= path(1:end-1)]);
                testCase.verifyEqual(out{n}, c(c ~= 1));
            end
            testCase.verifyError(@() CTCDecoder.greedy(Z, 1, [9 10 1]), 'OceanDL:Recurrent:CTCDecoder:BadLengths');
            testCase.verifyError(@() CTCDecoder.greedy(Z, 1, [9 1]), 'OceanDL:Recurrent:CTCDecoder:BadLengths');
        end

        function testLossEdgeCasesMatchPyTorch(testCase)
            % Batch: empty target, normal, T_n = 0 with a label (infeasible ->
            % zero_infinity), T_n = 0 with empty target (p = 1), repeated labels.
            import OceanDL.Layout.DimFormat
            fx = @(n) double(OceanData.Adapters.NpyReader.read(fullfile(fileparts(mfilename('fullpath')), ...
                'fixtures', ['ctcedge_' n '.npy'])));
            U = DimFormat.toCanonical(fx('logits_TNC'), "TBC");
            padded = fx('targets_padded'); tl = fx('target_lengths'); inLen = reshape(fx('input_lengths'), 1, []);
            targets = arrayfun(@(n) padded(n, 1:tl(n)) + 1, 1:numel(tl), 'UniformOutput', false);
            for red = ["sum", "mean"]
                u = OceanDL.Autograd.Tensor(U, 'RequiresGrad', true);
                op = OceanDL.Autograd.Functions.CTCLoss(targets, 'InputLengths', inLen, 'Reduction', red, ...
                    'ZeroInfinity', true);
                L = op.apply(u);
                testCase.verifyEqual(op.perSampleLoss(), reshape(fx('per_sample'), 1, []), 'AbsTol', 1e-10);   % before backward frees it
                L.backward();
                testCase.verifyEqual(L.Data, fx(char("loss_" + red)), 'AbsTol', 1e-10, char(red));
                gRef = DimFormat.toCanonical(fx(char("grad_" + red + "_TNC")), "TBC");
                testCase.verifyEqual(u.Grad, gRef, 'AbsTol', 1e-10, char(red));
                for n = 1:numel(inLen)                                   % padded steps: exactly zero
                    testCase.verifyEqual(nnz(u.Grad(:, n, inLen(n)+1:end)), 0);
                end
            end
            testCase.verifyError(@() OceanDL.Autograd.F.ctcLoss(U, targets, 'InputLengths', inLen), ...
                'OceanDL:Autograd:CTCLoss:Infeasible');                   % without ZeroInfinity
        end

        function testEmptyTargetIsProductOfBlanks(testCase)
            % p(empty | x) = prod_t y_blank(t): the only path is all blanks.
            s = RandStream('mt19937ar', 'Seed', 6);
            U = randn(s, 4, 1, 5);
            Y = exp(U - max(U, [], 1)); Y = Y ./ sum(Y, 1);
            L = OceanDL.Autograd.F.ctcLoss(U, {zeros(1, 0)}, 'Reduction', "sum");
            testCase.verifyEqual(L.Data, -sum(log(Y(1, 1, :))), 'AbsTol', 1e-12);
            L0 = OceanDL.Autograd.F.ctcLoss(U, {zeros(1, 0)}, 'InputLengths', 0);
            testCase.verifyEqual(L0.Data, 0);
        end

        function testGradcheckWithEmptyTargetAndZeroLength(testCase)
            targets = {zeros(1, 0), [2 3], 3, zeros(1, 0)};
            f = @(u) OceanDL.Autograd.F.ctcLoss(u, targets, 'InputLengths', [5 4 5 0]);
            r = OceanDL.Autograd.GradCheck.check(f, {randn(3, 4, 5)});
            testCase.verifyTrue(r.Passed, sprintf('max err %.3g', r.MaxAbsError));
            testCase.verifyEqual(r.Analytic{1}(:, 4, :), zeros(3, 1, 5));      % T_n = 0: no gradient
            testCase.verifyEqual(r.Analytic{1}(:, 2, 5), zeros(3, 1));         % step past T_n
        end

        function testSinglePrecision(testCase)
            s = RandStream('mt19937ar', 'Seed', 7);
            U = randn(s, 5, 3, 8);
            targets = {[2 3 3], 4, zeros(1, 0)};
            lens = [8 6 3];
            ud = OceanDL.Autograd.Tensor(U, 'RequiresGrad', true);
            us = OceanDL.Autograd.Tensor(single(U), 'RequiresGrad', true);
            Ld = OceanDL.Autograd.F.ctcLoss(ud, targets, 'InputLengths', lens); Ld.backward();
            Ls = OceanDL.Autograd.F.ctcLoss(us, targets, 'InputLengths', lens); Ls.backward();
            testCase.verifyClass(Ls.Data, 'single');
            testCase.verifyClass(us.Grad, 'single');
            testCase.verifyEqual(double(Ls.Data), Ld.Data, 'RelTol', 1e-5);
            testCase.verifyEqual(double(us.Grad), ud.Grad, 'AbsTol', 1e-5);
        end

        function testRejectsBadInputLengths(testCase)
            U = randn(3, 2, 4);
            testCase.verifyError(@() OceanDL.Autograd.F.ctcLoss(U, {2, 3}, 'InputLengths', [4 5]), ...
                'OceanDL:Autograd:CTCLoss:BadInputLengths');
            testCase.verifyError(@() OceanDL.Autograd.F.ctcLoss(U, {2, 3}, 'InputLengths', 4), ...
                'OceanDL:Autograd:CTCLoss:BadInputLengths');
        end
    end
end
