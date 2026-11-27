classdef SequenceMaskTest < OceanTest.BaseTestCase
    % SEQUENCEMASKTEST  fromLengths (right-padded validity mask) and apply
    %   (zero padded CBT steps, differentiable: zero gradient on padding).

    methods (Test)
        function testFromLengthsMarksLeadingSteps(testCase)
            lens = [3 0 5 1];
            M = OceanDL.Layout.SequenceMask.fromLengths(lens, 6);
            testCase.verifyClass(M, 'logical');
            testCase.verifySize(M, [4 6]);
            for n = 1:4
                testCase.verifyEqual(find(M(n, :)), 1:lens(n));       % valid steps first, then padding
            end
            testCase.verifySize(OceanDL.Layout.SequenceMask.fromLengths([2; 4]), [2 4]);   % T = max, column ok
            testCase.verifySize(OceanDL.Layout.SequenceMask.fromLengths([0 0], 0), [2 0]);
            testCase.verifySize(OceanDL.Layout.SequenceMask.fromLengths(zeros(1, 0), 3), [0 3]);
        end

        function testFromLengthsRejectsTooLong(testCase)
            testCase.verifyError(@() OceanDL.Layout.SequenceMask.fromLengths([2 7], 5), ...
                'OceanDL:Layout:SequenceMask:LengthExceedsT');
            testCase.verifyError(@() OceanDL.Layout.SequenceMask.fromLengths([2 -1], 5), ...
                'MATLAB:validators:mustBeNonnegative');
        end

        function testApplyOnArrayZeroesPaddingOnly(testCase)
            s = RandStream('mt19937ar', 'Seed', 1);
            X = randn(s, 3, 2, 4);
            M = OceanDL.Layout.SequenceMask.fromLengths([4 2], 4);
            Y = OceanDL.Layout.SequenceMask.apply(X, M);
            testCase.verifyEqual(Y(:, 1, :), X(:, 1, :));
            testCase.verifyEqual(Y(:, 2, 1:2), X(:, 2, 1:2));
            testCase.verifyEqual(Y(:, 2, 3:4), zeros(3, 1, 2));
            Ys = OceanDL.Layout.SequenceMask.apply(single(X), M);
            testCase.verifyClass(Ys, 'single');
        end

        function testApplyOnTensorIsDifferentiable(testCase)
            s = RandStream('mt19937ar', 'Seed', 2);
            X = OceanDL.Autograd.Tensor(randn(s, 3, 3, 5), 'RequiresGrad', true);
            M = OceanDL.Layout.SequenceMask.fromLengths([5 1 3], 5);
            Y = OceanDL.Layout.SequenceMask.apply(X, M);
            testCase.verifyClass(Y, 'OceanDL.Autograd.Tensor');
            R = randn(s, 3, 3, 5);
            L = sum(Y .* R, 'all');
            L.backward();
            expected = R .* reshape(double(M), 1, 3, 5);                  % dL/dX = mask .* dL/dY
            testCase.verifyEqual(X.Grad, expected, 'AbsTol', 1e-15);
            testCase.verifyEqual(X.Grad(:, 2, 2:5), zeros(3, 1, 4));     % padded steps: zero gradient
            r = OceanDL.Autograd.GradCheck.check(@(x) OceanDL.Layout.SequenceMask.apply(x, M), {randn(s, 3, 3, 5)});
            testCase.verifyTrue(r.Passed, sprintf('max err %.3g', r.MaxAbsError));
        end

        function testApplyRejectsMismatchedMask(testCase)
            M = OceanDL.Layout.SequenceMask.fromLengths([2 3], 3);
            testCase.verifyError(@() OceanDL.Layout.SequenceMask.apply(randn(4, 2, 5), M), ...
                'OceanDL:Layout:SequenceMask:SizeMismatch');
            testCase.verifyError(@() OceanDL.Layout.SequenceMask.apply(randn(4, 3, 3), M), ...
                'OceanDL:Layout:SequenceMask:SizeMismatch');
        end

        function testAgreesWithConv1dOutputLengths(testCase)
            % The intended pipeline: propagate lengths through Conv1d, mask its output.
            conv = OceanDL.nn.Conv1d(2, 2, 3, 'Stride', 2, 'Padding', 1, 'Seed', 1);
            lens = [9 4];
            Y = conv.forward(randn(2, 2, 9));
            outLens = conv.outputLength(lens);
            M = OceanDL.Layout.SequenceMask.fromLengths(outLens, size(Y, 3));
            Z = OceanDL.Layout.SequenceMask.apply(Y, M);
            testCase.verifyEqual(squeeze(any(Z.Data ~= 0, 1)), M);
        end
    end
end
