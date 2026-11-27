classdef VisionSequenceTest < OceanTest.BaseTestCase
    % VISIONSEQUENCETEST  CV side of the recurrent library: the USPS import
    %   contract on real data, image -> sequence adapters, and ReNet2D.

    methods (Test)
        function testUspsImportIsOrientedCorrectly(testCase)
            % Real-data orientation check: the mean "1" is a VERTICAL stroke,
            % so its intensity varies across columns (x) but hardly down rows
            % (y). A transposed import would silently swap the two.
            [X, y] = OceanTest.OceanDL.Recurrent.VisionSequenceTest.importUsps("test");
            testCase.verifySize(X, [16 16 1 2007]);
            ones_ = mean(X(:, :, 1, y == 1), 4);
            acrossColumns = var(mean(ones_, 1));          % profile along x
            downRows = var(mean(ones_, 2));               % profile along y
            testCase.verifyGreaterThan(acrossColumns, 5 * downRows);
            testCase.verifyEqual([min(X(:)) max(X(:))], [-1 1]);
        end

        function testRowsColumnsPixelsAdapters(testCase)
            X = reshape(1:4*3*2*2, 4, 3, 2, 2);          % H=4, W=3, C=2, N=2
            import OceanDL.Recurrent.ImageSequence
            S = ImageSequence.toSequence(X, "rows");
            testCase.verifySize(S, [6 2 4]);
            testCase.verifyEqual(S(:, 2, 3), reshape(X(3, :, :, 2), [], 1));      % row 3 of image 2
            S = ImageSequence.toSequence(X, "columns");
            testCase.verifySize(S, [8 2 3]);
            testCase.verifyEqual(S(:, 1, 2), reshape(X(:, 2, :, 1), [], 1));      % column 2 of image 1
            S = ImageSequence.toSequence(X, "pixels");
            testCase.verifySize(S, [2 2 12]);
            testCase.verifyEqual(S(:, 2, 7), squeeze(X(3, 2, :, 2)));              % pixel (3,2): t = 3 + 4*(2-1)
        end

        function testAdaptersAreDifferentiable(testCase)
            r = OceanDL.Autograd.GradCheck.check( ...
                @(x) OceanDL.Recurrent.ImageSequence.toSequence(x, "rows"), {randn(4, 3, 2, 2)});
            testCase.verifyTrue(r.Passed);
        end

        function testReNetShapeAndFullReceptiveField(testCase)
            % Every output position must depend on EVERY input pixel after the
            % vertical + horizontal bidirectional sweeps.
            net = OceanDL.Recurrent.ReNet2D(1, 2, 3, 4, 'Seed', 1);
            x = OceanDL.Autograd.Tensor(randn(8, 8, 1, 2), 'RequiresGrad', true);
            Y = net.forward(x);
            testCase.verifySize(Y.Data, [4 4 8 2]);
            backward(sum(Y(1, 1, :, 1), 'all'));          % one corner output of image 1
            testCase.verifyTrue(all(x.Grad(:, :, 1, 1) ~= 0, 'all'));
            testCase.verifyEqual(x.Grad(:, :, 1, 2), zeros(8, 8));   % other images untouched
        end

        function testReNetGradcheck(testCase)
            net = OceanDL.Recurrent.ReNet2D(1, 2, 2, 2, 'CellType', "elman", 'Seed', 2);
            r = OceanDL.Autograd.GradCheck.check(@(x) net.forward(x), {randn(4, 4, 1, 2)});
            testCase.verifyTrue(r.Passed, sprintf('max err %.3g', r.MaxAbsError));
        end
    end

    methods (Static)
        function [X, y] = importUsps(split)
            % The explicit import step: source rows -> canonical SSCB, validated.
            import OceanDL.Layout.DimFormat
            [rows, y, fmt] = OceanData.Adapters.UspsDigitsReader.read(split);
            X = DimFormat.fromRowMajor(reshape(rows.', 1, []), fmt.Shape);   % (N,16,16,1), index meaning kept
            X = DimFormat.toCanonical(X, fmt.Format);                          % -> 16 x 16 x 1 x N
            DimFormat.validate(X, "SSCB", 'FeatureDim', 1, 'BatchSize', numel(y));
        end
    end
end
