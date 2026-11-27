classdef DimFormatTest < OceanTest.BaseTestCase
    % DIMFORMATTEST  The import contract: axis relabelling to the canonical
    %   feature-by-observation layouts, row-major buffer decoding, validation
    %   messages, and NpyReader on files written by real NumPy.

    methods (Test)
        function testSequenceFromPyTorchBatchFirst(testCase)
            X = reshape(1:2*5*3, 2, 5, 3);                 % (N=2, T=5, D=3) = "BTC"
            [Y, ~, target] = OceanDL.Layout.DimFormat.toCanonical(X, "BTC");
            testCase.verifyEqual(target, "CBT");
            testCase.verifySize(Y, [3 2 5]);
            testCase.verifyEqual(Y(3, 2, 4), X(2, 4, 3));   % Y(d,n,t) = X(n,t,d)
        end

        function testImagesFromNCHW(testCase)
            X = reshape(1:2*3*4*5, 2, 3, 4, 5);            % (N, C, H, W) = "BCSS"
            Y = OceanDL.Layout.DimFormat.toCanonical(X, "BCSS");
            testCase.verifySize(Y, [4 5 3 2]);             % H x W x C x N
            testCase.verifyEqual(Y(3, 5, 2, 1), X(1, 2, 3, 5));
        end

        function testTableRowsBecomeColumns(testCase)
            T = table([1; 2; 3], [10; 20; 30], 'VariableNames', {'a', 'b'});
            testCase.verifyEqual(OceanDL.Layout.DimFormat.fromTable(T), [1 2 3; 10 20 30]);
            T.name = ["x"; "y"; "z"];
            testCase.verifyError(@() OceanDL.Layout.DimFormat.fromTable(T), ...
                'OceanDL:Layout:DimFormat:NonNumericColumns');
        end

        function testFromRowMajorKeepsIndexMeaning(testCase)
            % C-order buffer of shape (2,3,4): element (i,j,k) sits at (i-1)*12 + (j-1)*4 + (k-1).
            A = OceanDL.Layout.DimFormat.fromRowMajor(0:23, [2 3 4]);
            testCase.verifySize(A, [2 3 4]);
            testCase.verifyEqual(A(2, 3, 1), 1*12 + 2*4 + 0);
            testCase.verifyEqual(A(1, 2, 4), 0*12 + 1*4 + 3);
        end

        function testValidateCatchesTransposedFeatures(testCase)
            X = randn(64, 8, 10);                           % 64 obs x 8 feat x 10 steps: wrong layout
            try
                OceanDL.Layout.DimFormat.validate(X, "CBT", 'FeatureDim', 8);
                testCase.verifyFail('validate should have thrown');
            catch err
                testCase.verifyEqual(err.identifier, 'OceanDL:Layout:DimFormat:FeatureDimMismatch');
                testCase.verifySubstring(err.message, 'Axis 2 has size 8');
            end
        end

        function testValidateMaskRules(testCase)
            X = zeros(4, 2, 3);
            OceanDL.Layout.DimFormat.validate(X, "CBT", 'Mask', logical([1 1 0; 1 1 1]));
            testCase.verifyError(@() OceanDL.Layout.DimFormat.validate(X, "CBT", ...
                'Mask', logical([1 0 1; 1 1 1])), 'OceanDL:Layout:DimFormat:MaskNotRightPadded');
            testCase.verifyError(@() OceanDL.Layout.DimFormat.validate(X, "CBT", ...
                'Mask', true(3, 2)), 'OceanDL:Layout:DimFormat:MaskSize');
            testCase.verifyError(@() OceanDL.Layout.DimFormat.validate([1 NaN], "CB"), ...
                'OceanDL:Layout:DimFormat:NonFinite');
            testCase.verifyEqual(OceanDL.Layout.DimFormat.maskFromLengths([2; 3], 3), ...
                logical([1 1 0; 1 1 1]));
        end

        function testBadFormatsRejected(testCase)
            testCase.verifyError(@() OceanDL.Layout.DimFormat.toCanonical(zeros(2, 3), "BX"), ...
                'OceanDL:Layout:DimFormat:BadLabel');
            testCase.verifyError(@() OceanDL.Layout.DimFormat.toCanonical(zeros(2, 3, 4), "BC"), ...
                'OceanDL:Layout:DimFormat:BadRank');
        end

        function testNpyReaderOnNumpyFiles(testCase)
            fx = OceanTest.OceanDL.Layout.DimFormatTest.fixtures();
            [A, info] = OceanData.Adapters.NpyReader.read(fullfile(fx, 'npy_c_f32.npy'));
            testCase.verifyFalse(info.FortranOrder);
            testCase.verifyClass(A, 'single');
            testCase.verifyEqual(double(A), OceanDL.Layout.DimFormat.fromRowMajor(0:23, [2 3 4]));
            [B, info] = OceanData.Adapters.NpyReader.read(fullfile(fx, 'npy_f_i64.npy'));
            testCase.verifyTrue(info.FortranOrder);
            testCase.verifyEqual(B, int64([0 1 2; 3 4 5]));         % np.arange(6).reshape(2,3)
        end

        function testNpyReaderHandWrittenVersion1File(testCase)
            path = [tempname '.npy'];
            cleaner = onCleanup(@() delete(path));
            header = '{''descr'': ''<f8'', ''fortran_order'': False, ''shape'': (2, 2), }';
            header = [header, repmat(' ', 1, 64 - mod(10 + numel(header) + 1, 64)), newline];
            fid = fopen(path, 'w', 'ieee-le');
            fwrite(fid, [147 double('NUMPY') 1 0], 'uint8');
            fwrite(fid, numel(header), 'uint16');
            fwrite(fid, header, 'char');
            fwrite(fid, [1 2 3 4], 'double');                       % C order: [[1,2],[3,4]]
            fclose(fid);
            testCase.verifyEqual(OceanData.Adapters.NpyReader.read(path), [1 2; 3 4]);
        end
    end

    methods (Static)
        function d = fixtures()
            d = fullfile(fileparts(fileparts(mfilename('fullpath'))), '+Recurrent', 'fixtures');
        end
    end
end
