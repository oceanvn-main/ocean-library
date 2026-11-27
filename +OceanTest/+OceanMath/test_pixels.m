classdef test_pixels < OceanTest.BaseTestCase
    % TEST_PIXELS  Unit tests for modular pixel transformation package.
    
    methods (Test)
        function testLinearOperator(testCase)
            import OceanMath.Image.Pixels.LinearOperator
            
            % 1. Check double array scaling/bias
            I_double = [0.1, 0.5; 0.8, 1.0];
            op = LinearOperator('Scale', 2.0, 'Bias', -0.1, 'Clip', true);
            I_out = op.apply(I_double);
            % Expected: 2 * [0.1, 0.5; 0.8, 1.0] - 0.1 = [0.1, 0.9; 1.5, 1.9], clipped to [0, 1] -> [0.1, 0.9; 1.0, 1.0]
            expected = [0.1, 0.9; 1.0, 1.0];
            testCase.assertMatrixEquals(I_out, expected, 1e-12, 'LinearOperator on double failed.');
            
            % 2. Check uint8 array scaling/bias with clipping
            I_uint8 = uint8([100, 200; 50, 150]);
            op2 = LinearOperator('Scale', 1.5, 'Bias', 20.0, 'Clip', true);
            I_out2 = op2.apply(I_uint8);
            % Expected: 1.5 * [100, 200; 50, 150] + 20 = [170, 320; 95, 245], clipped -> uint8([170, 255; 95, 245])
            expected2 = uint8([170, 255; 95, 245]);
            testCase.verifyClass(I_out2, 'uint8', 'Output must remain uint8.');
            testCase.assertMatrixEquals(double(I_out2), double(expected2), 1e-12, 'LinearOperator on uint8 failed.');
        end
        
        function testGammaCorrection(testCase)
            import OceanMath.Image.Pixels.GammaCorrection
            
            % Test standard double gamma mapping
            I_double = [0.0, 0.25; 0.64, 1.0];
            op = GammaCorrection('Gamma', 0.5, 'C', 1.0);
            I_out = op.apply(I_double);
            % Expected: sqrt([0.0, 0.25; 0.64, 1.0]) = [0.0, 0.5; 0.8, 1.0]
            expected = [0.0, 0.5; 0.8, 1.0];
            testCase.assertMatrixEquals(I_out, expected, 1e-12, 'GammaCorrection failed.');
        end
        
        function testLookupTableTransform(testCase)
            import OceanMath.Image.Pixels.LookupTableTransform
            
            % 1. Create a threshold lookup table (threshold at 128)
            lut = zeros(256, 1);
            lut(129:256) = 255;
            op = LookupTableTransform(lut);
            
            I_uint8 = uint8([100, 130; 50, 200]);
            I_out = op.apply(I_uint8);
            expected = uint8([0, 255; 0, 255]);
            testCase.assertMatrixEquals(double(I_out), double(expected), 1e-12, 'LookupTableTransform failed.');
        end
        
        function testColorSpaceConverter(testCase)
            import OceanMath.Image.Pixels.ColorSpaceConverter
            
            % RGB to Gray conversion test
            I_rgb = cat(3, ones(3,3), zeros(3,3), zeros(3,3)); % Pure red image
            op = ColorSpaceConverter('RGB2Gray');
            I_gray = op.apply(I_rgb);
            
            % Expected luma: 0.298936021293775 * R + 0.587043074451121 * G + 0.114020904255104 * B
            expected = 0.298936021293775 * ones(3,3);
            testCase.assertMatrixEquals(I_gray, expected, 1e-6, 'ColorSpaceConverter RGB2Gray failed.');
            
            % Gray to RGB channel replication test
            op2 = ColorSpaceConverter('Gray2RGB');
            I_rgb_out = op2.apply(I_gray);
            testCase.verifySize(I_rgb_out, [3, 3, 3], 'Output size must be 3-channel RGB.');
            testCase.assertMatrixEquals(I_rgb_out(:,:,1), I_gray, 1e-12, 'Red channel mismatch.');
            testCase.assertMatrixEquals(I_rgb_out(:,:,2), I_gray, 1e-12, 'Green channel mismatch.');
            testCase.assertMatrixEquals(I_rgb_out(:,:,3), I_gray, 1e-12, 'Blue channel mismatch.');
        end
        
        function testSpatialFilter(testCase)
            import OceanMath.Image.Pixels.SpatialFilter
            
            % Box blur averaging filter test
            I = [10, 10, 10; 10, 90, 10; 10, 10, 10];
            h = 1/9 * ones(3,3);
            op = SpatialFilter('Kernel', h, 'Boundary', 0);
            I_out = op.apply(I);
            
            % Expected central pixel: 1/9 * (10 * 8 + 90) = 170 / 9 = 18.8889
            testCase.verifyEqual(I_out(2,2), 170/9, 'RelTol', 1e-6, 'SpatialFilter averaging failed.');
        end
        
        function testCrossCorrelation(testCase)
            import OceanMath.Image.Pixels.CrossCorrelation
            
            % Asymmetric impulse image
            I = [ 0,  0,  0,  0,  0;
                  0,  0,  0,  0,  0;
                  0,  0, 10,  0,  0;
                  0,  0,  0,  0,  0;
                  0,  0,  0,  0,  0 ];
            
            % Shift kernel (1 at right side)
            h = [ 0,  0,  0;
                  0,  1,  0;
                  0,  0,  0 ];
            % Wait, standard shift: if 1 is on the right side of the center [0, 0, 0; 0, 0, 1; 0, 0, 0]
            h_shift = [ 0,  0,  0;
                        0,  0,  1;
                        0,  0,  0 ];
            
            op = CrossCorrelation('Kernel', h_shift, 'Boundary', 0);
            I_out = op.apply(I);
            
            % Cross-correlation: Out(y, x) = sum_{i, j} I_in(y+i, x+j) * h(i,j)
            % The 10 at (3,3) should shift left to (3,2)
            testCase.verifyEqual(I_out(3,2), 10.0, 'Cross-correlation failed: 10 did not shift left to (3,2).');
            testCase.verifyEqual(I_out(3,3), 0.0, 'Cross-correlation failed: original position should be 0.');
        end
        
        function testTrueConvolution(testCase)
            import OceanMath.Image.Pixels.TrueConvolution
            
            % Asymmetric impulse image
            I = [ 0,  0,  0,  0,  0;
                  0,  0,  0,  0,  0;
                  0,  0, 10,  0,  0;
                  0,  0,  0,  0,  0;
                  0,  0,  0,  0,  0 ];
            
            % Shift kernel (1 at right side)
            h_shift = [ 0,  0,  0;
                        0,  0,  1;
                        0,  0,  0 ];
            
            op = TrueConvolution('Kernel', h_shift, 'Boundary', 0);
            I_out = op.apply(I);
            
            % True convolution: flips kernel 180 degrees -> 1 is at left side [0 0 0; 1 0 0; 0 0 0]
            % The 10 at (3,3) should shift right to (3,4)
            testCase.verifyEqual(I_out(3,4), 10.0, 'True convolution failed: 10 did not shift right to (3,4).');
            testCase.verifyEqual(I_out(3,3), 0.0, 'True convolution failed: original position should be 0.');
        end
        
        function testSeparableConvolution(testCase)
            import OceanMath.Image.Pixels.TrueConvolution
            
            % Generate a random 3D image
            I = rand(24, 24, 3);
            
            % Create a separable 2D kernel (e.g. 5x5 box blur)
            K_sep = 1/25 * ones(5, 5);
            
            % A rank-2 version with a tiny perturbation to bypass the SVD separability check
            K_nonsep = K_sep;
            K_nonsep(1, 1) = K_nonsep(1, 1) + 1e-15;
            
            op_sep = TrueConvolution('Kernel', K_sep, 'Boundary', 'replicate');
            op_nonsep = TrueConvolution('Kernel', K_nonsep, 'Boundary', 'replicate');
            
            I_out_sep = op_sep.apply(I);
            I_out_nonsep = op_nonsep.apply(I);
            
            testCase.assertMatrixEquals(I_out_sep, I_out_nonsep, 1e-12, 'Separable convolution output mismatch.');
        end
        
        function testSeparableCrossCorrelation(testCase)
            import OceanMath.Image.Pixels.CrossCorrelation
            
            % Generate a random 3D image
            I = rand(24, 24, 3);
            
            % Create a separable 2D kernel (e.g. 5x5 box blur)
            K_sep = 1/25 * ones(5, 5);
            
            % A rank-2 version with a tiny perturbation to bypass the SVD separability check
            K_nonsep = K_sep;
            K_nonsep(1, 1) = K_nonsep(1, 1) + 1e-15;
            
            op_sep = CrossCorrelation('Kernel', K_sep, 'Boundary', 'symmetric');
            op_nonsep = CrossCorrelation('Kernel', K_nonsep, 'Boundary', 'symmetric');
            
            I_out_sep = op_sep.apply(I);
            I_out_nonsep = op_nonsep.apply(I);
            
            testCase.assertMatrixEquals(I_out_sep, I_out_nonsep, 1e-12, 'Separable cross-correlation output mismatch.');
        end
        
        function testTransformPipeline(testCase)
            import OceanMath.Image.Pixels.ColorSpaceConverter
            import OceanMath.Image.Pixels.LinearOperator
            import OceanMath.Image.Pixels.TransformPipeline
            
            % Chain: RGB to Gray -> scale by 2.0
            I_rgb = cat(3, 0.5 * ones(2,2), zeros(2,2), zeros(2,2)); % R = 0.5
            pipeline = TransformPipeline({ ...
                ColorSpaceConverter('RGB2Gray'), ...
                LinearOperator('Scale', 2.0, 'Bias', 0.0, 'Clip', true) ...
            });
            I_out = pipeline.apply(I_rgb);
            
            % Expected: 2.0 * (0.298936021293775 * 0.5) = 0.298936021293775
            expected = 0.298936021293775 * ones(2,2);
            testCase.assertMatrixEquals(I_out, expected, 1e-12, 'TransformPipeline chaining failed.');
        end
        
        function testMinMaxStretch(testCase)
            import OceanMath.Image.Pixels.MinMaxStretch
            
            I = [0.2, 0.4; 0.6, 0.8];
            op = MinMaxStretch('TargetRange', [0, 1]);
            I_out = op.apply(I);
            
            expected = [0.0, 1/3; 2/3, 1.0];
            testCase.assertMatrixEquals(I_out, expected, 1e-12, 'MinMaxStretch failed.');
        end
        
        function testMeanMatchStretch(testCase)
            import OceanMath.Image.Pixels.MeanMatchStretch
            
            I = [0.2, 0.4; 0.6, 0.8]; % Mean = 0.5
            op = MeanMatchStretch('TargetMean', 0.5, 'Scale', 1.5);
            I_out = op.apply(I);
            
            expected = [0.05, 0.35; 0.65, 0.95];
            testCase.assertMatrixEquals(I_out, expected, 1e-12, 'MeanMatchStretch failed.');
        end
        
        function testHistogramEqualization(testCase)
            import OceanMath.Image.Pixels.HistogramEqualization
            
            % 4x4 image with 4 distinct gray values
            I = uint8(repmat([50, 100; 150, 200], 2, 2));
            op = HistogramEqualization('Alpha', 1.0);
            I_out = op.apply(I);
            
            % Expected full equalization mapping:
            % 50 -> 255 * 4/16 = 63.75 -> uint8 64
            % 100 -> 255 * 8/16 = 127.5 -> uint8 128
            % 150 -> 255 * 12/16 = 191.25 -> uint8 191
            % 200 -> 255 * 16/16 = 255.0 -> uint8 255
            expected = uint8(repmat([64, 128; 191, 255], 2, 2));
            testCase.assertMatrixEquals(double(I_out), double(expected), 1e-12, 'HistogramEqualization failed.');
            
            % Check property values
            testCase.verifySize(op.Histogram, [256, 1], 'Histogram property size mismatch.');
            testCase.verifySize(op.CDF, [256, 1], 'CDF property size mismatch.');
            testCase.verifySize(op.TransferFunction, [256, 1], 'TransferFunction property size mismatch.');
            
            testCase.verifyEqual(op.Histogram(51), 4.0, 'Histogram count at 50 failed.');
            testCase.verifyEqual(op.CDF(101), 8.0, 'CDF count at 100 failed.');
            testCase.verifyEqual(op.TransferFunction(151), 191.25, 'Transfer function at 150 failed.');
        end
        
        function testAdaptiveHistogramEqualization(testCase)
            import OceanMath.Image.Pixels.AdaptiveHistogramEqualization
            
            % 1. Check double array range preservation and dynamic bounds
            I = [0.1 * ones(4,4), 0.2 * ones(4,4); 0.3 * ones(4,4), 0.4 * ones(4,4)]; % 8x8 image
            op = AdaptiveHistogramEqualization('NumTiles', [2, 2], 'ClipLimit', 0.05);
            I_out = op.apply(I);
            
            testCase.verifySize(I_out, [8, 8], 'Output size mismatch.');
            testCase.verifyClass(I_out, 'double', 'Output class mismatch.');
            testCase.verifyTrue(all(I_out(:) >= 0.0 & I_out(:) <= 1.0), 'Values must be in range [0, 1].');
            
            % 2. Check uniform image returns uniform output
            I_uniform = uint8(128 * ones(16, 16));
            op2 = AdaptiveHistogramEqualization('NumTiles', [4, 4], 'ClipLimit', 0.02);
            I_out2 = op2.apply(I_uniform);
            
            testCase.verifyClass(I_out2, 'uint8', 'Output must remain uint8.');
            testCase.verifyTrue(all(I_out2(:) == I_out2(1)), 'Output of uniform image must be uniform.');
        end
        
        function testBoundaryPadder(testCase)
            import OceanMath.Image.Pixels.BoundaryPadder
            
            % Test matrix
            I = [1, 2; 3, 4];
            
            % 1. Zero/Constant Padding
            I_zero = BoundaryPadder.pad(I, [1, 1], 'zero');
            expected_zero = [0, 0, 0, 0; 0, 1, 2, 0; 0, 3, 4, 0; 0, 0, 0, 0];
            testCase.assertMatrixEquals(I_zero, expected_zero, 1e-12, 'Zero padding failed.');
            
            % 2. Replicate/Clamp Padding
            I_repl = BoundaryPadder.pad(I, [1, 1], 'replicate');
            expected_repl = [1, 1, 2, 2; 1, 1, 2, 2; 3, 3, 4, 4; 3, 3, 4, 4];
            testCase.assertMatrixEquals(I_repl, expected_repl, 1e-12, 'Replicate padding failed.');
            
            % 3. Symmetric Padding
            I_sym = BoundaryPadder.pad(I, [1, 1], 'symmetric');
            expected_sym = [1, 1, 2, 2; 1, 1, 2, 2; 3, 3, 4, 4; 3, 3, 4, 4];
            testCase.assertMatrixEquals(I_sym, expected_sym, 1e-12, 'Symmetric padding failed.');
            
            % 4. Circular/Wrap Padding
            I_circ = BoundaryPadder.pad(I, [1, 1], 'circular');
            expected_circ = [4, 3, 4, 3; 2, 1, 2, 1; 4, 3, 4, 3; 2, 1, 2, 1];
            testCase.assertMatrixEquals(I_circ, expected_circ, 1e-12, 'Circular padding failed.');
            
            % 5. Reflect/Mirror Padding
            I_refl = BoundaryPadder.pad(I, [1, 1], 'reflect');
            expected_refl = [4, 3, 4, 3; 2, 1, 2, 1; 4, 3, 4, 3; 2, 1, 2, 1];
            testCase.assertMatrixEquals(I_refl, expected_refl, 1e-12, 'Reflect padding failed.');
            
            % 6. Extend Padding
            I_ext = BoundaryPadder.pad(I, [1, 1], 'extend');
            expected_ext = [
                -2, -1,  0,  1;
                 0,  1,  2,  3;
                 2,  3,  4,  5;
                 4,  5,  6,  7
            ];
            testCase.assertMatrixEquals(I_ext, expected_ext, 1e-12, 'Extend padding failed.');
            
            % 7. Class Preservation
            I_uint8 = uint8(I);
            I_pad_uint8 = BoundaryPadder.pad(I_uint8, [1, 1], 'extend');
            testCase.verifyClass(I_pad_uint8, 'uint8', 'Output must remain uint8.');
        end
        
        function testFilterKernels(testCase)
            import OceanMath.Signal.Kernels.*
            import OceanMath.Image.Pixels.TrueConvolution
            import OceanMath.Image.Pixels.CrossCorrelation
            
            % 1. BoxKernel test
            kb = BoxKernel(5);
            testCase.verifyEqual(kb.getSize(), [5, 5], 'BoxKernel size incorrect.');
            testCase.verifyTrue(kb.isSeparable(), 'BoxKernel must be separable.');
            [h_b, v_b] = kb.get1DDecomposition();
            testCase.assertMatrixEquals(kb.getMatrix(), v_b * h_b, 1e-12, 'BoxKernel decomposition incorrect.');
            
            % 2. TriangleKernel test
            kt = TriangleKernel(3);
            testCase.verifyTrue(kt.isSeparable(), 'TriangleKernel must be separable.');
            [h_t, v_t] = kt.get1DDecomposition();
            testCase.assertMatrixEquals(kt.getMatrix(), v_t * h_t, 1e-12, 'TriangleKernel decomposition incorrect.');
            
            % 3. GaussianKernel test
            kg = GaussianKernel('Size', 5);
            testCase.verifyTrue(kg.isSeparable(), 'GaussianKernel must be separable.');
            [h_g, v_g] = kg.get1DDecomposition();
            testCase.assertMatrixEquals(kg.getMatrix(), v_g * h_g, 1e-12, 'GaussianKernel decomposition incorrect.');
            
            % 4. SobelKernel test
            ks = SobelKernel('Direction', 'horizontal', 'Size', 3);
            testCase.verifyTrue(ks.isSeparable(), 'SobelKernel must be separable.');
            [h_s, v_s] = ks.get1DDecomposition();
            testCase.assertMatrixEquals(ks.getMatrix(), v_s * h_s, 1e-12, 'SobelKernel decomposition incorrect.');
            
            % 5. SecondDerivativeKernel test
            kc = SecondDerivativeKernel(3);
            testCase.verifyTrue(kc.isSeparable(), 'SecondDerivativeKernel must be separable.');
            [h_c, v_c] = kc.get1DDecomposition();
            testCase.assertMatrixEquals(kc.getMatrix(), v_c * h_c, 1e-12, 'SecondDerivativeKernel decomposition incorrect.');
            
            % 6. Integration with TrueConvolution and CrossCorrelation
            I = rand(10, 10);
            op_conv = TrueConvolution('Kernel', kb, 'Boundary', 'replicate');
            op_corr = CrossCorrelation('Kernel', kb, 'Boundary', 'replicate');
            
            % Verify that they apply and return outputs of same sizes
            I_conv = op_conv.apply(I);
            I_corr = op_corr.apply(I);
            testCase.verifySize(I_conv, [10, 10]);
            testCase.verifySize(I_corr, [10, 10]);
            
            % 7. CustomKernel adapter test
            k_mat = [1, 2, 1; 2, 4, 2; 1, 2, 1] / 16;
            k_cust = CustomKernel(k_mat);
            testCase.verifyTrue(k_cust.isSeparable(), 'Custom separable kernel not recognized as separable.');
            [h_cust, v_cust] = k_cust.get1DDecomposition();
            testCase.assertMatrixEquals(k_cust.getMatrix(), v_cust * h_cust, 1e-12, 'CustomKernel decomposition mismatch.');
            
            % 8. KernelFactory resolution test
            k_fact1 = KernelFactory.resolve('triangle');
            testCase.verifyClass(k_fact1, 'OceanMath.Signal.Kernels.TriangleKernel', 'Factory failed to resolve string to TriangleKernel.');
            
            k_fact2 = KernelFactory.resolve(k_mat);
            testCase.verifyClass(k_fact2, 'OceanMath.Signal.Kernels.CustomKernel', 'Factory failed to resolve numeric matrix to CustomKernel.');
            
            k_fact3 = KernelFactory.resolve(kb);
            testCase.verifyEqual(k_fact3, kb, 'Factory failed to resolve existing IFilterKernel.');
        end
    end
end

