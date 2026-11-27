classdef test_filters < OceanTest.BaseTestCase
    % TEST_FILTERS  Unit tests for spatial filters, bilateral grid, and wavelet lifting.
    
    methods (Test)
        function testBoxFilter(testCase)
            import OceanMath.Image.Filters.BoxFilter
            
            % Setup test image
            I = [ 10, 20, 30, 40;
                  50, 60, 70, 80;
                  90, 80, 70, 60;
                  50, 40, 30, 20 ];
            
            % 3x3 Box filter with constant zero boundary padding
            f_box = BoxFilter('Size', [3, 3], 'Boundary', 'zero');
            I_out = f_box.apply(I);
            
            % Compute expected value manually for center pixel (2, 2)
            % Neighborhood:
            %   10, 20, 30
            %   50, 60, 70
            %   90, 80, 70
            % Sum = 10+20+30+50+60+70+90+80+70 = 480
            % Average = 480 / 9 = 53.3333
            expected_val = 480 / 9;
            testCase.verifyEqual(I_out(2, 2), expected_val, 'RelTol', 1e-10, 'BoxFilter average at (2,2) incorrect.');
            
            % Check O(1) integral image result matches for different sizes
            f_box_5 = BoxFilter('Size', 5, 'Boundary', 'replicate');
            I_out_5 = f_box_5.apply(I);
            testCase.verifySize(I_out_5, [4, 4], 'BoxFilter size incorrect.');
        end
        
        function testMedianFilter(testCase)
            import OceanMath.Image.Filters.MedianFilter
            
            % Setup impulse noise test image
            I = [ 10, 12, 11, 13;
                  12, 99, 14, 12;
                  11, 10, 13, 11;
                  14, 12, 11, 10 ]; % 99 is a salt noise pixel
              
            f_med = MedianFilter('Size', 3, 'Boundary', 'replicate');
            I_out = f_med.apply(I);
            
            % Expected central pixel (2, 2) neighborhood:
            %   10, 12, 11
            %   12, 99, 14
            %   11, 10, 13
            % Sorted elements: [10, 10, 11, 11, 12, 12, 13, 14, 99]
            % Median = 12 (removes 99)
            testCase.verifyEqual(I_out(2, 2), 12.0, 'MedianFilter failed to remove impulse noise.');
        end
        
        function testBilateralFilterAndGrid(testCase)
            import OceanMath.Image.Filters.BilateralFilter
            
            % Setup step edge image
            I = [ 0.1 * ones(8, 4), 0.9 * ones(8, 4) ];
            
            % Direct bilateral filter
            f_dir = BilateralFilter('SpatialSigma', 2.0, 'RangeSigma', 0.15, 'Method', 'direct');
            I_dir = f_dir.apply(I);
            
            % Grid bilateral filter
            f_grid = BilateralFilter('SpatialSigma', 2.0, 'RangeSigma', 0.15, 'Method', 'grid', ...
                                     'SpatialSampling', 1.0, 'RangeSampling', 0.02);
            I_grid = f_grid.apply(I);
            
            % Edge preservation: pixels at col 3 (0.1) and col 5 (0.9) must remain close to original
            testCase.verifyTrue(I_dir(4, 3) < 0.25, 'Direct bilateral filter blurred step edge too much.');
            testCase.verifyTrue(I_dir(4, 5) > 0.75, 'Direct bilateral filter blurred step edge too much.');
            
            testCase.verifyTrue(I_grid(4, 3) < 0.25, 'Grid bilateral filter blurred step edge too much.');
            testCase.verifyTrue(I_grid(4, 5) > 0.75, 'Grid bilateral filter blurred step edge too much.');
            
            % Check that direct and grid methods are close
            testCase.assertMatrixEquals(I_dir, I_grid, 0.15, 'Bilateral Grid output diverges from Direct search.');
        end
        
        function testWaveletLifting(testCase)
            import OceanMath.Signal.Lifting.LiftingScheme
            
            % 1. 1D Signal Haar & CDF 5/3 Lossless Reconstruction
            x = [12; 45; 67; 89; 100; 24; 88; 56];
            
            % Haar DWT
            wl_haar = LiftingScheme('WaveletType', 'Haar', 'Levels', 1);
            coeffs_haar = wl_haar.forward(x);
            x_recon_haar = wl_haar.inverse(coeffs_haar);
            testCase.assertMatrixEquals(x, x_recon_haar, 1e-12, '1D Haar reconstruction failed.');
            
            % CDF 5/3 DWT
            wl_cdf = LiftingScheme('WaveletType', 'CDF53', 'Levels', 2);
            coeffs_cdf = wl_cdf.forward(x);
            x_recon_cdf = wl_cdf.inverse(coeffs_cdf);
            testCase.assertMatrixEquals(x, x_recon_cdf, 1e-12, '1D CDF 5/3 reconstruction failed.');
            
            % 2. 2D Image Haar & CDF 5/3 Lossless Reconstruction
            I = [ 10,  20,  30,  40;
                  50,  60,  70,  80;
                  90,  100, 110, 120;
                  130, 140, 150, 160 ];
              
            coeffs_2d = wl_cdf.forward(I);
            I_recon = wl_cdf.inverse(coeffs_2d);
            testCase.assertMatrixEquals(I, I_recon, 1e-12, '2D CDF 5/3 reconstruction failed.');
        end
    end
end
