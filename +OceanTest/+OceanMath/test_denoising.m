classdef test_denoising < OceanTest.BaseTestCase
    % TEST_DENOISING  Unit tests for spatial denoising filters (Gaussian and NL-Means).
    
    methods (Test)
        function testGaussianFilter(testCase)
            import OceanMath.Image.Filters.GaussianFilter
            
            % 1. Check double array range preservation and smoothing
            I = [0.1, 0.1, 0.1; 0.1, 0.9, 0.1; 0.1, 0.1, 0.1];
            flt = GaussianFilter('Sigma', 0.5);
            I_out = flt.apply(I);
            
            testCase.verifySize(I_out, [3, 3], 'Output size must match input size.');
            testCase.verifyClass(I_out, 'double', 'Output class must remain double.');
            
            % Smoothing should lower the central peak
            testCase.verifyTrue(I_out(2,2) < 0.9, 'Center pixel should be smoothed.');
            testCase.verifyTrue(I_out(2,2) > 0.1, 'Center pixel should not drop to background.');
            
            % Average intensity should remain close (due to replicate padding preserving boundaries)
            testCase.verifyEqual(mean(I_out(:)), mean(I(:)), 'RelTol', 0.25, 'GaussianFilter should roughly preserve intensity mean.');
            
            % 2. Check uint8 class preservation
            I_uint8 = uint8(255 * I);
            I_out_uint8 = flt.apply(I_uint8);
            testCase.verifyClass(I_out_uint8, 'uint8', 'Output must remain uint8.');
        end
        
        function testNonLocalMeansFilter(testCase)
            import OceanMath.Image.Filters.NonLocalMeansFilter
            
            % Create a simple 2D image
            I = [0.2 * ones(5, 5), 0.8 * ones(5, 5); 0.2 * ones(5, 5), 0.8 * ones(5, 5)]; % 10x10 step edge
            flt = NonLocalMeansFilter('SearchRadius', 3, 'PatchRadius', 1, 'FilteringParameter', 0.1);
            I_out = flt.apply(I);
            
            testCase.verifySize(I_out, [10, 10], 'Output size must match input size.');
            testCase.verifyClass(I_out, 'double', 'Output class must remain double.');
            testCase.verifyTrue(all(I_out(:) >= 0.0 & I_out(:) <= 1.0), 'Values must remain in bounds.');
            
            % Flat regions should be smoothed and maintain close to original values
            testCase.verifyEqual(I_out(2,2), 0.2, 'AbsTol', 0.05, 'Left flat region should remain close to 0.2.');
            testCase.verifyEqual(I_out(2,8), 0.8, 'AbsTol', 0.05, 'Right flat region should remain close to 0.8.');
            
            % Check uint8 class preservation
            I_uint8 = uint8(255 * I);
            I_out_uint8 = flt.apply(I_uint8);
            testCase.verifyClass(I_out_uint8, 'uint8', 'Output must remain uint8.');
        end
    end
end
