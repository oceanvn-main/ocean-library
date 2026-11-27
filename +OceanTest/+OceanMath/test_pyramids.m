classdef test_pyramids < OceanTest.BaseTestCase
    % TEST_PYRAMIDS  Unit tests for Gaussian and Laplacian Image Pyramids.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-19
    
    methods (Test)
        function testGaussianReductionSize(testCase)
            import OceanMath.Image.MultiResolution.PyramidOperator
            
            % Test even size
            I_even = rand(64, 64, 3);
            I_red_even = PyramidOperator.reduce(I_even);
            testCase.verifySize(I_red_even, [32, 32, 3], 'Reduction of even image size incorrect.');
            
            % Test odd size
            I_odd = rand(65, 65, 3);
            I_red_odd = PyramidOperator.reduce(I_odd);
            testCase.verifySize(I_red_odd, [33, 33, 3], 'Reduction of odd image size incorrect.');
        end
        
        function testLaplacianPerfectReconstruction(testCase)
            import OceanMath.Image.MultiResolution.LaplacianPyramid
            
            % Test sizes to verify even and odd dimensions
            test_sizes = {[64, 64, 3], [65, 65, 3], [128, 128], [129, 129]};
            
            for idx = 1:length(test_sizes)
                sz = test_sizes{idx};
                I = rand(sz);
                
                % Build a 4-level Laplacian pyramid
                pyramid = LaplacianPyramid('Boundary', 'replicate');
                pyramid.build(I, 4);
                
                % Reconstruct the image
                I_rec = pyramid.reconstruct();
                
                % Verify size
                testCase.verifySize(I_rec, sz, 'Reconstructed image size mismatch.');
                
                % Verify value match (perfect reconstruction down to numerical precision)
                testCase.assertMatrixEquals(I, I_rec, 1e-12, ...
                    sprintf('Laplacian reconstruction failed for size: [%s]', num2str(sz)));
            end
        end
        
        function testGaussianPyramidConsistency(testCase)
            import OceanMath.Image.MultiResolution.GaussianPyramid
            
            I = rand(128, 128, 3);
            
            % Build a 3-level Gaussian pyramid
            pyramid = GaussianPyramid('Boundary', 'replicate');
            pyramid.build(I, 3);
            
            % Check level counts and sizes
            testCase.verifyEqual(pyramid.NumLevels, 3, 'Gaussian pyramid levels count mismatch.');
            testCase.verifySize(pyramid.Levels{1}, [128, 128, 3]);
            testCase.verifySize(pyramid.Levels{2}, [64, 64, 3]);
            testCase.verifySize(pyramid.Levels{3}, [32, 32, 3]);
            
            % Reconstructing a Gaussian pyramid just returns the original first level
            I_rec = pyramid.reconstruct();
            testCase.assertMatrixEquals(I, I_rec, 1e-15, 'Gaussian reconstruction failed.');
        end
        
        function testCustomPyramidKernel(testCase)
            import OceanMath.Image.MultiResolution.LaplacianPyramid
            
            I = rand(64, 64, 3);
            
            % Custom 3-tap triangle filter [1 2 1]/4
            custom_kernel = [1.0, 2.0, 1.0] / 4.0;
            
            pyramid = LaplacianPyramid('Kernel', custom_kernel);
            pyramid.build(I, 3);
            
            I_rec = pyramid.reconstruct();
            testCase.assertMatrixEquals(I, I_rec, 1e-12, 'Laplacian reconstruction with custom kernel failed.');
        end
        
        function testObjectOrientedKernels(testCase)
            import OceanMath.Image.MultiResolution.*
            import OceanMath.Signal.Kernels.*
            
            I = rand(64, 64, 3);
            
            % Test 1: GaussianKernel object
            g_kernel = GaussianKernel('Size', 5);
            g_pyramid = GaussianPyramid('Kernel', g_kernel);
            g_pyramid.build(I, 3);
            testCase.verifyEqual(g_pyramid.NumLevels, 3);
            
            % Test 2: TriangleKernel object
            t_kernel = TriangleKernel(3);
            l_pyramid = LaplacianPyramid('Kernel', t_kernel);
            l_pyramid.build(I, 3);
            I_rec = l_pyramid.reconstruct();
            testCase.assertMatrixEquals(I, I_rec, 1e-12, 'Object-oriented TriangleKernel reconstruction failed.');
            
            % Test 3: String-resolved BoxKernel
            box_pyramid = LaplacianPyramid('Kernel', 'box');
            box_pyramid.build(I, 3);
            I_rec_box = box_pyramid.reconstruct();
            testCase.assertMatrixEquals(I, I_rec_box, 1e-12, 'String-resolved BoxKernel reconstruction failed.');
        end
        
        function testEvenSizedKernel(testCase)
            import OceanMath.Image.MultiResolution.GaussianPyramid
            import OceanMath.Signal.Kernels.BoxKernel
            
            I = rand(64, 64, 3);
            
            % Test an even sized kernel (e.g. 2x2 Block) which previously caused 
            % fraction-padding error because pad_cols = (2-1)/2 = 0.5.
            kernel = BoxKernel(2);
            pyramid = GaussianPyramid('Kernel', kernel);
            pyramid.build(I, 3);
            
            testCase.verifyEqual(pyramid.NumLevels, 3, 'Pyramid with even-sized kernel failed to build.');
            testCase.verifySize(pyramid.Levels{2}, [32, 32, 3], 'Incorrect reduction size with even-sized kernel.');
        end
    end
end
