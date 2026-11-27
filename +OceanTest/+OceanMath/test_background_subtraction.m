classdef test_background_subtraction < OceanTest.BaseTestCase
    % TEST_BACKGROUND_SUBTRACTION Tests the BackgroundSubtractor class.
    %
    %   Author: Ocean Library System
    %   Date: 2026-07-08
    
    methods (Test)
        function testConstructor(testCase)
            import OceanMath.Image.Resampling.BackgroundSubtractor
            
            % Generate dummy training frames: 10 frames of 20x20x3
            H = 20; W = 20; C = 3; N = 10;
            trainFrames = rand(H, W, C, N);
            
            % Instantiate
            bs = BackgroundSubtractor(trainFrames);
            
            % Verify dimensions
            testCase.verifyEqual(size(bs.CentralTendency), [H, W, C], 'CentralTendency size is incorrect');
            testCase.verifyEqual(size(bs.Variance), [H, W, C], 'Variance size is incorrect');
            
            % Verify variance is lower-bounded to prevent division by zero
            testCase.verifyTrue(all(bs.Variance(:) >= 1e-4), 'Variance not properly lower-bounded');
        end
        
        function testApply(testCase)
            import OceanMath.Image.Resampling.BackgroundSubtractor
            
            % Create static background with zero variance (will be regularized to 1e-4)
            H = 10; W = 10; C = 3; N = 5;
            baseFrame = 0.5 * ones(H, W, C);
            trainFrames = repmat(baseFrame, [1, 1, 1, N]);
            
            bs = BackgroundSubtractor(trainFrames);
            
            % Test 1: Frame identical to mean should yield zero foreground mask
            [fgMask, foreground] = bs.apply(baseFrame, 11.34);
            testCase.verifyEqual(sum(fgMask(:)), 0, 'Static frame should yield zero foreground pixels');
            testCase.verifyEqual(foreground, zeros(H, W, 4), 'Foreground should be completely black RGBA');
            
            % Test 2: Frame with a significant change should yield foreground detections
            testFrame = baseFrame;
            testFrame(3, 4, :) = 1.0; % High intensity pixel (change = 0.5)
            % distance = 3 * (0.5^2) / 1e-4 = 7500 > 11.34
            
            [fgMask, foreground] = bs.apply(testFrame, 11.34);
            testCase.verifyTrue(fgMask(3, 4), 'Changed pixel should be classified as foreground');
            testCase.verifyEqual(foreground(3, 4, :), reshape([1.0, 1.0, 1.0, 1.0], [1, 1, 4]), 'Foreground pixel should match test frame RGBA');
            
            % Remaining pixels in foreground should be zero (black)
            foreground(3, 4, :) = 0;
            testCase.verifyEqual(sum(foreground(:)), 0, 'Background pixels in foreground image should be zeroed');
            
            % Test 3: blendOn
            newBg = 0.8 * ones(H, W, C);
            [~, fgRGBA] = bs.apply(testFrame, 11.34);
            composite = bs.blendOn(fgRGBA, newBg);
            % Pixel (3,4) is foreground (alpha = 1), so composite(3,4,:) should be [1.0, 1.0, 1.0]
            testCase.verifyEqual(composite(3, 4, :), reshape([1.0, 1.0, 1.0], [1, 1, 3]), 'BlendOn foreground pixel is incorrect');
            % Pixel (1,1) is background (alpha = 0), so composite(1,1,:) should be newBg(1,1,:) = 0.8
            testCase.verifyEqual(composite(1, 1, :), reshape([0.8, 0.8, 0.8], [1, 1, 3]), 'BlendOn background pixel is incorrect');
        end
        
        function testPostProcess(testCase)
            import OceanMath.Image.Resampling.BackgroundSubtractor
            
            % Instantiate using trivial background
            H = 50; W = 50; C = 3; N = 2;
            bs = BackgroundSubtractor(zeros(H, W, C, N));
            
            % Create raw mask with noise and one solid component
            % Component 1: 5x5 block at (10:14, 20:24) -> Area = 25, Centroid = (22, 12)
            % Component 2: single isolated noise pixel at (30, 30) -> Area = 1
            fgMask = false(H, W);
            fgMask(10:14, 20:24) = true;
            fgMask(30, 30) = true;
            
            % Run postProcess with MinArea = 10 (filters out isolated noise)
            [cleanMask, centroids] = bs.postProcess(fgMask, 'MinArea', 10, 'MorphSize', 1);
            
            % Verify isolated noise was removed
            testCase.verifyFalse(cleanMask(30, 30), 'Noise pixel should be removed by opening');
            
            % Verify centroid coordinates
            % MATLAB's regionprops centroids are in (x, y) format.
            % Column is X (20:24 center is 22), Row is Y (10:14 center is 12)
            testCase.verifyEqual(size(centroids, 1), 1, 'Should find exactly one component');
            testCase.verifyEqual(centroids(1, :), [22, 12], 'Centroid calculation is incorrect');
        end
        
        function testRobustEstimation(testCase)
            import OceanMath.Image.Resampling.BackgroundSubtractor
            
            % Generate training sequence of 10 frames with static background (0.5)
            H = 8; W = 8; C = 3; N = 10;
            trainFrames = 0.5 * ones(H, W, C, N);
            
            % Introduce a foreground outlier object (1.0) in only 2 of the frames (20% duration)
            trainFrames(3:5, 3:5, :, 1) = 1.0;
            trainFrames(3:5, 3:5, :, 2) = 1.0;
            
            % 1. Train Standard Model (Mean)
            bsStandard = BackgroundSubtractor(trainFrames, 'Method', 'MeanStd');
            % Arithmetic mean: (8 * 0.5 + 2 * 1.0) / 10 = 0.6
            testCase.verifyEqual(bsStandard.CentralTendency(4, 4, 1), 0.6, 'AbsTol', 1e-12, 'Arithmetic mean failed to compute expected polluted value');
            
            % 2. Train Robust Model (MedianMAD)
            bsRobust = BackgroundSubtractor(trainFrames, 'Method', 'MedianMAD');
            % Robust median should be immune to <50% outlier duration: exactly 0.5
            testCase.verifyEqual(bsRobust.CentralTendency(4, 4, 1), 0.5, 'AbsTol', 1e-12, 'Robust median was polluted by outlier');
            
            % Robust variance should also be immune to the outlier (since 80% of deviations from median are 0, MAD is 0)
            % 0 is lower-bounded to 1e-4
            testCase.verifyEqual(bsRobust.Variance(4, 4, 1), 1e-4, 'AbsTol', 1e-12, 'Robust variance was polluted by outlier');
        end
    end
end
