function tests = test_rainbow_inpainting
    % TEST_RAINBOW_INPAINTING Unit tests for the Exercise 3.29 inpainting pipeline.
    tests = functiontests(localfunctions);
end

function testGammaInvertibility(testCase)
    % Create random image double array
    I = rand(64, 64, 3);
    
    % Perform linearization (expansion)
    L = I .^ 2.2;
    
    % Perform compression
    I_recon = L .^ (1/2.2);
    
    % Verify reconstruction is identical within standard floating tolerance
    testCase.verifyEqual(I_recon, I, 'AbsTol', 1e-12);
end

function testBackgroundInpainting(testCase)
    % Create a 32x32 image with a flat constant color (e.g., 0.5)
    I_clean = ones(32, 32) * 0.5;
    
    % Create a logical mask at the center (10x10 square)
    mask = false(32, 32);
    mask(12:21, 12:21) = true;
    
    % Corrupt the image inside the mask with random values
    I_corrupted = I_clean;
    I_corrupted(mask) = rand(100, 1);
    
    % Perform background synthesis using regionfill
    I_synthesized = regionfill(I_corrupted, mask);
    
    % Since the surrounding sky was constant (0.5), Laplacian interpolation
    % must reconstruct a flat 0.5 across the entire masked region.
    testCase.verifyEqual(I_synthesized, I_clean, 'AbsTol', 1e-10);
end

function testZeroClamping(testCase)
    % Generate dummy total and background matrices
    L_total = [0.2, 0.4; 0.5, 0.1];
    L_bg = [0.3, 0.3; 0.4, 0.2];
    
    % Perform subtraction with clamping
    L_rainbow = max(L_total - L_bg, 0);
    
    % Check that minimum is strictly non-negative
    testCase.verifyTrue(all(L_rainbow(:) >= 0.0));
    
    % Check individual elements
    expected = [0.0, 0.1; 0.1, 0.0];
    testCase.verifyEqual(L_rainbow, expected, 'AbsTol', 1e-12);
end
