function tests = test_rainbow
    % TEST_RAINBOW Unit tests for RainbowSeparator and related algorithms.
    tests = functiontests(localfunctions);
end

function testCircleFitting(testCase)
    import OceanMath.Image.Resampling.RainbowSeparator;
    
    % Define exact circle parameters
    xc_gt = 120.5;
    yc_gt = -45.2;
    R_gt = 75.3;
    
    % Generate 6 points along the circle
    theta = linspace(0, 1.5*pi, 6);
    pts_x = xc_gt + R_gt * cos(theta);
    pts_y = yc_gt + R_gt * sin(theta);
    pts = [pts_x; pts_y];
    
    % Fit circle
    [xc_fit, yc_fit, R_fit] = RainbowSeparator.fitCircle(pts);
    
    % Verify parameters (absolute tolerance 1e-10)
    testCase.verifyEqual(xc_fit, xc_gt, 'AbsTol', 1e-10);
    testCase.verifyEqual(yc_fit, yc_gt, 'AbsTol', 1e-10);
    testCase.verifyEqual(R_fit, R_gt, 'AbsTol', 1e-10);
end

function testPolarUnrolling(testCase)
    import OceanMath.Image.Resampling.RainbowSeparator;
    
    % Create a simple 100x100 RGB image
    I = rand(100, 100, 3);
    
    xc = 50; yc = 50;
    r_range = [10, 40];
    theta_range = [-pi/2, pi/2];
    num_r = 30;
    num_theta = 90;
    
    I_polar = RainbowSeparator.unrollPolar(I, xc, yc, r_range, theta_range, num_r, num_theta);
    
    % Verify shape
    testCase.verifyEqual(size(I_polar), [num_r, num_theta, 3]);
    % Verify that values are within standard double intensity range [0, 1]
    testCase.verifyTrue(all(I_polar(:) >= 0.0) && all(I_polar(:) <= 1.0));
end

function testBackgroundSeparation(testCase)
    import OceanMath.Image.Resampling.RainbowSeparator;
    
    r_grid = 1:100;
    
    % Define true background (quadratic trend)
    bg_true = 0.5 - 0.00003 * (r_grid - 50).^2;
    
    % Define true additive rainbow signal (Gaussian peak at r = 50)
    peak_true = 0.35 * exp(-((r_grid - 50).^2) / (2 * 4^2));
    
    % Total captured intensity profile
    profile_total = bg_true + peak_true;
    
    % Format into dummy unrolled polar image shape: [num_r x num_theta x channels]
    % We will replicate the profile along the theta dimension (since it is constant along theta)
    num_theta = 10;
    I_polar = repmat(reshape(profile_total, [100, 1, 1]), [1, num_theta, 1]);
    
    % Rainbow range for background exclusion during fitting
    r_rainbow_range = [35, 65];
    
    % Perform separation
    poly_order = 2;
    [R_est, B_est] = RainbowSeparator.separateProfile(I_polar, r_grid, r_rainbow_range, poly_order);
    
    % Verify background reconstruction outside the peak region is accurate
    % We check points far away from the peak, e.g. at r = 10 and r = 90
    testCase.verifyEqual(B_est(10), bg_true(10), 'RelTol', 1e-4);
    testCase.verifyEqual(B_est(90), bg_true(90), 'RelTol', 1e-4);
    
    % Verify that the isolated peak is positive and matches the original peak
    testCase.verifyTrue(all(R_est >= 0));
    testCase.verifyEqual(R_est(50), peak_true(50), 'RelTol', 1e-2);
end

function test2DReconstruction(testCase)
    import OceanMath.Image.Resampling.RainbowSeparator;
    
    r_grid = 10:50;
    R_profile = ones(numel(r_grid), 3); % Uniform profile
    
    H = 80; W = 80;
    xc = 40; yc = 40;
    theta_range = [-pi, pi];
    
    R_2D = RainbowSeparator.reconstruct2D(R_profile, r_grid, H, W, xc, yc, theta_range);
    
    % Verify shape
    testCase.verifyEqual(size(R_2D), [H, W, 3]);
    
    % Check coordinate at center (radius is 0, which is outside r_grid=[10, 50], so should interpolate to 0)
    testCase.verifyEqual(R_2D(40, 40, 1), 0.0);
    
    % Check coordinate at radius 30 (which is inside r_grid=[10, 50], so should interpolate to 1.0)
    % Point at (40, 70) has radius 30 from (40, 40)
    testCase.verifyEqual(R_2D(40, 70, 1), 1.0, 'AbsTol', 1e-5);
end
