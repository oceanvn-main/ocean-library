function I_synth = generate_synthetic_test_image()
    % GENERATE_SYNTHETIC_TEST_IMAGE  Generates a multi-pattern test image for equalization.
    %
    %   Creates an 800x800 RGB synthetic image specifically designed to evaluate
    %   the 6 steps of Szeliski Exercise 3.7:
    %     - Top-Left: Low-contrast radial luma gradient with noise (evaluates gain-limiting).
    %     - Top-Right: Saturated color patches (evaluates color ratio preservation).
    %     - Bottom-Left: Under-exposed gradient (evaluates 5% black/white point punching).
    %     - Bottom-Right: Unequalized color gradients that clip early (evaluates highlight blending).
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-07-09

    W = 800;
    H = 800;
    I_synth = zeros(H, W, 3);
    
    % --- Quadrant 1 (Top-Left): Low-contrast, noisy radial gradient ---
    [x, y] = meshgrid(1:W/2, 1:H/2);
    cx = W/4; cy = H/4;
    dist = sqrt((x-cx).^2 + (y-cy).^2);
    max_d = max(dist(:));
    radial = 40 + 30 * (1 - dist / max_d); % intensity 40 to 70 (low contrast)
    % Add standard Gaussian noise
    noise = randn(size(radial)) * 1.5;
    radial_noisy = max(0, min(255, radial + noise));
    I_synth(1:H/2, 1:W/2, 1) = radial_noisy;
    I_synth(1:H/2, 1:W/2, 2) = radial_noisy;
    I_synth(1:H/2, 1:W/2, 3) = radial_noisy;
    
    % --- Quadrant 2 (Top-Right): Color grid for color ratio test ---
    patch_w = W/8;
    patch_h = H/8;
    colors = [
        1.0, 0.0, 0.0;  % Red
        0.0, 1.0, 0.0;  % Green
        0.0, 0.0, 1.0;  % Blue
        1.0, 1.0, 0.0;  % Yellow
        1.0, 0.0, 1.0;  % Magenta
        0.0, 1.0, 1.0;  % Cyan
        0.5, 0.5, 0.5;  % Grey
        1.0, 1.0, 1.0;  % White
    ];
    for r = 0:3
        for c = 0:3
            color_idx = mod(r*4 + c, 8) + 1;
            col = colors(color_idx, :) * 100; % mid-intensity to test boosting
            r_start = r*patch_h + 1;
            r_end = (r+1)*patch_h;
            c_start = W/2 + c*patch_w + 1;
            c_end = W/2 + (c+1)*patch_w;
            I_synth(r_start:r_end, c_start:c_end, 1) = col(1);
            I_synth(r_start:r_end, c_start:c_end, 2) = col(2);
            I_synth(r_start:r_end, c_start:c_end, 3) = col(3);
        end
    end
    
    % --- Quadrant 3 (Bottom-Left): Under-exposed linear gradient ---
    [x_bl, ~] = meshgrid(1:W/2, 1:H/2);
    under_exposed = 10 + 40 * (x_bl / (W/2)); % values 10 to 50
    I_synth(H/2+1:end, 1:W/2, 1) = under_exposed;
    I_synth(H/2+1:end, 1:W/2, 2) = under_exposed * 0.8; % slight green tint
    I_synth(H/2+1:end, 1:W/2, 3) = under_exposed * 0.6; % slight blue tint
    
    % --- Quadrant 4 (Bottom-Right): Highlight saturated gradients ---
    [x_br, y_br] = meshgrid(1:W/2, 1:H/2);
    grad_R = 150 + 150 * (x_br / (W/2)); % saturates to 255 halfway
    grad_G = 120 + 200 * (y_br / (H/2)); % saturates to 255 halfway
    grad_B = 50 + 100 * (x_br / (W/2));  % does not saturate
    
    I_synth(H/2+1:end, W/2+1:end, 1) = min(255, grad_R);
    I_synth(H/2+1:end, W/2+1:end, 2) = min(255, grad_G);
    I_synth(H/2+1:end, W/2+1:end, 3) = min(255, grad_B);
    
    I_synth = uint8(I_synth);
end
