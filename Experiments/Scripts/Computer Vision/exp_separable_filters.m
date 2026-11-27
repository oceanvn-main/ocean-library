% exp_separable_filters.m
%
% Scientific Experiment: Replicating Szeliski's Figure 3.14 (Separable Linear Filters)
%
% Description:
%   This script filters a cropped region of the grayscale ocean image using
%   five separable filter kernels generated via the new IFilterKernel architecture:
%     1. Box filter (5x5)
%     2. Bilinear/Triangle filter (3x3)
%     3. Gaussian filter (5x5 binomial approximation)
%     4. Sobel filter (3x3 horizontal derivative)
%     5. Corner filter (3x3 second-order cross-derivative)
%
%   The filtered results are displayed in a 3x5 grid showing the 2D kernels,
%   the 1D horizontal kernels, and the output images. Signed derivative outputs 
%   (Sobel and Corner) are scaled and shifted to a mid-gray offset for clear visibility.
%
% Author: Antigravity Code Assistant
% Date: 2026-06-15

clear;
close all;
clc;

% Import the custom convolution and kernels
import OceanMath.Image.Pixels.TrueConvolution
import OceanMath.Signal.Kernels.*

%% 1. Set Up Directories and File Paths
script_dir = fileparts(mfilename('fullpath'));
results_dir = fullfile(script_dir, 'Results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
output_snapshot_path = fullfile(results_dir, 'separable_filters_ocean.png');

%% 2. Load and Crop the Grayscale Ocean Image
image_path = fullfile(script_dir, '..', '..', '..', 'Data', 'images', 'Squid-game-symbols.jpg');
if ~exist(image_path, 'file')
    error('exp_separable_filters:ImageNotFound', 'Required image not found: %s', image_path);
end
I_raw = imread(image_path);
if size(I_raw, 3) == 3
    I_raw = im2gray(I_raw);
end
I_full = double(I_raw) / 255.0;

% Resize the full image to 256x256 to observe the entire scene and reduce computation
I_in = imresize(I_full, [256, 256]);

%% 3. Instantiate the 5 Separable Kernels
k_box = BoxKernel(5);
k_bil = TriangleKernel(3);
k_gauss = GaussianKernel('Size', 5); % Defaults to binomial [1 4 6 4 1]
k_sobel = SobelKernel('Direction', 'horizontal', 'Size', 3);
k_corner = SecondDerivativeKernel(3);

%% 4. Apply Filters using TrueConvolution (bypasses SVD for IFilterKernels)
op1 = TrueConvolution('Kernel', k_box, 'Boundary', 'replicate');
I_out1 = op1.apply(I_in);

op2 = TrueConvolution('Kernel', k_bil, 'Boundary', 'replicate');
I_out2 = op2.apply(I_in);

op3 = TrueConvolution('Kernel', k_gauss, 'Boundary', 'replicate');
I_out3 = op3.apply(I_in);

op4 = TrueConvolution('Kernel', k_sobel, 'Boundary', 'replicate');
I_out4_raw = op4.apply(I_in);
% Scale and shift Sobel output (horizontal derivatives range [-1, 1] roughly)
I_out4 = min(max(2.0 * I_out4_raw + 0.5, 0.0), 1.0);

op5 = TrueConvolution('Kernel', k_corner, 'Boundary', 'replicate');
I_out5_raw = op5.apply(I_in);
% Scale and shift Corner output (second derivatives range [-2, 2] roughly)
I_out5 = min(max(4.0 * I_out5_raw + 0.5, 0.0), 1.0);

%%% 5. Render Figure Layout (Dark Premium Theme)
fig = figure('Name', 'Szeliski Figure 3.14 Replicated on Elon Musk Image', ...
             'Color', [0.03, 0.04, 0.05], ...
             'Position', [50, 50, 1280, 800], ...
             'Visible', 'on');

% Prepare grids, titles, and multipliers matching the textbook style
titles = {'(a) box, K = 5 (1/25)', '(b) bilinear (1/16)', '(c) "Gaussian" (1/256)', '(d) Sobel (1/8)', '(e) corner (1/4)'};
titles_1d = {'1D box (1/5)', '1D bilinear (1/4)', '1D "Gaussian" (1/16)', '1D Sobel (1/2)', '1D corner (1/2)'};

% Multiply double kernels by their common denominators to get clean integer grids
K2D_grids = { ...
    25 * k_box.getMatrix(), ...
    16 * k_bil.getMatrix(), ...
    256 * k_gauss.getMatrix(), ...
    8 * k_sobel.getMatrix(), ...
    4 * k_corner.getMatrix() ...
};

[h_box, ~] = k_box.get1DDecomposition();
[h_bil, ~] = k_bil.get1DDecomposition();
[h_gauss, ~] = k_gauss.get1DDecomposition();
[h_sobel, ~] = k_sobel.get1DDecomposition();
[h_corner, ~] = k_corner.get1DDecomposition();

K1D_grids = { ...
    5 * h_box, ...
    4 * h_bil, ...
    16 * h_gauss, ...
    2 * h_sobel, ...
    2 * h_corner ...
};

images_out = {I_out1, I_out2, I_out3, I_out4, I_out5};

% Layout plotting loop
for col = 1:5
    % Row 1: 2D Kernels
    ax_top = subplot(3, 5, col);
    draw_kernel_grid(ax_top, K2D_grids{col});
    title(ax_top, titles{col}, 'Color', [0.9, 0.9, 0.9], 'FontSize', 11, 'FontWeight', 'bold');
    
    % Row 2: 1D Horizontal Kernels
    ax_mid = subplot(3, 5, col + 5);
    draw_kernel_grid(ax_mid, K1D_grids{col});
    title(ax_mid, titles_1d{col}, 'Color', [0.8, 0.8, 0.8], 'FontSize', 10, 'FontWeight', 'normal');
    
    % Row 3: Filtered Output Images
    ax_bot = subplot(3, 5, col + 10);
    imshow(images_out{col}, 'Parent', ax_bot);
    axis(ax_bot, 'image');
    set(ax_bot, 'XColor', [0.2, 0.2, 0.2], 'YColor', [0.2, 0.2, 0.2], ...
                'XTick', [], 'YTick', []);
end

% Set overall title
annotation(fig, 'textbox', [0.05, 0.93, 0.9, 0.06], ...
           'String', 'SEPARABLE LINEAR FILTERS', ...
           'Color', [1.0, 1.0, 1.0], ...
           'FontSize', 16, ...
           'FontWeight', 'bold', ...
           'HorizontalAlignment', 'center', ...
           'LineStyle', 'none');

drawnow;

% Save the visual result forcing high-resolution, unclipped page dimensions
set(fig, 'PaperUnits', 'inches');
set(fig, 'PaperPosition', [0, 0, 12.8, 8.0]);
set(fig, 'PaperSize', [12.8, 8.0]);
print(fig, output_snapshot_path, '-dpng', '-r100');


fprintf('Separable filters replication completed successfully.\n');
fprintf('Snapshot generated successfully: %s\n', output_snapshot_path);

%% Helper Function to Draw Kernel Heatmaps/Grids
function draw_kernel_grid(ax, K_int)
    [rows, cols] = size(K_int);
    
    % Establish symmetric limits around max magnitude for derivative colormaps
    max_val = max(abs(K_int(:)));
    if max_val == 0
        max_val = 1;
    end
    
    % Draw colored grid squares
    imagesc(ax, K_int, [-max_val, max_val]);
    colormap(ax, 'cool'); % Cyan-to-magenta premium range
    axis(ax, 'image');
    set(ax, 'XTick', [], 'YTick', [], 'XColor', 'none', 'YColor', 'none', ...
            'Color', [0.03, 0.04, 0.05]);
    hold(ax, 'on');
    
    % Draw grid line borders
    for r = 0.5:rows+0.5
        line([0.5, cols+0.5], [r, r], 'Color', [0.3, 0.3, 0.35], 'LineWidth', 1.0, 'Parent', ax);
    end
    for c = 0.5:cols+0.5
        line([c, c], [0.5, rows+0.5], 'Color', [0.3, 0.3, 0.35], 'LineWidth', 1.0, 'Parent', ax);
    end
    
    % Overlay integer text coefficients
    for r = 1:rows
        for c = 1:cols
            val = K_int(r, c);
            text(c, r, num2str(val), 'Color', [1.0, 1.0, 1.0], 'FontWeight', 'bold', ...
                 'FontSize', 10, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Parent', ax);
        end
    end
end
