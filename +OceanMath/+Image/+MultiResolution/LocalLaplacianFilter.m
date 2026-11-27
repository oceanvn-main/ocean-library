classdef LocalLaplacianFilter < handle
    % LOCALLAPLACIANFILTER  Implements the fast discretized Local Laplacian Filter (Paris et al. 2011).
    %
    %   A Local Laplacian Filter (LLF) is a powerful, edge-preserving multiresolution
    %   filter used for local contrast enhancement, detail smoothing (denoising),
    %   and HDR tone mapping. It avoids halos by convolving remapped versions of
    %   the image and interpolating coefficients level-by-level.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-07-15
    
    properties (Access = public)
        NumLevels double {mustBeInteger, mustBePositive} = 4 % Number of pyramid levels
        NumBins double {mustBeInteger, mustBePositive} = 8  % Number of discrete intensity bins
        Sigma double {mustBeNumeric, mustBePositive} = 0.1  % Edge threshold boundary
        Alpha double {mustBeNumeric, mustBePositive} = 1.0  % Detail control (alpha < 1: boost, alpha > 1: smooth)
        Beta double {mustBeNumeric, mustBePositive} = 1.0   % Edge/tone control (beta < 1: compress edges)
        
        % Stored pyramids for visualization
        InputGaussianPyramid                                % GaussianPyramid of the input luminance channel
        InputLaplacianPyramid                               % LaplacianPyramid of the input luminance channel
        BinLaplacianPyramids                                % Cell array of LaplacianPyramid objects for each bin
        OutputLaplacianPyramid                              % Interpolated LaplacianPyramid of the output luminance channel
    end
    
    methods
        function obj = LocalLaplacianFilter(options)
            % LOCALLAPLACIANFILTER  Constructs a Local Laplacian Filter instance.
            %
            %   obj = LocalLaplacianFilter('NumLevels', 4, 'NumBins', 8, 'Sigma', 0.1, 'Alpha', 1.0, 'Beta', 1.0)
            arguments
                options.NumLevels double {mustBeInteger, mustBePositive} = 4
                options.NumBins double {mustBeInteger, mustBePositive} = 8
                options.Sigma double {mustBeNumeric, mustBePositive} = 0.1
                options.Alpha double {mustBeNumeric, mustBePositive} = 1.0
                options.Beta double {mustBeNumeric, mustBePositive} = 1.0
            end
            obj.NumLevels = options.NumLevels;
            obj.NumBins = options.NumBins;
            obj.Sigma = options.Sigma;
            obj.Alpha = options.Alpha;
            obj.Beta = options.Beta;
        end
        
        function I_out = filter(obj, I)
            % FILTER  Applies the Local Laplacian Filter to an input image.
            %
            %   I_out = filter(obj, I)
            %   Automatically detects color vs grayscale, converts color to CIE L*a*b*,
            %   filters the L* luminance channel, and converts back to RGB.
            arguments
                obj
                I {mustBeNumeric}
            end
            
            % Check input type and size
            if size(I, 3) == 3
                % Color Image: Convert to CIE L*a*b* to process luminance only
                lab = rgb2lab(I);
                L = lab(:, :, 1) / 100.0; % Normalize L* channel from [0, 100] to [0, 1]
                
                % Filter luminance
                L_filtered = obj.filterLuminance(L);
                
                % Reassemble L*a*b* and convert back to RGB
                lab_out = lab;
                lab_out(:, :, 1) = L_filtered * 100.0;
                I_out = lab2rgb(lab_out);
                I_out = max(0.0, min(1.0, I_out)); % Clamp output to valid RGB range
            else
                % Grayscale Image: Filter directly
                I_dbl = double(I);
                is_normalized = (max(I_dbl(:)) <= 1.0 && min(I_dbl(:)) >= 0.0);
                if ~is_normalized
                    I_dbl = I_dbl / 255.0;
                end
                
                L_filtered = obj.filterLuminance(I_dbl);
                
                if ~is_normalized
                    I_out = uint8(L_filtered * 255.0);
                else
                    I_out = L_filtered;
                end
            end
        end
        
        function I_out = apply(obj, I)
            % APPLY  Alias for filter to support standard IPixelTransform pipelines.
            I_out = obj.filter(I);
        end
        
        function L_out = filterLuminance(obj, L)
            % FILTERLUMINANCE  Applies fast discretized LLF to a single-channel [0, 1] image.
            arguments
                obj
                L double
            end
            
            % 1. Build standard Gaussian pyramid for the original input channel L
            GPyr_L = OceanMath.Image.MultiResolution.GaussianPyramid();
            GPyr_L.build(L, obj.NumLevels);
            obj.InputGaussianPyramid = GPyr_L;
            
            % Build standard Laplacian pyramid for input reference
            LPyr_L = OceanMath.Image.MultiResolution.LaplacianPyramid();
            LPyr_L.build(L, obj.NumLevels);
            obj.InputLaplacianPyramid = LPyr_L;
            
            % 2. Define discrete reference bins
            bins = linspace(0.0, 1.0, obj.NumBins);
            
            % 3. Precompute Laplacian Pyramids for all remapped bin values
            LPyr_bins = cell(1, obj.NumBins);
            for k = 1:obj.NumBins
                % Remap original image around the bin reference intensity
                L_remap = obj.remap(L, bins(k));
                
                % Build standard Laplacian pyramid for this remapped version
                LPyr_bins{k} = OceanMath.Image.MultiResolution.LaplacianPyramid();
                LPyr_bins{k}.build(L_remap, obj.NumLevels);
            end
            obj.BinLaplacianPyramids = LPyr_bins;
            
            % 4. Interpolate output Laplacian pyramid coefficients level-by-level
            LPyr_out = OceanMath.Image.MultiResolution.LaplacianPyramid();
            LPyr_out.NumLevels = obj.NumLevels;
            LPyr_out.Levels = cell(1, obj.NumLevels);
            
            for l = 1:obj.NumLevels
                G_l = GPyr_L.Levels{l};
                [h_l, w_l, c_l] = size(G_l);
                
                % Clamp local reference intensities to [0, 1]
                g_clamped = max(0.0, min(1.0, G_l));
                
                % Calculate continuous bin indices
                bin_idx = g_clamped * (obj.NumBins - 1) + 1;
                k_idx = floor(bin_idx);
                k_idx = max(1, min(obj.NumBins - 1, k_idx)); % Clamp to [1, NumBins - 1]
                
                % Linear interpolation weight
                w = bin_idx - k_idx;
                
                % Interpolate output coefficients
                level_out = zeros(h_l, w_l, c_l);
                for k_val = 1:(obj.NumBins - 1)
                    mask = (k_idx == k_val);
                    if any(mask(:))
                        % Get details for bins k_val and k_val+1
                        level_k = LPyr_bins{k_val}.Levels{l};
                        level_kp1 = LPyr_bins{k_val+1}.Levels{l};
                        
                        % Replicate weight across channels if necessary (single channel input here)
                        w_sub = w(mask);
                        level_out(mask) = (1 - w_sub) .* level_k(mask) + w_sub .* level_kp1(mask);
                    end
                end
                LPyr_out.Levels{l} = level_out;
            end
            obj.OutputLaplacianPyramid = LPyr_out;
            
            % 5. Reconstruct final image from the interpolated Laplacian pyramid
            L_out = LPyr_out.reconstruct();
            L_out = max(0.0, min(1.0, L_out));
        end
        
        function remap_img = remap(obj, x, g)
            % REMAP  Vectorized pointwise remapping function r(x; g) centered at g.
            arguments
                obj
                x double
                g double
            end
            
            d = x - g;
            abs_d = abs(d);
            sgn = sign(d);
            y = zeros(size(x));
            
            % Detail region (x - g <= Sigma)
            mask_detail = (abs_d <= obj.Sigma);
            if any(mask_detail(:))
                y(mask_detail) = obj.Sigma * (abs_d(mask_detail) / obj.Sigma) .^ obj.Alpha;
            end
            
            % Edge region (x - g > Sigma)
            mask_edge = ~mask_detail;
            if any(mask_edge(:))
                y(mask_edge) = obj.Beta * (abs_d(mask_edge) - obj.Sigma) + obj.Sigma;
            end
            
            remap_img = g + sgn .* y;
        end
        
        function fig = plotPyramids(obj)
            % PLOTPYRAMIDS  Visualizes the stored Gaussian, input Laplacian, and output Laplacian pyramids.
            arguments
                obj
            end
            
            if isempty(obj.InputGaussianPyramid) || isempty(obj.OutputLaplacianPyramid)
                error('No pyramids stored. Run filter() on an image first before plotting.');
            end
            
            viz = OceanViz.Core('Theme', 'Dark');
            
            fig = figure('Name', 'Local Laplacian Filter - Pyramid Visualization', ...
                         'Position', [100, 100, 1200, 750]);
            set(fig, 'Color', viz.Colors.Background);
            
            L = obj.NumLevels;
            
            % Grid size: 3 rows (Input GP, Input LP, Output LP) x L columns (Levels)
            for l = 1:L
                % --- Row 1: Input Gaussian Pyramid ---
                ax_gp = subplot(3, L, l);
                imshow(obj.InputGaussianPyramid.Levels{l});
                title(sprintf('Input GP Lvl %d', l), 'Color', viz.Colors.Text, 'FontSize', 9, 'FontWeight', 'bold');
                viz.styleAxes(ax_gp);
                
                % --- Row 2: Input Laplacian Pyramid ---
                ax_il = subplot(3, L, L + l);
                % Laplacian visualization needs offset and scale
                if l == L
                    imshow(obj.InputLaplacianPyramid.Levels{l}); % Residual is low-pass
                else
                    imshow(max(0.0, min(1.0, 2.0 * obj.InputLaplacianPyramid.Levels{l} + 0.5)));
                end
                title(sprintf('Input LP Lvl %d', l), 'Color', viz.Colors.Text, 'FontSize', 9, 'FontWeight', 'bold');
                viz.styleAxes(ax_il);
                
                % --- Row 3: Output Laplacian Pyramid ---
                ax_ol = subplot(3, L, 2*L + l);
                if l == L
                    imshow(obj.OutputLaplacianPyramid.Levels{l});
                else
                    imshow(max(0.0, min(1.0, 2.0 * obj.OutputLaplacianPyramid.Levels{l} + 0.5)));
                end
                title(sprintf('Output LP Lvl %d', l), 'Color', viz.Colors.Text, 'FontSize', 9, 'FontWeight', 'bold');
                viz.styleAxes(ax_ol);
            end
            
            sgtitle('Local Laplacian Filtering: Pyramids Visualizer', ...
                    'Color', viz.Colors.Text, 'FontSize', 14, 'FontWeight', 'bold');
            
            % Add explanatory caption at the bottom
            annotation(fig, 'textbox', [0.1, 0.01, 0.8, 0.04], ...
                       'String', 'Note: Laplacian bands are offset by +0.5 and scaled by 2.0 for visualization. Last level is the low-frequency residual.', ...
                       'Color', viz.Colors.Text, 'FontSize', 10, 'FontAngle', 'italic', ...
                       'LineStyle', 'none', 'HorizontalAlignment', 'center');
        end
    end
end
