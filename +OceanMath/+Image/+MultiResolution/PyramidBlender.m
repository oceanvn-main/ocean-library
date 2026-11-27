classdef PyramidBlender < handle
    % PYRAMIDBLENDER  Multiresolution Laplacian Pyramid Image Blending (Szeliski Ex 3.18).
    %
    %   Blends N images using their Laplacian pyramids, weighted by a Gaussian
    %   pyramid built from corresponding soft mask images.
    %
    %   Algorithm:
    %     1. Build a Laplacian pyramid LP_i for each source image I_i.
    %     2. Build a Gaussian pyramid GP_Wi for each mask W_i.
    %     3. At each pyramid level l, form the blended band:
    %           B_l = sum_i( LP_i{l} .* GP_Wi{l} )
    %        (Optionally normalize by sum of mask weights to handle non-unity masks.)
    %     4. Reconstruct the final blended image from B_l using the standard
    %        Laplacian collapsing sum.
    %
    %   Renormalization note:
    %     If masks sum to 1.0 at every pixel, no renormalization is needed
    %     because the Gaussian filter is a normalized linear operator.
    %     Set Normalize=true only when masks may not partition unity.
    %
    %   Example:
    %     blender = OceanMath.Image.MultiResolution.PyramidBlender( ...
    %         'Kernel', OceanMath.Signal.Kernels.GaussianKernel('Size', 5), ...
    %         'NumLevels', 6);
    %     I_blend = blender.blend({imgA, imgB}, {maskA, maskB});
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-07-13
    
    properties (Access = public)
        Kernel                       % IFilterKernel for pyramid construction
        Boundary char = 'replicate'  % Boundary padding mode
        NumLevels double = 6         % Number of pyramid levels
        Normalize logical = false    % Normalize by sum of mask pyramids at each level
        BypassFilter logical = false % Skip low-pass filter (demonstrates aliasing in blends)
    end
    
    methods
        function obj = PyramidBlender(options)
            % PYRAMIDBLENDER  Construct the blender with configuration.
            arguments
                options.Kernel = OceanMath.Signal.Kernels.GaussianKernel('Size', 5)
                options.Boundary char = 'replicate'
                options.NumLevels double = 6
                options.Normalize logical = false
                options.BypassFilter logical = false
            end
            obj.Kernel = OceanMath.Signal.Kernels.KernelFactory.resolve(options.Kernel);
            obj.Boundary = options.Boundary;
            obj.NumLevels = options.NumLevels;
            obj.Normalize = options.Normalize;
            obj.BypassFilter = options.BypassFilter;
        end
        
        function I_blend = blend(obj, images, masks)
            % BLEND  Produce a Laplacian pyramid blend of N images.
            %
            %   I_blend = blend(obj, images, masks)
            %
            %   Input:
            %       images - 1xN cell array of double images (all same size)
            %       masks  - 1xN cell array of double soft mask images in [0,1].
            %                Each mask must have same H x W as the images.
            %                Can be single-channel (grayscale) or 3-channel (RGB).
            %                Masks need not sum to 1 if Normalize=true.
            %
            %   Output:
            %       I_blend - Blended double image in [0,1] range.
            
            arguments
                obj
                images (1,:) cell
                masks  (1,:) cell
            end
            
            N = numel(images);
            assert(numel(masks) == N, 'PyramidBlender:Mismatch', ...
                'Number of masks must equal number of images (got %d images, %d masks).', N, numel(masks));
            assert(N >= 1, 'PyramidBlender:Empty', 'At least one image is required.');
            
            % Clamp number of levels to valid range for image size
            [H, W, ~] = size(images{1});
            max_levels = floor(log2(min(H, W)));
            nL = min(obj.NumLevels, max_levels);
            
            % --- 1. Build Laplacian pyramids for all images ---
            LP = cell(1, N);
            for i = 1:N
                lap = OceanMath.Image.MultiResolution.LaplacianPyramid( ...
                    'Kernel', obj.Kernel, ...
                    'Boundary', obj.Boundary, ...
                    'BypassFilter', obj.BypassFilter);
                lap.build(double(images{i}), nL);
                LP{i} = lap.Levels;
            end
            
            % --- 2. Build Gaussian pyramids for all masks ---
            GP_W = cell(1, N);
            for i = 1:N
                % Expand mask to match image channels if needed
                M = double(masks{i});
                [~, ~, C_img] = size(images{1});
                [~, ~, C_mask] = size(M);
                if C_mask == 1 && C_img > 1
                    M = repmat(M, [1, 1, C_img]);
                end
                
                gp = OceanMath.Image.MultiResolution.GaussianPyramid( ...
                    'Kernel', obj.Kernel, ...
                    'Boundary', obj.Boundary, ...
                    'BypassFilter', obj.BypassFilter);
                gp.build(M, nL);
                GP_W{i} = gp.Levels;
            end
            
            % --- 3. Blend Laplacian bands using Gaussian mask weights ---
            B_levels = cell(1, nL);
            for l = 1:nL
                B_l = zeros(size(LP{1}{l}));
                W_sum = zeros(size(LP{1}{l}));
                
                for i = 1:N
                    B_l = B_l + LP{i}{l} .* GP_W{i}{l};
                    W_sum = W_sum + GP_W{i}{l};
                end
                
                % Normalize by sum of mask weights at this level (if requested)
                if obj.Normalize
                    eps_val = 1e-8;
                    B_levels{l} = B_l ./ (W_sum + eps_val);
                else
                    B_levels{l} = B_l;
                end
            end
            
            % --- 4. Reconstruct from blended Laplacian pyramid ---
            % Start from the top (smallest, coarsest) level
            I_blend = B_levels{nL};
            
            for level = (nL - 1):-1:1
                I_rec_expanded = OceanMath.Image.MultiResolution.PyramidOperator.expand(...
                    I_blend, size(B_levels{level}), obj.Boundary, obj.Kernel, ...
                    BypassFilter=obj.BypassFilter);
                I_blend = B_levels{level} + I_rec_expanded;
            end
        end
    end
end
