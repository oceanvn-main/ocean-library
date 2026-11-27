classdef LaplacianPyramid < OceanMath.Image.MultiResolution.IPyramid
    % LAPLACIANPYRAMID  Concrete implementation of the Laplacian Pyramid from scratch.
    %
    %   A Laplacian pyramid represents the band-pass (detail) levels of an 
    %   image at multiple resolutions. It supports perfect reconstruction of 
    %   the original image.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-19
    
    properties (Access = public)
        Boundary char = 'replicate'  % Padding boundary mode
        Kernel                       % Resolved IFilterKernel instance
        BypassFilter logical = false % Skip low-pass filter (shows aliasing artifacts)
    end
    
    methods
        function obj = LaplacianPyramid(options)
            % LAPLACIANPYRAMID  Constructs a Laplacian Pyramid object.
            arguments
                options.Boundary char = 'replicate'
                options.Kernel = OceanMath.Signal.Kernels.GaussianKernel('Size', 5)
                options.BypassFilter logical = false
            end
            obj.Boundary = options.Boundary;
            obj.Kernel = OceanMath.Signal.Kernels.KernelFactory.resolve(options.Kernel);
            obj.BypassFilter = options.BypassFilter;
        end
        
        function build(obj, I_in, num_levels)
            % BUILD  Constructs the Laplacian pyramid levels.
            %
            %   build(obj, I_in, num_levels)
            arguments
                obj
                I_in {mustBeNumeric}
                num_levels double {mustBeInteger, mustBePositive}
            end
            
            obj.NumLevels = num_levels;
            obj.Levels = cell(1, num_levels);
            
            % 1. Build a temporary Gaussian Pyramid using strided reduction
            G = cell(1, num_levels);
            G{1} = double(I_in);
            for level = 2:num_levels
                G{level} = OceanMath.Image.MultiResolution.PyramidOperator.reduce(...
                    G{level-1}, obj.Boundary, obj.Kernel, BypassFilter=obj.BypassFilter);
            end
            
            % 2. Calculate Laplacian levels (difference between Gaussian level and expanded next level)
            for level = 1:(num_levels-1)
                G_next_expanded = OceanMath.Image.MultiResolution.PyramidOperator.expand(...
                    G{level+1}, size(G{level}), obj.Boundary, obj.Kernel, BypassFilter=obj.BypassFilter);
                obj.Levels{level} = G{level} - G_next_expanded;
            end
            
            % The smallest/last level is simply the smallest Gaussian level
            obj.Levels{num_levels} = G{num_levels};
        end
        
        function I_rec = reconstruct(obj)
            % RECONSTRUCT  Reconstructs the original image from the Laplacian pyramid.
            arguments
                obj
            end
            if isempty(obj.Levels)
                error('LaplacianPyramid:EmptyPyramid', 'Pyramid must be built before reconstruction.');
            end
            
            % Start with the smallest Gaussian level at the top of the pyramid
            I_rec = obj.Levels{obj.NumLevels};
            
            % Recursively expand and add detail bands going down to the base
            for level = (obj.NumLevels-1):-1:1
                I_rec_expanded = OceanMath.Image.MultiResolution.PyramidOperator.expand(...
                    I_rec, size(obj.Levels{level}), obj.Boundary, obj.Kernel, BypassFilter=obj.BypassFilter);
                I_rec = obj.Levels{level} + I_rec_expanded;
            end
        end
        
        function report(obj)
            % REPORT  Prints a summary explanation of the Laplacian Pyramid characteristics and overcompleteness.
            arguments
                obj
            end
            
            fprintf('\n=== The Laplacian Pyramid ===\n');
            fprintf('The Laplacian pyramid is an overcomplete (redundant) representation.\n');
            fprintf('It works by smoothing an image (using a Gaussian filter), subsampling it,\n');
            fprintf('and then upsampling it back to the original size to subtract from the original image.\n');
            fprintf('The resulting "band" contains only the high-frequency residual (edges and textures).\n');
            fprintf('Because it is overcomplete, the number of pixels in the pyramid exceeds the number of pixels in the original image.\n');
            
            if ~isempty(obj.Levels)
                % Calculate exact pixel counts
                n_orig = numel(obj.Levels{1}); % Base level has same dimensions as original input
                n_pyr = 0;
                for l = 1:obj.NumLevels
                    n_pyr = n_pyr + numel(obj.Levels{l});
                end
                overhead_pct = (n_pyr - n_orig) / n_orig * 100.0;
                fprintf('\n  Empirical Complexity Audit:\n');
                fprintf('    Original Image Pixels : %d\n', n_orig);
                fprintf('    Pyramid Total Pixels  : %d\n', n_pyr);
                fprintf('    Redundancy Overhead   : %.2f%%\n', overhead_pct);
            end
            fprintf('=============================\n\n');
        end
    end
end
