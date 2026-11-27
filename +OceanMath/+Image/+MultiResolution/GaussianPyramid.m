classdef GaussianPyramid < OceanMath.Image.MultiResolution.IPyramid
    % GAUSSIANPYRAMID  Concrete implementation of the Gaussian Pyramid from scratch.
    %
    %   A Gaussian pyramid represents the low-pass, decimation representation 
    %   of an image at multiple resolutions.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-19
    
    properties (Access = public)
        Boundary char = 'replicate'  % Padding boundary mode
        Kernel                       % Resolved IFilterKernel instance
        BypassFilter logical = false % Skip low-pass filter before decimation (shows aliasing)
    end
    
    methods
        function obj = GaussianPyramid(options)
            % GAUSSIANPYRAMID  Constructs a Gaussian Pyramid object.
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
            % BUILD  Constructs the Gaussian pyramid levels.
            %
            %   build(obj, I_in, num_levels)
            arguments
                obj
                I_in {mustBeNumeric}
                num_levels double {mustBeInteger, mustBePositive}
            end
            
            obj.NumLevels = num_levels;
            obj.Levels = cell(1, num_levels);
            
            % Level 1 is the high-resolution input image (as double)
            obj.Levels{1} = double(I_in);
            
            % Recursively reduce for subsequent levels using strided convolution
            for level = 2:num_levels
                obj.Levels{level} = OceanMath.Image.MultiResolution.PyramidOperator.reduce(...
                    obj.Levels{level-1}, obj.Boundary, obj.Kernel, BypassFilter=obj.BypassFilter);
            end
        end
        
        function I_rec = reconstruct(obj)
            % RECONSTRUCT  Returns the base layer (original image).
            arguments
                obj
            end
            if isempty(obj.Levels)
                error('GaussianPyramid:EmptyPyramid', 'Pyramid must be built before reconstruction.');
            end
            I_rec = obj.Levels{1};
        end
    end
end
