classdef (Abstract) IPyramid < handle
    % IPYRAMID  Abstract base class for multi-resolution image pyramids.
    %
    %   This class defines the interface and common properties for both
    %   Gaussian and Laplacian pyramids.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-19
    
    properties (Access = public)
        Levels     cell    % Cell array holding the image at each level
        NumLevels  double  % Total number of scales in the pyramid
    end
    
    methods (Abstract)
        % BUILD  Abstract method to construct the pyramid from the input image.
        build(obj, I_in, num_levels)
        
        % RECONSTRUCT  Abstract method to reconstruct the original image.
        I_rec = reconstruct(obj)
    end
end
