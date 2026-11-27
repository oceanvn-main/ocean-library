classdef (Abstract) IPixelTransform < handle
    % IPIXELTRANSFORM  Interface for all image pixel-level transformations.
    %
    %   All custom pixel operations, point operators, color space converters,
    %   and neighborhood filters must inherit from this class and implement
    %   the apply() method.
    %
    %   Author: Ocean Library System
    %   Date: 2026-06-11
    
    properties (Access = public)
        Description char = ''
    end
    
    methods (Abstract)
        % APPLY Apply the transformation to the input image.
        %
        %   I_out = apply(obj, I_in)
        %
        %   Inputs:
        %       I_in  - [H x W x C] numeric array representing the input image.
        %               Usually uint8 or double.
        %
        %   Outputs:
        %       I_out - [H x W x C] numeric array of the transformed image,
        %               maintaining the original data type or casting appropriately.
        I_out = apply(obj, I_in)
    end
end
