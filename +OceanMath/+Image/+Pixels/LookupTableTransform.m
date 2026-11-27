classdef LookupTableTransform < OceanMath.Image.Pixels.IPixelTransform
    % LOOKUPTABLETRANSFORM  Applies O(1) intensity mappings using lookup tables (LUT).
    %
    %   Useful for fast arbitrary point operations (e.g. contrast curves, 
    %   histogram equalization, thresholds, inversions).
    %
    %   Constructor:
    %       LookupTableTransform(lut)
    %
    %   Inputs:
    %       lut - [256 x 1] or [256 x 3] array of mapping values.
    %             If [256 x 3], maps each color channel independently.
    %
    %   Author: Ocean Library System
    %   Date: 2026-06-11
    
    properties (Access = private)
        LUT_ % [256 x 1] or [256 x 3] lookup array
    end
    
    methods
        function obj = LookupTableTransform(lut)
            % LOOKUPTABLETRANSFORM Construct a LookupTableTransform.
            arguments
                lut (:,:) double = (0:255)'
            end
            
            [rows, cols] = size(lut);
            if rows ~= 256
                error('OceanMath:Vision:Pixels:InvalidLUTSize', ...
                    'Lookup table must have exactly 256 rows.');
            end
            if cols ~= 1 && cols ~= 3
                error('OceanMath:Vision:Pixels:InvalidLUTDimensions', ...
                    'Lookup table must have either 1 column or 3 columns.');
            end
            
            obj.LUT_ = lut;
        end
        
        function I_out = apply(obj, I_in)
            % APPLY Map input pixels to output pixels using the LUT.
            arguments
                obj
                I_in {mustBeNumeric}
            end
            
            is_uint8 = isa(I_in, 'uint8');
            I_double = double(I_in);
            
            % Assert input values are in [0, 255] range for table indexing
            if max(I_double(:)) > 255.0 || min(I_double(:)) < 0.0
                error('OceanMath:Vision:Pixels:RangeMismatch', ...
                    'LookupTableTransform requires input values in the [0, 255] range.');
            end
            
            % Standard 1-based indexing for MATLAB
            indices = round(I_double) + 1;
            indices = max(1, min(256, indices));
            
            if size(obj.LUT_, 2) == 3 && size(I_in, 3) == 3
                % Independent channel mapping
                I_res = zeros(size(I_in));
                for c = 1:3
                    I_res(:, :, c) = obj.LUT_(indices(:, :, c), c);
                end
            else
                % Single channel mapping applied to all channels
                I_res = obj.LUT_(indices);
            end
            
            % Return matching class type
            if is_uint8
                I_out = uint8(I_res);
            else
                I_out = I_res;
            end
        end
    end
end
