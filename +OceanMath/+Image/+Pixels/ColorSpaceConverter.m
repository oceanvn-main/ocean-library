classdef ColorSpaceConverter < OceanMath.Image.Pixels.IPixelTransform
    % COLORSPACECONVERTER  Applies color space transformations to images.
    %
    %   Supported transformations:
    %       - "RGB2Gray": 3-channel RGB to 1-channel grayscale
    %       - "Gray2RGB": 1-channel grayscale to 3-channel RGB (replicated)
    %       - "RGB2HSV" : RGB to Hue-Saturation-Value
    %       - "HSV2RGB" : HSV back to RGB
    %       - "RGB2YCbCr": RGB to Y-Cb-Cr
    %       - "YCbCr2RGB": YCbCr back to RGB
    %
    %   Author: Ocean Library System
    %   Date: 2026-06-11
    
    properties (Access = private)
        Type (1,1) string = "RGB2Gray"
    end
    
    methods
        function obj = ColorSpaceConverter(type)
            % COLORSPACECONVERTER Construct a ColorSpaceConverter.
            arguments
                type (1,1) string {mustBeMember(type, ...
                    ["RGB2Gray", "Gray2RGB", "RGB2HSV", "HSV2RGB", "RGB2YCbCr", "YCbCr2RGB"])} = "RGB2Gray"
            end
            obj.Type = type;
        end
        
        function I_out = apply(obj, I_in)
            % APPLY Performs the selected color space mapping.
            arguments
                obj
                I_in {mustBeNumeric}
            end
            
            is_uint8 = isa(I_in, 'uint8');
            
            switch obj.Type
                case "RGB2Gray"
                    if size(I_in, 3) == 1
                        I_out = I_in; % Already grayscale
                        return;
                    end
                    try
                        I_out = rgb2gray(I_in);
                    catch
                        % Manual fallback using Rec. 601 luma coefficients
                        I_double = double(I_in);
                        gray_double = 0.299 * I_double(:,:,1) + ...
                                      0.587 * I_double(:,:,2) + ...
                                      0.114 * I_double(:,:,3);
                        if is_uint8
                            I_out = uint8(gray_double);
                        else
                            I_out = gray_double;
                        end
                    end
                    
                case "Gray2RGB"
                    if size(I_in, 3) == 3
                        I_out = I_in; % Already RGB
                        return;
                    end
                    I_out = cat(3, I_in, I_in, I_in);
                    
                case "RGB2HSV"
                    I_out = rgb2hsv(I_in);
                    
                case "HSV2RGB"
                    I_out = hsv2rgb(I_in);
                    
                case "RGB2YCbCr"
                    I_out = rgb2ycbcr(I_in);
                    
                case "YCbCr2RGB"
                    I_out = ycbcr2rgb(I_in);
            end
        end
    end
end
