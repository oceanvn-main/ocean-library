classdef HistogramEqualization < OceanMath.Image.Pixels.IPixelTransform
    % HISTOGRAMEQUALIZATION  Applies full or partial histogram equalization.
    %
    %   Dynamically calculates the histogram and cumulative distribution function 
    %   (CDF) of the input image, derives the equalization transfer functions,
    %   and maps intensities in O(1) time using lookup tables.
    %
    %   Constructor:
    %       HistogramEqualization('Alpha', alpha)
    %
    %   Inputs:
    %       Alpha - Blending factor (0.0 = original, 1.0 = full equalization).
    %               Default: 1.0.
    %
    %   Properties (SetAccess = private):
    %       Histogram        - [256 x C] array of computed channel histograms
    %       CDF              - [256 x C] array of cumulative counts
    %       TransferFunction - [256 x C] array of intensity mapping functions
    %
    %   Author: Ocean Library System
    %   Date: 2026-06-11
    
    properties
        Alpha (1,1) double = 1.0
    end
    
    properties (SetAccess = private, GetAccess = public)
        Histogram        (:,:) double
        CDF              (:,:) double
        TransferFunction (:,:) double
    end
    
    methods
        function obj = HistogramEqualization(options)
            % HISTOGRAMEQUALIZATION Construct a HistogramEqualization operator.
            arguments
                options.Alpha (1,1) double = 1.0
            end
            obj.Alpha = options.Alpha;
        end
        
        function I_out = apply(obj, I_in)
            % APPLY Computes histogram/CDF and maps pixel values.
            arguments
                obj
                I_in {mustBeNumeric}
            end
            
            is_uint8 = isa(I_in, 'uint8');
            I_double = double(I_in);
            
            [height, width, num_channels] = size(I_in);
            num_pixels = height * width;
            
            % Initialize properties to store computed metrics
            obj.Histogram = zeros(256, num_channels);
            obj.CDF = zeros(256, num_channels);
            obj.TransferFunction = zeros(256, num_channels);
            
            I_res = zeros(size(I_in));
            
            for c = 1:num_channels
                channel_data = I_double(:, :, c);
                
                % Check if the values are normalized [0, 1]
                is_normalized = ~is_uint8 && max(I_double(:)) <= 1.0;
                if is_normalized
                    channel_data = channel_data * 255.0;
                end
                
                % Round and clamp to ensure valid indexing
                channel_rounded = round(channel_data);
                channel_rounded = max(0, min(255, channel_rounded));
                
                % Compute histogram explicitly
                h = zeros(256, 1);
                for val = 0:255
                    h(val + 1) = sum(channel_rounded(:) == val);
                end
                obj.Histogram(:, c) = h;
                
                % Compute CDF
                cdf = cumsum(h);
                obj.CDF(:, c) = cdf;
                
                % Compute standard equalization mapping
                T = 255.0 * (cdf / num_pixels);
                
                % Blend with identity mapping using Alpha
                T_final = (1.0 - obj.Alpha) * (0:255)' + obj.Alpha * T;
                obj.TransferFunction(:, c) = T_final;
                
                % Map pixels using lookup indexing
                indices = channel_rounded + 1;
                channel_out = T_final(indices);
                
                if is_normalized
                    channel_out = channel_out / 255.0;
                end
                
                I_res(:, :, c) = channel_out;
            end
            
            if is_uint8
                I_out = uint8(I_res);
            else
                I_out = I_res;
            end
        end
    end
end
