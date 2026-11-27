classdef BoxFilter < OceanMath.Image.Filters.INeighborhoodFilter
    % BOXFILTER  Applies highly optimized 2D Box (average) filtering in O(1) time.
    %
    %   This class uses the Integral Image (Summed-Area Table) computational
    %   trick to compute local neighborhood sums in constant time, independent
    %   of the filter size.
    %
    %   Constructor:
    %       BoxFilter('Size', [kH, kW], 'Boundary', border)
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-17
    
    properties (Access = private)
        Size (1,2) double = [5, 5]
      end
      
      methods
          function obj = BoxFilter(options)
              % BOXFILTER Construct a BoxFilter.
              arguments
                  options.Size = [5, 5]
                  options.Boundary = 'replicate'
                  options.ConstantValue = 0.0
              end
              obj = obj@OceanMath.Image.Filters.INeighborhoodFilter(...
                  'Boundary', options.Boundary, 'ConstantValue', options.ConstantValue);
              
              sz = options.Size;
              if isscalar(sz)
                  sz = [sz, sz];
              end
              if any(mod(sz, 2) == 0) || any(sz <= 0)
                  error('BoxFilter:InvalidSize', 'Size must be odd positive integers.');
              end
              obj.Size = sz;
          end
      end
      
      methods (Access = protected)
          function pad_size = getPadSize(obj)
              % GETPADSIZE Calculates the symmetric padding size.
              pad_size = [floor(obj.Size(1)/2), floor(obj.Size(2)/2)];
          end
          
          function I_out = applyFilter(obj, I_padded, I_in)
              % APPLYFILTER Applies box filter using Integral Image.
              [H, W, C] = size(I_in);
              kH = obj.Size(1);
              kW = obj.Size(2);
              
              % Compute 2D Cumulative Sums (Integral Image) along rows & columns
              % for all channels simultaneously
              intImg = cumsum(cumsum(double(I_padded), 1), 2);
              
              % Prepend a zero row and column to avoid boundary conditions in indexing
              [H_pad, W_pad, ~] = size(I_padded);
              intImg_padded = zeros(H_pad + 1, W_pad + 1, C);
              intImg_padded(2:end, 2:end, :) = intImg;
              
              % Coordinate indices for window corners in the padded image
              R_start = (1 : H)';
              R_end = (kH : H + kH - 1)';
              C_start = (1 : W);
              C_end = (kW : W + kW - 1);
              
              % Evaluate the O(1) box sum for each pixel
              term1 = intImg_padded(R_end + 1, C_end + 1, :);
              term2 = intImg_padded(R_start, C_end + 1, :);
              term3 = intImg_padded(R_end + 1, C_start, :);
              term4 = intImg_padded(R_start, C_start, :);
              
              windowSum = term1 - term2 - term3 + term4;
              
              % Divide by the spatial footprint area to get the average
              area = kH * kW;
              I_out = windowSum / area;
          end
      end
  end
