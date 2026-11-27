classdef SteerableFilter < OceanMath.Image.Pixels.IPixelTransform
    % STEERABLEFILTER  Steerable filter responses for edge/feature extraction.
    %
    %   Implements Freeman and Adelson's steerable filter algorithm.
    %   Convolves the image with basis filters and allows O(1) steering to any
    %   arbitrary orientation angle.
    %
    %   Constructor:
    %       SteerableFilter('Sigma', sigma, 'Order', order, 'Size', [H, W], 'BoundaryMode', mode)
    %
    %   Inputs:
    %       Sigma        - Gaussian scale parameter (default: 2.0).
    %       Order        - Filter order (1 for G1 derivative, 2 for G2 derivative) (default: 1).
    %       Size         - [1 x 2] kernel grid dimensions (default: [15, 15]).
    %       BoundaryMode - Padding strategy (default: 'replicate').
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-07-13
    
    properties (Access = private)
        Sigma        (1,1) double = 2.0
        Order        (1,1) double = 1
        Size         (1,2) double = [15, 15]
        BoundaryMode (1,1) string = 'replicate'
    end
    
    properties (SetAccess = private)
        BasisResponses struct
    end
    
    methods
        function obj = SteerableFilter(options)
            % STEERABLEFILTER Construct a steerable filter operator.
            arguments
                options.Sigma        (1,1) double = 2.0
                options.Order        (1,1) double = 1
                options.Size         (1,2) double = [15, 15]
                options.BoundaryMode (1,1) string = 'replicate'
            end
            obj.Sigma = options.Sigma;
            obj.Order = options.Order;
            obj.Size = options.Size;
            obj.BoundaryMode = options.BoundaryMode;
            
            if obj.Order ~= 1 && obj.Order ~= 2
                error('Filter Order must be either 1 or 2.');
            end
        end
        
        function I_out = apply(obj, I_in)
            % APPLY Computes convolved basis responses and returns a multi-banded image.
            %
            %   For Order 1: Returns [H x W x 2] convolved responses (Rx, Ry).
            %   For Order 2: Returns [H x W x 3] convolved responses (R0, R90, R45).
            arguments
                obj
                I_in {mustBeNumeric}
            end
            
            is_uint8 = isa(I_in, 'uint8');
            I_double = double(I_in);
            
            % Convert color to luminance if RGB
            if ndims(I_double) == 3 && size(I_double, 3) == 3
                I_gray = 0.299 * I_double(:,:,1) + 0.587 * I_double(:,:,2) + 0.114 * I_double(:,:,3);
            else
                I_gray = I_double;
            end
            
            % Setup base filters
            obj.BasisResponses = struct();
            
            if obj.Order == 1
                % First-order basis: 0 deg (x-deriv) and 90 deg (y-deriv)
                k_x = OceanMath.Signal.Kernels.GaussianDerivativeKernel(...
                    'Size', obj.Size, 'Sigma', obj.Sigma, 'Order', 1, 'Angle', 0);
                k_y = OceanMath.Signal.Kernels.GaussianDerivativeKernel(...
                    'Size', obj.Size, 'Sigma', obj.Sigma, 'Order', 1, 'Angle', 90);
                
                conv_x = OceanMath.Image.Pixels.TrueConvolution('Kernel', k_x, 'Boundary', obj.BoundaryMode);
                conv_y = OceanMath.Image.Pixels.TrueConvolution('Kernel', k_y, 'Boundary', obj.BoundaryMode);
                
                obj.BasisResponses.Rx = conv_x.apply(I_gray);
                obj.BasisResponses.Ry = conv_y.apply(I_gray);
                
                I_out = cat(3, obj.BasisResponses.Rx, obj.BasisResponses.Ry);
            else
                % Second-order basis: 0 deg (xx-deriv), 90 deg (yy-deriv) and 45 deg
                k_0 = OceanMath.Signal.Kernels.GaussianDerivativeKernel(...
                    'Size', obj.Size, 'Sigma', obj.Sigma, 'Order', 2, 'Angle', 0);
                k_90 = OceanMath.Signal.Kernels.GaussianDerivativeKernel(...
                    'Size', obj.Size, 'Sigma', obj.Sigma, 'Order', 2, 'Angle', 90);
                k_45 = OceanMath.Signal.Kernels.GaussianDerivativeKernel(...
                    'Size', obj.Size, 'Sigma', obj.Sigma, 'Order', 2, 'Angle', 45);
                
                conv_0 = OceanMath.Image.Pixels.TrueConvolution('Kernel', k_0, 'Boundary', obj.BoundaryMode);
                conv_90 = OceanMath.Image.Pixels.TrueConvolution('Kernel', k_90, 'Boundary', obj.BoundaryMode);
                conv_45 = OceanMath.Image.Pixels.TrueConvolution('Kernel', k_45, 'Boundary', obj.BoundaryMode);
                
                obj.BasisResponses.R0 = conv_0.apply(I_gray);
                obj.BasisResponses.R90 = conv_90.apply(I_gray);
                obj.BasisResponses.R45 = conv_45.apply(I_gray);
                
                I_out = cat(3, obj.BasisResponses.R0, obj.BasisResponses.R90, obj.BasisResponses.R45);
            end
        end
        
        function R_theta = steer(obj, angle_deg)
            % STEER Computes the filter response at an arbitrary angle in O(1) time.
            if isempty(fields(obj.BasisResponses))
                error('SteerableFilter must be applied to an image before calling steer().');
            end
            
            theta = deg2rad(angle_deg);
            
            if obj.Order == 1
                R_theta = cos(theta) * obj.BasisResponses.Rx + sin(theta) * obj.BasisResponses.Ry;
            else
                % Steering formula for 2nd order using basis responses at 0, 90, and 45 degrees
                c = cos(theta);
                s = sin(theta);
                R_theta = obj.BasisResponses.R0 * (c^2 - c*s) + ...
                          obj.BasisResponses.R90 * (s^2 - c*s) + ...
                          obj.BasisResponses.R45 * (2*c*s);
            end
        end
        
        function [theta_max, mag] = computeDominantOrientation(obj)
            % COMPUTEDOMINANTORIENTATION Computes local orientation and max gradient response.
            %
            %   Outputs:
            %       theta_max - [H x W] matrix of dominant local orientation in degrees.
            %       mag       - [H x W] matrix of maximum filter response magnitude.
            if isempty(fields(obj.BasisResponses))
                error('SteerableFilter must be applied to an image before calling computeDominantOrientation().');
            end
            
            if obj.Order == 1
                Rx = obj.BasisResponses.Rx;
                Ry = obj.BasisResponses.Ry;
                
                % Gradient magnitude
                mag = sqrt(Rx.^2 + Ry.^2);
                
                % Local dominant orientation (angle of gradient)
                theta_max = rad2deg(atan2(Ry, Rx));
            else
                R0 = obj.BasisResponses.R0;
                R90 = obj.BasisResponses.R90;
                R45 = obj.BasisResponses.R45;
                
                % Map G2 steering coefficients
                A = R0;
                B = R90;
                C = 2 * R45 - R0 - R90;
                
                % Double angle formulas
                theta_2 = atan2(C, A - B);
                theta_max = rad2deg(0.5 * theta_2);
                
                % Maximum response magnitude
                mag = 0.5 * (A + B) + 0.5 * sqrt((A - B).^2 + C.^2);
            end
        end
    end
end
