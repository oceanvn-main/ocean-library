classdef LensDistortion < handle
    % LENSDISTORTION  Models lens aberrations using radial and tangential models.
    
    properties
        Radial (1,3) double = [0, 0, 0] % [k1, k2, k3]
        Tangential (1,2) double = [0, 0] % [p1, p2]
    end
    
    methods
        function obj = LensDistortion(options)
            arguments
                options.Radial (1,3) double = [0, 0, 0]
                options.Tangential (1,2) double = [0, 0]
            end
            obj.Radial = options.Radial;
            obj.Tangential = options.Tangential;
        end
        
        function ptsOut = distort(obj, ptsIn)
            % DISTORT Apply radial and tangential distortion on normalized coordinates.
            %
            % INPUTS:
            %   ptsIn - [2 x N] normalized camera coordinates (x_n = X_c/Z_c, y_n = Y_c/Z_c)
            %
            % OUTPUTS:
            %   ptsOut - [2 x N] distorted normalized camera coordinates
            arguments
                obj
                ptsIn (2,:) double
            end
            
            x = ptsIn(1, :);
            y = ptsIn(2, :);
            
            k1 = obj.Radial(1);
            k2 = obj.Radial(2);
            k3 = obj.Radial(3);
            p1 = obj.Tangential(1);
            p2 = obj.Tangential(2);
            
            % Compute squared radius
            r2 = x.^2 + y.^2;
            
            % Radial distortion factor
            radialFactor = 1 + k1 * r2 + k2 * r2.^2 + k3 * r2.^3;
            
            % Apply radial distortion
            xRad = x .* radialFactor;
            yRad = y .* radialFactor;
            
            % Compute tangential distortion
            xTang = 2 * p1 * x .* y + p2 * (r2 + 2 * x.^2);
            yTang = p1 * (r2 + 2 * y.^2) + 2 * p2 * x .* y;
            
            % Combine radial and tangential distortions
            ptsOut = [xRad + xTang; ...
                      yRad + yTang];
        end
        
        function ptsImgOut = distortImageCoordinates(obj, ptsImgIn, K)
            % DISTORTIMAGECOORDINATES Warp image coordinates using camera intrinsics.
            %
            % Unprojects image coordinates to normalized coordinate space, applies
            % distortion, and projects back to image space.
            %
            % INPUTS:
            %   ptsImgIn - [2 x N] raw image plane coordinates
            %   K        - [3 x 3] camera intrinsic matrix
            %
            % OUTPUTS:
            %   ptsImgOut - [2 x N] distorted image plane coordinates
            arguments
                obj
                ptsImgIn (2,:) double
                K (3,3) double
            end
            
            N = size(ptsImgIn, 2);
            
            % Convert to homogeneous coordinates
            ptsHom = [ptsImgIn; ones(1, N)];
            
            % Un-project: ptsNorm = K \ ptsHom
            ptsNormHom = K \ ptsHom;
            ptsNorm = ptsNormHom(1:2, :);
            
            % Distort normalized coordinates
            ptsDistNorm = obj.distort(ptsNorm);
            
            % Project back to image coordinates
            ptsDistHom = K * [ptsDistNorm; ones(1, N)];
            
            % Slice out x and y image coordinates
            ptsImgOut = [ptsDistHom(1, :); ...
                         ptsDistHom(2, :)];
        end
    end
end
