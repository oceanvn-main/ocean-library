classdef CameraProjection < handle
    % CAMERAPROJECTION  Models 3D to 2D camera coordinate projections.
    %
    % Supports Perspective, Orthographic, and Weak Perspective models.
    
    properties
        Intrinsics (3,3) double = [500, 0, 320; 0, 500, 240; 0, 0, 1]
        Extrinsics (4,4) double = eye(4) % T_w2c = [R_w2c, t_w2c; 0 0 0 1]
        ProjectionType (1,1) string = "Perspective"
    end
    
    methods
        function obj = CameraProjection(options)
            arguments
                options.Intrinsics (3,3) double = [500, 0, 320; 0, 500, 240; 0, 0, 1]
                options.Extrinsics (4,4) double = eye(4)
                options.ProjectionType (1,1) string {mustBeMember(options.ProjectionType, ["Perspective", "Orthographic", "WeakPerspective", "PureOrthographic", "Paraperspective"])} = "Perspective"
            end
            obj.Intrinsics = options.Intrinsics;
            obj.Extrinsics = options.Extrinsics;
            obj.ProjectionType = options.ProjectionType;
        end
        
        function [pts2D, inFront] = project(obj, pts3D)
            % PROJECT Project 3D points onto the 2D image plane.
            %
            % INPUTS:
            %   pts3D  - [3 x N] matrix of 3D world coordinates
            %
            % OUTPUTS:
            %   pts2D  - [2 x N] projected image coordinates
            %   inFront - [1 x N] logical mask showing points in front of camera plane (z > 0)
            arguments
                obj
                pts3D (3,:) double
            end
            
            N = size(pts3D, 2);
            
            % Transform to camera coordinates: P_cam = R * P_world + t
            R_w2c = obj.Extrinsics(1:3, 1:3);
            t_w2c = obj.Extrinsics(1:3, 4);
            
            P_cam = R_w2c * pts3D + repmat(t_w2c, 1, N);
            
            % Check if points are in front of camera sensor
            inFront = P_cam(3, :) > 0;
            
            fx = obj.Intrinsics(1,1);
            fy = obj.Intrinsics(2,2);
            cx = obj.Intrinsics(1,3);
            cy = obj.Intrinsics(2,3);
            s  = obj.Intrinsics(1,2);
            
            switch obj.ProjectionType
                case "Perspective"
                    % standard homography projection
                    % pts2D_hom = K * P_cam
                    pts2D_hom = obj.Intrinsics * P_cam;
                    
                    % Normalize by z component (homogeneous w)
                    z = pts2D_hom(3, :);
                    z(abs(z) < 1e-15) = 1e-15;
                    
                    pts2D = [pts2D_hom(1, :) ./ z; ...
                             pts2D_hom(2, :) ./ z];
                             
                case "Orthographic"
                    % Orthographic projection scaled by focal length + principal point offset
                    % x = fx * X_c + s * Y_c + cx
                    % y = fy * Y_c + cy
                    pts2D = [fx * P_cam(1, :) + s * P_cam(2, :) + cx; ...
                             fy * P_cam(2, :) + cy];
                             
                case "WeakPerspective"
                    % Scaled orthographic projection based on the average depth Z_avg
                    Z_avg = mean(P_cam(3, :));
                    if Z_avg < 1e-3
                        Z_avg = 1e-3; % Prevent division by zero
                    end
                    
                    % x = (fx * X_c + s * Y_c) / Z_avg + cx
                    % y = (fy * Y_c) / Z_avg + cy
                    pts2D = [(fx * P_cam(1, :) + s * P_cam(2, :)) / Z_avg + cx; ...
                             (fy * P_cam(2, :)) / Z_avg + cy];
                             
                case "PureOrthographic"
                    % Pure orthographic projection: simply drop the z coordinate
                    % x = X_c
                    % y = Y_c
                    pts2D = P_cam(1:2, :);
                    
                case "Paraperspective"
                    % Paraperspective projection: projects points onto a reference plane
                    % along rays parallel to the line of sight to the object centroid.
                    % P_avg = centroid of all points
                    P_avg = mean(P_cam, 2);
                    X_avg = P_avg(1);
                    Y_avg = P_avg(2);
                    Z_avg = P_avg(3);
                    if Z_avg < 1e-3
                        Z_avg = 1e-3; % Prevent division by zero
                    end
                    
                    % X_para = X - X_avg * Z / Z_avg
                    % Y_para = Y - Y_avg * Z / Z_avg
                    X_para = P_cam(1, :) - X_avg * P_cam(3, :) / Z_avg;
                    Y_para = P_cam(2, :) - Y_avg * P_cam(3, :) / Z_avg;
                    
                    % Project to image plane
                    pts2D = [(fx * X_para + s * Y_para) / Z_avg + cx; ...
                             (fy * Y_para) / Z_avg + cy];
            end
        end
    end
end
