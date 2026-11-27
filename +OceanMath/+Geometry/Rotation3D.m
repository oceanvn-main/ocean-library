classdef Rotation3D
    % ROTATION3D  Converts between various 3D rotation representations.
    
    methods (Static)
        function R = eulerToMatrix(roll_deg, pitch_deg, yaw_deg)
            % ZYX Euler convention (Yaw, then Pitch, then Roll)
            arguments
                roll_deg (1,1) double
                pitch_deg (1,1) double
                yaw_deg (1,1) double
            end
            r = deg2rad(roll_deg);
            p = deg2rad(pitch_deg);
            y = deg2rad(yaw_deg);
            
            % Rotation Z (yaw)
            cy = cos(y); sy = sin(y);
            Rz = [cy, -sy, 0; ...
                  sy,  cy, 0; ...
                   0,   0, 1];
               
            % Rotation Y (pitch)
            cp = cos(p); sp = sin(p);
            Ry = [cp, 0, sp; ...
                   0, 1,  0; ...
                 -sp, 0, cp];
             
            % Rotation X (roll)
            cr = cos(r); sr = sin(r);
            Rx = [1,  0,   0; ...
                  0, cr, -sr; ...
                  0, sr,  cr];
              
            R = Rz * Ry * Rx;
        end
        
        function [roll_deg, pitch_deg, yaw_deg] = matrixToEuler(R)
            arguments
                R (3,3) double
            end
            % ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
            % R31 = -sin(pitch)
            % R32 = sin(roll)*cos(pitch)
            % R33 = cos(roll)*cos(pitch)
            % R21 = sin(yaw)*cos(pitch)
            % R11 = cos(yaw)*cos(pitch)
            
            pitch = asin(max(-1.0, min(1.0, -R(3,1))));
            
            if abs(cos(pitch)) > 1e-6
                roll = atan2(R(3,2), R(3,3));
                yaw = atan2(R(2,1), R(1,1));
            else
                % Gimbal lock: pitch is +/- 90 degrees
                roll = 0;
                yaw = atan2(-R(1,2), R(2,2));
            end
            
            roll_deg = rad2deg(roll);
            pitch_deg = rad2deg(pitch);
            yaw_deg = rad2deg(yaw);
        end
        
        function R = quatToMatrix(q)
            arguments
                q (1,4) double
            end
            R = OceanMath.Quat.quat_to_rotm(q);
        end
        
        function q = matrixToQuat(R)
            arguments
                R (3,3) double
            end
            % Converts 3x3 orthonormal rotation matrix to [w, x, y, z] unit quaternion.
            tr = trace(R);
            if tr > 0
                S = sqrt(tr + 1.0) * 2; % S = 4*qw
                qw = 0.25 * S;
                qx = (R(3,2) - R(2,3)) / S;
                qy = (R(1,3) - R(3,1)) / S;
                qz = (R(2,1) - R(1,2)) / S;
            elseif (R(1,1) > R(2,2)) && (R(1,1) > R(3,3))
                S = sqrt(1.0 + R(1,1) - R(2,2) - R(3,3)) * 2; % S = 4*qx
                qw = (R(3,2) - R(2,3)) / S;
                qx = 0.25 * S;
                qy = (R(1,2) + R(2,1)) / S;
                qz = (R(1,3) + R(3,1)) / S;
            elseif R(2,2) > R(3,3)
                S = sqrt(1.0 + R(2,2) - R(1,1) - R(3,3)) * 2; % S = 4*qy
                qw = (R(1,3) - R(3,1)) / S;
                qx = (R(1,2) + R(2,1)) / S;
                qy = 0.25 * S;
                qz = (R(2,3) + R(3,2)) / S;
            else
                S = sqrt(1.0 + R(3,3) - R(1,1) - R(2,2)) * 2; % S = 4*qz
                qw = (R(2,1) - R(1,2)) / S;
                qx = (R(1,3) + R(3,1)) / S;
                qy = (R(2,3) + R(3,2)) / S;
                qz = 0.25 * S;
            end
            q = [qw, qx, qy, qz];
            q = q / norm(q);
        end
        
        function R = axisAngleToMatrix(axis_vec, theta_deg)
            arguments
                axis_vec (1,3) double
                theta_deg (1,1) double
            end
            % Rodrigues' rotation formula
            v = axis_vec(:);
            vNorm = norm(v);
            if vNorm < 1e-12
                R = eye(3);
                return;
            end
            v = v / vNorm;
            theta = deg2rad(theta_deg);
            
            c = cos(theta);
            s = sin(theta);
            
            % Skew-symmetric cross product matrix [v]_x
            vx = [  0,   -v(3),  v(2); ...
                  v(3),   0,   -v(1); ...
                 -v(2),  v(1),   0  ];
             
            R = c * eye(3) + (1 - c) * (v * v') + s * vx;
        end
        
        function [axis_vec, theta_deg] = matrixToAxisAngle(R)
            arguments
                R (3,3) double
            end
            % Trace check
            tr = trace(R);
            theta = acos(max(-1.0, min(1.0, (tr - 1) / 2)));
            
            if theta < 1e-6
                axis_vec = [0, 0, 1];
                theta_deg = 0;
            elseif theta > pi - 1e-6
                % Angle is close to 180 degrees. Axis is recovered from eigenvectors
                % or diagonal elements of R
                [V, D] = eig(R);
                [~, idx] = min(abs(diag(D) - 1));
                axis_vec = real(V(:, idx))';
                axis_vec = axis_vec / norm(axis_vec);
                theta_deg = 180;
            else
                axis_vec = [R(3,2) - R(2,3), R(1,3) - R(3,1), R(2,1) - R(1,2)];
                axis_vec = axis_vec / norm(axis_vec);
                theta_deg = rad2deg(theta);
            end
        end
        
        function R = expMapToMatrix(v_3x1)
            arguments
                v_3x1 (3,1) double
            end
            theta_rad = norm(v_3x1);
            if theta_rad < 1e-12
                R = eye(3);
            else
                axis_vec = (v_3x1 / theta_rad)';
                R = OceanMath.Geometry.Rotation3D.axisAngleToMatrix(axis_vec, rad2deg(theta_rad));
            end
        end
        
        function v_3x1 = matrixToExpMap(R)
            arguments
                R (3,3) double
            end
            [axis_vec, theta_deg] = OceanMath.Geometry.Rotation3D.matrixToAxisAngle(R);
            v_3x1 = axis_vec(:) * deg2rad(theta_deg);
        end
        
        function q = slerp(q1, q2, t)
            arguments
                q1 (1,4) double
                q2 (1,4) double
                t (1,1) double {mustBeInRange(t, 0, 1)}
            end
            q = OceanMath.Slerp.interpolate(q1, q2, t);
        end
    end
end
