classdef MatrixRotation3D < OceanMath.Geometry.Rotations.RotationRepresentation3D
    % MATRIXROTATION3D  Object-oriented 3x3 rotation matrix strategy.
    
    properties (Access = private)
        R_  % [3 x 3] orthonormal rotation matrix (SO3)
    end
    
    methods
        function obj = MatrixRotation3D(R)
            arguments
                R (3,3) double
            end
            obj.R_ = R;
        end
        
        function R = toMatrix(obj)
            R = obj.R_;
        end
        
        function q = toQuat(obj)
            q = OceanMath.Geometry.Rotation3D.matrixToQuat(obj.R_);
        end
        
        function [roll, pitch, yaw] = toEuler(obj)
            [roll, pitch, yaw] = OceanMath.Geometry.Rotation3D.matrixToEuler(obj.R_);
        end
        
        function [axisVec, theta] = toAxisAngle(obj)
            [axisVec, theta] = OceanMath.Geometry.Rotation3D.matrixToAxisAngle(obj.R_);
        end
        
        function v = toExpMap(obj)
            v = OceanMath.Geometry.Rotation3D.matrixToExpMap(obj.R_);
        end
        
        function qInterp = interpolate(obj, other, t)
            % Converts to unit quaternion for smooth SLERP geodesic path, then maps back
            arguments
                obj
                other OceanMath.Geometry.Rotations.RotationRepresentation3D
                t (1,1) double {mustBeInRange(t, 0, 1)}
            end
            q1 = obj.toQuat();
            q2 = other.toQuat();
            q_res = OceanMath.Slerp.interpolate(q1, q2, t);
            R_res = OceanMath.Quat.quat_to_rotm(q_res);
            qInterp = OceanMath.Geometry.Rotations.MatrixRotation3D(R_res);
        end
    end
end
