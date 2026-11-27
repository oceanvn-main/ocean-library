classdef QuaternionRotation3D < OceanMath.Geometry.Rotations.RotationRepresentation3D
    % QUATERNIONROTATION3D  Object-oriented unit quaternion 3D rotation strategy.
    
    properties (Access = private)
        Q_  % [1 x 4] unit quaternion [w, x, y, z]
    end
    
    methods
        function obj = QuaternionRotation3D(q)
            % Constructor: Normalizes input double vector to form a unit quaternion.
            arguments
                q (1,4) double
            end
            qNorm = norm(q);
            if qNorm < 1e-12
                obj.Q_ = [1, 0, 0, 0];
            else
                obj.Q_ = q / qNorm;
            end
        end
        
        function R = toMatrix(obj)
            % Reuse existing procedurals from OceanMath.Quat
            R = OceanMath.Quat.quat_to_rotm(obj.Q_);
        end
        
        function q = toQuat(obj)
            q = obj.Q_;
        end
        
        function [roll, pitch, yaw] = toEuler(obj)
            R = obj.toMatrix();
            [roll, pitch, yaw] = OceanMath.Geometry.Rotation3D.matrixToEuler(R);
        end
        
        function [axisVec, theta] = toAxisAngle(obj)
            R = obj.toMatrix();
            [axisVec, theta] = OceanMath.Geometry.Rotation3D.matrixToAxisAngle(R);
        end
        
        function v = toExpMap(obj)
            R = obj.toMatrix();
            v = OceanMath.Geometry.Rotation3D.matrixToExpMap(R);
        end
        
        function qInterp = interpolate(obj, other, t)
            % Reuse existing procedurals from OceanMath.Slerp
            arguments
                obj
                other OceanMath.Geometry.Rotations.RotationRepresentation3D
                t (1,1) double {mustBeInRange(t, 0, 1)}
            end
            q2 = other.toQuat();
            q_res = OceanMath.Slerp.interpolate(obj.Q_, q2, t);
            qInterp = OceanMath.Geometry.Rotations.QuaternionRotation3D(q_res);
        end
    end
end
