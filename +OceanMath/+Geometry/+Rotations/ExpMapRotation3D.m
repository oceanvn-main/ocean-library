classdef ExpMapRotation3D < OceanMath.Geometry.Rotations.RotationRepresentation3D
    % EXPMAPROTATION3D  Object-oriented exponential map 3D rotation strategy.
    % Directly wraps Rodrigues' rotation formula (Ayache 1989).
    
    properties (Access = private)
        V_  % [3 x 1] exponential map vector (theta * axis)
    end
    
    methods
        function obj = ExpMapRotation3D(v)
            arguments
                v (3,1) double
            end
            obj.V_ = v;
        end
        
        function R = toMatrix(obj)
            % Directly wraps the existing Rodrigues' rotation formula procedural
            R = OceanMath.Geometry.Rotation3D.expMapToMatrix(obj.V_);
        end
        
        function q = toQuat(obj)
            R = obj.toMatrix();
            q = OceanMath.Geometry.Rotation3D.matrixToQuat(R);
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
            v = obj.V_;
        end
        
        function qInterp = interpolate(obj, other, t)
            % Performs linear interpolation of the so(3) Lie Algebra vector
            arguments
                obj
                other OceanMath.Geometry.Rotations.RotationRepresentation3D
                t (1,1) double {mustBeInRange(t, 0, 1)}
            end
            v2 = other.toExpMap();
            v_t = obj.V_ + t * (v2 - obj.V_);
            qInterp = OceanMath.Geometry.Rotations.ExpMapRotation3D(v_t);
        end
    end
end
