classdef EulerRotation3D < OceanMath.Geometry.Rotations.RotationRepresentation3D
    % EULERROTATION3D  Object-oriented ZYX Euler angle 3D rotation strategy.
    
    properties (Access = private)
        Roll_   % Roll in degrees
        Pitch_  % Pitch in degrees
        Yaw_    % Yaw in degrees
    end
    
    methods
        function obj = EulerRotation3D(roll, pitch, yaw)
            arguments
                roll (1,1) double
                pitch (1,1) double
                yaw (1,1) double
            end
            obj.Roll_ = roll;
            obj.Pitch_ = pitch;
            obj.Yaw_ = yaw;
        end
        
        function R = toMatrix(obj)
            R = OceanMath.Geometry.Rotation3D.eulerToMatrix(obj.Roll_, obj.Pitch_, obj.Yaw_);
        end
        
        function q = toQuat(obj)
            R = obj.toMatrix();
            q = OceanMath.Geometry.Rotation3D.matrixToQuat(R);
        end
        
        function [roll, pitch, yaw] = toEuler(obj)
            roll = obj.Roll_;
            pitch = obj.Pitch_;
            yaw = obj.Yaw_;
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
            % For Euler representation, linear interpolation of angles is performed
            arguments
                obj
                other OceanMath.Geometry.Rotations.RotationRepresentation3D
                t (1,1) double {mustBeInRange(t, 0, 1)}
            end
            [r2, p2, y2] = other.toEuler();
            roll_t = obj.Roll_ + t * (r2 - obj.Roll_);
            pitch_t = obj.Pitch_ + t * (p2 - obj.Pitch_);
            yaw_t = obj.Yaw_ + t * (y2 - obj.Yaw_);
            qInterp = OceanMath.Geometry.Rotations.EulerRotation3D(roll_t, pitch_t, yaw_t);
        end
    end
end
