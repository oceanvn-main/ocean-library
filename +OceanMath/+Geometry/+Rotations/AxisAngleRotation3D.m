classdef AxisAngleRotation3D < OceanMath.Geometry.Rotations.RotationRepresentation3D
    % AXISANGLEROTATION3D  Object-oriented axis-angle 3D rotation strategy.
    
    properties (Access = private)
        AxisVec_  % [1 x 3] unit axis vector
        Theta_    % Rotation angle in degrees
    end
    
    methods
        function obj = AxisAngleRotation3D(axisVec, theta)
            arguments
                axisVec (1,3) double
                theta (1,1) double
            end
            v = axisVec(:)';
            vNorm = norm(v);
            if vNorm < 1e-12
                obj.AxisVec_ = [0, 0, 1];
            else
                obj.AxisVec_ = v / vNorm;
            end
            obj.Theta_ = theta;
        end
        
        function R = toMatrix(obj)
            R = OceanMath.Geometry.Rotation3D.axisAngleToMatrix(obj.AxisVec_, obj.Theta_);
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
            axisVec = obj.AxisVec_;
            theta = obj.Theta_;
        end
        
        function v = toExpMap(obj)
            R = obj.toMatrix();
            v = OceanMath.Geometry.Rotation3D.matrixToExpMap(R);
        end
        
        function qInterp = interpolate(obj, other, t)
            % Performs linear interpolation of the axis-angle components
            arguments
                obj
                other OceanMath.Geometry.Rotations.RotationRepresentation3D
                t (1,1) double {mustBeInRange(t, 0, 1)}
            end
            [ax2, th2] = other.toAxisAngle();
            ax_t = obj.AxisVec_ + t * (ax2 - obj.AxisVec_);
            th_t = obj.Theta_ + t * (th2 - obj.Theta_);
            qInterp = OceanMath.Geometry.Rotations.AxisAngleRotation3D(ax_t, th_t);
        end
    end
end
