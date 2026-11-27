classdef (Abstract) RotationRepresentation3D < handle
    % ROTATIONREPRESENTATION3D  Abstract contract for all 3D rotation representations.
    
    methods (Abstract)
        % Returns standard [3 x 3] orthonormal rotation matrix (SO3)
        R = toMatrix(obj)
        
        % Returns unit quaternion [w, x, y, z]
        q = toQuat(obj)
        
        % Returns ZYX Euler angles [roll, pitch, yaw] in degrees
        [roll, pitch, yaw] = toEuler(obj)
        
        % Returns unit axis [ax, ay, az] and angle in degrees
        [axisVec, theta] = toAxisAngle(obj)
        
        % Returns [3 x 1] exponential map (so3 Lie algebra)
        v = toExpMap(obj)
        
        % Interpolates between this and another rotation at time t in [0, 1]
        % Uses the representation's native optimal path (e.g. SLERP for Quat)
        qInterp = interpolate(obj, other, t)
    end
end
