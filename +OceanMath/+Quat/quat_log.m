function v = quat_log(q)
%MANUAL_LOG Computes the Principal Logarithm of a unit quaternion.
%
%   v = manual_log(q)
%
%   Description:
%       Maps a unit quaternion from the Hypersphere (S^3) to the Tangent Space (R^3)
%       at the identity. This operation is the inverse of the Exponential map.
%       Geometrically, it converts a rotation quaternion into a scaled rotation
%       vector (Axis * Half-Angle).
%
%   Mathematical Definition:
%       If q = [cos(theta), u * sin(theta)], then:
%       log(q) = u * theta
%       Where 'u' is the unit axis and 'theta' is the half-angle.
%   Input:
%       q - [1x4 double] Unit quaternion in format [w, x, y, z].
%   Output:
%       v - [1x3 double] Rotation vector [vx, vy, vz].
%           The magnitude norm(v) equals the half-angle of rtation (theta).
%
%   Safety Features:
%       1. Enforces the "Principal Value" (Shortest Path): Ensures the returned
%          rotation vector has a magnitude <= pi/2 (representing a physical
%          rotation <= 180 degrees).
%       2. Taylor Series Fallback: Handles angles close to zero to prevent
%          singularity (division by zero) and derivative discontinuities.
%
%--- Execution ---
    % 1. Enforce Principal Logarithm (Shortest Path)
    % The logarithm is multi-valued (like angle + 2*k*pi).
    % We strictly want the "principal value" corresponding to the minimal rotation.
    % If w < 0, the quaternion represents the "long way" (> 180 deg).
    % We flip it to -q to find the equivalent "short way" rotation.
    if q(1) < 0
        q = -q;
    end
    w = q(1); %Scalar part (cos(theta))
    vec = q(2:4); %Vector part (u * sin(theta))

    %Robust Angle Calculation
    %We calculate theta (half-angle) using atan2 fornumerical stability.
    %norm(vec) is sin(theta), w is cos(theta).
    %atan2 is safe for all quadrants and small angles.
    vec_norm = norm(vec);
    theta = atan2(vec_norm, w);

    %Numerical Stability (Taylor Series Expansion)
    %Notice we need v_hat * theta. But we have v^ sin(theta)
    %Multiply v_hat * sin(theta) by theta / sin(theta) = v_hat * theta
    % We need to compute v = vec * (theta / sin(theta)).
    % Let k = theta / sin(theta)
    % Problem: As theta -> 0, sin(theta) -> 0, causing 0/0 division.
    % Solution: Use Taylor Series for Sinc function inverse near zero.
    % Expansion: x/sin(x) approx 1 + x^2/6 + 7x^4/360 ...
    if vec_norm < 1e-4
    %Use 2nd-order Taylor Approximation
    % Note: We use 'theta' as x. Since theta approx sin(theta) approx
    % vec_norm for small angles, this is highly accurate.
    % limit theta approaches 0 of sin( theta) / theta = 1 ( Calculus) 
        k = 1 + (theta^2 / 6);
    else 
        k = theta / vec_norm;
    end

    %4. Compute the Final Tangent Vector
    v = vec * k;
end
