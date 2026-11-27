function v_out = quat_rotate_vector(q, v)
%QUAT_ROTATE_VECTOR Rotates a 3D vector by a quaternion.
%
%   v_out = quat_rotate_vector(q, v)
%
%   Description:
%       Directly applies a quaternion rotation to a vector without converting
%       to a rotation matrix first. This function implements the optimized 
%       "Rodrigues-like" formula, which is significantly faster for single 
%       vector operations (e.g., rotating a velocity vector, a camera forward 
%       vector, or a sensor offset).
%
%   Mathematical Formula:
%       v' = v + 2*w*(u x v) + 2*(u x (u x v))
%
%       This is algebraically equivalent to the standard sandwich product:
%           p' = q * p * q^-1
%       But reduces computational cost from ~32 ops to ~15 ops.
%
%   Inputs:
%       q - [1x4 double] Unit Quaternion [w, x, y, z]
%       v - [1x3 or 3x1 double] 3D Vector to rotate
%
%   Output:
%       v_out - [1x3 or 3x1 double] Rotated vector (maintains input shape)
%
%   See also: quat_to_rotm

    % 1. Input Shape Standardization
    % Force 'v' to be a column vector [3x1] for consistent cross-product math.
    % We store the original shape state to restore it later.
    was_row = isrow(v);
    v_in = v(:);
    
    % 2. Extract Quaternion Components
    % q = [w, u_x, u_y, u_z]
    w = q(1);        % Scalar part (cos(theta/2))
    u = q(2:4)';     % Vector part (axis * sin(theta/2)) - ensure column
    
    % 3. Compute Cross Products (The Geometric Interaction)
    % term1 represents the vector perpendicular to both axis 'u' and vector 'v'
    uv = cross(u, v_in);
    
    % term2 represents the vector perpendicular to axis 'u' and 'uv'
    % This term points "inward" towards the rotation axis
    uuv = cross(u, uv);
    
    % 4. Apply the Rodrigues Formula
    % v' = v + 2*w*(u x v) + 2*(u x (u x v))
    % The factors of '2' account for the half-angle nature of quaternions.
    v_prime = v_in + (2 * w * uv) + (2 * uuv);
    
    % 5. Output Formatting
    % Return the vector in the same orientation (row/col) as the input.
    if was_row
        v_out = v_prime';
    else
        v_out = v_prime;
    end
end