function R = quat_to_rotm(q)
%QUAT_TO_ROTM Converts a Unit Quaternion to a 3x3 Rotation Matrix.
%
%   R = quat_to_rotm(q)
%
%   Description:
%       Converts a quaternion into a rotation matrix by calculating where the
%       standard basis axes (X, Y, Z) end up after the rotation.
%
%   -----------------------------------------------------------------------
%   DERIVATION NOTE (For Future Reference):
%   How do we get the formula for r11 = 1 - 2y^2 - 2z^2?
%
%   1. Concept:
%      The first column of a rotation matrix represents the new position
%      of the world X-axis vector i = [1, 0, 0].
%
%   2. Formula:
%      We rotate vector 'i' using the quaternion sandwich product:
%      v' = q * i * q_conjugate
%      Let q = w + xi + yj + zk
%
%   3. Expansion:
%      (w + xi + yj + zk) * i * (w - xi - yj - zk)
%      ... expanding terms ...
%      Result x-component = w^2 + x^2 - y^2 - z^2
%
%   4. Optimization (The "Trick"):
%      Since q is unit length: w^2 + x^2 + y^2 + z^2 = 1
%      Therefore: w^2 + x^2 = 1 - y^2 - z^2
%      Substitute this into the result:
%      r11 = (1 - y^2 - z^2) - y^2 - z^2
%      r11 = 1 - 2y^2 - 2z^2
%   -----------------------------------------------------------------------
%
%   Input:
%       q - [1x4 double] Unit Quaternion [w, x, y, z]
%
%   Output:
%       R - [3x3 double] Orthonormal Rotation Matrix

    % 1. Input Robustness
    % Force q to be a row vector and normalize it.
    % Normalization is critical to prevent the matrix from scaling the object.
    q = q(:)';
    q = q / norm(q);

    % 2. Extract Components
    w = q(1); x = q(2); y = q(3); z = q(4);

    % 3. Pre-calculate squared terms
    % These appear repeatedly in the diagonal elements.
    xx = x^2;  yy = y^2;  zz = z^2;

    % 4. Construct the Matrix Columns
    
    % --- Column 1: The New X-Axis ---
    % Corresponds to q * [1 0 0] * q'
    r11 = 1 - 2*yy - 2*zz;  % (See Derivation Note above)
    r21 = 2*x*y + 2*z*w;
    r31 = 2*x*z - 2*y*w;

    % --- Column 2: The New Y-Axis ---
    % Corresponds to q * [0 1 0] * q'
    r12 = 2*x*y - 2*z*w;
    r22 = 1 - 2*xx - 2*zz;
    r32 = 2*y*z + 2*x*w;

    % --- Column 3: The New Z-Axis ---
    % Corresponds to q * [0 0 1] * q'
    r13 = 2*x*z + 2*y*w;
    r23 = 2*y*z - 2*x*w;
    r33 = 1 - 2*xx - 2*yy;

    % 5. Assemble Matrix
    R = [r11, r12, r13;
         r21, r22, r23;
         r31, r32, r33];
end