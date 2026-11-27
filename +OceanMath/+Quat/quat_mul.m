function q_out = quat_mul(q1, q2)
%MUL Computes the Hamilton Product of two quaternions.
%
%   q_out = quat.mul(q1, q2)
%
%   Description:
%       Multiplies two quaternions. If inputs are unit quaternions representing
%       rotations, this computes the composite rotation.
%   Reference: https://www.3dgep.com/understanding-quaternions/
%
%   -----------------------------------------------------------------------
%   DERIVATION NOTE
%   The product of two quaternions q1 = (s1, v1) and q2 = (s2, v2) is derived
%   by distributing terms and applying basis vector rules (i^2=-1, ij=k, etc).
%
%   Expansion:
%   (s1 + v1) * (s2 + v2)
%   = s1*s2 + s1*v2 + s2*v1 + v1*v2
%
%   The term v1*v2 breaks down into:
%   - Dot product (Real part):  -(v1 . v2)
%   - Cross product (Vector part): (v1 x v2)
%
%   Final Formula:
%   Scalar Part: s1*s2 - dot(v1, v2)
%   Vector Part: s1*v2 + s2*v1 + cross(v1, v2)
%   -----------------------------------------------------------------------
%
%   Input:
%       q1, q2 - [1x4 double] Quaternions [w, x, y, z]
%
%   Output:
%       q_out  - [1x4 double] Composite Quaternion

    % 1. Robustness: Force row vectors to support [w, v] concatenation
    q1 = q1(:)';
    q2 = q2(:)';

    % 2. Extract Components
    % q = [w, v_x, v_y, v_z]
    w1 = q1(1); v1 = q1(2:4);
    w2 = q2(1); v2 = q2(2:4);

    % 3. Compute Scalar Part
    % Corresponds to: s1*s2 - (v1 . v2)
    w = w1*w2 - dot(v1, v2);

    % 4. Compute Vector Part
    % Corresponds to: s1*v2 + s2*v1 + (v1 x v2)
    v = w1*v2 + w2*v1 + cross(v1, v2);

    % 5. Combine Output
    q_out = [w, v];
end