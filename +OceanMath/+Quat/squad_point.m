function si = squad_point(prev, curr, next)
% CALCULATE_SI_VECTOR Computes the inner control point (si) for Squad
% Interpolation
% si = calculate_si_vector(prev, curr, next)
% Description:
% This function calculates the helper control point 'si' (also denoted 'ai'
% in some literature) required for Spherical Quadrangle Interpolation
% (SQUAD).
% In a quaternion spline, simply connecting knots q_i with SLERP produces
% sharp changes in angular velocity at the knots. SQUAD uses these
% intermediate points 'si' to define the tangent at q_i, ensuring C^1
% continuity (smooth velocity) through the sequence.
% Mathematical Formula. The function implemetns the Shoemake heuristic:
%       s_i = exp( - (log(q_{i+1} * q_i^-1) + log(q_{i-1} * q_i^-1)) / 4 ) * q_i
%
%       Where:
%           q_{i-1} = prev
%           q_i     = curr
%           q_{i+1} = next
%
%   Inputs:
%       prev - [1x4 double] Unit quaternion q_{i-1} (The previous frame)
%       curr - [1x4 double] Unit quaternion q_i     (The current frame)
%       next - [1x4 double] Unit quaternion q_{i+1} (The next frame)
%
%       *Format:* [w, x, y, z] (Scalar first).
%       *Note:* Inputs must be unit length for valid rotation logic.
%   Ouputs:
%       si: - [1x4 double] The intermediate control quaternion s_i.
%       This point lies on the tangent sphere surface near 'curr'.
%   Dependencies: This function requires the following helper functions
%       - quat_mult(q1, q2) : Quaternion multiplication
%       - quat_log(q)       : Quaternion logarithm map (S^3 -> R^3) From 4D
%       to 3D.
%       - quat_exp(v)       : Quaternion exponential map (R^3 -> S^3) Map
%       from 3D to 4D
%   Example:
%       % Define 3 keyframes
%       q_prev = [1, 0, 0, 0];
%       q_curr = [0.7071, 0, 0.7071, 0];
%       q_next = [0, 0, 1, 0];
%
%       % Calculate the control point for q_curr
%       s_curr = calculate_si_vector(q_prev, q_curr, q_next);
%
%   References:
%       Shoemake, Ken. "Animating rotation with quaternion curves." 
%       SIGGRAPH Comput. Graph. 19.3 (1985).

%Execution
    % Invert the current quaternion to compute relative offsets. By property of
    % quaternion. q^-1 = q* / |q|^2 . When |q|^2 = 1, we have q^-1 = q*, where
    % q* is conjugate. More understanding https://www.3dgep.com/understanding-quaternions/
    % q_inv = [w , -x, -y, -z]
    % 1. Prepare Inverse for Global Difference Calculation
    %    We want to solve for 'diff' in:  next = diff * curr ( diff on the
    %    left, apply diff on curr. Global rotation)
    %   But to find that diff we need to multiply by curr^-1 on the Right
    %    Algebra: next * curr^-1 = diff
    %    Therefore, we need the inverse on the RIGHT.
    inv_curr = [curr(1), -curr(2:4)];

    %Calculate relative rotation in the Global Frame
    %Computes the "difference" rotation from Current to Next/Prev
    %delta q= q1^-1 * q2
    %2. Calculate Relative Rotations (Global Differences)
    %    q_next_rel = next * inv_curr  (Difference to go from Curr -> Next in World)
    %    q_prev_rel = prev * inv_curr  (Difference to go from Curr -> Prev in World)
    q_next_rel = quat.quat_mul(next, inv_curr); 
    q_prev_rel = quat.quat_mul(prev, inv_curr);

    %3. Convert to Tangent Space (3D Angular Velocity)
    %These vectors represent the axis of rotation in the WORLD frame.
    v_next = quat.quat_log(q_next_rel);
    v_prev = quat.quat_log(q_prev_rel);

    % 4. Average the Tangents (Shoemake Heuristic)
    %    We average the velocities to ensure the curve passes through 'curr' smoothly.
    %    Note: The negative sign flips 'v_prev' so it points 'forward'.
    v_tan = - (v_next + v_prev) / 4;

    %5. Convert Tangent back to Rotation
    %    q_tan is a small rotation representing the tangent direction.
    q_tan = quat.quat_exp(v_tan);

    %6. Apply Tangent to Current Frame
    % Since 'q_tan' was derived in the Global Frame, we must Apply it on
    % the LEFT.     si = q_tan * curr
    si = quat.quat_mul(q_tan, curr);

    %7. Renormalized for numerical stability
    si = si / norm(si);
end


