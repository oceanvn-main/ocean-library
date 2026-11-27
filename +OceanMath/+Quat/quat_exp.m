function q = quat_exp(v)
%MANUAL_EXP Computes the exponential map from tangent space to unit quaternion.
%
%   q = manual_exp(v)
%
%   Description:
%       Maps a rotation vector in R^3 (Lie Algebra so(3)) to a Unit Quaternion
%       in S^3 (Lie Group SO(3)). This operation essentially "wraps" a straight
%       line from tangent space onto the surface of the hypersphere.
%
%   Mathematical Definition:
%       Given a rotation vector v = theta * u (where u is a unit axis):
%       exp(v) = [cos(theta), u * sin(theta)]
%       *Note:* This function assumes the standard half-angle convention where
%       norm(v) = theta (half the total rotation angle).
%
%   Input:
%       v - [1x3 or 3x1 double] Rotation vector.
%
%   Output:
%       q - [1x4 double] Unit quaternion [w, x, y, z].
%
%   Robustness Features:
%       - Taylor Series approximation for small angles (avoids 0/0 division).
%       - Input shape validation (forces row vector output).
%       - Zero-vector optimization (immediate identity return).
%       - Final renormalization to combat floating point drift. 

% --- Execution ---
    % 1. Input Validation & Sanitization
    %Ensure input is real-valued
    if ~isreal(v)
        error('quat_log:ComplexInput', 'Input vector must be real-valued.');
    end

    %Forve v to be a row vector [1x3] for consistent math
    v = v(:)';
    if length(v) ~= 3
        error('manual_exp:InvalidShape', 'Input must be a 3-element vector.');
    end

    % 2. Calculate Half-Angle (theta)
    theta = norm(v);

    % 3. Optimization: Exact Zero Check
    % If the rotation is exactly zero, return Identity immediately.
    if theta == 0
        q = [1, 0, 0, 0];
        return;
    end

    % 4. Numerical Stability (Taylor Series)
    % We need to compute coefficient k = sin(theta) / theta.
    % As theta -> 0, this approaches 0/0.
    % Threshold 1e-4 is chosen where the 4th-order error term is < machine eps.
    if theta < 1e-4 
        % 2nd order Taylor Series expansion for sin(x)/x:
        % sin(x)/x approx 1 - x^2/6
        k = 1 - (theta^2 / 6);
    else
        k = sin(theta) / theta;
    end

    % 5. Construct Quaternion
    % q = [cos(theta) , v*k]
    % Note: v * k is equivalent to (u * theta) * (sin(theta)/theta) = u * sin(theta)
    q = [cos(theta), v * k];

    % 6. Final Safety Normalization
    % Theoretically, q is already unit length.
    % Practically, floating point error in cos/sin/k can drift slightly off 1.0.
    q = q / norm(q);

end