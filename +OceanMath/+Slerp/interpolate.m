function q_out = interpolate(q1, q2, t)
%INTERPOLATE Performs Spherical Linear Interpolation (SLERP).
%
%   q_out = slerp.interpolate(q1, q2, t)
%
%   Description:
%       Computes the shortest path interpolation between two unit quaternions.
%       Handles the "Double Cover" property (q = -q) to ensure the rotation
%       takes the shortest arc (always < 180 degrees).
%
%   Theory:
%       q(t) = (sin((1-t)O)/sin(O)) * q1 + (sin(tO)/sin(O)) * q2
%
%   Input:
%       q1, q2 - [1x4 double] Unit Quaternions [w, x, y, z]
%       t      - [Scalar] Interpolation factor [0..1]
%
%   Output:
%       q_out  - [1x4 double] Interpolated Quaternion

    % 1. Robustness: Force row vectors (Pure Double Array)
    % Converting to double handles both 'double' and 'quaternion' objects input
    v1 = double(q1(:)');
    v2 = double(q2(:)');

    % 2. Compute Cosine of Angle (Dot Product)
    cosTheta = dot(v1, v2);

    % 3. Shortest Path Check (The "Double Cover" Fix)
    % If dot < 0, quaternions are > 90 deg apart on the 4D sphere.
    % We flip one vector to find the antipodal point that represents
    % the exact same rotation but is closer.
    if cosTheta < 0
        v1 = -v1;
        cosTheta = -cosTheta;
    end

    % 4. Edge Case: Linear Fallback (Small Angles)
    % If quaternions are very close, sin(theta) -> 0, causing division by zero.
    % We use Linear Interpolation (LERP) approximation for stability.
    if cosTheta > 0.9995
        % LERP: (1-t)v1 + t*v2
        qt_vec = (1 - t) * v1 + t * v2;
        q_out = qt_vec / norm(qt_vec); % Normalize
    else
        % 5. Standard SLERP
        theta = acos(cosTheta);
        sinTheta = sin(theta);
        
        weight1 = sin((1-t) * theta) / sinTheta;
        weight2 = sin(t * theta) / sinTheta;
        
        qt_vec = weight1 * v1 + weight2 * v2;
        q_out = qt_vec / norm(qt_vec); % Normalize result
    end
end