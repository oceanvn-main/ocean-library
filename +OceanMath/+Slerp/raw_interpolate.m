function q_out = raw_interpolate(q1, q2, t)
%RAW_INTERPOLATE Performs SLERP without "Shortest Path" enforcement.
%
%   q_out = slerp.raw_interpolate(q1, q2, t)
%
%   Description:
%       Standard SLERP forces the "shortest path" (flipping signs if dot < 0).
%       This function SKIPS that check. It is designed for SQUAD, where
%       'quat.unroll_quaternions' has already determined the correct path alignment.

    % 1. Calculate Cosine
    cosTheta = dot(q1, q2);
    
    % --- CRITICAL: NO FLIP CHECK HERE ---
    % We strictly respect the sign passed in.

    % 2. Linear Fallback (if inputs are identical)
    if abs(cosTheta) >= 1.0
        q_out = q1;
        return;
    end
    
    % 3. Standard SLERP Math
    sinTheta = sqrt(1.0 - cosTheta^2);
    
    % Avoid division by zero
    if abs(sinTheta) < 1e-4
        q_out = 0.5 * q1 + 0.5 * q2;
        q_out = q_out / norm(q_out);
        return;
    end

    angle = atan2(sinTheta, cosTheta);
    
    ratio1 = sin((1 - t) * angle) / sinTheta;
    ratio2 = sin(t * angle) / sinTheta;
    
    q_out = ratio1 * q1 + ratio2 * q2;
end