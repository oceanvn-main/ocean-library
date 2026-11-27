function q_aligned = unroll_quaternions(q_raw)
%UNROLL_QUATERNIONS Aligns a quaternion sequence for continuous rotation.
%
%   q_aligned = quat.unroll_quaternions(q_raw)
%
%   Description:
%       Solves the "Double Cover" ambiguity. It iterates through the sequence
%       and ensures that q(i) is always on the same hemisphere as q(i-1).
%       This prevents the "Long Way Around" rotation and visual flips.
%
%   Algorithm:
%       1. Start at index 2.
%       2. Check dot product with previous frame.
%       3. If negative, flip current frame (-q).
%       4. SAVE the flip so the next iteration compares against the new value.

    % 1. Handle Input Type (Object vs Double)
    isObj = isa(q_raw, 'quaternion');
    if isObj
        q_data = double(q_raw);
    else
        q_data = q_raw;
    end
    
    [n_frames, ~] = size(q_data);

    % 2. Sequential Alignment Loop
    % We start at 2 because we need a "Previous" to compare against.
    for i = 2:n_frames
        
        q_prev = q_data(i-1, :);
        q_curr = q_data(i, :);
        
        % Calculate alignment
        if dot(q_prev, q_curr) < 0
            % FLIP: Move q_curr to the valid hemisphere
            q_data(i, :) = -q_curr;
        end
        
        % CRITICAL NOTE: 
        % We modified q_data(i) in place. The next iteration (i+1) will
        % now compare itself against this *corrected* value. This propagates
        % the smoothness down the entire chain.
    end
    
    % 3. Return Correct Format
    if isObj
        q_aligned = quaternion(q_data);
    else
        q_aligned = q_data;
    end
end