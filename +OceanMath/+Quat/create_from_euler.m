function q = create_from_euler(varargin)
%MANUAL_QUAT Creates a unit quaternion fro raw components or Euler angles.
% 
% Assumptions:
% - For Euler angles, the rotation sequence is Z-Y-X (Yaw-Pitch-Roll).
% - Input angles are assumed to be in Radians.
% - The output follows the [w, x, y, z] convention (scalar-first).

    %Check if the user provided 4 components inputs
    if nargin == 4
        %Assign inputs directly into 1x4 row vector
        q = [varargin{1}, varargin{2}, varargin{3}, varargin{4}];

    elseif nargin == 1
        %Extract angles fro mthe input vectors [Yaw, Pitch, Roll]
        angles = varargin{1};
        yaw = angles(1);    % Rotation around Z-axis (psi)
        pitch = angles(2);  % Rotation around Y-axis (theta)
        roll = angles(3);   % Rotation around X-axis (phi)
        %q_z = [cy, 0, 0, sy]
        %q_y = [cp, 0, sp, 0]
        %Scalar part. cycp - v_z dot v_y = cycp - 0 = cycp
        %Vector part: scalar of qz times vector v_y + scalar of qy times vector v_z + vector part z cross vector part y 
        
        %Calculate half-angles as quaternions represent rotations by theta/2
        % Compute cosines (c) and sines (s) for each half-angle
        cy = cos(yaw * 0.5);   sy = sin(yaw * 0.5);   % Yaw half-angle
        cp = cos(pitch * 0.5); sp = sin(pitch * 0.5); % Pitch half-angle
        cr = cos(roll * 0.5);  sr = sin(roll * 0.5);  % Roll half-angle


        %Shortcut. Derived from Yaw cross Pitch = result, then (Yaw x pitch) x roll
        %Combine the three rotations into a single quaternion
        q = [cy*cp*cr + sy*sp*sr, ... % Scalar part (w): Combined cosine products
             cy*cp*sr - sy*sp*cr, ... % Vector part (x): Primarily Roll component
             cy*sp*cr + sy*cp*sr, ... % Vector part (y): Primarily Pitch component
             sy*cp*cr - cy*sp*sr];    % Vector part (z): Primarily Yaw 

    end
    %Prevent floting-point error (creep error)
    q  = q/norm(q);
end