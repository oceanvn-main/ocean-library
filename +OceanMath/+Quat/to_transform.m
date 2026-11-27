function M = to_transform(q, pos)
%TO_TRANSFORM Constructs a 4x4 Homogeneous Transformation Matrix.
%
%   M = quat.to_transform(q, pos)
%
%   Description:
%       Combines an object's Orientation (Quaternion) and Position (Vector)
%       into a single 4x4 "Rigid Body Transform" matrix.
%
%   Why do we need this? (The Role):
%       1. Graphics Engines: MATLAB's 'hgtransform' and OpenGL require a
%          single matrix to draw objects. They cannot accept a separate
%          quaternion and vector.
%       2. Kinematic Chains: To calculate the position of a robot hand, you
%          multiply the matrices of every joint: M_total = M_shoulder * M_elbow * M_hand.
%       3. Uniform Math: It allows Translation to be treated as Multiplication,
%          unifying rotation and movement into one operation.
%
%   Theory (The Structure):
%       The matrix is built from four distinct blocks:
%
%       M = [  R (3x3)   |   T (3x1)  ]
%           [  0 (1x3)   |   1 (1x1)  ]
%
%       - R (Rotation Block): Defines the Local X, Y, Z axes directions.
%         This is derived from the quaternion.
%       - T (Translation Block): Defines where the Local Origin (0,0,0)
%         is located in the World.
%       - [0 0 0] (Perspective): Used for 3D projection (cameras). For
%         physics/robotics, these are always 0.
%       - [1] (Scale/Homogeneous): Allows us to distinguish between
%         Points (w=1) and Directions (w=0).
%
%   Input:
%       q   - [1x4 double] Unit Quaternion [w, x, y, z]
%       pos - [1x3 double] Position Vector [x, y, z]
%
%   Output:
%       M   - [4x4 double] Homogeneous Transformation Matrix

    % 1. Construct Rotation Block (The "Orientation")
    %    Instead of rewriting the math, we call our verified library function.
    %    This ensures 'M' is always consistent with 'quat.to_rotm'.
    R = quat.quat_to_rotm(q);
    
    % 2. Construct Translation Block (The "Position")
    %    Force 'pos' to be a column vector (3x1) to fit on the right side.
    T = pos(:);
    
    % 3. Assemble the Matrix
    %    [ R  T ]
    %    [ 0  1 ]
    M = [R, T; 0 0 0 1];
end