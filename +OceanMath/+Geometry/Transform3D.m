classdef Transform3D
    % TRANSFORM3D  Generates 3D homogeneous coordinate transformation matrices.
    
    methods (Static)
        function T = translation(tx, ty, tz)
            arguments
                tx (1,1) double
                ty (1,1) double
                tz (1,1) double
            end
            T = [1, 0, 0, tx; ...
                 0, 1, 0, ty; ...
                 0, 0, 1, tz; ...
                 0, 0, 0,  1];
        end
        
        function R = rotation(R_3x3)
            arguments
                R_3x3 (3,3) double
            end
            R = [R_3x3, zeros(3,1); ...
                 zeros(1,3), 1];
        end
        
        function S = scale(sx, sy, sz)
            arguments
                sx (1,1) double
                sy (1,1) double
                sz (1,1) double
            end
            S = [sx,  0,  0, 0; ...
                  0, sy,  0, 0; ...
                  0,  0, sz, 0; ...
                  0,  0,  0, 1];
        end
        
        function Sh = shear(sh_xy, sh_xz, sh_yx, sh_yz, sh_zx, sh_zy)
            % SHEAR  Constructs 3D shear matrix (6 DOF).
            % Displaces coordinates proportionally to the other axes:
            %   x' = x + sh_xy * y + sh_xz * z
            %   y' = y + sh_yx * x + sh_yz * z
            %   z' = z + sh_zx * x + sh_zy * y
            %
            % Matrix form:
            %   [   1    sh_xy  sh_xz    0 ]
            %   [ sh_yx    1    sh_yz    0 ]
            %   [ sh_zx  sh_zy    1      0 ]
            %   [   0      0      0      1 ]
            arguments
                sh_xy (1,1) double
                sh_xz (1,1) double
                sh_yx (1,1) double
                sh_yz (1,1) double
                sh_zx (1,1) double
                sh_zy (1,1) double
            end
            Sh = [   1,  sh_xy,  sh_xz, 0; ...
                  sh_yx,      1,  sh_yz, 0; ...
                  sh_zx,  sh_zy,      1, 0; ...
                      0,      0,      0, 1];
        end
        
        function SE = rigid(tx, ty, tz, rotationObj)
            % RIGID  Constructs 3D rigid body motion / Euclidean transformation matrix (6 DOF).
            % rotationObj can be any subclass of RotationRepresentation3D or a 3x3 matrix.
            % Composition: T(tx, ty, tz) * R(R_3x3)
            %
            % Matrix form:
            %   [  R_3x3   t ]
            %   [    0     1 ]
            arguments
                tx (1,1) double
                ty (1,1) double
                tz (1,1) double
                rotationObj
            end
            if isa(rotationObj, 'OceanMath.Geometry.Rotations.RotationRepresentation3D')
                R_3x3 = rotationObj.toMatrix();
            elseif ismatrix(rotationObj) && all(size(rotationObj) == [3, 3])
                R_3x3 = rotationObj;
            else
                error('rotationObj must be an instance of RotationRepresentation3D or a 3x3 matrix.');
            end
            
            T = OceanMath.Geometry.Transform3D.translation(tx, ty, tz);
            R = OceanMath.Geometry.Transform3D.rotation(R_3x3);
            SE = T * R;
        end
        
        function Sim = similarity(tx, ty, tz, rotationObj, s)
            % SIMILARITY  Constructs 3D similarity transform matrix (7 DOF).
            % rotationObj can be any subclass of RotationRepresentation3D or a 3x3 matrix.
            % Composition: T(tx, ty, tz) * R(R_3x3) * S(s, s, s)
            %
            % Matrix form:
            %   [ s*R_3x3  t ]
            %   [    0     1 ]
            arguments
                tx (1,1) double
                ty (1,1) double
                tz (1,1) double
                rotationObj
                s (1,1) double
            end
            if isa(rotationObj, 'OceanMath.Geometry.Rotations.RotationRepresentation3D')
                R_3x3 = rotationObj.toMatrix();
            elseif ismatrix(rotationObj) && all(size(rotationObj) == [3, 3])
                R_3x3 = rotationObj;
            else
                error('rotationObj must be an instance of RotationRepresentation3D or a 3x3 matrix.');
            end
            
            T = OceanMath.Geometry.Transform3D.translation(tx, ty, tz);
            R = OceanMath.Geometry.Transform3D.rotation(R_3x3);
            S = OceanMath.Geometry.Transform3D.scale(s, s, s);
            Sim = T * R * S;
        end
        
        % (Note: No LaTeX in code comments per user rules)
        function A = affine(A_3x3, tx, ty, tz)
            % AFFINE  Constructs 3D general affine transformation matrix (12 DOF).
            % A_3x3 represents any 3x3 linear transformation (rotation, scale, shear).
            %
            % Matrix form:
            %   [ A_3x3  t ]
            %   [   0    1 ]
            arguments
                A_3x3 (3,3) double
                tx (1,1) double
                ty (1,1) double
                tz (1,1) double
            end
            A = [A_3x3, [tx; ty; tz]; ...
                 0, 0, 0, 1];
        end
        
        function H = projective(H_4x4)
            % PROJECTIVE  Constructs 3D projective transformation matrix (15 DOF).
            % Normalizes the homogeneous matrix so the last element is 1.
            arguments
                H_4x4 (4,4) double
            end
            H = H_4x4;
            if H(4,4) ~= 0
                H = H / H(4,4);
            end
        end
        
        function ptsOut = transformPoints(T, ptsIn)
            % TRANSFORMPOINTS Apply 4x4 homogeneous transformation matrix T to 3D points.
            %
            % INPUTS:
            %   T     - [4 x 4] transformation matrix
            %   ptsIn - [3 x N] or [4 x N] input coordinates
            %
            % OUTPUTS:
            %   ptsOut - [3 x N] transformed and normalized coordinates
            arguments
                T (4,4) double
                ptsIn double
            end
            [rows, cols] = size(ptsIn);
            if rows == 3
                ptsHom = [ptsIn; ones(1, cols)];
            elseif rows == 4
                ptsHom = ptsIn;
            else
                error('ptsIn must be of size [3 x N] or [4 x N].');
            end
            
            ptsTrans = T * ptsHom;
            
            % Normalize by homogeneous coordinate w
            w = ptsTrans(4, :);
            w(abs(w) < 1e-15) = 1e-15;
            
            ptsOut = [ptsTrans(1, :) ./ w; ...
                      ptsTrans(2, :) ./ w; ...
                      ptsTrans(3, :) ./ w];
        end
        
        function planeOut = transformPlane(T, planeIn)
            % TRANSFORMPLANE Apply 4x4 homogeneous transformation matrix T to 3D homogeneous plane co-vectors.
            % Formula: pi' = T^(-T) * pi
            %
            % INPUTS:
            %   T       - [4 x 4] transformation matrix
            %   planeIn - [4 x N] homogeneous plane coordinates [a; b; c; d]
            %
            % OUTPUTS:
            %   planeOut - [4 x N] transformed homogeneous plane coordinates
            arguments
                T (4,4) double
                planeIn (4, :) double
            end
            planeOut = (inv(T)') * planeIn;
        end
    end
end
