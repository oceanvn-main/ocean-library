classdef Transform2D
    % TRANSFORM2D  Generates 2D homogeneous coordinate transformation matrices.
    %
    % Generic mathematical functions for translation, rotation, scale, shear,
    % affine, and projective (homography) transforms.
    
    methods (Static)
        function T = translation(tx, ty)
            % TRANSLATION  Constructs 2D homogeneous translation matrix (2 DOF).
            % Maps point (x, y) to (x + tx, y + ty).
            %
            % Matrix form:
            %   [ 1  0  tx ]
            %   [ 0  1  ty ]
            %   [ 0  0   1 ]
            arguments
                tx (1,1) double
                ty (1,1) double
            end
            T = [1, 0, tx; ...
                 0, 1, ty; ...
                 0, 0,  1];
        end
        
        function R = rotation(theta_deg)
            % ROTATION  Constructs 2D homogeneous rotation matrix (1 DOF).
            % Performs counter-clockwise rotation by theta degrees around the origin (0,0).
            %
            % Matrix form:
            %   [ cos(theta)  -sin(theta)  0 ]
            %   [ sin(theta)   cos(theta)  0 ]
            %   [     0            0       1 ]
            arguments
                theta_deg (1,1) double
            end
            theta = deg2rad(theta_deg);
            c = cos(theta);
            s = sin(theta);
            R = [c, -s, 0; ...
                 s,  c, 0; ...
                 0,  0, 1];
        end
        
        function S = scale(sx, sy)
            % SCALE  Constructs 2D homogeneous scaling matrix (2 DOF).
            % Multiplies coordinates by non-uniform scaling factors sx and sy.
            %
            % Matrix form:
            %   [ sx   0   0 ]
            %   [  0  sy   0 ]
            %   [  0   0   1 ]
            arguments
                sx (1,1) double
                sy (1,1) double
            end
            S = [sx,  0, 0; ...
                  0, sy, 0; ...
                  0,  0, 1];
        end
        
        function Sh = shear(shx, shy)
            % SHEAR  Constructs 2D homogeneous shear matrix (2 DOF).
            % Displaces coordinates proportionally to the other axis:
            %   x' = x + shx * y
            %   y' = y + shy * x
            %
            % Matrix form:
            %   [  1  shx  0 ]
            %   [ shy   1  0 ]
            %   [  0    0  1 ]
            arguments
                shx (1,1) double
                shy (1,1) double
            end
            Sh = [1, shx, 0; ...
                  shy, 1, 0; ...
                  0,   0, 1];
        end
        
        function SE = rigid(tx, ty, theta_deg)
            % RIGID  Constructs 2D rigid body motion / Euclidean transformation matrix (3 DOF).
            % Preserves Euclidean distances and angles (no scale, no shear).
            % Composition: T(tx, ty) * R(theta)
            %
            % Matrix form:
            %   [ cos(theta)  -sin(theta)  tx ]
            %   [ sin(theta)   cos(theta)  ty ]
            %   [     0            0        1 ]
            arguments
                tx (1,1) double
                ty (1,1) double
                theta_deg (1,1) double
            end
            T = OceanMath.Geometry.Transform2D.translation(tx, ty);
            R = OceanMath.Geometry.Transform2D.rotation(theta_deg);
            SE = T * R;
        end
        
        function Sim = similarity(tx, ty, theta_deg, s)
            % SIMILARITY  Constructs 2D similarity transform matrix (4 DOF).
            % Preserves angles between lines, scaling is uniform.
            % Composition: T(tx, ty) * R(theta) * S(s, s)
            %
            % Matrix form:
            %   [ s*cos(theta)  -s*sin(theta)  tx ]
            %   [ s*sin(theta)   s*cos(theta)  ty ]
            %   [      0              0         1 ]
            arguments
                tx (1,1) double
                ty (1,1) double
                theta_deg (1,1) double
                s (1,1) double
            end
            T = OceanMath.Geometry.Transform2D.translation(tx, ty);
            R = OceanMath.Geometry.Transform2D.rotation(theta_deg);
            S = OceanMath.Geometry.Transform2D.scale(s, s);
            Sim = T * R * S;
        end
        
        function SS = stretchSquash(tx, ty, sx, sy)
            % STRETCHSQUASH  Constructs aspect-ratio (stretch/squash) transformation matrix (4 DOF).
            % Applies non-uniform scaling followed by translation.
            % Composition: T(tx, ty) * S(sx, sy)
            %
            % Matrix form:
            %   [ sx   0   tx ]
            %   [  0  sy   ty ]
            %   [  0   0    1 ]
            arguments
                tx (1,1) double
                ty (1,1) double
                sx (1,1) double
                sy (1,1) double
            end
            T = OceanMath.Geometry.Transform2D.translation(tx, ty);
            S = OceanMath.Geometry.Transform2D.scale(sx, sy);
            SS = T * S;
        end
        
        function A = affine(tx, ty, theta_deg, sx, sy, shx, shy)
            % AFFINE  Constructs general affine transformation matrix (6 DOF).
            % Preserves parallel lines but not angles or distances.
            % Composition: T(tx, ty) * R(theta) * Sh(shx, shy) * S(sx, sy)
            arguments
                tx (1,1) double
                ty (1,1) double
                theta_deg (1,1) double
                sx (1,1) double
                sy (1,1) double
                shx (1,1) double
                shy (1,1) double
            end
            T = OceanMath.Geometry.Transform2D.translation(tx, ty);
            R = OceanMath.Geometry.Transform2D.rotation(theta_deg);
            Sh = OceanMath.Geometry.Transform2D.shear(shx, shy);
            S = OceanMath.Geometry.Transform2D.scale(sx, sy);
            A = T * R * Sh * S;
        end
        
        function ptsOut = transformPoints(H, ptsIn)
            % TRANSFORMPOINTS  Applies 3x3 homography matrix H to standard 2D vectors (points).
            % Converts point to homogeneous coordinates [x; y; 1] (if 2xN),
            % performs linear warp, and normalizes by the third homogeneous coordinate w.
            %
            % Formula:
            %   x_hom' = H * [x; y; 1]
            %   x' = x_hom'(1)/x_hom'(3), y' = x_hom'(2)/x_hom'(3)
            %
            % INPUTS:
            %   H     - [3 x 3] projective transformation matrix
            %   ptsIn - [2 x N] or [3 x N] input coordinates
            %
            % OUTPUTS:
            %   ptsOut - [2 x N] transformed and normalized coordinates
            arguments
                H (3,3) double
                ptsIn double
            end
            [rows, cols] = size(ptsIn);
            if rows == 2
                ptsHom = [ptsIn; ones(1, cols)];
            elseif rows == 3
                ptsHom = ptsIn;
            else
                error('ptsIn must be of size [2 x N] or [3 x N].');
            end
            
            ptsTrans = H * ptsHom;
            
            % Normalize by homogeneous coordinate w (Depth/perspective scalar)
            w = ptsTrans(3, :);
            
            % Avoid divide-by-zero for points at infinity
            w(abs(w) < 1e-15) = 1e-15;
            
            ptsOut = [ptsTrans(1, :) ./ w; ...
                      ptsTrans(2, :) ./ w];
        end
        
        function lineOut = transformLine(H, lineIn)
            % TRANSFORMLINE  Applies 3x3 homography matrix H to co-vectors (line equations).
            % Uses the transposed inverse matrix (H^-T) to preserve algebraic and physical
            % intersection properties (l^T * x = 0) under non-orthogonal transforms.
            %
            % Formula:
            %   l' = H^(-T) * l
            %
            % INPUTS:
            %   H      - [3 x 3] projective transformation matrix
            %   lineIn - [3 x N] homogeneous line coordinates [a; b; c]
            %
            % OUTPUTS:
            %   lineOut - [3 x N] transformed homogeneous line coordinates
            arguments
                H (3,3) double
                lineIn (3, :) double
            end
            lineOut = (inv(H)') * lineIn;
        end
    end
end
