classdef BoxGeometry < OceanMath.Vision.Geometry.IGeometry
    % BOXGEOMETRY  Oriented (rotated) rectangular box -- the building block of
    %   the procedural furniture shapes used as 3-D ground truth.
    %
    %   A box is  {x : |R^T (x - c)|_i <= h_i, i = 1..3}  with centre c,
    %   half-extents h and orientation R (columns = the box's local axes in
    %   world coordinates).
    %
    % RAY INTERSECTION (slab method, Kay & Kajiya 1986), in the box frame
    %   o' = R^T (o - c),  d' = R^T d:
    %     t1_i = (-h_i - o'_i) / d'_i,   t2_i = (h_i - o'_i) / d'_i
    %     t_near = max_i min(t1_i, t2_i),   t_far = min_i max(t1_i, t2_i)
    %     hit  <=>  t_near <= t_far  and  t_far > 0
    %   The hit face is the axis i* attaining t_near; its outward normal is
    %   -sign(d'_{i*}) e_{i*}, rotated back to world by R.
    %   (If the origin is inside the box, the exit point t_far is returned.)
    %
    % USAGE:
    %   box = OceanMath.Vision.Geometry.BoxGeometry('Center', [0;0;0], 'HalfSize', [1;0.5;0.2]);
    %   [t, n] = box.intersect([0;0;-5], [0;0;1]);            % IGeometry single-ray contract
    %   [t, N] = box.intersectRays(O, D);                    % vectorised, O/D: (P x 3)
    %   [P, N] = box.sampleSurface(1000);                    % uniform-by-area surface points
    %
    % See also: OceanMath.Vision.Geometry.BoxAssembly, OceanMath.Vision.Geometry.IGeometry

    properties
        Center   (3,1) double = [0; 0; 0]
        HalfSize (3,1) double {mustBePositive} = [0.5; 0.5; 0.5]
        Rotation (3,3) double = eye(3)      % columns: local axes in world frame
    end

    methods
        function obj = BoxGeometry(options)
            arguments
                options.Center   (3,1) double = [0; 0; 0]
                options.HalfSize (3,1) double {mustBePositive} = [0.5; 0.5; 0.5]
                options.Rotation (3,3) double = eye(3)
            end
            obj.Center = options.Center;
            obj.HalfSize = options.HalfSize;
            obj.Rotation = options.Rotation;
        end

        % ---------------------------------------------------------------- %
        %  IGeometry contract
        % ---------------------------------------------------------------- %
        function [t, normal] = intersect(obj, rayOrigin, rayDir)
            arguments
                obj
                rayOrigin (3,1) double
                rayDir (3,1) double
            end
            [t, N] = obj.intersectRays(rayOrigin.', (rayDir / norm(rayDir)).');
            normal = N.';
            if isnan(t)
                normal = [0; 0; 0];
            end
        end

        function [vertices, faces] = toMesh(obj)
            % TOMESH  8 corners, 12 outward-oriented triangles (3 x V, 3 x F).
            [V, F] = OceanMath.Vision.Geometry.BoxGeometry.unitCubeMesh();   % cube [-1,1]^3
            V = (obj.Rotation * (obj.HalfSize .* V.')) + obj.Center;          % 3 x 8
            vertices = V;
            faces = F.';
        end

        function normal = getNormal(obj, point)
            % GETNORMAL  Normal of the face closest to the point (largest
            %   normalised local coordinate |x'_i| / h_i).
            local = obj.Rotation.' * (point - obj.Center);
            [~, i] = max(abs(local) ./ obj.HalfSize);
            nLocal = zeros(3, 1);
            nLocal(i) = sign(local(i));
            normal = obj.Rotation * nLocal;
        end

        % ---------------------------------------------------------------- %
        %  Vectorised queries
        % ---------------------------------------------------------------- %
        function [t, N] = intersectRays(obj, O, D)
            % INTERSECTRAYS  O, D: (P x 3) ray origins / directions (world).
            %   t: (P x 1) hit distance in units of ||d|| (NaN = miss);
            %   N: (P x 3) outward world normal at the hit.
            arguments
                obj
                O (:,3) double
                D (:,3) double
            end
            R = obj.Rotation;
            Ol = (O - obj.Center.') * R;                 % rows: R^T (o - c)
            Dl = D * R;                                  % rows: R^T d
            Dl(abs(Dl) < 1e-15) = 1e-15;                 % parallel slab -> +-inf, handled by min/max
            h = obj.HalfSize.';
            t1 = (-h - Ol) ./ Dl;
            t2 = ( h - Ol) ./ Dl;
            tMin = min(t1, t2);
            tMax = max(t1, t2);
            [tNear, axisNear] = max(tMin, [], 2);
            tFar = min(tMax, [], 2);
            hit = (tNear <= tFar) & (tFar > 0);

            originInside = tNear <= 0;                   % start inside: report the exit
            [~, axisFar] = min(tMax, [], 2);
            t = tNear;
            t(originInside) = tFar(originInside);
            hitAxis = axisNear;
            hitAxis(originInside) = axisFar(originInside);
            t(~hit) = NaN;

            P = size(O, 1);
            lin = sub2ind([P 3], (1:P).', hitAxis);
            Nl = zeros(P, 3);
            s = -sign(Dl(lin));                          % entering face faces the ray
            s(originInside) = -s(originInside);          % exit face faces along the ray
            Nl(lin) = s;
            N = Nl * R.';                                % rows: R n_local
            N(~hit, :) = 0;
        end

        function tf = inside(obj, P, tol)
            % INSIDE  (M x 1) logical: point inside the box grown by tol.
            arguments
                obj
                P (:,3) double
                tol (1,1) double = 0
            end
            local = (P - obj.Center.') * obj.Rotation;
            tf = all(abs(local) <= obj.HalfSize.' + tol, 2);
        end

        function [P, N, faceId] = sampleSurface(obj, n, stream)
            % SAMPLESURFACE  n points uniform by area on the 6 faces.
            %   faceId in 1..6 = (axis-1)*2 + (1 for +, 2 for -).
            arguments
                obj
                n (1,1) double {mustBeInteger, mustBeNonnegative}
                stream = []
            end
            h = obj.HalfSize.';
            faceArea = 4 * [h(2)*h(3), h(2)*h(3), h(1)*h(3), h(1)*h(3), h(1)*h(2), h(1)*h(2)];
            cdf = cumsum(faceArea) / sum(faceArea);
            cdf(end) = 1;
            u = OceanMath.Vision.Geometry.BoxGeometry.uniform(stream, n, 3);
            faceId = discretize(u(:, 1), [0, cdf]);
            axisIdx = ceil(faceId / 2);
            sgn = 1 - 2 * (mod(faceId, 2) == 0);         % odd -> +1, even -> -1

            % Local point: the face axis is pinned to +-h, the two free axes
            % are uniform in [-h, h] (u(:,2), u(:,3) are independent of the face choice).
            other1 = mod(axisIdx, 3) + 1;
            other2 = mod(axisIdx + 1, 3) + 1;
            Lp = zeros(n, 3);
            rows = (1:n).';
            Lp(sub2ind([n 3], rows, axisIdx)) = sgn .* h(axisIdx).';
            Lp(sub2ind([n 3], rows, other1)) = (2 * u(:, 2) - 1) .* h(other1).';
            Lp(sub2ind([n 3], rows, other2)) = (2 * u(:, 3) - 1) .* h(other2).';

            Nl = zeros(n, 3);
            Nl(sub2ind([n 3], rows, axisIdx)) = sgn;
            P = Lp * obj.Rotation.' + obj.Center.';
            N = Nl * obj.Rotation.';
        end

        function a = surfaceArea(obj)
            h = obj.HalfSize;
            a = 8 * (h(1)*h(2) + h(2)*h(3) + h(1)*h(3));
        end

        function v = volume(obj)
            v = 8 * prod(obj.HalfSize);
        end

        function C = corners(obj)
            % CORNERS  (8 x 3) world-space corners.
            [V, ~] = OceanMath.Vision.Geometry.BoxGeometry.unitCubeMesh();
            C = (V .* obj.HalfSize.') * obj.Rotation.' + obj.Center.';
        end

        function b = transformed(obj, R, t)
            % TRANSFORMED  Rigid motion x -> R x + t applied to the box.
            arguments
                obj
                R (3,3) double
                t (3,1) double = [0; 0; 0]
            end
            b = OceanMath.Vision.Geometry.BoxGeometry('Center', R * obj.Center + t, ...
                'HalfSize', obj.HalfSize, 'Rotation', R * obj.Rotation);
        end

        function b = scaled(obj, s)
            % SCALED  Uniform scaling about the origin.
            arguments
                obj
                s (1,1) double {mustBePositive}
            end
            b = OceanMath.Vision.Geometry.BoxGeometry('Center', s * obj.Center, ...
                'HalfSize', s * obj.HalfSize, 'Rotation', obj.Rotation);
        end
    end

    methods (Static)
        function [V, F] = unitCubeMesh()
            % UNITCUBEMESH  Cube [-1,1]^3: V (8 x 3), F (12 x 3) outward CCW.
            V = [-1 -1 -1;  1 -1 -1;  1  1 -1; -1  1 -1; ...
                 -1 -1  1;  1 -1  1;  1  1  1; -1  1  1];
            F = [1 4 3; 1 3 2;     % z = -1
                 5 6 7; 5 7 8;     % z = +1
                 1 2 6; 1 6 5;     % y = -1
                 4 8 7; 4 7 3;     % y = +1
                 1 5 8; 1 8 4;     % x = -1
                 2 3 7; 2 7 6];    % x = +1
        end
    end

    methods (Static, Access = private)
        function u = uniform(stream, rows, cols)
            if isempty(stream)
                u = rand(rows, cols);
            else
                u = rand(stream, rows, cols);
            end
        end
    end
end
