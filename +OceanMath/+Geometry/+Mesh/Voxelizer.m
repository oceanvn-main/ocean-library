classdef Voxelizer < handle
    % VOXELIZER  Solid -> voxel occupancy grid (the ground-truth target of a
    %   voxel branch, and the inverse check of Cubify).
    %
    %   Two ways to say what "inside" means:
    %     (a) an implicit inside-test  insideFn(P) -> logical, P (M x 3)
    %         (exact for analytic solids, e.g. a union of boxes), or
    %     (b) a closed triangle mesh, via the RAY-PARITY rule (Jordan curve
    %         theorem in 3-D): a point is inside iff a ray from it crosses the
    %         surface an odd number of times.
    %
    % OCCUPANCY RULE (supersampled):
    %   Each voxel is probed at s^3 regularly spaced sub-cell centres; the
    %   voxel is occupied iff the inside FRACTION >= MinFraction.
    %     MinFraction = 0.5      "majority" (volume-preserving on average)
    %     MinFraction = 1/s^3    "any sub-sample inside" (keeps thin parts,
    %                            e.g. chair legs thinner than one voxel)
    %
    % RAY-PARITY TEST (vectorised):
    %   Rays are cast along a fixed GENERIC direction d (not an axis). The
    %   scene is first rotated so that d becomes +x; then, for a triangle
    %   with corners (y,z)-projected to a, b, c and a query point q:
    %     det  = (b - a) x (c - a)                          (2-D cross product)
    %     l_b  = ((q - a) x (c - a)) / det,  l_c = ((b - a) x (q - a)) / det,
    %     l_a  = 1 - l_b - l_c
    %   The +x ray from q crosses the triangle iff l_a, l_b, l_c >= 0 and
    %   x_hit = l_a x_a + l_b x_b + l_c x_c  >  qx.  Triangles parallel to the
    %   ray (det = 0) are skipped.
    %   WHY NOT AN AXIS RAY: for a lattice-aligned mesh (Cubify output) an
    %   axis ray from a voxel centre pierces each square face exactly at its
    %   centre -- ON the diagonal shared by its two triangles -- so it is
    %   counted twice and the parity flips. A generic direction makes such
    %   edge/vertex hits a measure-zero event instead of a systematic one.
    %
    % GRID CONVENTION: identical to Cubify -- occ(d,h,w), w->x, h->y, d->z,
    %   grid spanning Bounds = [xmin xmax; ymin ymax; zmin zmax].
    %
    % USAGE:
    %   occ = OceanMath.Geometry.Mesh.Voxelizer.fromInsideFunction(@(P) vecnorm(P,2,2) < 0.8, [16 16 16]);
    %   occ = OceanMath.Geometry.Mesh.Voxelizer.fromMesh(mesh, [16 16 16], 'Supersample', 2);
    %   tf  = OceanMath.Geometry.Mesh.Voxelizer.insideMesh(V, F, P);
    %
    % See also: OceanMath.Geometry.Mesh.Cubify

    methods (Static)
        function [occ, fraction] = fromInsideFunction(insideFn, gridSize, opts)
            % FROMINSIDEFUNCTION  occ (D x H x W logical), fraction (D x H x W).
            arguments
                insideFn function_handle
                gridSize (1,3) double {mustBeInteger, mustBePositive}
                opts.Bounds (3,2) double = [-1 1; -1 1; -1 1]
                opts.Supersample (1,1) double {mustBeInteger, mustBePositive} = 1
                opts.MinFraction (1,1) double {mustBeGreaterThan(opts.MinFraction, 0), mustBeLessThanOrEqual(opts.MinFraction, 1)} = 0.5
            end
            D = gridSize(1); H = gridSize(2); W = gridSize(3);
            s = opts.Supersample;
            b = opts.Bounds;

            % Sub-cell centres along each axis: (k - 0.5)/(n s), k = 1..n s.
            xs = b(1,1) + ((1:W*s) - 0.5) / (W*s) * (b(1,2) - b(1,1));
            ys = b(2,1) + ((1:H*s) - 0.5) / (H*s) * (b(2,2) - b(2,1));
            zs = b(3,1) + ((1:D*s) - 0.5) / (D*s) * (b(3,2) - b(3,1));
            [Z, Y, X] = ndgrid(zs, ys, xs);                 % (D s) x (H s) x (W s)
            inside = insideFn([X(:), Y(:), Z(:)]);
            inside = reshape(inside, D*s, H*s, W*s);

            % Average the s x s x s probes that fall into each voxel.
            fraction = reshape(inside, s, D, s, H, s, W);
            fraction = squeeze(mean(fraction, [1 3 5]));
            fraction = reshape(fraction, D, H, W);
            occ = fraction >= opts.MinFraction - 1e-12;
        end

        function [occ, fraction] = fromMesh(mesh, gridSize, opts)
            % FROMMESH  Voxelise a closed TriMesh with the ray-parity rule.
            arguments
                mesh OceanMath.Geometry.Mesh.TriMesh
                gridSize (1,3) double {mustBeInteger, mustBePositive}
                opts.Bounds (3,2) double = [-1 1; -1 1; -1 1]
                opts.Supersample (1,1) double {mustBeInteger, mustBePositive} = 1
                opts.MinFraction (1,1) double = 0.5
            end
            V = mesh.Vertices; F = mesh.Faces;
            [occ, fraction] = OceanMath.Geometry.Mesh.Voxelizer.fromInsideFunction( ...
                @(P) OceanMath.Geometry.Mesh.Voxelizer.insideMesh(V, F, P), gridSize, ...
                'Bounds', opts.Bounds, 'Supersample', opts.Supersample, 'MinFraction', opts.MinFraction);
        end

        function tf = insideMesh(V, F, P, opts)
            % INSIDEMESH  Ray-parity inside test for query points P (M x 3)
            %   against a closed mesh (V, F). Returns (M x 1) logical.
            arguments
                V (:,3) double
                F (:,3) double
                P (:,3) double
                opts.ChunkElements (1,1) double = 4e6   % max (points x faces) per chunk
            end
            % Rotate the generic ray direction onto +x (rows of Rd: d, u, w orthonormal).
            Rd = OceanMath.Geometry.Mesh.Voxelizer.rayFrame();
            V = V * Rd.';
            P = P * Rd.';
            A = V(F(:,1), :); B = V(F(:,2), :); C = V(F(:,3), :);
            % Projection onto the (y,z) plane: row vectors of length nF.
            ay = A(:,2).'; az = A(:,3).';
            e1y = (B(:,2) - A(:,2)).'; e1z = (B(:,3) - A(:,3)).';   % b - a
            e2y = (C(:,2) - A(:,2)).'; e2z = (C(:,3) - A(:,3)).';   % c - a
            det2 = e1y .* e2z - e1z .* e2y;
            valid = abs(det2) > 1e-14;                                % not parallel to +x
            det2(~valid) = 1;                                         % avoid 0/0; masked below
            xa = A(:,1).'; xb = B(:,1).'; xc = C(:,1).';

            M = size(P, 1);
            tf = false(M, 1);
            chunk = max(1, floor(opts.ChunkElements / max(1, size(F, 1))));
            for first = 1:chunk:M
                idx = first:min(M, first + chunk - 1);
                qy = P(idx, 2) - ay;                                  % (m x nF): q - a
                qz = P(idx, 3) - az;
                lb = (qy .* e2z - qz .* e2y) ./ det2;                 % ((q-a) x (c-a)) / det
                lc = (e1y .* qz - e1z .* qy) ./ det2;                 % ((b-a) x (q-a)) / det
                la = 1 - lb - lc;
                hit = valid & la >= 0 & lb >= 0 & lc >= 0;
                xHit = la .* xa + lb .* xb + lc .* xc;
                crossings = sum(hit & (xHit > P(idx, 1)), 2);
                tf(idx) = mod(crossings, 2) == 1;
            end
        end

        function Rd = rayFrame()
            % RAYFRAME  Orthonormal rows [d; u; w] with d the generic ray direction
            %   normalize([1, sqrt(2)/97, sqrt(3)/89]) (irrational slopes, so no
            %   lattice edge is hit exactly by rays from rational grid points).
            d = [1, sqrt(2)/97, sqrt(3)/89];
            d = d / norm(d);
            u = cross(d, [0 0 1]);
            u = u / norm(u);
            w = cross(d, u);
            Rd = [d; u; w];
        end

        function meta = getMetadata()
            meta = struct();
            meta.SymbolicLatex = "o(c) = \Big[\tfrac{1}{s^3}\sum_{k=1}^{s^3} \mathbb{1}[\mathbf{q}_{c,k} \in \Omega] \ge \phi\Big], \quad \mathbf{q} \in \Omega \iff \#\{\text{ray crossings}\} \text{ odd}";
            meta.Symbols = [ ...
                struct('Symbol', "o(c)", 'Description', "occupancy of voxel c"), ...
                struct('Symbol', "\mathbf{q}_{c,k}", 'Description', "k-th sub-cell probe point of voxel c (s^3 per voxel)"), ...
                struct('Symbol', "\Omega", 'Description', "the solid (implicit inside-function or closed mesh interior)"), ...
                struct('Symbol', "\phi", 'Description', "MinFraction: required inside fraction")];
            meta.Assumptions = {'mesh input must be closed (watertight) for the parity rule to be meaningful', ...
                                'rays are cast along +x; faces parallel to x are ignored'};
            meta.IsSymbolic = true;
            meta.Reference = "Nooruddin & Turk (2003), Simplification and Repair of Polygonal Models Using Volumetric Techniques";
        end
    end
end
