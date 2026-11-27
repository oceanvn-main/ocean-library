classdef Cubify < handle
    % CUBIFY  Voxel occupancy grid -> watertight triangle mesh.
    %   The bridge between the voxel branch and the mesh-refinement branch of
    %   Mesh R-CNN (Gkioxari, Malik & Johnson, ICCV 2019, Sec. 3.1 "Cubify").
    %
    % ALGORITHM (the paper's three steps):
    %   1. Binarise:  voxel c is occupied iff p(c) > Threshold.
    %   2. Replace every occupied voxel by an axis-aligned cube
    %      (8 vertices, 6 square faces = 12 triangles).
    %   3. Delete every face shared by two occupied voxels (it is interior),
    %      and merge vertices that sit on the same lattice corner.
    %   The result is the boundary surface of the union of occupied cubes:
    %   closed and consistently oriented, with arbitrary topology (a voxelised
    %   chair back with a slot gives a genus-1 mesh -- the point of the paper's
    %   voxel branch, which a sphere-deforming method cannot produce).
    %
    %   Step 3 is vectorised per axis direction e in {+-x, +-y, +-z}:
    %     face (c, e) is kept  <=>  occ(c) AND NOT occ(c + e)
    %   evaluated against a copy of the grid padded with one layer of empty
    %   voxels, so faces on the grid boundary are always kept.
    %
    % GRID CONVENTION (same D x H x W volume layout as
    %   OceanDL.Pooling.MaxPool3D, i.e. a voxel head's output):
    %     occ(d, h, w):  w -> x (image column),  h -> y (image row),  d -> z (depth)
    %   The grid spans Bounds = [xmin xmax; ymin ymax; zmin zmax]; voxel
    %   (d,h,w) occupies x in xmin + [w-1, w] * (xmax - xmin)/W, likewise y, z.
    %
    %   Merged vertices are lattice corners (lx, ly, lz), lx in 0..W, etc.:
    %     vertex = [xmin + lx*sx, ymin + ly*sy, zmin + lz*sz],  sx = (xmax-xmin)/W ...
    %
    % CAVEAT (shared with the paper's implementation): two cubes that touch
    %   only along an edge or at a corner share those lattice vertices, which
    %   makes the mesh non-manifold there (an edge used by 4 faces). The mesh
    %   is still closed and consistently oriented.
    %
    % USAGE:
    %   cub  = OceanMath.Geometry.Mesh.Cubify('Threshold', 0.2);
    %   mesh = cub.forward(probs);                      % probs: D x H x W
    %   [V, F] = OceanMath.Geometry.Mesh.Cubify.compute(occ, 0.5, [-1 1; -1 1; -1 1]);
    %   batch = cub.forwardBatch(probs5D);              % D x H x W x 1 x N -> MeshBatch
    %
    % See also: OceanMath.Geometry.Mesh.TriMesh, OceanMath.Geometry.Mesh.Voxelizer

    properties
        Threshold (1,1) double = 0.2                 % paper value for Mesh R-CNN
        Bounds    (3,2) double = [-1 1; -1 1; -1 1]  % [xmin xmax; ymin ymax; zmin zmax]
    end

    methods
        function obj = Cubify(opts)
            arguments
                opts.Threshold (1,1) double = 0.2
                opts.Bounds    (3,2) double = [-1 1; -1 1; -1 1]
            end
            obj.Threshold = opts.Threshold;
            obj.Bounds = opts.Bounds;
        end

        function mesh = forward(obj, occ)
            % FORWARD  (D x H x W) occupancy / probability grid -> TriMesh.
            arguments
                obj
                occ (:,:,:) {mustBeNumericOrLogical}
            end
            [V, F] = OceanMath.Geometry.Mesh.Cubify.compute(occ, obj.Threshold, obj.Bounds);
            mesh = OceanMath.Geometry.Mesh.TriMesh(V, F);
        end

        function batch = forwardBatch(obj, occ)
            % FORWARDBATCH  (D x H x W x 1 x N) grids -> packed MeshBatch of N meshes.
            arguments
                obj
                occ {mustBeNumericOrLogical}
            end
            N = size(occ, 5);
            meshes = cell(N, 1);
            for n = 1:N
                meshes{n} = obj.forward(occ(:, :, :, 1, n));
            end
            batch = OceanMath.Geometry.Mesh.MeshBatch.fromMeshes(meshes);
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = "\mathcal{F} = \{(c,\mathbf{e}) : o(c) = 1 \wedge o(c+\mathbf{e}) = 0\}, \quad o(c) = [\,p(c) > \tau\,]";
            meta.Symbols = [ ...
                struct('Symbol', "p(c)", 'Description', "predicted occupancy probability of voxel c"), ...
                struct('Symbol', "\tau", 'Description', sprintf('binarisation threshold (Threshold = %g)', obj.Threshold)), ...
                struct('Symbol', "\mathbf{e}", 'Description', "one of the 6 axis directions; face (c,e) is the cube face of c facing e"), ...
                struct('Symbol', "\mathcal{F}", 'Description', "kept (exposed) cube faces, each split into 2 triangles")];
            meta.Assumptions = {'grid layout occ(d,h,w) with w->x, h->y, d->z', ...
                                'voxels outside the grid are empty (boundary faces are kept)', ...
                                'vertices on the same lattice corner are merged (edge-touching cubes give non-manifold edges)'};
            meta.IsSymbolic = true;
            meta.Reference = "Gkioxari, Malik & Johnson (2019), Mesh R-CNN, ICCV, Sec. 3.1";
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.Inputs  = struct('Name', "occ", 'Domain', "voxel-grid", 'Dtype', "double|logical", 'Shape', "D x H x W");
            spec.Outputs = struct('Name', "mesh", 'Domain', "triangle-mesh", 'Dtype', "TriMesh", 'Shape', "V x 3, F x 3");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', sprintf('cubify(%g)', obj.Threshold));
            spec.ComposesFrom = {};
        end
    end

    methods (Static)
        function [V, F] = compute(occ, threshold, bounds)
            % COMPUTE  Functional core: occupancy grid -> (V x 3, F x 3).
            arguments
                occ (:,:,:) {mustBeNumericOrLogical}
                threshold (1,1) double = 0.5
                bounds (3,2) double = [-1 1; -1 1; -1 1]
            end
            [D, H, W] = size(occ, 1:3);
            filled = occ > threshold;

            % Occupied cells as zero-based lattice coordinates of their lower
            % corner, in (x, y, z) order.
            [d, h, w] = ind2sub([D H W], find(filled));
            cellLo = [w, h, d] - 1;

            padded = false(D + 2, H + 2, W + 2);
            padded(2:end-1, 2:end-1, 2:end-1) = filled;

            quads = OceanMath.Geometry.Mesh.Cubify.faceQuads();
            triCorners = cell(numel(quads), 1);   % each (2n x 3 x 3): triangle, corner, xyz
            for q = 1:numel(quads)
                e = quads(q).Direction;           % [ex ey ez]
                neighbourOccupied = padded(sub2ind(size(padded), ...
                    d + 1 + e(3), h + 1 + e(2), w + 1 + e(1)));
                c = cellLo(~neighbourOccupied, :);
                n = size(c, 1);
                corners = reshape(c, n, 1, 3) + reshape(quads(q).Corners, 1, 4, 3);   % (n, 4, 3)
                % Square (1,2,3,4) -> triangles (1,2,3) and (1,3,4): same winding.
                triCorners{q} = [corners(:, [1 2 3], :); corners(:, [1 3 4], :)];
            end
            T = vertcat(triCorners{:});
            if isempty(T)
                V = zeros(0, 3);
                F = zeros(0, 3);
                return;
            end

            % Merge vertices: identical lattice corners get identical keys.
            key = T(:, :, 1) + (W + 1) * (T(:, :, 2) + (H + 1) * T(:, :, 3));   % (nTri x 3)
            [uniqueKey, ~, vertexIdx] = unique(key(:));
            F = reshape(vertexIdx, [], 3);

            lx = mod(uniqueKey, W + 1);
            rest = floor(uniqueKey / (W + 1));
            ly = mod(rest, H + 1);
            lz = floor(rest / (H + 1));
            step = (bounds(:, 2) - bounds(:, 1)).' ./ [W H D];
            V = bounds(:, 1).' + [lx, ly, lz] .* step;
        end

        function quads = faceQuads()
            % FACEQUADS  The 6 cube faces as unit squares on the lattice.
            %   For axis a and sign s the face lies at offset (s > 0) along a;
            %   the other two axes enumerate (0,0),(1,0),(1,1),(0,1). The corner
            %   order is then REVERSED if needed so that the right-hand normal
            %   (c2 - c1) x (c3 - c1) equals the outward direction e.
            unitSquare = [0 0; 1 0; 1 1; 0 1];
            quads = repmat(struct('Direction', zeros(1, 3), 'Corners', zeros(4, 3)), 6, 1);
            k = 0;
            for axisIdx = 1:3
                others = setdiff(1:3, axisIdx);
                for s = [1 -1]
                    C = zeros(4, 3);
                    C(:, axisIdx) = (s > 0);
                    C(:, others) = unitSquare;
                    e = zeros(1, 3);
                    e(axisIdx) = s;
                    if dot(cross(C(2,:) - C(1,:), C(3,:) - C(1,:)), e) < 0
                        C = C([1 4 3 2], :);
                    end
                    k = k + 1;
                    quads(k) = struct('Direction', e, 'Corners', C);
                end
            end
        end
    end
end
