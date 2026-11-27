classdef TriMesh < handle
    % TRIMESH  Indexed triangle mesh with the topology and differential-geometry
    %   queries used by mesh-prediction pipelines (Mesh R-CNN, Pixel2Mesh) and
    %   by mesh losses / metrics (Chamfer, normal consistency, edge length).
    %
    % CONVENTIONS:
    %   Vertices (V x 3) double -- one row per vertex [x y z] (the layout of
    %            MATLAB patch / triangulation, NOT IGeometry.toMesh's 3 x V;
    %            use TriMesh.fromGeometry to convert).
    %   Faces    (F x 3) 1-based vertex indices, ordered COUNTER-CLOCKWISE when
    %            seen from outside, so (v2 - v1) x (v3 - v1) points outward.
    %
    % FORMULAS (per face f with corners v1, v2, v3):
    %   c_f  = (v2 - v1) x (v3 - v1)              un-normalised normal, ||c_f|| = 2 A_f
    %   n_f  = c_f / ||c_f||                      unit outward normal
    %   A_f  = ||c_f|| / 2                        area
    %   Vol  = (1/6) sum_f v1 . (v2 x v3)         enclosed volume (divergence theorem,
    %                                             closed & consistently oriented only)
    %   chi  = V - E + F                          Euler characteristic
    %   g    = (2K - chi) / 2                     genus for a closed orientable surface
    %                                             with K connected components
    %   n_v  = normalise( sum_{f incident to v} c_f )   area-weighted vertex normal
    %
    % USAGE:
    %   m = OceanMath.Geometry.Mesh.TriMesh(V, F);
    %   m.isWatertight(), m.volume(), m.genus(), m.edges(), m.adjacency()
    %   m2 = m.transformed(T4x4);   m3 = m.withVertices(Vnew);
    %
    % See also: OceanMath.Geometry.Mesh.Cubify, OceanMath.Geometry.Mesh.MeshBatch,
    %           OceanMath.Geometry.Mesh.SurfaceSampler, OceanMath.Vision.Geometry.IGeometry

    properties (SetAccess = private)
        Vertices (:,3) double = zeros(0, 3)
        Faces    (:,3) double = zeros(0, 3)
    end

    methods
        function obj = TriMesh(vertices, faces)
            arguments
                vertices (:,3) double = zeros(0, 3)
                faces    (:,3) double {mustBeInteger, mustBePositive} = zeros(0, 3)
            end
            if ~isempty(faces) && max(faces, [], 'all') > size(vertices, 1)
                error('OceanMath:Geometry:Mesh:TriMesh:IndexOutOfRange', ...
                    'Faces reference vertex %d but only %d vertices exist.', ...
                    max(faces, [], 'all'), size(vertices, 1));
            end
            obj.Vertices = vertices;
            obj.Faces = faces;
        end

        % ---------------------------------------------------------------- %
        %  Sizes
        % ---------------------------------------------------------------- %
        function n = numVertices(obj)
            n = size(obj.Vertices, 1);
        end

        function n = numFaces(obj)
            n = size(obj.Faces, 1);
        end

        function tf = isEmpty(obj)
            tf = obj.numFaces() == 0;
        end

        % ---------------------------------------------------------------- %
        %  Topology
        % ---------------------------------------------------------------- %
        function [E, faceCount] = edges(obj)
            % EDGES  Unique undirected edges (E x 2, sorted so E(:,1) < E(:,2))
            %   and, per edge, the number of faces that contain it.
            F = obj.Faces;
            halfEdges = [F(:, [1 2]); F(:, [2 3]); F(:, [3 1])];
            [E, ~, ic] = unique(sort(halfEdges, 2), 'rows');
            faceCount = accumarray(ic, 1, [size(E, 1) 1]);
        end

        function A = adjacency(obj)
            % ADJACENCY  Sparse symmetric (V x V) vertex adjacency, A(i,j)=1 iff
            %   (i,j) is a mesh edge. This is the neighbourhood N(i) used by
            %   graph convolution: sum_{j in N(i)} f_j = (A * f)(i,:).
            E = obj.edges();
            nV = obj.numVertices();
            A = sparse([E(:,1); E(:,2)], [E(:,2); E(:,1)], 1, nV, nV);
        end

        function tf = isWatertight(obj)
            % ISWATERTIGHT  Every edge is shared by exactly two faces (closed,
            %   edge-manifold surface; no boundary, no fins).
            if obj.isEmpty()
                tf = false;
                return;
            end
            [~, faceCount] = obj.edges();
            tf = all(faceCount == 2);
        end

        function tf = isConsistentlyOriented(obj)
            % ISCONSISTENTLYORIENTED  Neighbouring faces traverse every shared
            %   edge in OPPOSITE directions:  |#(a->b) - #(b->a)| <= 1  per edge.
            %     manifold interior edge   1 vs 1   (coherent)   vs 2 vs 0 (flipped face)
            %     boundary edge            1 vs 0
            %     non-manifold edge        2 vs 2   (e.g. Cubify cubes touching along an edge)
            %   For a manifold mesh this is the usual "each directed half-edge
            %   occurs at most once"; the count form also covers non-manifold edges.
            F = obj.Faces;
            halfEdges = [F(:, [1 2]); F(:, [2 3]); F(:, [3 1])];
            [~, ~, ic] = unique(sort(halfEdges, 2), 'rows');
            forward = halfEdges(:, 1) < halfEdges(:, 2);            % a->b with a < b
            nForward = accumarray(ic, double(forward));
            nBackward = accumarray(ic, double(~forward));
            tf = all(abs(nForward - nBackward) <= 1);
        end

        function chi = eulerCharacteristic(obj)
            chi = obj.numVertices() - size(obj.edges(), 1) + obj.numFaces();
        end

        function k = numComponents(obj)
            % NUMCOMPONENTS  Connected components of the vertex graph (vertices
            %   not referenced by any face are ignored).
            used = unique(obj.Faces(:));
            if isempty(used)
                k = 0;
                return;
            end
            A = obj.adjacency();
            bins = conncomp(graph(A(used, used)));
            k = max(bins);
        end

        function g = genus(obj)
            % GENUS  Number of handles ("through-holes") of a closed orientable
            %   mesh: chi = 2K - 2g  =>  g = (2K - chi)/2.
            g = (2 * obj.numComponents() - obj.eulerCharacteristic()) / 2;
        end

        % ---------------------------------------------------------------- %
        %  Differential geometry
        % ---------------------------------------------------------------- %
        function C = faceCrossProducts(obj)
            % FACECROSSPRODUCTS  c_f = (v2 - v1) x (v3 - v1), (F x 3).
            V = obj.Vertices; F = obj.Faces;
            C = cross(V(F(:,2),:) - V(F(:,1),:), V(F(:,3),:) - V(F(:,1),:), 2);
        end

        function N = faceNormals(obj)
            C = obj.faceCrossProducts();
            N = C ./ max(vecnorm(C, 2, 2), eps);
        end

        function A = faceAreas(obj)
            A = 0.5 * vecnorm(obj.faceCrossProducts(), 2, 2);
        end

        function N = vertexNormals(obj)
            % VERTEXNORMALS  Area-weighted average of incident face normals.
            C = obj.faceCrossProducts();          % ||c_f|| = 2 A_f: area weighting for free
            nV = obj.numVertices();
            N = zeros(nV, 3);
            for corner = 1:3
                for dim = 1:3
                    N(:, dim) = N(:, dim) + accumarray(obj.Faces(:, corner), C(:, dim), [nV 1]);
                end
            end
            N = N ./ max(vecnorm(N, 2, 2), eps);
        end

        function s = surfaceArea(obj)
            s = sum(obj.faceAreas());
        end

        function vol = volume(obj)
            % VOLUME  Signed enclosed volume, (1/6) sum_f v1 . (v2 x v3).
            %   Positive for an outward-oriented closed mesh.
            V = obj.Vertices; F = obj.Faces;
            vol = sum(dot(V(F(:,1),:), cross(V(F(:,2),:), V(F(:,3),:), 2), 2)) / 6;
        end

        function [lo, hi] = boundingBox(obj)
            lo = min(obj.Vertices, [], 1);
            hi = max(obj.Vertices, [], 1);
        end

        function L = edgeLengths(obj)
            E = obj.edges();
            L = vecnorm(obj.Vertices(E(:,1),:) - obj.Vertices(E(:,2),:), 2, 2);
        end

        % ---------------------------------------------------------------- %
        %  Derived meshes (a TriMesh is never mutated in place)
        % ---------------------------------------------------------------- %
        function m = withVertices(obj, vertices)
            % WITHVERTICES  Same connectivity, new vertex positions -- exactly
            %   what a mesh-refinement stage produces (v' = v + delta v).
            arguments
                obj
                vertices (:,3) double
            end
            if size(vertices, 1) ~= obj.numVertices()
                error('OceanMath:Geometry:Mesh:TriMesh:VertexCountMismatch', ...
                    'Expected %d vertices, got %d.', obj.numVertices(), size(vertices, 1));
            end
            m = OceanMath.Geometry.Mesh.TriMesh(vertices, obj.Faces);
        end

        function m = transformed(obj, T)
            % TRANSFORMED  Apply a 4x4 homogeneous transform to every vertex.
            %   A transform with negative determinant (a reflection) flips the
            %   face winding so that normals stay outward.
            arguments
                obj
                T (4,4) double
            end
            Vh = [obj.Vertices, ones(obj.numVertices(), 1)] * T.';
            V = Vh(:, 1:3) ./ Vh(:, 4);
            F = obj.Faces;
            if det(T(1:3, 1:3)) < 0
                F = F(:, [1 3 2]);
            end
            m = OceanMath.Geometry.Mesh.TriMesh(V, F);
        end

        function m = flipped(obj)
            % FLIPPED  Reverse every face's winding (normals point inward).
            m = OceanMath.Geometry.Mesh.TriMesh(obj.Vertices, obj.Faces(:, [1 3 2]));
        end

        function m = compacted(obj)
            % COMPACTED  Drop vertices not referenced by any face and re-index.
            [used, ~, ic] = unique(obj.Faces(:));
            m = OceanMath.Geometry.Mesh.TriMesh(obj.Vertices(used, :), reshape(ic, [], 3));
        end

        % ---------------------------------------------------------------- %
        %  Introspection (3-pillar contract)
        % ---------------------------------------------------------------- %
        function s = describe(obj)
            s = sprintf(['TriMesh: V=%d  F=%d  E=%d  chi=%d  watertight=%d  ', ...
                         'area=%.4g  volume=%.4g'], ...
                obj.numVertices(), obj.numFaces(), size(obj.edges(), 1), ...
                obj.eulerCharacteristic(), obj.isWatertight(), obj.surfaceArea(), obj.volume());
            if nargout == 0
                disp(s);
                clear s;
            end
        end

        function meta = getMetadata(~)
            meta = struct();
            meta.SymbolicLatex = "\chi = V - E + F, \quad \mathrm{Vol} = \tfrac{1}{6}\sum_f \mathbf{v}_1 \cdot (\mathbf{v}_2 \times \mathbf{v}_3)";
            meta.Symbols = [ ...
                struct('Symbol', "\chi", 'Description', "Euler characteristic; 2 - 2g for a closed genus-g surface"), ...
                struct('Symbol', "V, E, F", 'Description', "number of vertices, unique undirected edges, triangular faces"), ...
                struct('Symbol', "\mathbf{v}_1,\mathbf{v}_2,\mathbf{v}_3", 'Description', "corners of face f, counter-clockwise seen from outside"), ...
                struct('Symbol', "\mathrm{Vol}", 'Description', "signed enclosed volume (divergence theorem), > 0 for outward orientation")];
            meta.Assumptions = {'faces are counter-clockwise seen from outside (outward normals)', ...
                                'volume/genus are only meaningful for closed (watertight) meshes'};
            meta.IsSymbolic = true;
            meta.Reference = "Botsch et al. (2010), Polygon Mesh Processing, Ch. 1-2";
        end
    end

    methods (Static)
        function m = fromGeometry(geometry)
            % FROMGEOMETRY  Convert any IGeometry.toMesh() output (3 x V, 3 x F)
            %   into the (V x 3, F x 3) TriMesh layout.
            arguments
                geometry OceanMath.Vision.Geometry.IGeometry
            end
            [V, F] = geometry.toMesh();
            m = OceanMath.Geometry.Mesh.TriMesh(V.', F.');
        end

        function m = concatenate(meshes)
            % CONCATENATE  Disjoint union of several meshes (vertices are NOT
            %   merged; face indices are offset).
            arguments
                meshes cell
            end
            nV = cellfun(@(x) x.numVertices(), meshes);
            offsets = [0, cumsum(nV(1:end-1))];
            Vs = cellfun(@(x) x.Vertices, meshes, 'UniformOutput', false);
            Fs = cellfun(@(x, o) x.Faces + o, meshes, num2cell(offsets), 'UniformOutput', false);
            m = OceanMath.Geometry.Mesh.TriMesh(vertcat(Vs{:}), vertcat(Fs{:}));
        end
    end
end
