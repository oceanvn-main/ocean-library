classdef MeshBatch < handle
    % MESHBATCH  A batch of N triangle meshes in PACKED layout: all vertices
    %   stacked in one (sum V_n x 3) matrix, all faces/edges re-indexed into it.
    %
    %   Meshes in a batch have different vertex counts (every cubified shape
    %   is different), so they cannot be stored as one (N x V x 3) array.
    %   The packed layout (PyTorch3D's "packed" representation) turns
    %   per-mesh operations into single vectorised ones:
    %     - graph convolution:  sum_{j in N(i)} f_j over the packed edge list
    %       never mixes meshes, because no edge connects two meshes;
    %     - VertAlign: vertex i samples the feature map of RoI MeshIdx(i);
    %     - losses: per-mesh reductions via accumarray over MeshIdx.
    %
    % FIELDS:
    %   Vertices     (Vt x 3)  packed vertices
    %   Faces        (Ft x 3)  packed faces, indices into Vertices
    %   Edges        (Et x 2)  packed unique undirected edges, indices into Vertices
    %   VertMeshIdx  (Vt x 1)  mesh id (1..N) of every vertex
    %   FaceMeshIdx  (Ft x 1)  mesh id of every face
    %   VertOffsets  (N+1 x 1) mesh n owns vertices VertOffsets(n)+1 : VertOffsets(n+1)
    %   FaceOffsets  (N+1 x 1) likewise for faces
    %
    % USAGE:
    %   b  = OceanMath.Geometry.Mesh.MeshBatch.fromMeshes({m1, m2, m3});
    %   m2 = b.getMesh(2);
    %   b2 = b.withVertices(b.Vertices + delta);      % same topology, new positions
    %
    % See also: OceanMath.Geometry.Mesh.TriMesh, OceanMath.Geometry.Mesh.Cubify

    properties (SetAccess = private)
        Vertices    (:,3) double = zeros(0, 3)
        Faces       (:,3) double = zeros(0, 3)
        Edges       (:,2) double = zeros(0, 2)
        VertMeshIdx (:,1) double = zeros(0, 1)
        FaceMeshIdx (:,1) double = zeros(0, 1)
        VertOffsets (:,1) double = 0
        FaceOffsets (:,1) double = 0
    end

    methods
        function n = numMeshes(obj)
            n = numel(obj.VertOffsets) - 1;
        end

        function n = numVerticesPerMesh(obj)
            n = diff(obj.VertOffsets);
        end

        function n = numFacesPerMesh(obj)
            n = diff(obj.FaceOffsets);
        end

        function mesh = getMesh(obj, n)
            % GETMESH  Unpack mesh n as a stand-alone TriMesh (local indices).
            arguments
                obj
                n (1,1) double {mustBeInteger, mustBePositive}
            end
            vRange = obj.VertOffsets(n)+1 : obj.VertOffsets(n+1);
            fRange = obj.FaceOffsets(n)+1 : obj.FaceOffsets(n+1);
            mesh = OceanMath.Geometry.Mesh.TriMesh(obj.Vertices(vRange, :), ...
                obj.Faces(fRange, :) - obj.VertOffsets(n));
        end

        function V = getVertices(obj, n)
            V = obj.Vertices(obj.VertOffsets(n)+1 : obj.VertOffsets(n+1), :);
        end

        function b = withVertices(obj, vertices)
            % WITHVERTICES  New batch, same topology (faces/edges/offsets reused).
            arguments
                obj
                vertices (:,3) double
            end
            if size(vertices, 1) ~= size(obj.Vertices, 1)
                error('OceanMath:Geometry:Mesh:MeshBatch:VertexCountMismatch', ...
                    'Expected %d packed vertices, got %d.', size(obj.Vertices, 1), size(vertices, 1));
            end
            b = OceanMath.Geometry.Mesh.MeshBatch();
            b.assign(vertices, obj.Faces, obj.Edges, obj.VertMeshIdx, obj.FaceMeshIdx, ...
                obj.VertOffsets, obj.FaceOffsets);
        end

        function A = adjacency(obj)
            % ADJACENCY  Sparse (Vt x Vt) block-diagonal vertex adjacency.
            nV = size(obj.Vertices, 1);
            A = sparse([obj.Edges(:,1); obj.Edges(:,2)], [obj.Edges(:,2); obj.Edges(:,1)], 1, nV, nV);
        end

        function s = describe(obj)
            s = sprintf('MeshBatch: N=%d meshes, %d vertices, %d faces, %d edges', ...
                obj.numMeshes(), size(obj.Vertices, 1), size(obj.Faces, 1), size(obj.Edges, 1));
            if nargout == 0
                disp(s);
                clear s;
            end
        end
    end

    methods (Static)
        function b = fromMeshes(meshes)
            % FROMMESHES  Pack a cell array of TriMesh objects.
            arguments
                meshes cell
            end
            N = numel(meshes);
            nV = zeros(N, 1); nF = zeros(N, 1);
            for n = 1:N
                nV(n) = meshes{n}.numVertices();
                nF(n) = meshes{n}.numFaces();
            end
            vOff = [0; cumsum(nV)];
            fOff = [0; cumsum(nF)];
            V = zeros(vOff(end), 3);
            F = zeros(fOff(end), 3);
            edgeCells = cell(N, 1);
            for n = 1:N
                V(vOff(n)+1 : vOff(n+1), :) = meshes{n}.Vertices;
                F(fOff(n)+1 : fOff(n+1), :) = meshes{n}.Faces + vOff(n);
                edgeCells{n} = meshes{n}.edges() + vOff(n);
            end
            E = vertcat(edgeCells{:});
            if isempty(E)
                E = zeros(0, 2);
            end
            vertMeshIdx = repelem((1:N).', nV);
            faceMeshIdx = repelem((1:N).', nF);
            b = OceanMath.Geometry.Mesh.MeshBatch();
            b.assign(V, F, E, vertMeshIdx, faceMeshIdx, vOff, fOff);
        end
    end

    methods (Access = private)
        function assign(obj, V, F, E, vIdx, fIdx, vOff, fOff)
            obj.Vertices = V;
            obj.Faces = F;
            obj.Edges = E;
            obj.VertMeshIdx = vIdx;
            obj.FaceMeshIdx = fIdx;
            obj.VertOffsets = vOff;
            obj.FaceOffsets = fOff;
        end
    end
end
