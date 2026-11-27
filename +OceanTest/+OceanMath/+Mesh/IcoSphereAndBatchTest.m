classdef IcoSphereAndBatchTest < OceanTest.BaseTestCase
    % ICOSPHEREANDBATCHTEST  IcoSphere closed-form counts / convergence, and
    %   MeshBatch packing invariants (offsets, no cross-mesh edges, round trip).

    methods (Test)
        function testIcoSphereCounts(testCase)
            for L = 0:3
                m = OceanMath.Geometry.Mesh.IcoSphere.create(L);
                testCase.verifyEqual(m.numVertices(), 10 * 4^L + 2);
                testCase.verifyEqual(m.numFaces(), 20 * 4^L);
                testCase.verifyEqual(size(m.edges(), 1), 30 * 4^L);
                testCase.verifyTrue(m.isWatertight());
                testCase.verifyTrue(m.isConsistentlyOriented());
                testCase.verifyGreaterThan(m.volume(), 0);
            end
        end

        function testIcoSphereOnSphereAndVolumeConverges(testCase)
            r = 0.6;
            m = OceanMath.Geometry.Mesh.IcoSphere.create(4, 'Radius', r, 'Center', [1 2 3]);
            testCase.verifyEqual(vecnorm(m.Vertices - [1 2 3], 2, 2), r * ones(m.numVertices(), 1), 'AbsTol', 1e-12);
            testCase.verifyEqual(m.volume(), 4/3 * pi * r^3, 'RelTol', 0.01);
        end

        function testBatchRoundTripAndOffsets(testCase)
            a = OceanTest.OceanMath.Mesh.TriMeshTest.cube();
            b = OceanMath.Geometry.Mesh.IcoSphere.create(1);
            batch = OceanMath.Geometry.Mesh.MeshBatch.fromMeshes({a, b, a});
            testCase.verifyEqual(batch.numMeshes(), 3);
            testCase.verifyEqual(batch.numVerticesPerMesh(), [8; 42; 8]);
            testCase.verifyEqual(batch.getMesh(2).Vertices, b.Vertices);
            testCase.verifyEqual(batch.getMesh(2).Faces, b.Faces);
            testCase.verifyEqual(batch.getMesh(3).volume(), 8, 'AbsTol', 1e-12);
        end

        function testNoEdgeCrossesMeshes(testCase)
            batch = OceanMath.Geometry.Mesh.MeshBatch.fromMeshes( ...
                {OceanTest.OceanMath.Mesh.TriMeshTest.cube(), OceanMath.Geometry.Mesh.IcoSphere.create(1)});
            idx = batch.VertMeshIdx;
            testCase.verifyEqual(idx(batch.Edges(:,1)), idx(batch.Edges(:,2)));
            testCase.verifyEqual(size(batch.Edges, 1), 18 + 120);
        end

        function testWithVerticesKeepsTopology(testCase)
            batch = OceanMath.Geometry.Mesh.MeshBatch.fromMeshes({OceanTest.OceanMath.Mesh.TriMeshTest.cube()});
            moved = batch.withVertices(2 * batch.Vertices);
            testCase.verifyEqual(moved.Faces, batch.Faces);
            testCase.verifyEqual(moved.getMesh(1).volume(), 64, 'AbsTol', 1e-12);
            testCase.verifyError(@() batch.withVertices(zeros(3, 3)), ...
                'OceanMath:Geometry:Mesh:MeshBatch:VertexCountMismatch');
        end
    end
end
