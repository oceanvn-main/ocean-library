classdef TriMeshTest < OceanTest.BaseTestCase
    % TRIMESHTEST  Topology / geometry queries of OceanMath.Geometry.Mesh.TriMesh
    %   checked against analytic facts of known solids (cube, icosphere).

    methods (Test)
        function testUnitCubeTopology(testCase)
            m = OceanTest.OceanMath.Mesh.TriMeshTest.cube();
            testCase.verifyEqual(m.numVertices(), 8);
            testCase.verifyEqual(m.numFaces(), 12);
            testCase.verifyEqual(size(m.edges(), 1), 18);
            testCase.verifyEqual(m.eulerCharacteristic(), 2);
            testCase.verifyEqual(m.genus(), 0);
            testCase.verifyTrue(m.isWatertight());
            testCase.verifyTrue(m.isConsistentlyOriented());
        end

        function testUnitCubeMeasures(testCase)
            m = OceanTest.OceanMath.Mesh.TriMeshTest.cube();                           % [-1,1]^3
            testCase.verifyEqual(m.volume(), 8, 'AbsTol', 1e-12);
            testCase.verifyEqual(m.surfaceArea(), 24, 'AbsTol', 1e-12);
            [lo, hi] = m.boundingBox();
            testCase.verifyEqual([lo hi], [-1 -1 -1 1 1 1]);
        end

        function testFaceNormalsPointOutward(testCase)
            m = OceanTest.OceanMath.Mesh.TriMeshTest.cube();
            V = m.Vertices; F = m.Faces;
            centroid = (V(F(:,1),:) + V(F(:,2),:) + V(F(:,3),:)) / 3;
            testCase.verifyTrue(all(dot(m.faceNormals(), centroid, 2) > 0));
        end

        function testFlipNegatesVolume(testCase)
            m = OceanTest.OceanMath.Mesh.TriMeshTest.cube();
            testCase.verifyEqual(m.flipped().volume(), -8, 'AbsTol', 1e-12);
        end

        function testTransformScalesVolumeByDeterminant(testCase)
            m = OceanTest.OceanMath.Mesh.TriMeshTest.cube();
            T = diag([2 3 4 1]);
            testCase.verifyEqual(m.transformed(T).volume(), 8 * 24, 'AbsTol', 1e-9);
            % A reflection must keep the mesh outward (winding flipped).
            R = diag([-1 1 1 1]);
            testCase.verifyEqual(m.transformed(R).volume(), 8, 'AbsTol', 1e-12);
        end

        function testAdjacencyIsSymmetricWithDegreeSum2E(testCase)
            m = OceanTest.OceanMath.Mesh.TriMeshTest.cube();
            A = m.adjacency();
            testCase.verifyEqual(nnz(A - A.'), 0);
            testCase.verifyEqual(full(sum(A, 'all')), 2 * size(m.edges(), 1));
        end

        function testIcosphereVertexNormalsAreRadial(testCase)
            m = OceanMath.Geometry.Mesh.IcoSphere.create(2);
            radial = m.Vertices ./ vecnorm(m.Vertices, 2, 2);
            testCase.verifyGreaterThan(min(dot(m.vertexNormals(), radial, 2)), 0.99);
        end

        function testCompactedDropsUnusedVertices(testCase)
            m = OceanTest.OceanMath.Mesh.TriMeshTest.cube();
            V = [m.Vertices; 5 5 5];
            m2 = OceanMath.Geometry.Mesh.TriMesh(V, m.Faces).compacted();
            testCase.verifyEqual(m2.numVertices(), 8);
            testCase.verifyEqual(m2.volume(), 8, 'AbsTol', 1e-12);
        end

        function testOpenMeshIsNotWatertight(testCase)
            m = OceanTest.OceanMath.Mesh.TriMeshTest.cube();
            open = OceanMath.Geometry.Mesh.TriMesh(m.Vertices, m.Faces(1:end-1, :));
            testCase.verifyFalse(open.isWatertight());
        end

        function testSingleFlippedFaceBreaksOrientation(testCase)
            m = OceanTest.OceanMath.Mesh.TriMeshTest.cube();
            F = m.Faces;
            F(5, :) = F(5, [1 3 2]);
            testCase.verifyFalse(OceanMath.Geometry.Mesh.TriMesh(m.Vertices, F).isConsistentlyOriented());
        end

        function testIndexOutOfRangeErrors(testCase)
            testCase.verifyError(@() OceanMath.Geometry.Mesh.TriMesh(zeros(3,3), [1 2 4]), ...
                'OceanMath:Geometry:Mesh:TriMesh:IndexOutOfRange');
        end

        function testWithVerticesRequiresSameCount(testCase)
            m = OceanTest.OceanMath.Mesh.TriMeshTest.cube();
            testCase.verifyError(@() m.withVertices(zeros(7, 3)), ...
                'OceanMath:Geometry:Mesh:TriMesh:VertexCountMismatch');
        end

        function testTwoDisjointCubesHaveTwoComponents(testCase)
            a = OceanTest.OceanMath.Mesh.TriMeshTest.cube();
            b = a.transformed([eye(3) [5;0;0]; 0 0 0 1]);
            m = OceanMath.Geometry.Mesh.TriMesh.concatenate({a, b});
            testCase.verifyEqual(m.numComponents(), 2);
            testCase.verifyEqual(m.eulerCharacteristic(), 4);
            testCase.verifyEqual(m.genus(), 0);
            testCase.verifyEqual(m.volume(), 16, 'AbsTol', 1e-12);
        end
    end

    methods (Static)
        function m = cube()
            [V, F] = OceanMath.Vision.Geometry.BoxGeometry.unitCubeMesh();
            m = OceanMath.Geometry.Mesh.TriMesh(V, F);
        end
    end
end
