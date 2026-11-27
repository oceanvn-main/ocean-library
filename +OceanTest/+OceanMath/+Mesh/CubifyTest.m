classdef CubifyTest < OceanTest.BaseTestCase
    % CUBIFYTEST  OceanMath.Geometry.Mesh.Cubify against closed-form counts,
    %   the divergence theorem (volume), Euler/genus topology, and an
    %   independent algorithm: ray-parity voxelisation must invert cubify.

    methods (Test)
        function testSingleVoxelIsACube(testCase)
            [V, F] = OceanMath.Geometry.Mesh.Cubify.compute(true, 0.5, [-1 1; -1 1; -1 1]);
            m = OceanMath.Geometry.Mesh.TriMesh(V, F);
            testCase.verifyEqual(m.numVertices(), 8);
            testCase.verifyEqual(m.numFaces(), 12);
            testCase.verifyTrue(m.isWatertight());
            testCase.verifyTrue(m.isConsistentlyOriented());
            testCase.verifyEqual(m.volume(), 8, 'AbsTol', 1e-12);   % outward orientation
        end

        function testTwoAdjacentVoxelsShareAFace(testCase)
            occ = true(1, 1, 2);                                     % two voxels along x
            [V, F] = OceanMath.Geometry.Mesh.Cubify.compute(occ, 0.5, [0 2; 0 1; 0 1]);
            m = OceanMath.Geometry.Mesh.TriMesh(V, F);
            testCase.verifyEqual(m.numVertices(), 12);               % 16 corners - 4 shared
            testCase.verifyEqual(m.numFaces(), 20);                  % 24 - 2 interior pairs
            testCase.verifyEqual(m.volume(), 2, 'AbsTol', 1e-12);
            testCase.verifyTrue(m.isWatertight());
        end

        function testSolidBlockCountsAndVolume(testCase)
            occ = false(7, 6, 5);
            occ(2:6, 2:5, 2:4) = true;                               % 5 (z) x 4 (y) x 3 (x) block
            cub = OceanMath.Geometry.Mesh.Cubify('Threshold', 0.5, 'Bounds', [0 5; 0 6; 0 7]);
            m = cub.forward(occ);                                    % unit voxels
            testCase.verifyEqual(m.numFaces(), 2 * 2 * (3*4 + 4*5 + 3*5));
            testCase.verifyEqual(m.eulerCharacteristic(), 2);
            testCase.verifyEqual(m.volume(), 60, 'AbsTol', 1e-9);
        end

        function testRingHasGenusOne(testCase)
            occ = false(1, 3, 3);
            occ(1, :, :) = true;
            occ(1, 2, 2) = false;                                    % 3x3 ring with a hole
            m = OceanMath.Geometry.Mesh.Cubify('Threshold', 0.5).forward(occ);
            testCase.verifyEqual(m.eulerCharacteristic(), 0);
            testCase.verifyEqual(m.genus(), 1);
        end

        function testAxisConvention(testCase)
            occ = false(4, 4, 4);
            occ(1, 2, 3) = true;                                     % d=1 (z), h=2 (y), w=3 (x)
            m = OceanMath.Geometry.Mesh.Cubify('Threshold', 0.5, 'Bounds', [0 4; 0 4; 0 4]).forward(occ);
            [lo, hi] = m.boundingBox();
            testCase.verifyEqual([lo; hi], [2 1 0; 3 2 1]);
        end

        function testThresholdIsStrict(testCase)
            probs = zeros(2, 2, 2);
            probs(1) = 0.3; probs(2) = 0.2;
            m = OceanMath.Geometry.Mesh.Cubify('Threshold', 0.2).forward(probs);
            testCase.verifyEqual(m.numVertices(), 8);                % only the 0.3 voxel
        end

        function testEmptyGridGivesEmptyMesh(testCase)
            m = OceanMath.Geometry.Mesh.Cubify().forward(zeros(3, 3, 3));
            testCase.verifyTrue(m.isEmpty());
            testCase.verifySize(m.Vertices, [0 3]);
        end

        function testVolumeEqualsVoxelCountForRandomBlobs(testCase)
            s = RandStream('mt19937ar', 'Seed', 11);
            for trial = 1:5
                occ = rand(s, 6, 7, 8) > 0.55;
                m = OceanMath.Geometry.Mesh.Cubify('Threshold', 0.5, 'Bounds', [0 8; 0 7; 0 6]).forward(occ);
                testCase.verifyEqual(m.volume(), nnz(occ), 'AbsTol', 1e-9);
                testCase.verifyTrue(m.isConsistentlyOriented());
            end
        end

        function testVoxelizeInvertsCubify(testCase)
            % Two independent algorithms: cubify (face exposure) and ray-parity
            % voxelisation must agree cell-for-cell at voxel centres.
            s = RandStream('mt19937ar', 'Seed', 5);
            occ = rand(s, 8, 9, 10) > 0.5;
            bounds = [-1 1; -1 1; -1 1];
            m = OceanMath.Geometry.Mesh.Cubify('Threshold', 0.5, 'Bounds', bounds).forward(occ);
            back = OceanMath.Geometry.Mesh.Voxelizer.fromMesh(m, size(occ), 'Bounds', bounds);
            testCase.verifyEqual(back, occ);
        end

        function testForwardBatchPacksMeshes(testCase)
            occ = false(3, 3, 3, 1, 2);
            occ(2, 2, 2, 1, 1) = true;
            occ(:, :, :, 1, 2) = true;
            b = OceanMath.Geometry.Mesh.Cubify('Threshold', 0.5).forwardBatch(occ);
            testCase.verifyEqual(b.numMeshes(), 2);
            testCase.verifyEqual(b.getMesh(1).numVertices(), 8);
            testCase.verifyEqual(b.getMesh(2).eulerCharacteristic(), 2);
        end
    end
end
