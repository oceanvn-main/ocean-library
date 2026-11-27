classdef VoxelizerTest < OceanTest.BaseTestCase
    % VOXELIZERTEST  Occupancy against analytic volumes, the ray-parity test
    %   against the implicit definition of a solid, and the MinFraction rule.

    methods (Test)
        function testSphereVolumeConverges(testCase)
            r = 0.7;
            G = 32;
            occ = OceanMath.Geometry.Mesh.Voxelizer.fromInsideFunction( ...
                @(P) vecnorm(P, 2, 2) < r, [G G G]);
            voxelVolume = (2 / G)^3;
            testCase.verifyEqual(nnz(occ) * voxelVolume, 4/3 * pi * r^3, 'RelTol', 0.02);
        end

        function testInsideMeshOnCube(testCase)
            m = OceanTest.OceanMath.Mesh.TriMeshTest.cube();
            s = RandStream('mt19937ar', 'Seed', 1);
            Pin = 1.9 * rand(s, 500, 3) - 0.95;                       % strictly inside
            Pout = (1.1 + rand(s, 500, 3)) .* sign(randn(s, 500, 3)); % |coord| > 1.1
            testCase.verifyTrue(all(OceanMath.Geometry.Mesh.Voxelizer.insideMesh(m.Vertices, m.Faces, Pin)));
            testCase.verifyFalse(any(OceanMath.Geometry.Mesh.Voxelizer.insideMesh(m.Vertices, m.Faces, Pout)));
        end

        function testMeshAndImplicitVoxelisationAgree(testCase)
            r = 0.8;
            G = 16;
            sphere = OceanMath.Geometry.Mesh.IcoSphere.create(4, 'Radius', r);
            byMesh = OceanMath.Geometry.Mesh.Voxelizer.fromMesh(sphere, [G G G]);
            byFn = OceanMath.Geometry.Mesh.Voxelizer.fromInsideFunction(@(P) vecnorm(P, 2, 2) < r, [G G G]);
            testCase.verifyGreaterThan(mean(byMesh(:) == byFn(:)), 0.99);
        end

        function testMinFractionKeepsThinParts(testCase)
            slab = @(P) abs(P(:, 2)) < 0.05;                           % 0.1 thick, voxel 0.25
            majority = OceanMath.Geometry.Mesh.Voxelizer.fromInsideFunction(slab, [8 8 8], ...
                'Supersample', 4, 'MinFraction', 0.5);
            anyRule = OceanMath.Geometry.Mesh.Voxelizer.fromInsideFunction(slab, [8 8 8], ...
                'Supersample', 4, 'MinFraction', 1/64);
            testCase.verifyEqual(nnz(majority), 0);
            testCase.verifyEqual(nnz(anyRule), 8 * 8 * 2);           % the two rows straddling y = 0
        end

        function testFractionIsInsideShare(testCase)
            halfSpace = @(P) P(:, 1) < 0;                             % x < 0
            [~, frac] = OceanMath.Geometry.Mesh.Voxelizer.fromInsideFunction(halfSpace, [2 2 2], ...
                'Supersample', 4);
            testCase.verifyEqual(squeeze(frac(1, 1, :)).', [1 0]);    % w = 1 is x < 0
        end
    end
end
