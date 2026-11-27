classdef BoxShapesTest < OceanTest.BaseTestCase
    % BOXSHAPESTEST  BoxGeometry / BoxAssembly ray casting, surface sampling
    %   and inside tests, plus the procedural FurnitureShapeFactory checked
    %   end-to-end through Voxelizer -> Cubify -> TriMesh.genus.

    methods (Test)
        function testAxisAlignedRayHit(testCase)
            box = OceanMath.Vision.Geometry.BoxGeometry('HalfSize', [1; 0.5; 0.25]);
            [t, n] = box.intersect([0.3; 0.1; -5], [0; 0; 1]);
            testCase.verifyEqual(t, 5 - 0.25, 'AbsTol', 1e-12);
            testCase.verifyEqual(n, [0; 0; -1]);
            [t, n] = box.intersect([0; 3; -5], [0; 0; 1]);             % passes above
            testCase.verifyTrue(isnan(t));
            testCase.verifyEqual(n, [0; 0; 0]);
        end

        function testOriginInsideReturnsExit(testCase)
            box = OceanMath.Vision.Geometry.BoxGeometry('HalfSize', [1; 1; 1]);
            [t, n] = box.intersect([0; 0; 0.5], [0; 0; 1]);
            testCase.verifyEqual(t, 0.5, 'AbsTol', 1e-12);
            testCase.verifyEqual(n, [0; 0; 1]);
        end

        function testRotatedBoxHitsLieOnSurface(testCase)
            R = expm([0 -1.1 -0.7; 1.1 0 -0.4; 0.7 0.4 0]);                % rotation = exp(skew)
            box = OceanMath.Vision.Geometry.BoxGeometry('Center', [0.2; -0.1; 0.3], ...
                'HalfSize', [0.8; 0.4; 0.3], 'Rotation', R);
            s = RandStream('mt19937ar', 'Seed', 3);
            O = 0.3 * randn(s, 400, 3) + [0 0 -4];
            D = repmat([0 0 1], 400, 1);
            [t, N] = box.intersectRays(O, D);
            hit = ~isnan(t);
            testCase.verifyGreaterThan(nnz(hit), 50);
            X = O(hit, :) + t(hit) .* D(hit, :);
            testCase.verifyTrue(all(box.inside(X, 1e-9)));
            testCase.verifyFalse(any(box.inside(X, -1e-9)));
            testCase.verifyTrue(all(N(hit, 3) < 0));                  % front faces face the camera
            for k = find(hit).'
                Xk = (O(k, :) + t(k) * D(k, :)).';
                testCase.verifyEqual(N(k, :).', box.getNormal(Xk), 'AbsTol', 1e-9);
            end
        end

        function testBoxMeshAndSampling(testCase)
            box = OceanMath.Vision.Geometry.BoxGeometry('HalfSize', [1; 0.5; 0.25]);
            m = OceanMath.Geometry.Mesh.TriMesh.fromGeometry(box);
            testCase.verifyEqual(m.volume(), box.volume(), 'AbsTol', 1e-12);
            testCase.verifyEqual(m.surfaceArea(), box.surfaceArea(), 'AbsTol', 1e-12);
            n = 30000;
            [P, N, faceId] = box.sampleSurface(n, RandStream('mt19937ar', 'Seed', 2));
            testCase.verifyTrue(all(box.inside(P, 1e-12)) && ~any(box.inside(P, -1e-12)));
            halfAlongFace = [1 1 0.5 0.5 0.25 0.25];
            testCase.verifyEqual(dot(P, N, 2), halfAlongFace(faceId).', 'AbsTol', 1e-12);
            share = accumarray(ceil(faceId / 2), 1, [3 1]).' / n;  % x, y, z face pairs
            expected = [1 2 4] / 7;                                   % face areas 4 h_j h_k per pair
            testCase.verifyEqual(share, expected, 'AbsTol', 0.015);
        end

        function testUnionSurfaceSamplingOfOverlappingBoxes(testCase)
            % A = [-1,1]^3, B = A shifted by +1 in x  => union = [-1,2]x[-1,1]x[-1,1].
            A = OceanMath.Vision.Geometry.BoxGeometry('HalfSize', [1; 1; 1]);
            B = OceanMath.Vision.Geometry.BoxGeometry('Center', [1; 0; 0], 'HalfSize', [1; 1; 1]);
            asm = OceanMath.Vision.Geometry.BoxAssembly({A, B});
            n = 30000;
            [P, N] = asm.sampleSurface(n, RandStream('mt19937ar', 'Seed', 6));
            U = OceanMath.Vision.Geometry.BoxGeometry('Center', [0.5; 0; 0], 'HalfSize', [1.5; 1; 1]);
            testCase.verifyTrue(all(U.inside(P, 1e-9)) && ~any(U.inside(P, -1e-9)));  % on the union boundary
            % Face shares of the union box 3x2x2: x-faces 4+4, y-faces 6+6, z-faces 6+6.
            axisShare = [mean(abs(N(:,1)) == 1), mean(abs(N(:,2)) == 1), mean(abs(N(:,3)) == 1)];
            testCase.verifyEqual(axisShare, [8 12 12] / 32, 'AbsTol', 0.015);
        end

        function testExactUnionVolume(testCase)
            A = OceanMath.Vision.Geometry.BoxGeometry('HalfSize', [1; 1; 1]);
            B = OceanMath.Vision.Geometry.BoxGeometry('Center', [1; 0; 0], 'HalfSize', [1; 1; 1]);
            C = OceanMath.Vision.Geometry.BoxGeometry('Center', [0; 5; 0], 'HalfSize', [0.5; 0.5; 0.5]);
            asm = OceanMath.Vision.Geometry.BoxAssembly({A, B, C});
            testCase.verifyEqual(asm.volume(), 3*2*2 + 1, 'AbsTol', 1e-12);   % overlap counted once
            R = expm([0 -0.3 0.2; 0.3 0 -0.5; -0.2 0.5 0]);
            testCase.verifyEqual(asm.transformed(R, [1; 2; 3]).volume(), 13, 'AbsTol', 1e-9);
            % Cross-check against dense voxel integration of the same solid.
            % Midpoint-rule error bound: each surface patch of area dA shifts the
            % estimate by at most (h/2) dA, so |err| / vol <= (h/2) * area / vol.
            unit = asm.normalized();
            [~, frac] = OceanMath.Geometry.Mesh.Voxelizer.fromInsideFunction(@(P) unit.inside(P), ...
                [64 64 64], 'Supersample', 2);
            h = 2 / 128;
            areaUpper = sum(cellfun(@(b) b.surfaceArea(), unit.Parts));
            testCase.verifyEqual(sum(frac(:)) * (2/64)^3, unit.volume(), ...
                'RelTol', 0.5 * h * areaUpper / unit.volume());
        end

        function testAssemblyNearestHitAndNormalization(testCase)
            A = OceanMath.Vision.Geometry.BoxGeometry('Center', [0; 0; 2], 'HalfSize', [1; 1; 1]);
            B = OceanMath.Vision.Geometry.BoxGeometry('Center', [0; 0; -2], 'HalfSize', [1; 1; 1]);
            asm = OceanMath.Vision.Geometry.BoxAssembly({A, B});
            [t, N, part] = asm.intersectRays([0 0 -10], [0 0 1]);
            testCase.verifyEqual([t part], [7 2]);
            testCase.verifyEqual(N, [0 0 -1]);
            [lo, hi] = asm.normalized().bounds();
            testCase.verifyEqual(max((hi - lo) / 2), 1, 'AbsTol', 1e-12);
            testCase.verifyEqual((hi + lo) / 2, [0 0 0], 'AbsTol', 1e-12);
        end

        function testFactoryIsReproducibleAndNormalized(testCase)
            f1 = OceanData.Generators.Shapes3D.FurnitureShapeFactory('Seed', 21);
            f2 = OceanData.Generators.Shapes3D.FurnitureShapeFactory('Seed', 21);
            for k = 1:6
                [a1, i1] = f1.createRandom();
                [a2, i2] = f2.createRandom();
                testCase.verifyEqual(i1.Class, i2.Class);
                [lo1, hi1] = a1.bounds(); [lo2, hi2] = a2.bounds();
                testCase.verifyEqual([lo1 hi1], [lo2 hi2]);
                testCase.verifyEqual(max((hi1 - lo1) / 2), 1, 'AbsTol', 1e-12);
            end
        end

        function testChairBackStyleControlsTopology(testCase)
            % Factory -> exact solid -> voxels -> cubify -> genus: a frame back
            % encloses a slot (genus 1), a solid back does not (genus 0).
            fac = OceanData.Generators.Shapes3D.FurnitureShapeFactory('Seed', 0);
            seen = strings(0);
            for k = 1:40
                [asm, info] = fac.create("chair");
                if any(seen == info.Style)
                    continue;
                end
                occ = OceanMath.Geometry.Mesh.Voxelizer.fromInsideFunction(@(P) asm.inside(P), ...
                    [32 32 32], 'Supersample', 2, 'MinFraction', 1/8);
                m = OceanMath.Geometry.Mesh.Cubify('Threshold', 0.5).forward(occ);
                testCase.verifyTrue(m.isWatertight(), info.Style);
                testCase.verifyEqual(m.genus(), double(info.Style == "frame"), info.Style);
                seen(end+1) = info.Style; %#ok<AGROW>
                if numel(seen) == 2
                    break;
                end
            end
            testCase.verifyEqual(sort(seen), ["frame" "solid"]);
        end
    end
end
