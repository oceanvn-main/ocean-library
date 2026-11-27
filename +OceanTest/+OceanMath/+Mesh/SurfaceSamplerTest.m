classdef SurfaceSamplerTest < OceanTest.BaseTestCase
    % SURFACESAMPLERTEST  Points lie on the surface, faces are chosen by area,
    %   the in-triangle density is uniform, and backward() is the exact
    %   derivative of the (fixed-draw) sampling map.

    methods (Test)
        function testSamplesLieOnCubeSurfaceWithMatchingNormals(testCase)
            m = OceanTest.OceanMath.Mesh.TriMeshTest.cube();
            [P, N] = OceanMath.Geometry.Mesh.SurfaceSampler(5000, 'Seed', 1).forward(m);
            testCase.verifyEqual(max(abs(P), [], 2), ones(5000, 1), 'AbsTol', 1e-12);
            % The normal's axis is the one where the point sits on the face.
            onFace = abs(sum(P .* N, 2) - 1) < 1e-12;
            testCase.verifyTrue(all(onFace));
        end

        function testFacesChosenProportionallyToArea(testCase)
            V = [0 0 0; 1 0 0; 0 2 0;  10 0 0; 13 0 0; 10 2 0];   % areas 1 and 3
            F = [1 2 3; 4 5 6];
            n = 40000;
            [~, ~, faceIdx] = OceanMath.Geometry.Mesh.SurfaceSampler.sample(V, F, n, ...
                RandStream('mt19937ar', 'Seed', 2));
            frac = mean(faceIdx == 1);
            sigma = sqrt(0.25 * 0.75 / n);
            testCase.verifyLessThan(abs(frac - 0.25), 5 * sigma);
        end

        function testUniformDensityInsideTriangle(testCase)
            % Uniform density => sample mean = centroid, and the fraction in
            % the sub-triangle near v1 (bary w1 > 1/2) is its area share 1/4.
            V = [0 0 0; 1 0 0; 0 1 0];
            n = 40000;
            [P, ~, ~, bary] = OceanMath.Geometry.Mesh.SurfaceSampler.sample(V, [1 2 3], n, ...
                RandStream('mt19937ar', 'Seed', 3));
            testCase.verifyEqual(mean(P, 1), [1/3 1/3 0], 'AbsTol', 0.01);
            testCase.verifyEqual(mean(bary(:,1) > 0.5), 0.25, 'AbsTol', 0.01);
        end

        function testBackwardMatchesFiniteDifferences(testCase)
            m = OceanMath.Geometry.Mesh.IcoSphere.create(1);
            sampler = OceanMath.Geometry.Mesh.SurfaceSampler(300, 'Seed', 4);
            [~, ~, faceIdx, bary] = OceanMath.Geometry.Mesh.SurfaceSampler.sample( ...
                m.Vertices, m.Faces, 300, RandStream('mt19937ar', 'Seed', 4));
            sampler.forward(m);
            W = randn(RandStream('mt19937ar', 'Seed', 9), 300, 3);    % L = sum(W .* P)
            dV = sampler.backward(W);

            Fs = m.Faces(faceIdx, :);
            pointsOf = @(V) bary(:,1).*V(Fs(:,1),:) + bary(:,2).*V(Fs(:,2),:) + bary(:,3).*V(Fs(:,3),:);
            loss = @(V) sum(W .* pointsOf(V), 'all');
            h = 1e-6;
            numeric = zeros(size(m.Vertices));
            for k = 1:numel(numeric)
                Vp = m.Vertices; Vp(k) = Vp(k) + h;
                Vm = m.Vertices; Vm(k) = Vm(k) - h;
                numeric(k) = (loss(Vp) - loss(Vm)) / (2 * h);
            end
            testCase.verifyEqual(dV, numeric, 'AbsTol', 1e-6);
        end

        function testSeedIsReproducible(testCase)
            m = OceanTest.OceanMath.Mesh.TriMeshTest.cube();
            P1 = OceanMath.Geometry.Mesh.SurfaceSampler(50, 'Seed', 8).forward(m);
            P2 = OceanMath.Geometry.Mesh.SurfaceSampler(50, 'Seed', 8).forward(m);
            testCase.verifyEqual(P1, P2);
        end

        function testBackwardBeforeForwardErrors(testCase)
            s = OceanMath.Geometry.Mesh.SurfaceSampler(10);
            testCase.verifyError(@() s.backward(zeros(10, 3)), ...
                'OceanMath:Geometry:Mesh:SurfaceSampler:NoForwardCache');
        end
    end
end
