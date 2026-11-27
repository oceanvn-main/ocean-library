classdef IcoSphere
    % ICOSPHERE  Geodesic sphere by recursive subdivision of an icosahedron.
    %
    %   The fixed-topology (genus-0) template that sphere-deforming methods
    %   (Pixel2Mesh, Wang et al. 2018) refine -- and the baseline Mesh R-CNN
    %   compares against: a sphere can never grow the hole in a chair back,
    %   while Cubify's voxel-derived mesh can.
    %
    %   Preferred over the latitude/longitude mesh of SphereGeometry.toMesh
    %   because every vertex has (almost) the same valence and the faces are
    %   nearly equilateral -- no pole singularities -- which matters for
    %   graph convolution (uniform neighbourhoods) and edge-length losses.
    %
    % CONSTRUCTION:
    %   Level 0 = regular icosahedron: 12 vertices (0, +-1, +-phi) and cyclic
    %   permutations, phi = (1 + sqrt(5))/2, 20 faces.
    %   Each subdivision splits every triangle (a,b,c) into 4 using the edge
    %   midpoints m_ab, m_bc, m_ca (shared between neighbouring faces), then
    %   projects the new vertices back onto the sphere:
    %     (a, m_ab, m_ca), (m_ab, b, m_bc), (m_ca, m_bc, c), (m_ab, m_bc, m_ca)
    %   Counts at level L:  V = 10 * 4^L + 2,  F = 20 * 4^L,  E = 30 * 4^L.
    %
    % USAGE:
    %   mesh = OceanMath.Geometry.Mesh.IcoSphere.create(2, 'Radius', 0.5);   % 162 V, 320 F
    %
    % See also: OceanMath.Geometry.Mesh.TriMesh, OceanMath.Vision.Geometry.SphereGeometry

    methods (Static)
        function mesh = create(level, opts)
            arguments
                level (1,1) double {mustBeInteger, mustBeNonnegative} = 2
                opts.Radius (1,1) double {mustBePositive} = 1
                opts.Center (1,3) double = [0 0 0]
            end
            [V, F] = OceanMath.Geometry.Mesh.IcoSphere.icosahedron();
            for k = 1:level
                [V, F] = OceanMath.Geometry.Mesh.IcoSphere.subdivide(V, F);
            end
            mesh = OceanMath.Geometry.Mesh.TriMesh(opts.Center + opts.Radius * V, F);
        end

        function [V, F] = icosahedron()
            % ICOSAHEDRON  Unit-radius icosahedron, faces oriented outward.
            phi = (1 + sqrt(5)) / 2;
            V = [-1  phi 0;  1  phi 0; -1 -phi 0;  1 -phi 0; ...
                  0 -1  phi;  0  1  phi;  0 -1 -phi;  0  1 -phi; ...
                  phi 0 -1;  phi 0  1; -phi 0 -1; -phi 0  1];
            V = V ./ vecnorm(V, 2, 2);
            F = [1 12 6; 1 6 2; 1 2 8; 1 8 11; 1 11 12; ...
                 2 6 10; 6 12 5; 12 11 3; 11 8 7; 8 2 9; ...
                 4 10 5; 4 5 3; 4 3 7; 4 7 9; 4 9 10; ...
                 5 10 6; 3 5 12; 7 3 11; 9 7 8; 10 9 2];
            F = OceanMath.Geometry.Mesh.IcoSphere.orientOutward(V, F);
        end

        function [V, F] = subdivide(V, F)
            % SUBDIVIDE  One 1-to-4 split with shared, re-projected midpoints.
            nV = size(V, 1);
            nF = size(F, 1);
            halfEdges = [F(:, [1 2]); F(:, [2 3]); F(:, [3 1])];
            [E, ~, edgeOfHalfEdge] = unique(sort(halfEdges, 2), 'rows');
            mid = (V(E(:,1), :) + V(E(:,2), :)) / 2;
            V = [V; mid ./ vecnorm(mid, 2, 2)];

            mAB = nV + edgeOfHalfEdge(1:nF);
            mBC = nV + edgeOfHalfEdge(nF+1:2*nF);
            mCA = nV + edgeOfHalfEdge(2*nF+1:3*nF);
            a = F(:,1); b = F(:,2); c = F(:,3);
            F = [a mAB mCA; mAB b mBC; mCA mBC c; mAB mBC mCA];
        end
    end

    methods (Static, Access = private)
        function F = orientOutward(V, F)
            % For a convex solid centred at the origin, face f is outward iff
            % its normal points away from the origin: n_f . centroid_f > 0.
            n = cross(V(F(:,2),:) - V(F(:,1),:), V(F(:,3),:) - V(F(:,1),:), 2);
            centroid = (V(F(:,1),:) + V(F(:,2),:) + V(F(:,3),:)) / 3;
            inward = dot(n, centroid, 2) < 0;
            F(inward, :) = F(inward, [1 3 2]);
        end
    end
end
