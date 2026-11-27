classdef SurfaceSampler < handle
    % SURFACESAMPLER  Uniform random point sampling on a triangle-mesh surface,
    %   differentiable w.r.t. the vertex positions.
    %
    %   Mesh losses (Chamfer, normal) compare POINT CLOUDS, so a predicted mesh
    %   is turned into points by sampling its surface uniformly by area
    %   (Mesh R-CNN Sec. 3.3; Smith et al. 2019, GEOMetrics).
    %
    % ALGORITHM (Osada, Funkhouser, Chazelle & Dobkin 2002, "Shape
    %   Distributions", Sec. 4.2):
    %   1. Pick face f with probability  A_f / sum_g A_g
    %      (inverse CDF: u ~ U(0,1), f = first face whose cumulative area share > u).
    %   2. Pick a uniform point inside it with r1, r2 ~ U(0,1):
    %        w1 = 1 - sqrt(r1),  w2 = sqrt(r1) (1 - r2),  w3 = sqrt(r1) r2
    %        p  = w1 v1 + w2 v2 + w3 v3
    %      (the sqrt makes the density uniform in area, not piled up at v1).
    %   3. The sample's normal is the face normal n_f.
    %
    % GRADIENT (why this is differentiable although sampling is random):
    %   For a FIXED draw (f, w1, w2, w3), p is LINEAR in the three vertices,
    %     dp/dv_k = w_k I_3   for the corners v_k of face f,
    %   so an upstream gradient dL/dp scatters back as
    %     dL/dv_i = sum_{samples s whose face contains v_i} w_{s,i} dL/dp_s.
    %   backward() implements exactly this with accumarray. (The discrete
    %   face choice itself has no gradient -- the standard "reparameterised
    %   sampling" used by PyTorch3D's sample_points_from_meshes.)
    %
    % USAGE:
    %   sampler = OceanMath.Geometry.Mesh.SurfaceSampler(2000, 'Seed', 7);
    %   [P, N]  = sampler.forward(mesh);        % P, N: (n x 3); caches (f, w)
    %   dV      = sampler.backward(dLdP);       % (V x 3) gradient on mesh vertices
    %   [P, N, faceIdx, bary] = OceanMath.Geometry.Mesh.SurfaceSampler.sample(V, F, n, stream);
    %
    % See also: OceanMath.Geometry.Mesh.TriMesh

    properties
        NumSamples (1,1) double {mustBeInteger, mustBePositive} = 2000
    end

    properties (Access = private)
        Stream_ = []          % RandStream (seeded) or [] = global stream
        FaceIdx_ = []         % (n x 1) sampled face per point, set by forward()
        Bary_ = []            % (n x 3) barycentric weights, set by forward()
        Faces_ = []           % (F x 3) faces of the mesh seen by forward()
        NumVertices_ = 0
    end

    methods
        function obj = SurfaceSampler(numSamples, opts)
            arguments
                numSamples (1,1) double {mustBeInteger, mustBePositive} = 2000
                opts.Seed double = []
            end
            obj.NumSamples = numSamples;
            if ~isempty(opts.Seed)
                obj.Stream_ = RandStream('mt19937ar', 'Seed', opts.Seed);
            end
        end

        function [P, N] = forward(obj, mesh)
            % FORWARD  Sample NumSamples points (and normals) on the mesh surface.
            arguments
                obj
                mesh OceanMath.Geometry.Mesh.TriMesh
            end
            [P, N, obj.FaceIdx_, obj.Bary_] = OceanMath.Geometry.Mesh.SurfaceSampler.sample( ...
                mesh.Vertices, mesh.Faces, obj.NumSamples, obj.Stream_);
            obj.Faces_ = mesh.Faces;
            obj.NumVertices_ = mesh.numVertices();
        end

        function dV = backward(obj, dP)
            % BACKWARD  Scatter dL/dP (n x 3) back onto the vertices (V x 3)
            %   through the barycentric weights cached by forward().
            arguments
                obj
                dP (:,3) double
            end
            if isempty(obj.FaceIdx_)
                error('OceanMath:Geometry:Mesh:SurfaceSampler:NoForwardCache', ...
                    'backward() called before forward().');
            end
            dV = OceanMath.Geometry.Mesh.SurfaceSampler.scatterToVertices( ...
                dP, obj.Faces_(obj.FaceIdx_, :), obj.Bary_, obj.NumVertices_);
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = "\mathbf{p} = (1-\sqrt{r_1})\,\mathbf{v}_1 + \sqrt{r_1}(1-r_2)\,\mathbf{v}_2 + \sqrt{r_1}\,r_2\,\mathbf{v}_3, \quad f \sim A_f / \textstyle\sum_g A_g";
            meta.Symbols = [ ...
                struct('Symbol', "r_1, r_2", 'Description', "independent U(0,1) draws"), ...
                struct('Symbol', "\mathbf{v}_1,\mathbf{v}_2,\mathbf{v}_3", 'Description', "corners of the sampled face f"), ...
                struct('Symbol', "A_f", 'Description', "area of face f (face chosen proportionally to area)"), ...
                struct('Symbol', "\mathbf{p}", 'Description', sprintf('one of %d surface samples', obj.NumSamples))];
            meta.Assumptions = {'the face choice is not differentiated; points are linear in vertices for a fixed draw', ...
                                'zero-area faces are never sampled'};
            meta.IsSymbolic = true;
            meta.Reference = "Osada et al. (2002), Shape Distributions, ACM TOG, Sec. 4.2";
        end
    end

    methods (Static)
        function [P, N, faceIdx, bary] = sample(V, F, n, stream)
            % SAMPLE  Functional core (see class doc). stream: RandStream or [].
            arguments
                V (:,3) double
                F (:,3) double
                n (1,1) double {mustBeInteger, mustBePositive}
                stream = []
            end
            C = cross(V(F(:,2),:) - V(F(:,1),:), V(F(:,3),:) - V(F(:,1),:), 2);
            area2 = vecnorm(C, 2, 2);                 % 2 * face area
            candidates = find(area2 > 0);
            if isempty(candidates)
                error('OceanMath:Geometry:Mesh:SurfaceSampler:ZeroArea', ...
                    'Mesh has no face with positive area to sample from.');
            end

            % 1. Face choice by inverse CDF over cumulative area shares.
            cdf = cumsum(area2(candidates)) / sum(area2(candidates));
            cdf(end) = 1;                             % guard against round-off
            u = OceanMath.Geometry.Mesh.SurfaceSampler.uniform(stream, n, 1);
            faceIdx = candidates(discretize(u, [0; cdf]));

            % 2. Uniform barycentric coordinates inside the chosen triangle.
            r = OceanMath.Geometry.Mesh.SurfaceSampler.uniform(stream, n, 2);
            s1 = sqrt(r(:, 1));
            bary = [1 - s1, s1 .* (1 - r(:, 2)), s1 .* r(:, 2)];

            Fs = F(faceIdx, :);
            P = bary(:,1) .* V(Fs(:,1),:) + bary(:,2) .* V(Fs(:,2),:) + bary(:,3) .* V(Fs(:,3),:);

            % 3. Normals of the sampled faces.
            N = C(faceIdx, :) ./ area2(faceIdx);
        end

        function dV = scatterToVertices(dP, facesOfSamples, bary, numVertices)
            % SCATTERTOVERTICES  dL/dv_i = sum_s w_{s,i} dL/dp_s (see class doc).
            dV = zeros(numVertices, 3);
            for corner = 1:3
                idx = facesOfSamples(:, corner);
                for dim = 1:3
                    dV(:, dim) = dV(:, dim) + accumarray(idx, bary(:, corner) .* dP(:, dim), [numVertices 1]);
                end
            end
        end
    end

    methods (Static, Access = private)
        function u = uniform(stream, rows, cols)
            if isempty(stream)
                u = rand(rows, cols);
            else
                u = rand(stream, rows, cols);
            end
        end
    end
end
