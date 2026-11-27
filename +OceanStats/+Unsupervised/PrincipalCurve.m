classdef PrincipalCurve < handle
    % PRINCIPALCURVE Hastie & Stuetzle (1989) Principal Curves algorithm.
    %
    % PURPOSE:
    %   Discovers a 1D non-linear manifold (curve) passing through the "middle"
    %   of a data cloud by alternating between expectation (smoothing) and 
    %   projection steps.
    %
    % USAGE:
    %   import OceanStats.Unsupervised.PrincipalCurve
    %   import OceanStats.NonParametric.KernelSmoother
    %
    %   pc = PrincipalCurve('Smoother', KernelSmoother('Bandwidth', 0.5));
    %   model = pc.fit(X);
    %   
    %   % model.Curve          - (M x p) Points defining the discrete curve
    %   % model.ArcLengths     - (M x 1) Cumulative arc length of Curve points
    %   % model.DataProjection - (N x 1) Projected arc length for each data point
    %
    % AUTHOR: Ocean Library System

    properties
        Smoother  % OceanStats.Base.IUnivariateSmoother
        MaxIter (1,1) double {mustBePositive, mustBeInteger} = 20
        Tol     (1,1) double {mustBePositive} = 1e-4
        Verbose (1,1) logical = true
    end

    methods
        function obj = PrincipalCurve(opts)
            arguments
                opts.Smoother = OceanStats.NonParametric.KernelSmoother('Bandwidth', 0.5)
                opts.MaxIter (1,1) double = 20
                opts.Tol     (1,1) double = 1e-4
                opts.Verbose (1,1) logical = true
            end
            obj.Smoother = opts.Smoother;
            obj.MaxIter  = opts.MaxIter;
            obj.Tol      = opts.Tol;
            obj.Verbose  = opts.Verbose;
        end

        function model = fit(obj, X)
            [N, p] = size(X);

            % 1. Initialization (Linear Principal Component)
            mu = mean(X, 1);
            X_c = X - repmat(mu, N, 1);
            [V, ~] = pca(X_c);
            v1 = V(:, 1);
            
            % Initial projections onto PC1
            lambda = X_c * v1;

            iter = 0;
            delta = inf;
            prev_dist = inf;

            while iter < obj.MaxIter && delta > obj.Tol
                iter = iter + 1;

                % --- EXPECTATION STEP.  Freeze the projection parameters (lambda),
                %and move the curve to the "middle" (expected value) of the data.
                %Sort to trace the point a long the curve in order
                [lambda_sort, sort_idx] = sort(lambda);
                X_sort = X(sort_idx, :);

                curve_points = zeros(N, p);
                
                % Smooth each dimension independently against lambda
                for j = 1:p
                    sm_model = obj.Smoother.fit(lambda_sort, X_sort(:, j));
                    curve_points(:, j) = obj.Smoother.predict(sm_model, lambda_sort);
                end

                % Recalculate true arc-length of the discrete curve
                diffs = diff(curve_points, 1, 1); %Different between consecutive points on the curve
                segment_lengths = sqrt(sum(diffs.^2, 2)); %Pythagorean for lenght of those tiny segment
                arc_lengths = [0; cumsum(segment_lengths)]; %Accumulate

                % --- PROJECTION STEP ---
                % Project each original data point onto the piecewise linear curve
                [lambda_new, dists] = obj.projectToCurve(X, curve_points, arc_lengths, segment_lengths);

                % --- CONVERGENCE CHECK ---
                mean_dist = mean(dists);
                if prev_dist == inf
                    delta = inf;
                else
                    delta = abs(prev_dist - mean_dist) / max(prev_dist, 1e-6);
                end

                if obj.Verbose
                    fprintf('[PrincipalCurve] Iter %2d | Mean Orthogonal Dist: %.4f | Delta: %.4e\n', ...
                        iter, mean_dist, delta);
                end

                prev_dist = mean_dist;
                lambda = lambda_new;
            end

            model.Curve = curve_points;
            model.ArcLengths = arc_lengths;
            model.DataProjection = lambda_new;
            model.MeanDistance = prev_dist;
        end
    end

    methods (Access = private)
        function [lambda_new, dists] = projectToCurve(~, X, V, arc_lengths, segment_lengths)
            % Project each point in X onto the piecewise linear curve defined by vertices V.
            % Derived via parameterize the line as parametric eq, optimize to find distance of point on the line to x_i. Take derivative.
            %t=∥p2​−p1​∥2(xi​−p1​)⋅(p2​−p1​)​
            %Geometrically, this find the sign length of vector of interest (xi - p1) points a long the segment direction (p2-p1). Scale by the rules magnitude of (p2-p1)
            N = size(X, 1);
            num_segments = size(V, 1) - 1;
            
            lambda_new = zeros(N, 1);
            dists = zeros(N, 1);

            for i = 1:N
                xi = X(i, :);
                
                min_sq_dist = inf;
                best_lambda = 0;

                % Check all segments
                for k = 1:num_segments
                    p1 = V(k, :);
                    p2 = V(k+1, :);
                    L = segment_lengths(k);
                    
                    if L < 1e-12
                        % Degenerate segment
                        sq_dist = sum((xi - p1).^2);
                        t = 0;
                    else
                        v = p2 - p1;
                        % Projection scalar
                        t = dot(xi - p1, v) / (L^2);
                        % Clamp to segment
                        t = max(0, min(1, t));
                        
                        pc = p1 + t * v;
                        sq_dist = sum((xi - pc).^2);
                    end

                    if sq_dist < min_sq_dist
                        min_sq_dist = sq_dist;
                        best_lambda = arc_lengths(k) + t * L;
                    end
                end

                dists(i) = sqrt(min_sq_dist);
                lambda_new(i) = best_lambda;
            end
        end
    end
end
