classdef KMeans < handle
    % KMEANS Implementation of Lloyd's algorithm with multiple restarts.
    %
    % USAGE:
    %   km = OceanStats.Unsupervised.KMeans();
    %   labels = km.fit(X, K);                           % standard (5 restarts)
    %   labels = km.fit(X, K, 'InitCentroids', C0);      % fixed initialization, 1 restart
    %   labels = km.fit(X, K, 'TrackHistory', true);      % record centroid trajectory
    %   labels = km.fit(X, K, 'OnIteration', @(it) ...);  % callback per iteration (recorders)
    %
    % OnIteration receives struct(Restart, Iteration, Centroids, Labels): Iteration 0 =
    % the initial centroids with their nearest-centroid assignment; Iteration m >= 1 =
    % after step 2 of Lloyd's algorithm (assignment L_m, centroids = cluster means of L_m).
    % It observes only: results are identical with or without it.
    %
    % After fit:
    %   km.Centroids   — (K x p) final centroids
    %   km.Labels      — (N x 1) cluster assignments
    %   km.History     — struct array with .Centroids per iteration (if TrackHistory=true)
    
    properties
        Centroids
        Labels
        History   % struct array: History(t).Centroids = (K x p) at iteration t
    end
    
    methods
        function labels = fit(obj, X, K, opts)
            % FIT  Lloyd's algorithm with optional fixed init and trajectory tracking.
            arguments
                obj
                X           double
                K           (1,1) double {mustBePositive, mustBeInteger}
                opts.InitCentroids  double = []       % (K x p) fixed starting centroids
                opts.TrackHistory   (1,1) logical = false
                opts.MaxIter        (1,1) double = 50
                opts.NumRestarts    (1,1) double = 5
                opts.OnIteration    = []                % optional @(struct) observer
            end
            
            useFixedInit = ~isempty(opts.InitCentroids);
            if useFixedInit
                num_restarts = 1;  % no restarts when user supplies centroids
            else
                num_restarts = opts.NumRestarts;
            end
            
            best_inertia = inf;
            best_history = [];
            
            for r = 1:num_restarts
                % Initialization
                if useFixedInit
                    C = opts.InitCentroids;
                else
                    ridx = randperm(size(X, 1), K);
                    C = X(ridx, :);
                end
                
                % Pre-allocate trajectory storage
                if opts.TrackHistory
                    traj = struct('Centroids', cell(opts.MaxIter + 1, 1));
                    traj(1).Centroids = C;
                end
                
                L = zeros(size(X, 1), 1);
                if ~isempty(opts.OnIteration)
                    [~, L0] = min(obj.computeDist(X, C), [], 2);
                    opts.OnIteration(struct('Restart', r, 'Iteration', 0, 'Centroids', C, 'Labels', L0));
                end
                for iter = 1:opts.MaxIter
                    % Assignment
                    dist = obj.computeDist(X, C);
                    [~, L_new] = min(dist, [], 2);
                    
                    if isequal(L_new, L), break; end
                    L = L_new;
                    
                    % Update
                    for k = 1:K
                        if any(L == k)
                            C(k, :) = mean(X(L == k, :), 1);
                        end
                    end
                    
                    if opts.TrackHistory
                        traj(iter + 1).Centroids = C;
                    end
                    if ~isempty(opts.OnIteration)
                        opts.OnIteration(struct('Restart', r, 'Iteration', iter, 'Centroids', C, 'Labels', L));
                    end
                end
                
                % Truncate trajectory
                if opts.TrackHistory
                    traj = traj(1:iter);
                end
                
                % Check inertia
                inertia = 0;
                for k = 1:K
                    inertia = inertia + sum(sum((X(L == k, :) - C(k, :)).^2));
                end
                
                if inertia < best_inertia
                    best_inertia = inertia;
                    obj.Centroids = C;
                    obj.Labels = L;
                    if opts.TrackHistory
                        best_history = traj;
                    end
                end
            end
            
            labels = obj.Labels;
            if opts.TrackHistory
                obj.History = best_history;
            else
                obj.History = [];
            end
        end
    end
    
    methods (Access = private)
        function dist = computeDist(~, X, C)
            D2 = sum(X.^2, 2) + sum(C.^2, 2)' - 2 * (X * C');
            dist = sqrt(max(D2, 0));
        end
    end
end
