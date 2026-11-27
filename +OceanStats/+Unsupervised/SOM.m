classdef SOM < handle
    % SOM  Self-Organizing Map (Kohonen, 1982).
    %
    % ALGORITHM:
    %   For each iteration t = 1 .. NumPasses * N:
    %     1. Pick a random training sample x_i.
    %     2. Find the Best Matching Unit (BMU): argmin_j ||x_i - w_j||. Data space
    %     3. Update all prototypes:
    %            w_j <- w_j + alpha(t) * h(j, bmu, t) * (x_i - w_j) 
    %   error: x_i - w_j in data space. h(...) in grid space. How much error to pull (direction) on grid space
    %        where h is a Gaussian neighborhood kernel:
    %            h(j, bmu, t) = exp(-||grid_j - grid_bmu||^2 / (2*r(t)^2))
    %     4. alpha(t) and r(t) decay linearly from initial to final values.
    %
    % USAGE:
    %   som = OceanStats.Unsupervised.SOM('GridSize', [5, 5], 'NumPasses', 40);
    %   som.fit(X);
    %   labels = som.predict(X_new);
    %   [gx, gy] = som.getGridCoords();
    %
    % After fit:
    %   som.Weights   - (Q1*Q2 x p) final prototype weight vectors
    %   som.Labels    - (N x 1) BMU assignment for each training sample
    %   som.GridSize  - [Q1, Q2]
    %   som.History   - struct array: History(t).Weights per iteration (if TrackHistory)
    %
    % Ref: Kohonen (1982). Self-organized formation of topologically correct
    %      feature maps. Biological Cybernetics, 43(1), 59-69.
    %      Hastie et al. (2009), ESL Section 14.4.
    
    properties
        Weights         % (Q1*Q2 x p) prototype weight vectors (codebook)
        Labels          % (N x 1)     BMU assignments for training data
        GridSize        % [Q1, Q2]    rectangular grid dimensions
        History         % struct array: History(t).Weights (if TrackHistory)
        ErrorHistory    % (NumPasses+1 x 1) reconstruction error per pass (if TrackHistory)
    end
    
    properties (SetAccess = private)
        InitRadius_     (1,1) double = 2
        FinalRadius_    (1,1) double = 0.1
        InitLR_         (1,1) double = 0.5
        FinalLR_        (1,1) double = 0.01
        NumPasses_      (1,1) double = 40
    end
    
    methods
        function obj = SOM(opts)
            % SOM Constructor.
            %
            %   som = OceanStats.Unsupervised.SOM()
            %   som = OceanStats.Unsupervised.SOM('GridSize', [5,5], 'NumPasses', 40)
            arguments
                opts.GridSize     (1,2) double {mustBePositive, mustBeInteger} = [5, 5]
                opts.InitRadius   (1,1) double {mustBePositive} = 2
                opts.FinalRadius  (1,1) double {mustBeNonnegative} = 0.1
                opts.InitLR       (1,1) double {mustBePositive} = 0.5
                opts.FinalLR      (1,1) double {mustBeNonnegative} = 0.01
                opts.NumPasses    (1,1) double {mustBePositive, mustBeInteger} = 40
            end
            obj.GridSize     = opts.GridSize;
            obj.InitRadius_  = opts.InitRadius;
            obj.FinalRadius_ = opts.FinalRadius;
            obj.InitLR_      = opts.InitLR;
            obj.FinalLR_     = opts.FinalLR;
            obj.NumPasses_   = opts.NumPasses;
        end
        
        function labels = fit(obj, X, opts)
            % FIT  Train the SOM on data X.
            %
            %   som.fit(X)
            %   som.fit(X, 'TrackHistory', true)
            %   som.fit(X, 'InitWeights', W0)  % fixed initialization
            %   som.fit(X, 'Mode', 'batch')    % batch SOM (ESL Eq 14.48)
            %
            %   X - (N x p) data matrix
            %
            %   Mode:
            %     'online' (default) — stochastic Kohonen update, one sample at a time.
            %     'batch'            — ESL Eq 14.48: assign all points to BMUs, then
            %                          update each prototype as the neighborhood-weighted
            %                          mean. No learning rate needed; only r(t) decays.
            arguments
                obj
                X              double
                opts.TrackHistory  (1,1) logical = false
                opts.InitWeights   double = []
                opts.Verbose       (1,1) logical = false
                opts.Mode          (1,1) string {mustBeMember(opts.Mode, ["online","batch"])} = "online"
            end
            
            [N, p] = size(X);
            Q1 = obj.GridSize(1);
            Q2 = obj.GridSize(2);
            M  = Q1 * Q2;  % total number of prototypes
            
            % --- Grid coordinates (integer positions on Q1 x Q2 grid) ---
            % Stored as (M x 2) for on-the-fly distance computation.
            % Grid distances are computed per-BMU as needed: O(M) per call
            % instead of O(M^2) precomputed matrix.  Scales to large grids.
            [gy, gx] = ndgrid(1:Q1, 1:Q2);
            gcX = gx(:);   % (M x 1) grid column indices
            gcY = gy(:);   % (M x 1) grid row indices
            
            % --- Initialize weights ---
            if ~isempty(opts.InitWeights)
                obj.Weights = opts.InitWeights;
            else
                % Random samples from data as initial prototypes
                ridx = randperm(N, min(M, N)); 
                obj.Weights = X(ridx, :);
                % If M > N, fill remaining with noisy copies
                if M > N
                    extra = M - N;
                    obj.Weights(N+1:M, :) = X(randi(N, extra, 1), :) + ...
                        0.01 * randn(extra, p);
                end
            end
            
            % --- Total iterations ---
            totalIter = obj.NumPasses_ * N;
            
            % --- History storage ---
            if opts.TrackHistory
                % Record at start + after each full pass
                histLen = obj.NumPasses_ + 1;
                traj = struct('Weights', cell(histLen, 1));
                traj(1).Weights = obj.Weights;
                histIdx = 1;
                
                % Reconstruction error = total sum of squared distances to BMU
                errHist = zeros(histLen, 1);
                D0 = sum(X.^2, 2) + sum(obj.Weights.^2, 2)' - 2*(X * obj.Weights');
                D0 = max(D0, 0);
                errHist(1) = sum(min(D0, [], 2));
            end
            
            % --- Main training loop ---
            if opts.Mode == "online"
                % =============================================================
                % ONLINE SOM (Kohonen, 1982)
                %   One sample at a time, stochastic gradient-like update.
                %   Both alpha(t) and r(t) decay linearly over all iterations.
                % =============================================================
                iterCount = 0;
                for pass = 1:obj.NumPasses_
                    perm = randperm(N);
                    for s = 1:N
                        iterCount = iterCount + 1;
                        
                        % Linear decay fraction: 0 at start, 1 at end
                        frac = (iterCount - 1) / max(totalIter - 1, 1);
                        
                        % Current learning rate and neighborhood radius
                        alpha = obj.InitLR_      + frac * (obj.FinalLR_     - obj.InitLR_);
                        r     = obj.InitRadius_  + frac * (obj.FinalRadius_ - obj.InitRadius_);
                        
                        % Pick sample
                        xi = X(perm(s), :);
                        
                        % Find BMU (Best Matching Unit). Data space
                        dists = sum((obj.Weights - xi).^2, 2);
                        [~, bmu] = min(dists);
                        
                        %Need a function that maximize at BMU, and fade out smoothly, We choose Gaussian in this case
                        % Gaussian neighborhood function (on-the-fly grid distance). Grid, Latent Space
                        d2 = (gcX - gcX(bmu)).^2 + (gcY - gcY(bmu)).^2;
                        h = exp(-d2 / (2 * r^2));
                        
                        % Update all prototypes
                        delta = xi - obj.Weights;        % (M x p)
                        obj.Weights = obj.Weights + alpha * (h .* delta);
                    end
                    
                    if opts.TrackHistory
                        histIdx = histIdx + 1;
                        traj(histIdx).Weights = obj.Weights;
                        Dp = sum(X.^2, 2) + sum(obj.Weights.^2, 2)' - 2*(X * obj.Weights');
                        Dp = max(Dp, 0);
                        errHist(histIdx) = sum(min(Dp, [], 2));
                    end
                    
                    if opts.Verbose
                        allDists = obj.computeDist_(X);
                        [minD, ~] = min(allDists, [], 2);
                        mqe = mean(minD);
                        fprintf('[SOM-online] Pass %3d/%d | alpha=%.4f | r=%.3f | MQE=%.4f\n', ...
                            pass, obj.NumPasses_, alpha, r, mqe);
                    end
                end
                
            else
                % =============================================================
                % BATCH SOM (ESL Eq 14.48)
                %   Assign ALL points to BMUs, then update each prototype as
                %   the neighborhood-weighted mean of all data points.
                %   No learning rate — each update is a deterministic weighted
                %   average.  Only r(t) decays (linearly per pass).
                %
                %   m_j = sum_k [ h(j, bmu(k)) * x_k ] / sum_k [ h(j, bmu(k)) ]
                % =============================================================
                for pass = 1:obj.NumPasses_
                    % Decay fraction per pass (0 at start, 1 at end)
                    frac = (pass - 1) / max(obj.NumPasses_ - 1, 1);
                    r = obj.InitRadius_ + frac * (obj.FinalRadius_ - obj.InitRadius_);
                    
                    % Step 1: Assign all points to their BMU
                    D2 = sum(X.^2, 2) + sum(obj.Weights.^2, 2)' - 2*(X * obj.Weights');
                    D2 = max(D2, 0);
                    [~, bmuAll] = min(D2, [], 2);   % (N x 1)
                    
                    % Step 2: Neighborhood-weighted mean for each prototype
                    %   H(k, j) = exp(-||grid_j - grid_bmu(k)||^2 / (2*r^2))
                    %   Computed on-the-fly: O(N*M) work, no O(M^2) storage
                    dx = gcX(bmuAll) - gcX';   % (N x M)
                    dy = gcY(bmuAll) - gcY';   % (N x M)
                    H = exp(-(dx.^2 + dy.^2) / (2 * r^2));   % (N x M)
                    
                    sumH = sum(H, 1)';              % (M x 1)
                    newWeights = (H' * X);           % (M x p)
                    
                    % Only update prototypes with non-zero weight
                    active = sumH > 1e-12;
                    newWeights(active, :) = newWeights(active, :) ./ sumH(active);
                    newWeights(~active, :) = obj.Weights(~active, :);
                    
                    obj.Weights = newWeights;
                    
                    if opts.TrackHistory
                        histIdx = histIdx + 1;
                        traj(histIdx).Weights = obj.Weights;
                        Dp = sum(X.^2, 2) + sum(obj.Weights.^2, 2)' - 2*(X * obj.Weights');
                        Dp = max(Dp, 0);
                        errHist(histIdx) = sum(min(Dp, [], 2));
                    end
                    
                    if opts.Verbose
                        allDists = obj.computeDist_(X);
                        [minD, ~] = min(allDists, [], 2);
                        mqe = mean(minD);
                        fprintf('[SOM-batch]  Pass %3d/%d | r=%.3f | MQE=%.4f\n', ...
                            pass, obj.NumPasses_, r, mqe);
                    end
                end
            end
            
            % --- Final assignment ---
            allDists = obj.computeDist_(X);
            [~, obj.Labels] = min(allDists, [], 2);
            labels = obj.Labels;
            
            if opts.TrackHistory
                obj.History = traj(1:histIdx);
                obj.ErrorHistory = errHist(1:histIdx);
            else
                obj.History = [];
                obj.ErrorHistory = [];
            end
        end
        
        function labels = predict(obj, X)
            % PREDICT  Assign each row of X to its Best Matching Unit.
            %
            %   labels = som.predict(X_new)
            arguments
                obj
                X double
            end
            allDists = obj.computeDist_(X);
            [~, labels] = min(allDists, [], 2);
        end
        
        function [gx, gy] = getGridCoords(obj)
            % GETGRIDCOORDS  Returns the 2D integer grid coordinates for each prototype.
            %
            %   [gx, gy] = som.getGridCoords()
            %   gx, gy - (M x 1) column vectors of grid positions
            Q1 = obj.GridSize(1);
            Q2 = obj.GridSize(2);
            [gy_mat, gx_mat] = ndgrid(1:Q1, 1:Q2);
            gx = gx_mat(:);
            gy = gy_mat(:);
        end
        
        function mqe = meanQuantizationError(obj, X)
            % MEANQUANTIZATIONERROR  Average Euclidean distance to BMU.
            %
            %   mqe = som.meanQuantizationError(X)
            arguments
                obj
                X double
            end
            allDists = obj.computeDist_(X);
            [minD, ~] = min(allDists, [], 2);
            mqe = mean(minD);
        end
        
        function te = topographicError(obj, X)
            % TOPOGRAPHICERROR  Fraction of samples whose 1st and 2nd BMU
            %   are NOT adjacent on the grid.
            %
            %   te = som.topographicError(X)
            %
            % Adjacency = Manhattan distance == 1 on the integer grid.
            arguments
                obj
                X double
            end
            [gx, gy] = obj.getGridCoords();
            allDists = obj.computeDist_(X);
            N = size(X, 1);
            
            nErrors = 0;
            for i = 1:N
                [~, sortIdx] = sort(allDists(i, :));
                bmu1 = sortIdx(1);
                bmu2 = sortIdx(2);
                % Manhattan distance on grid
                gridDist = abs(gx(bmu1) - gx(bmu2)) + abs(gy(bmu1) - gy(bmu2));
                if gridDist > 1
                    nErrors = nErrors + 1;
                end
            end
            te = nErrors / N;
        end
        
        function U = umatrix(obj)
            % UMATRIX  Compute the U-matrix (unified distance matrix).
            %
            %   U = som.umatrix()
            %Scan through the grid, create copy of SOM, and write into it the average displacement in data space
            % U(i,j) = mean Euclidean distance from prototype at grid position
            %          (i,j) to its direct grid neighbors. Reveals cluster
            %          boundaries in the projected space.
            Q1 = obj.GridSize(1);
            Q2 = obj.GridSize(2);
            W = reshape(obj.Weights, [Q1, Q2, size(obj.Weights, 2)]);
            U = zeros(Q1, Q2);
            
            for i = 1:Q1
                for j = 1:Q2
                    wi = squeeze(W(i, j, :))';
                    neighbors = [];
                    if i > 1,  neighbors = [neighbors; squeeze(W(i-1, j, :))']; end
                    if i < Q1, neighbors = [neighbors; squeeze(W(i+1, j, :))']; end
                    if j > 1,  neighbors = [neighbors; squeeze(W(i, j-1, :))']; end
                    if j < Q2, neighbors = [neighbors; squeeze(W(i, j+1, :))']; end
                    
                    diffs = neighbors - wi; %Displacement from this knot to each neighbor
                    U(i, j) = mean(sqrt(sum(diffs.^2, 2))); %avg distance
                end
            end
        end
    end
    
    methods (Access = private)
        function D = computeDist_(obj, X)
            % COMPUTEDIST_  Euclidean distance from each sample to each prototype.
            %   D - (N x M) distance matrix
            D2 = sum(X.^2, 2) + sum(obj.Weights.^2, 2)' - 2 * (X * obj.Weights');
            D = sqrt(max(D2, 0));
        end
    end
end
