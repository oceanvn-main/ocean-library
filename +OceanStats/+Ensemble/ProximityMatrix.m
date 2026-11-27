classdef ProximityMatrix
    % PROXIMITYMATRIX Co-leaf proximity between training samples.
    %
    % OVERVIEW:
    %   For a trained forest (RandomForest or ExtraTreesForest), two samples
    %   are "proximate" if they land in the same terminal leaf across many
    %   trees.  Proximity(i,j) = (# trees where i and j reach the same leaf) / B.
    %
    %   The resulting N×N symmetric matrix in [0,1] encodes pairwise
    %   structural similarity according to the ensemble.  Uses:
    %     - Outlier detection (low average proximity = likely outlier)
    %     - Missing data imputation (weight by proximity)
    %     - Cluster visualization (MDS on 1 - Proximity distance matrix)
    %
    %   Complexity: O(N * max_leaf_size * B) — warn if N > 1000.
    %
    % USAGE:
    %   P   = OceanStats.Ensemble.ProximityMatrix.compute(rf, X_train);
    %   out = OceanStats.Ensemble.ProximityMatrix.outlierScores(P);
    %   OceanStats.Ensemble.ProximityMatrix.visualize(P, ax, Y);
    %
    % REFERENCE:
    %   Breiman (2001). Random Forests. Machine Learning, 45(1), 5-32.  §2.3
    %
    % Author: Ocean Library
    % Date:   2026-05-12

    properties (Constant, Access = private)
        LARGE_N_WARN = 1000   % Warn for N above this threshold
    end

    methods (Static)

        function P = compute(forest, X, options)
            % COMPUTE N×N proximity matrix for all rows in X.
            %
            % Algorithm:
            %   For each tree b:
            %     1. Route all N samples to leaf IDs via DecisionTree.getLeafIds().
            %     2. For each leaf, all samples in that leaf are pairwise proximate:
            %        P(grp, grp) += 1.
            %   Normalize P by B. Symmetrize (P + P') / 2.
            %
            % Args:
            %   forest  : Fitted RandomForest or ExtraTreesForest object.
            %   X       : (N x p double) Feature matrix (usually training data).
            %   Verbose (logical, optional): Print progress. Default false.
            %
            % Returns:
            %   P (N x N double): Symmetric proximity matrix in [0, 1].
            %                     P(i,i) = 1 by definition.

            arguments
                forest
                X      (:,:) double
                options.Verbose (1,1) logical = false
            end

            N = size(X, 1);
            trees = forest.Trees_;
            B     = numel(trees);

            if N > OceanStats.Ensemble.ProximityMatrix.LARGE_N_WARN
                warning('ProximityMatrix:LargeN', ...
                    ['N=%d > %d: O(N^2) proximity computation may be slow. ' ...
                     'Consider passing a subsample of X.'], ...
                    N, OceanStats.Ensemble.ProximityMatrix.LARGE_N_WARN);
            end

            P = zeros(N, N);

            for b = 1:B
                if options.Verbose && mod(b, max(1, floor(B/5))) == 0
                    fprintf('[ProximityMatrix] Tree %d / %d\n', b, B);
                end

                % Route all samples to leaf nodes
                [leafIds, ~] = trees{b}.getLeafIds(X);

                % For each unique leaf, mark all pairwise members as proximate
                uLeaves = unique(leafIds);
                for ll = 1:numel(uLeaves)
                    grp = find(leafIds == uLeaves(ll));
                    P(grp, grp) = P(grp, grp) + 1;
                end
            end

            % Normalize and symmetrize
            P = P / B;
            P = (P + P') / 2;   % numerical symmetry guarantee
        end

        % -----------------------------------------------------------------
        function scores = outlierScores(P)
            % OUTLIERSCORES Compute per-sample outlier score from proximity.
            %
            % Outlier score(i) = 1 - mean_j(P(i,j))  (excluding self)
            % Range [0, 1]: high score means sample i is rarely co-leaf with
            % any other sample — likely an outlier.
            %
            % Args:
            %   P (N x N double): Proximity matrix from compute().
            %
            % Returns:
            %   scores (N x 1 double): Outlier scores, higher = more outlying.

            N = size(P, 1);
            % Sum of each row excluding the diagonal (self-proximity = 1)
            row_sum   = sum(P, 2) - diag(P);
            avg_prox  = row_sum / max(1, N - 1);
            scores    = 1 - avg_prox;   % high = low avg proximity = outlier
        end

        % -----------------------------------------------------------------
        function visualize(P, ax, Y, options)
            % VISUALIZE Plot the proximity matrix as a sorted heatmap.
            %
            % Sorts rows/columns by class label (if Y supplied) so that
            % within-class proximity blocks appear on the diagonal.
            %
            % Args:
            %   P  (N x N double): Proximity matrix.
            %   ax (axes, optional): Target axes. Default: gca().
            %   Y  (N x 1, optional): Class labels for sorting. [] = no sort.
            %   Title (string, optional): Axes title. Default: 'Proximity Matrix'.
            %   Colormap (string, optional): Colormap name. Default: 'parula'.

            arguments
                P       (:,:) double
                ax                   = []
                Y                    = []
                options.Title    string  = 'Proximity Matrix'
                options.Colormap string  = 'parula'
            end

            if isempty(ax) || ~isgraphics(ax)
                ax = gca();
            end

            N = size(P, 1);

            % Sort by class label if provided
            if ~isempty(Y) && numel(Y) == N
                [~, ord] = sort(Y);
            else
                ord = 1:N;
            end

            P_sorted = P(ord, ord);

            imagesc(ax, P_sorted);
            colormap(ax, options.Colormap);
            colorbar(ax);
            caxis(ax, [0, 1]);
            axis(ax, 'square');
            title(ax, options.Title, 'FontSize', 13, 'FontWeight', 'bold');
            xlabel(ax, 'Sample index (sorted)');
            ylabel(ax, 'Sample index (sorted)');
        end

        % -----------------------------------------------------------------
        function [coords, stress] = mdsCoords(P, ndim)
            % MDSCOORDS Classical MDS on the (1 - P) dissimilarity matrix.
            %
            % Allows 2D/3D scatter visualization of forest-defined distances.
            %
            % Args:
            %   P     (N x N double): Proximity matrix.
            %   ndim  (double): Embedding dimension (2 or 3). Default 2.
            %
            % Returns:
            %   coords (N x ndim): MDS coordinates.
            %   stress (double):   Kruskal stress (lower = better fit).

            arguments
                P    (:,:) double
                ndim (1,1) double = 2
            end

            D = 1 - P;          % dissimilarity: same leaf = 0, never co-leaf = 1
            D = max(D, 0);      % numerical safety

            try
                coords = cmdscale(D, ndim);
                % Compute Kruskal stress-1
                d_mds  = pdist(coords);
                d_orig = squareform(D);
                d_orig = d_orig(tril(true(size(D)), -1));
                stress = sqrt(sum((d_mds(:) - d_orig(:)).^2) / sum(d_orig(:).^2));
            catch ME
                warning('ProximityMatrix:MDSFailed', ...
                    'cmdscale failed: %s. Returning zeros.', ME.message);
                coords = zeros(size(P, 1), ndim);
                stress = NaN;
            end
        end

    end
end
