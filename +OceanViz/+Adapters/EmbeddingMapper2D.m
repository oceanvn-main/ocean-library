classdef EmbeddingMapper2D < OceanViz.Adapters.ISpatialMapper
    % EMBEDDINGMAPPER2D  Maps high-dimensional state features to 2D via embedding.
    %
    % This mapper is essential when state indices alone are insufficient to
    % describe the geometry of the state space. If each state is associated
    % with a high-dimensional feature vector (e.g., from a neural network
    % activation, a sensor reading, or a probability distribution), this
    % mapper uses dimensionality reduction to project them onto a 2D plane
    % for visualization.
    %
    % Supported embedding methods:
    %   'pca'   - Principal Component Analysis (linear, fast, exact)
    %             Projects onto the two directions of maximum variance.
    %   'tsne'  - t-SNE (non-linear, preserves local cluster structure)
    %             Requires Statistics and Machine Learning Toolbox.
    %   'umap'  - UMAP (non-linear, preserves global + local topology)
    %             Requires a UMAP MATLAB implementation to be on the path.
    %
    % Usage:
    %   % 10-state feature matrix (each state has a 50-dim feature vector):
    %   features = randn(10, 50);   % (K x F) matrix
    %   mapper   = OceanViz.Adapters.EmbeddingMapper2D(features, 'Method', 'pca');
    %   coords   = mapper.mapState(3);     % [x, y] — 2D projection of state 3
    %
    %   % mapSequence is vectorized (no re-projection)
    %   coordSeq = mapper.mapSequence(hist.getStateSequence());
    %
    % Notes:
    %   - All embedding is computed ONCE at construction and cached.
    %   - mapState() and mapSequence() are pure O(1) / O(N) lookups.
    %   - The embedding is fixed. If features change, create a new mapper.
    %
    % See also: OceanViz.Adapters.ISpatialMapper
    %           OceanViz.Adapters.GraphMapper

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        EmbeddedPositions_ (:,2) double   % (K x 2) pre-projected coordinates
        Method_            (1,:) char      % embedding method name
        VarianceExplained_ (1,:) double    % for PCA: variance explained ratio
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = EmbeddingMapper2D(features, options)
            % EMBEDDINGMAPPER2D  Compute embedding and cache all 2D positions.
            %
            % INPUTS:
            %   features - (K x F) double matrix.
            %       K = number of states, F = feature dimensionality.
            %       Each row is the feature vector of a state.
            %
            % Name-Value:
            %   Method (char): 'pca' (default), 'tsne', or 'umap'.
            %   Normalize (logical): z-score normalize features first. Default: true.
            arguments
                features (:,:) double
                options.Method    (1,:) char    = 'pca'
                options.Normalize (1,1) logical = true
            end

            [K, F] = size(features);
            if K < 2
                error('OceanViz:Adapters:EmbeddingMapper2D:TooFew', ...
                    'Need at least 2 states. Got %d.', K);
            end
            if F < 2
                error('OceanViz:Adapters:EmbeddingMapper2D:LowDim', ...
                    'Feature dimensionality must be >= 2. Got %d.', F);
            end

            obj.Method_ = lower(options.Method);

            % --- Normalize features ---
            if options.Normalize
                mu_f = mean(features, 1);
                sd_f = std(features, 0, 1);
                sd_f(sd_f < 1e-10) = 1;   % avoid division by zero for const columns
                features = (features - mu_f) ./ sd_f;
            end

            % --- Compute embedding (once, cached) ---
            switch obj.Method_
                case 'pca'
                    [~, scores, ~, ~, explained] = pca(features);
                    obj.EmbeddedPositions_ = scores(:, 1:2);
                    obj.VarianceExplained_ = explained(1:2)';

                case 'tsne'
                    if ~license('test', 'statistics_toolbox')
                        error('OceanViz:Adapters:EmbeddingMapper2D:NoToolbox', ...
                            't-SNE requires the Statistics and Machine Learning Toolbox.');
                    end
                    obj.EmbeddedPositions_ = tsne(features, 'NumDimensions', 2);
                    obj.VarianceExplained_  = NaN(1, 2);   % t-SNE has no variance explained

                case 'umap'
                    % UMAP requires a 3rd-party run_umap() function on the path.
                    if ~exist('run_umap', 'file')
                        error('OceanViz:Adapters:EmbeddingMapper2D:NoUMAP', ...
                            ['UMAP requires ''run_umap'' to be on the MATLAB path. ' ...
                             'Download from: https://www.math.uci.edu/icamp/courses/math77c/demos/hist_eq.pdf']);
                    end
                    [obj.EmbeddedPositions_, ~] = run_umap(features, 'n_components', 2, 'verbose', 'none');
                    obj.VarianceExplained_      = NaN(1, 2);

                otherwise
                    error('OceanViz:Adapters:EmbeddingMapper2D:UnknownMethod', ...
                        'Unknown method: "%s". Use pca, tsne, or umap.', options.Method);
            end
        end

        % ---- ISpatialMapper Implementation ---- %

        function coords = mapState(obj, stateIndex)
            % MAPSTATE  Return the 2D embedding coordinates of a state. O(1).
            if ~obj.isValidState(stateIndex)
                error('OceanViz:Adapters:EmbeddingMapper2D:InvalidState', ...
                    'State %d is out of range [1, %d].', stateIndex, obj.getNumStates());
            end
            coords = obj.EmbeddedPositions_(stateIndex, :);
        end

        function coordSeq = mapSequence(obj, stateSeq)
            % MAPSEQUENCE  Vectorized batch lookup — no re-projection overhead.
            idx      = double(stateSeq(:));
            coordSeq = obj.EmbeddedPositions_(idx, :);
        end

        function D = getDimensions(~)
            D = 2;
        end

        function N = getNumStates(obj)
            N = size(obj.EmbeddedPositions_, 1);
        end

        function disp(obj)
            fprintf('EmbeddingMapper2D  [%d states, method: %s]\n', ...
                obj.getNumStates(), obj.Method_);
            if ~any(isnan(obj.VarianceExplained_))
                fprintf('  Variance explained: PC1=%.1f%%  PC2=%.1f%%\n', ...
                    obj.VarianceExplained_(1), obj.VarianceExplained_(2));
            end
        end
    end
end
