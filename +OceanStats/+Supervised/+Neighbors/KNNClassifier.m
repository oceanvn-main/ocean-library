classdef KNNClassifier < handle
    % KNNCLASSIFIER K-Nearest Neighbour multi-class classifier.
    %
    % THEORY (ESL Section 13.3):
    %   For a query point x, find the K training points closest by the chosen
    %   metric. Assign the majority class label (or distance-weighted vote).
    %
    % BACKENDS:
    %   UseKDTree=true  (default when N_train > 500):
    %     Builds an OceanAlgo.Spatial.KDTree for O(log N) per-query lookup.
    %     Supports 'euclidean', 'manhattan', 'chebyshev'.
    %   UseKDTree=false:
    %     Vectorized full distance matrix O(N_train * N_test).
    %     Required when a custom DistanceFcn is supplied.
    %
    % USAGE:
    %   knn = OceanStats.Supervised.Neighbors.KNNClassifier('K', 7);
    %   knn.fit(X_train, y_train);
    %   y_pred = knn.predict(X_test);
    %   probs  = knn.predictProba(X_test);   % (N x nClasses)
    %
    % REGRESSION:
    %   knn = OceanStats.Supervised.Neighbors.KNNClassifier( ...
    %       'K', 5, 'Task', 'regression');
    %   knn.fit(X_train, Y_train);
    %   Y_hat = knn.predict(X_test);
    %
    % Author: Ocean Library
    % Date:   Modified 2026-05-12 — KDTree backend, distance weighting, regression

    properties
        K           (1,1) double {mustBePositive, mustBeInteger} = 1
        Metric      string  = 'euclidean'   % 'euclidean'|'manhattan'|'chebyshev'
        Weights     string  = 'uniform'     % 'uniform'|'distance'
        Task        string  = 'auto'        % 'auto'|'classification'|'regression'
        UseKDTree   logical = true          % false forces brute-force matrix
        LeafSize    (1,1) double = 10
        DistanceFcn = []   % (optional) @(A,B) -> (N_A x N_B) squared-dist matrix
    end

    properties (SetAccess = private)
        X_tr    double
        y_tr    double
        Classes double
        Tree_          % OceanAlgo.Spatial.KDTree (empty when UseKDTree=false)
        Task_   string % resolved task
    end

    methods

        function obj = KNNClassifier(varargin)
            % KNNCLASSIFIER Constructor (name-value pairs, legacy varargin).
            p = inputParser;
            addParameter(p, 'K',           1);
            addParameter(p, 'Metric',      'euclidean');
            addParameter(p, 'Weights',     'uniform');
            addParameter(p, 'Task',        'auto');
            addParameter(p, 'UseKDTree',   true);
            addParameter(p, 'LeafSize',    10);
            addParameter(p, 'DistanceFcn', []);
            parse(p, varargin{:});
            obj.K           = p.Results.K;
            obj.Metric      = p.Results.Metric;
            obj.Weights     = p.Results.Weights;
            obj.Task        = p.Results.Task;
            obj.UseKDTree   = p.Results.UseKDTree;
            obj.LeafSize    = p.Results.LeafSize;
            obj.DistanceFcn = p.Results.DistanceFcn;
        end

        % -----------------------------------------------------------------
        function fit(obj, X, y)
            % FIT Store training data; build KDTree if applicable.
            obj.X_tr = X;
            obj.y_tr = y(:);

            % Resolve task
            if strcmpi(obj.Task, 'auto')
                isInt   = all(y == floor(y));
                nUnique = numel(unique(y));
                obj.Task_ = 'regression';
                if isInt && nUnique <= 20, obj.Task_ = 'classification'; end
            else
                obj.Task_ = obj.Task;
            end

            if strcmpi(obj.Task_, 'classification')
                obj.Classes = unique(y(:))';
            else
                obj.Classes = [];
            end

            % Build KDTree backend if suitable
            useFast = obj.UseKDTree && isempty(obj.DistanceFcn) && ...
                      any(strcmpi(obj.Metric, {'euclidean','manhattan','chebyshev'}));
            if useFast
                obj.Tree_ = OceanAlgo.Spatial.KDTree(X, ...
                    'Metric', obj.Metric, 'LeafSize', obj.LeafSize);
            else
                obj.Tree_ = [];
            end
        end

        % -----------------------------------------------------------------
        function y_pred = predict(obj, X_test)
            % PREDICT Hard labels (classif) or values (regression).
            if ~isempty(obj.Tree_)
                y_pred = obj.predictKDTree_(X_test);
            else
                y_pred = obj.predictBrute_(X_test);
            end
        end

        % -----------------------------------------------------------------
        function probs = predictProba(obj, X_test)
            % PREDICTPROBA (N x nClasses) fraction matrix, rows sum to 1.
            % Classification only.
            if strcmpi(obj.Task_, 'regression')
                error('KNNClassifier:NotClassification', ...
                    'predictProba is only for classification tasks.');
            end
            if ~isempty(obj.Tree_)
                probs = obj.probaKDTree_(X_test);
            else
                probs = obj.probaBrute_(X_test);
            end
        end

    end % public

    % =====================================================================
    methods (Access = private)

        % ---- KDTree path ------------------------------------------------
        function y_pred = predictKDTree_(obj, X_test)
            M      = size(X_test, 1);
            y_pred = zeros(M, 1);
            for i = 1:M
                [ids, dists] = obj.Tree_.findKNearest(X_test(i,:), obj.K);
                valid = ids > 0;
                nbY   = obj.y_tr(ids(valid));
                dv    = dists(valid);
                if strcmpi(obj.Task_, 'classification')
                    y_pred(i) = obj.voteClass_(nbY, dv);
                else
                    y_pred(i) = obj.aggregateReg_(nbY, dv);
                end
            end
        end

        function probs = probaKDTree_(obj, X_test)
            M  = size(X_test, 1);
            nC = numel(obj.Classes);
            probs = zeros(M, nC);
            for i = 1:M
                [ids, dists] = obj.Tree_.findKNearest(X_test(i,:), obj.K);
                valid = ids > 0;
                nbY   = obj.y_tr(ids(valid));
                dv    = dists(valid);
                w     = obj.computeWeights_(dv);
                for c = 1:nC
                    probs(i,c) = sum(w(nbY == obj.Classes(c)));
                end
                s = sum(probs(i,:));
                if s > 0, probs(i,:) = probs(i,:) / s; end
            end
        end

        % ---- Brute-force path -------------------------------------------
        function y_pred = predictBrute_(obj, X_test)
            D2 = obj.distMatrix_(X_test);
            K_eff = min(obj.K, size(obj.X_tr, 1));
            [~, idx] = sort(D2, 2);
            nn_idx   = idx(:, 1:K_eff);
            M        = size(X_test, 1);
            y_pred   = zeros(M, 1);
            for i = 1:M
                nbY = obj.y_tr(nn_idx(i,:));
                dv  = sqrt(max(D2(i, nn_idx(i,:)), 0))';
                if strcmpi(obj.Task_, 'classification')
                    y_pred(i) = obj.voteClass_(nbY, dv);
                else
                    y_pred(i) = obj.aggregateReg_(nbY, dv);
                end
            end
        end

        function probs = probaBrute_(obj, X_test)
            D2   = obj.distMatrix_(X_test);
            nC   = numel(obj.Classes);
            M    = size(X_test, 1);
            K_eff = min(obj.K, size(obj.X_tr, 1));
            [~, idx] = sort(D2, 2);
            nn_idx   = idx(:, 1:K_eff);
            probs    = zeros(M, nC);
            nn_y     = reshape(obj.y_tr(nn_idx), M, K_eff);
            for ci = 1:nC
                probs(:, ci) = mean(nn_y == obj.Classes(ci), 2);
            end
        end

        function D2 = distMatrix_(obj, X_test)
            if ~isempty(obj.DistanceFcn)
                D2 = obj.DistanceFcn(X_test, obj.X_tr);
            else
                % Vectorized squared Euclidean
                D2 = sum(X_test.^2,2) + sum(obj.X_tr.^2,2)' ...
                     - 2*(X_test * obj.X_tr');
                D2 = max(D2, 0);
            end
        end

        % ---- Aggregation helpers ----------------------------------------
        function lab = voteClass_(obj, nbY, dists)
            w = obj.computeWeights_(dists);
            nC = numel(obj.Classes);
            votes = zeros(1, nC);
            for c = 1:nC
                votes(c) = sum(w(nbY == obj.Classes(c)));
            end
            [~, idx] = max(votes);
            lab = obj.Classes(idx);
        end

        function val = aggregateReg_(obj, nbY, dists)
            w = obj.computeWeights_(dists);
            val = sum(w .* nbY(:)) / sum(w);
        end

        function w = computeWeights_(obj, dists)
            if strcmpi(obj.Weights, 'distance')
                exact = dists == 0;
                if any(exact), w = double(exact);
                else,          w = 1 ./ dists; end
            else
                w = ones(size(dists));
            end
            w = w(:);
        end

    end
end
