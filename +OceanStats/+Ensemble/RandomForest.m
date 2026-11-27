classdef RandomForest < handle
    % RANDOMFOREST Random Forest ensemble (Breiman, 2001).
    %
    % OVERVIEW:
    %   Trains B decision trees, each on a bootstrap sample of the training
    %   data and each node considering only `Mtry` randomly chosen features.
    %   Predictions are made by majority vote (classification) or mean
    %   (regression).  OOB error and feature importance are computed as
    %   by-products of training at no extra data cost.
    %
    %   Tree training is parallelised with parfor when the Parallel Computing
    %   Toolbox is available AND NumTrees >= MinTreesForParallel (default 20),
    %   as parallel overhead dominates for small forests.
    %
    % USAGE (Classification):
    %   rf = OceanStats.Ensemble.RandomForest('NumTrees', 100, 'Criterion', 'gini');
    %   rf.fit(X_train, Y_train);
    %   Y_pred = rf.predict(X_test);
    %   rf.printSummary();
    %
    % USAGE (Regression):
    %   rf = OceanStats.Ensemble.RandomForest('NumTrees', 200, 'Criterion', 'mse');
    %   rf.fit(X, Y);
    %
    % REFERENCE:
    %   Breiman (2001). Random Forests. Machine Learning, 45(1), 5-32.
    %   ESL §15 (Hastie, Tibshirani, Friedman).
    %
    % Author: Ocean Library
    % Date:   2026-05-12

    % =====================================================================
    %  Properties
    % =====================================================================
    properties
        %--- Hyperparameters ---
        NumTrees  (1,1) double   = 100   % Number of trees in the forest
        Mtry      (1,1) double   = 0     % Features per node (0 = auto)
        MaxDepth  (1,1) double   = Inf   % Inf = grow fully (RF default)
        MinLeafSize (1,1) double = 1     % Minimum samples per leaf
        SubsampleFraction (1,1) double {mustBeGreaterThan(SubsampleFraction, 0), mustBeLessThanOrEqual(SubsampleFraction, 1)} = 1.0

        % Criterion: 'gini' | 'entropy' | 'misclassification' (classif)
        %             'regression' | 'mse'                       (regress)
        Criterion string = 'gini'

        %--- Options ---
        ComputeOOB        (1,1) logical = true   % Compute OOB error
        ComputeImportance (1,1) logical = true   % Compute feature importance
        ImportanceMethod  string        = 'MDA'  % 'MDA' or 'MDI'
        Seed              (1,1) double  = 0      % 0 = no seed
        MinTreesForParallel (1,1) double = 20    % Threshold to use parfor
        Verbose           (1,1) logical = false
    end

    properties (SetAccess = private)
        %--- Fitted state (read-only after fit) ---
        Trees_       cell    % {1 x B} fitted DecisionTree objects
        OOBIndices_  cell    % {1 x B} OOB row index vectors
        OOBError_    double  % Scalar OOB error
        Importance_  double  % (1 x p) feature importance scores
        Classes_     double  % Unique class labels (classification)
        Task_        string  % 'classification' | 'regression'
        IsFitted_    logical = false
        P_           double  % Number of features (set at fit-time)
    end

    % =====================================================================
    %  Public API
    % =====================================================================
    methods

        function obj = RandomForest(options)
            % RANDOMFOREST Constructor.
            %
            % Name-Value Args:
            %   NumTrees   (double): Number of trees. Default 100.
            %   Mtry       (double): Features sampled per node. Default 0 (auto).
            %   MaxDepth   (double): Max tree depth. Default Inf (full tree).
            %   MinLeafSize(double): Min leaf size. Default 1.
            %   Criterion  (string): Impurity criterion. Default 'gini'.
            %   ComputeOOB (logical): Compute OOB error. Default true.
            %   ComputeImportance (logical): Compute importance. Default true.
            %   ImportanceMethod  (string): 'MDA' or 'MDI'. Default 'MDA'.
            %   Seed       (double): RNG seed (0 = no seed). Default 0.
            %   Verbose    (logical): Print progress. Default false.

            arguments
                options.NumTrees           (1,1) double  = 100
                options.Mtry               (1,1) double  = 0
                options.MaxDepth           (1,1) double  = Inf
                options.MinLeafSize        (1,1) double  = 1
                options.Criterion          string        = 'gini'
                options.ComputeOOB         (1,1) logical = true
                options.ComputeImportance  (1,1) logical = true
                options.ImportanceMethod   string        = 'MDA'
                options.SubsampleFraction  (1,1) double  = 1.0
                options.Seed               (1,1) double  = 0
                options.MinTreesForParallel (1,1) double = 20
                options.Verbose            (1,1) logical = false
            end

            obj.NumTrees            = options.NumTrees;
            obj.Mtry                = options.Mtry;
            obj.MaxDepth            = options.MaxDepth;
            obj.MinLeafSize         = options.MinLeafSize;
            obj.Criterion           = options.Criterion;
            obj.ComputeOOB          = options.ComputeOOB;
            obj.ComputeImportance   = options.ComputeImportance;
            obj.ImportanceMethod    = options.ImportanceMethod;
            obj.SubsampleFraction   = options.SubsampleFraction;
            obj.Seed                = options.Seed;
            obj.MinTreesForParallel = options.MinTreesForParallel;
            obj.Verbose             = options.Verbose;
        end

        % -----------------------------------------------------------------
        function fit(obj, X, Y)
            % FIT Train the Random Forest on data (X, Y).
            %
            % Args:
            %   X (N x p double): Feature matrix.
            %   Y (N x 1):        Labels (categorical/double) or responses.

            [N, p] = size(X);
            obj.P_ = p;

            % Seed RNG for reproducibility
            if obj.Seed ~= 0
                rng(obj.Seed);
            end

            % Resolve task type
            obj.Task_    = obj.resolveTask_();
            obj.Classes_ = [];
            if strcmpi(obj.Task_, 'classification')
                obj.Classes_ = unique(Y);
            end

            % Resolve auto-Mtry
            mtry_eff = obj.resolveMtry_(p);

            K = max(1, numel(obj.Classes_));   % 1 for regression

            if obj.Verbose
                fprintf('[RandomForest] Task=%s  B=%d  mtry=%d  N=%d  p=%d\n', ...
                    obj.Task_, obj.NumTrees, mtry_eff, N, p);
            end

            % ----------------------------------------------------------
            % Tree Training (parallel if worth it)
            % ----------------------------------------------------------
            useParallel = obj.shouldParallelize_();

            trees_out   = cell(1, obj.NumTrees);
            oob_out     = cell(1, obj.NumTrees);

            if useParallel
                if obj.Verbose
                    fprintf('[RandomForest] Using parfor (%d trees).\n', obj.NumTrees);
                end
                % Pre-generate all bootstrap index arrays on the main thread
                % (parfor workers cannot share RNG state)
                inBag_all  = cell(1, obj.NumTrees);
                outBag_all = cell(1, obj.NumTrees);
                for b = 1:obj.NumTrees
                    [inBag_all{b}, outBag_all{b}] = obj.bootstrapIndices_(N); 
                end

                % Snapshot hyperparams for broadcast to workers
                maxD  = obj.MaxDepth;
                minL  = obj.MinLeafSize;
                crit  = obj.Criterion;
                mtry  = mtry_eff;

                parfor b = 1:obj.NumTrees
                    trees_out{b} = OceanStats.Ensemble.RandomForest.trainOneTree_( ...
                        X, Y, inBag_all{b}, maxD, minL, crit, mtry);
                    oob_out{b}   = outBag_all{b};
                end
            else
                for b = 1:obj.NumTrees
                    [inBag, outBag] = obj.bootstrapIndices_(N);
                    trees_out{b}    = RandomForest.trainOneTree_( ...
                        X, Y, inBag, obj.MaxDepth, obj.MinLeafSize, ...
                        obj.Criterion, mtry_eff);
                    oob_out{b}      = outBag;

                    if obj.Verbose && mod(b, max(1, floor(obj.NumTrees/10))) == 0
                        fprintf('[RandomForest] Tree %d / %d done.\n', b, obj.NumTrees);
                    end
                end
            end

            obj.Trees_      = trees_out;
            obj.OOBIndices_ = oob_out;

            % ----------------------------------------------------------
            % OOB Error
            % ----------------------------------------------------------
            if obj.ComputeOOB
                tracker = OceanStats.Ensemble.OOBTracker(N, K);
                for b = 1:obj.NumTrees
                    tracker.accumulate(trees_out{b}, oob_out{b}, X, obj.Task_);
                end
                obj.OOBError_ = tracker.oobError(Y, obj.Task_, obj.Classes_);
                if obj.Verbose
                    fprintf('[RandomForest] OOB Error = %.4f\n', obj.OOBError_);
                end
            end

            % ----------------------------------------------------------
            % Feature Importance
            % ----------------------------------------------------------
            if obj.ComputeImportance
                if strcmpi(obj.ImportanceMethod, 'MDA')
                    obj.Importance_ = OceanStats.Ensemble.ImportanceCalculator.mda( ...
                        trees_out, oob_out, X, Y, obj.Task_, obj.Classes_);
                else
                    obj.Importance_ = OceanStats.Ensemble.ImportanceCalculator.mdi( ...
                        trees_out, p);
                end
            end

            obj.IsFitted_ = true;
        end

        % -----------------------------------------------------------------
        function Y_pred = predict(obj, X)
            % PREDICT Aggregate predictions (majority vote / mean).
            %
            % Args:
            %   X (M x p double): Feature matrix.
            %
            % Returns:
            %   Y_pred (M x 1): Hard class labels or regression values.

            obj.assertFitted_();
            [M, ~] = size(X);

            if strcmpi(obj.Task_, 'classification')
                P = obj.predictProba(X);          % M x K
                [~, idx] = max(P, [], 2);
                Y_pred   = obj.Classes_(idx);
            else
                acc = zeros(M, 1);
                for b = 1:obj.NumTrees
                    acc = acc + obj.Trees_{b}.predict(X);
                end
                Y_pred = acc / obj.NumTrees;
            end
        end

        % -----------------------------------------------------------------
        function P = predictProba(obj, X)
            % PREDICTPROBA Soft-vote class probability matrix.
            %
            % Args:
            %   X (M x p double): Feature matrix.
            %
            % Returns:
            %   P (M x K double): Row-normalised class probability matrix.

            obj.assertFitted_();
            if strcmpi(obj.Task_, 'regression')
                error('RandomForest:NotClassification', ...
                    'predictProba is only available for classification tasks.');
            end

            M = size(X, 1);
            K = numel(obj.Classes_);
            P = zeros(M, K);

            for b = 1:obj.NumTrees
                P = P + obj.Trees_{b}.predictProb(X);
            end

            P = P / obj.NumTrees;   % normalise to [0,1], rows sum to 1
        end

        % -----------------------------------------------------------------
        function err = oobError(obj)
            % OOBERROR Return the computed OOB error (scalar).
            obj.assertFitted_();
            if ~obj.ComputeOOB
                error('RandomForest:OOBNotComputed', ...
                    'Set ComputeOOB=true before calling fit().');
            end
            err = obj.OOBError_;
        end

        % -----------------------------------------------------------------
        function scores = featureImportance(obj)
            % FEATUREIMPORTANCE Return the (1 x p) importance vector.
            obj.assertFitted_();
            if ~obj.ComputeImportance
                error('RandomForest:ImportanceNotComputed', ...
                    'Set ComputeImportance=true before calling fit().');
            end
            scores = obj.Importance_;
        end

        % -----------------------------------------------------------------
        function printSummary(obj)
            % PRINTSUMMARY Print a formatted training summary.
            obj.assertFitted_();
            fprintf('\n=== Random Forest Summary ===\n');
            fprintf('  Task        : %s\n', obj.Task_);
            fprintf('  Trees       : %d\n', obj.NumTrees);
            fprintf('  Mtry (eff)  : %d\n', obj.resolveMtry_(obj.P_));
            fprintf('  Max Depth   : %s\n', num2str(obj.MaxDepth));
            fprintf('  Min Leaf    : %d\n', obj.MinLeafSize);
            fprintf('  Criterion   : %s\n', obj.Criterion);
            if obj.ComputeOOB && ~isnan(obj.OOBError_)
                if strcmpi(obj.Task_, 'classification')
                    fprintf('  OOB Error   : %.4f  (%.2f%%)\n', ...
                        obj.OOBError_, obj.OOBError_ * 100);
                else
                    fprintf('  OOB MSE     : %.6f\n', obj.OOBError_);
                end
            end
            if obj.ComputeImportance
                [~, ord] = sort(obj.Importance_, 'descend');
                top_n = min(5, obj.P_);
                fprintf('  Top %d features (by %s):\n', top_n, obj.ImportanceMethod);
                for k = 1:top_n
                    fprintf('    Feature %3d : %.4f\n', ord(k), obj.Importance_(ord(k)));
                end
            end
            fprintf('=============================\n\n');
        end

    end

    % =====================================================================
    %  Private helpers
    % =====================================================================
    methods (Access = private)

        function obj = assertFitted_(obj)
            if ~obj.IsFitted_
                error('RandomForest:NotFitted', ...
                    'Call fit() before making predictions.');
            end
        end

        function task = resolveTask_(obj)
            classifCriteria = {'gini','entropy','misclassification','crossentropy'};
            if any(strcmpi(obj.Criterion, classifCriteria))
                task = 'classification';
            else
                task = 'regression';
            end
        end

        function mtry = resolveMtry_(obj, p) 
            if obj.Mtry <= 0 %default sqrt(p)
                if strcmpi(obj.Task_, 'classification')
                    mtry = max(1, floor(sqrt(p)));
                else
                    mtry = max(1, floor(p / 3));
                end
            else
                mtry = min(obj.Mtry, p);
            end
        end

        function use = shouldParallelize_(obj)
            % Use parfor only when:
            %   (a) Parallel Computing Toolbox is installed.
            %   (b) NumTrees >= MinTreesForParallel (overhead not worth it below).
            use = false;
            if obj.NumTrees < obj.MinTreesForParallel
                return;
            end
            try
                p = gcp('nocreate');
                if isempty(p)
                    % Try to open a pool — will throw if toolbox absent
                    parpool('local', 'SpmdEnabled', false);
                end
                use = true;
            catch
                use = false;   % Toolbox not available — silent fallback
            end
        end

        function [inBag, outBag] = bootstrapIndices_(obj, N)
            % Draw N samples with replacement (Bootstrap) OR nSub without replacement
            if obj.SubsampleFraction == 1.0
                inBag  = randi(N, N, 1);
            else
                %Draw without replacement. Still accurate with less 50% data.
                nSub  = max(1, floor(obj.SubsampleFraction * N));
                inBag = randperm(N, nSub)';
            end
            outBag = setdiff(1:N, inBag)';
        end

    end

    % =====================================================================
    %  Private static helpers
    % =====================================================================
    methods (Static, Access = private)

        function tree = trainOneTree_(X, Y, inBagIdx, maxDepth, minLeaf, criterion, mtry)
            % TRAINONETREE_ Build and fit one bootstrap tree.
            % Static so it can be called inside parfor without capturing obj.

            gen  = OceanStats.Supervised.Tree.RandomFeatureCandidateGenerator(mtry);

            % Handle Inf MaxDepth — DecisionTree stores as double, Inf is valid
            tree = OceanStats.Supervised.Tree.DecisionTree( ...
                'MaxDepth',    maxDepth, ...
                'MinLeafSize', minLeaf, ...
                'Criterion',   criterion, ...
                'CandidateGenerator', gen);

            tree.fit(X(inBagIdx, :), Y(inBagIdx));
        end

    end
end
