classdef ExtraTreesForest < handle
    % EXTRATREESFOREST Extremely Randomized Trees ensemble (Geurts et al., 2006).
    %
    % OVERVIEW:
    %   Extends the Random Forest idea with two key differences:
    %     1. No bootstrap (default): each tree trains on the FULL dataset.
    %        This reduces variance from subsampling and lets the algorithm
    %        focus its randomization entirely on the split selection.
    %     2. Split thresholds are RANDOM (not optimal): for each of the `Mtry`
    %        randomly selected features, one threshold is drawn uniformly from
    %        the observed feature range at that node — no search needed.
    %
    %   Combined effect vs Random Forest:
    %     - Faster training (no threshold search; no bootstrap sampling overhead)
    %     - Higher bias per tree, but lower variance of the ensemble average
    %     - Often competitive in accuracy; virtually eliminates overfitting
    %
    %   Parallelism: same parfor threshold logic as RandomForest.
    %   Importance: defaults to MDI (since MDA requires OOB or a holdout set,
    %               and OOB is unavailable when UseBootstrap=false).
    %               Use UseBootstrap=true to enable MDA.
    %
    % USAGE (Classification):
    %   et = OceanStats.Ensemble.ExtraTreesForest('NumTrees', 100);
    %   et.fit(X_train, Y_train);
    %   Y_pred = et.predict(X_test);
    %   et.printSummary();
    %
    % USAGE (Regression):
    %   et = OceanStats.Ensemble.ExtraTreesForest( ...
    %       'NumTrees', 200, 'Criterion', 'regression');
    %   et.fit(X, Y);
    %
    % REFERENCE:
    %   Geurts, Ernst & Wehenkel (2006). Extremely Randomized Trees.
    %   Machine Learning, 63(1), 3-42.
    %
    % Author: Ocean Library
    % Date:   2026-05-12

    % =====================================================================
    %  Properties
    % =====================================================================
    properties
        NumTrees        (1,1) double  = 100     % Number of trees
        Mtry            (1,1) double  = 0       % Features per node (0 = auto)
        NumCandidates   (1,1) double  = 1       % Random thresholds per feature (K in paper)
        MaxDepth        (1,1) double  = Inf     % Inf = grow fully (default)
        MinLeafSize     (1,1) double  = 2       % Geurts (2006) uses min_samples=2

        % Criterion: 'gini' | 'entropy' | 'misclassification' (classif)
        %             'regression'                               (regress)
        Criterion       string        = 'gini'

        % UseBootstrap: false (canonical ExtraTrees) or true (enables OOB)
        UseBootstrap    (1,1) logical = false

        ComputeOOB        (1,1) logical = false  % true only valid when UseBootstrap=true
        ComputeImportance (1,1) logical = true
        ImportanceMethod  string        = 'MDI'  % 'MDI' (default) or 'MDA' (needs bootstrap)
        Seed              (1,1) double  = 0
        MinTreesForParallel (1,1) double = 20
        Verbose           (1,1) logical = false
    end

    properties (SetAccess = private)
        Trees_      cell    % {1 x B} fitted DecisionTree objects
        OOBIndices_ cell    % {1 x B} OOB index vectors (empty when UseBootstrap=false)
        OOBError_   double
        Importance_ double  % (1 x p)
        Classes_    double
        Task_       string
        IsFitted_   logical = false
        P_          double
    end

    % =====================================================================
    %  Public API
    % =====================================================================
    methods

        function obj = ExtraTreesForest(options)
            % EXTRATREESFOREST Constructor.
            %
            % Name-Value Args:
            %   NumTrees        (double): Trees. Default 100.
            %   Mtry            (double): Features per node. Default 0 (auto sqrt(p)).
            %   NumCandidates   (double): Random thresholds per feature. Default 1.
            %   MaxDepth        (double): Max depth. Default Inf.
            %   MinLeafSize     (double): Min samples per leaf. Default 2.
            %   Criterion       (string): 'gini'|'entropy'|'regression'. Default 'gini'.
            %   UseBootstrap    (logical): Bootstrap sampling. Default false.
            %   ComputeOOB      (logical): Requires UseBootstrap=true. Default false.
            %   ComputeImportance (logical): Compute importance. Default true.
            %   ImportanceMethod  (string): 'MDI'|'MDA'. Default 'MDI'.
            %   Seed            (double): RNG seed (0=no seed). Default 0.
            %   Verbose         (logical): Print progress. Default false.

            arguments
                options.NumTrees          (1,1) double  = 100
                options.Mtry              (1,1) double  = 0
                options.NumCandidates     (1,1) double  = 1
                options.MaxDepth          (1,1) double  = Inf
                options.MinLeafSize       (1,1) double  = 2
                options.Criterion         string        = 'gini'
                options.UseBootstrap      (1,1) logical = false
                options.ComputeOOB        (1,1) logical = false
                options.ComputeImportance (1,1) logical = true
                options.ImportanceMethod  string        = 'MDI'
                options.Seed              (1,1) double  = 0
                options.MinTreesForParallel (1,1) double = 20
                options.Verbose           (1,1) logical = false
            end

            obj.NumTrees          = options.NumTrees;
            obj.Mtry              = options.Mtry;
            obj.NumCandidates     = options.NumCandidates;
            obj.MaxDepth          = options.MaxDepth;
            obj.MinLeafSize       = options.MinLeafSize;
            obj.Criterion         = options.Criterion;
            obj.UseBootstrap      = options.UseBootstrap;
            obj.ComputeOOB        = options.ComputeOOB;
            obj.ComputeImportance = options.ComputeImportance;
            obj.ImportanceMethod  = options.ImportanceMethod;
            obj.Seed              = options.Seed;
            obj.MinTreesForParallel = options.MinTreesForParallel;
            obj.Verbose           = options.Verbose;

            % Guard: OOB requires bootstrap
            if obj.ComputeOOB && ~obj.UseBootstrap
                error('ExtraTreesForest:OOBRequiresBootstrap', ...
                    'ComputeOOB=true requires UseBootstrap=true.');
            end

            % Guard: MDA requires OOB
            if strcmpi(obj.ImportanceMethod, 'MDA') && ~obj.UseBootstrap
                error('ExtraTreesForest:MDARequiresBootstrap', ...
                    ['MDA importance requires OOB samples. ' ...
                     'Set UseBootstrap=true or use ImportanceMethod=''MDI''.']);
            end
        end

        % -----------------------------------------------------------------
        function fit(obj, X, Y)
            % FIT Train the Extra-Trees forest on (X, Y).

            [N, p] = size(X);
            obj.P_ = p;

            if obj.Seed ~= 0, rng(obj.Seed); end

            obj.Task_    = obj.resolveTask_();
            obj.Classes_ = [];
            if strcmpi(obj.Task_, 'classification')
                obj.Classes_ = unique(Y);
            end

            mtry_eff = obj.resolveMtry_(p);
            K        = max(1, numel(obj.Classes_));

            if obj.Verbose
                fprintf('[ExtraTreesForest] Task=%s  B=%d  mtry=%d  K=%d  N=%d  p=%d\n', ...
                    obj.Task_, obj.NumTrees, mtry_eff, obj.NumCandidates, N, p);
                if ~obj.UseBootstrap
                    fprintf('[ExtraTreesForest] Bootstrap=OFF (full dataset per tree)\n');
                end
            end

            % ----------------------------------------------------------
            % Prepare sample indices per tree
            % ----------------------------------------------------------
            inBag_all  = cell(1, obj.NumTrees);
            outBag_all = cell(1, obj.NumTrees);
            for b = 1:obj.NumTrees
                if obj.UseBootstrap
                    [inBag_all{b}, outBag_all{b}] = obj.sampleIndices_(N);
                else
                    inBag_all{b}  = (1:N)';     % full dataset
                    outBag_all{b} = [];          % no OOB
                end
            end

            % ----------------------------------------------------------
            % Tree Training
            % ----------------------------------------------------------
            useParallel = obj.shouldParallelize_();
            trees_out   = cell(1, obj.NumTrees);

            % Snapshot scalars for parfor broadcast
            maxD   = obj.MaxDepth;
            minL   = obj.MinLeafSize;
            crit   = obj.Criterion;
            mtry   = mtry_eff;
            nCand  = obj.NumCandidates;

            if useParallel
                if obj.Verbose
                    fprintf('[ExtraTreesForest] Using parfor (%d trees).\n', obj.NumTrees);
                end
                parfor b = 1:obj.NumTrees
                    trees_out{b} = ExtraTreesForest.trainOneTree_( ...
                        X, Y, inBag_all{b}, maxD, minL, crit, mtry, nCand);
                end
            else
                for b = 1:obj.NumTrees
                    trees_out{b} = ExtraTreesForest.trainOneTree_( ...
                        X, Y, inBag_all{b}, maxD, minL, crit, mtry, nCand);
                    if obj.Verbose && mod(b, max(1, floor(obj.NumTrees/10))) == 0
                        fprintf('[ExtraTreesForest] Tree %d / %d done.\n', b, obj.NumTrees);
                    end
                end
            end

            obj.Trees_      = trees_out;
            obj.OOBIndices_ = outBag_all;

            % ----------------------------------------------------------
            % OOB Error (bootstrap mode only)
            % ----------------------------------------------------------
            if obj.ComputeOOB && obj.UseBootstrap
                tracker = OceanStats.Ensemble.OOBTracker(N, K);
                for b = 1:obj.NumTrees
                    tracker.accumulate(trees_out{b}, outBag_all{b}, X, obj.Task_);
                end
                obj.OOBError_ = tracker.oobError(Y, obj.Task_, obj.Classes_);
                if obj.Verbose
                    fprintf('[ExtraTreesForest] OOB Error = %.4f\n', obj.OOBError_);
                end
            end

            % ----------------------------------------------------------
            % Feature Importance
            % ----------------------------------------------------------
            if obj.ComputeImportance
                if strcmpi(obj.ImportanceMethod, 'MDA')
                    obj.Importance_ = OceanStats.Ensemble.ImportanceCalculator.mda( ...
                        trees_out, outBag_all, X, Y, obj.Task_, obj.Classes_);
                else
                    obj.Importance_ = OceanStats.Ensemble.ImportanceCalculator.mdi( ...
                        trees_out, p);
                end
            end

            obj.IsFitted_ = true;
        end

        % -----------------------------------------------------------------
        function Y_pred = predict(obj, X)
            % PREDICT Majority vote (classification) or mean (regression).
            obj.assertFitted_();
            [M, ~] = size(X);

            if strcmpi(obj.Task_, 'classification')
                P = obj.predictProba(X);
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
            % PREDICTPROBA Soft-vote probability matrix (classification only).
            obj.assertFitted_();
            if strcmpi(obj.Task_, 'regression')
                error('ExtraTreesForest:NotClassification', ...
                    'predictProba is only available for classification tasks.');
            end
            M = size(X, 1);
            K = numel(obj.Classes_);
            P = zeros(M, K);
            for b = 1:obj.NumTrees
                P = P + obj.Trees_{b}.predictProb(X);
            end
            P = P / obj.NumTrees;
        end

        % -----------------------------------------------------------------
        function err = oobError(obj)
            % OOBERROR OOB error (requires UseBootstrap=true and ComputeOOB=true).
            obj.assertFitted_();
            if ~obj.ComputeOOB
                error('ExtraTreesForest:OOBNotComputed', ...
                    'Set UseBootstrap=true and ComputeOOB=true before fit().');
            end
            err = obj.OOBError_;
        end

        % -----------------------------------------------------------------
        function scores = featureImportance(obj)
            % FEATUREIMPORTANCE Return (1 x p) importance vector.
            obj.assertFitted_();
            if ~obj.ComputeImportance
                error('ExtraTreesForest:ImportanceNotComputed', ...
                    'Set ComputeImportance=true before fit().');
            end
            scores = obj.Importance_;
        end

        % -----------------------------------------------------------------
        function printSummary(obj)
            % PRINTSUMMARY Formatted training summary.
            obj.assertFitted_();
            fprintf('\n=== Extra-Trees Forest Summary ===\n');
            fprintf('  Task         : %s\n', obj.Task_);
            fprintf('  Trees        : %d\n', obj.NumTrees);
            fprintf('  Mtry (eff)   : %d\n', obj.resolveMtry_(obj.P_));
            fprintf('  K (thresholds): %d\n', obj.NumCandidates);
            fprintf('  Max Depth    : %s\n', num2str(obj.MaxDepth));
            fprintf('  Min Leaf     : %d\n', obj.MinLeafSize);
            fprintf('  Criterion    : %s\n', obj.Criterion);
            fprintf('  Bootstrap    : %s\n', mat2str(obj.UseBootstrap));
            if obj.ComputeOOB && obj.UseBootstrap && ~isnan(obj.OOBError_)
                fprintf('  OOB Error    : %.4f  (%.2f%%)\n', ...
                    obj.OOBError_, obj.OOBError_ * 100);
            end
            if obj.ComputeImportance
                [~, ord] = sort(obj.Importance_, 'descend');
                top_n = min(5, obj.P_);
                fprintf('  Top %d features (by %s):\n', top_n, obj.ImportanceMethod);
                for k = 1:top_n
                    fprintf('    Feature %3d : %.4f\n', ord(k), obj.Importance_(ord(k)));
                end
            end
            fprintf('==================================\n\n');
        end

    end

    % =====================================================================
    %  Private helpers
    % =====================================================================
    methods (Access = private)

        function assertFitted_(obj)
            if ~obj.IsFitted_
                error('ExtraTreesForest:NotFitted', 'Call fit() first.');
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
            if obj.Mtry <= 0
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
            use = false;
            if obj.NumTrees < obj.MinTreesForParallel, return; end
            try
                pool = gcp('nocreate');
                if isempty(pool)
                    parpool('local', 'SpmdEnabled', false);
                end
                use = true;
            catch
                use = false;
            end
        end

        function [inBag, outBag] = sampleIndices_(~, N)
            inBag  = randi(N, N, 1);
            outBag = setdiff(1:N, inBag)';
        end

    end

    % =====================================================================
    %  Private static helpers
    % =====================================================================
    methods (Static, Access = private)

        function tree = trainOneTree_(X, Y, inBagIdx, maxDepth, minLeaf, ...
                                     criterion, mtry, numCandidates)
            % TRAINONETREE_ Build and fit one ExtraTrees tree.
            % Static — safe for parfor without capturing obj.

            gen  = OceanStats.Supervised.Tree.ExtraTreesCandidateGenerator( ...
                mtry, numCandidates);

            tree = OceanStats.Supervised.Tree.DecisionTree( ...
                'MaxDepth',          maxDepth, ...
                'MinLeafSize',       minLeaf, ...
                'Criterion',         criterion, ...
                'CandidateGenerator', gen);

            tree.fit(X(inBagIdx, :), Y(inBagIdx));
        end

    end
end
