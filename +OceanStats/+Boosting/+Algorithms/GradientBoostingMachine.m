classdef GradientBoostingMachine < handle
    % GRADIENTBOOSTINGMACHINE  Friedman's (2001) Gradient Boosted Trees.
    %
    %   OVERVIEW:
    %     Implements Algorithm 10.3 (regression/binary) and Algorithm 10.4
    %     (multiclass MART) from ESL (Hastie, Tibshirani, Friedman).
    %     An additive ensemble of regression trees, each trained sequentially
    %     to correct the residuals of the current ensemble using the gradient
    %     of any injected ILoss function.
    %
    %   STOCHASTIC GBM / ISLE EXTENSION (Friedman 2002, Friedman & Hall 2007):
    %     Setting SubsampleFraction (eta) < 1 enables the ISLE "randomness knob".
    %     Each tree is trained on eta*N rows sampled WITHOUT replacement.
    %     The F running tally is still updated using all N samples.
    %     Recommended ISLE recipe: LearningRate=0.1, SubsampleFraction<=0.5
    %
    %   REGRESSION / BINARY PATH (single tree per round):
    %     Loss.initPrediction() returns scalar → F is N×1.
    %     Trees_ is M×1 cell. predict() returns N×1.
    %
    %   MULTICLASS MART PATH (K trees per round):
    %     Loss.initPrediction() returns K×1 (K > 1) → F is N×K.
    %     Trees_ is M×K cell. predict() returns N×K probabilities (softmax).
    %     Use MulticlassDevianceLoss for ESL §10.14.3 replication.
    %
    %   STRATEGY PATTERN:
    %     Loss is injected via constructor. Any ILoss implementation works:
    %       MSELoss                → standard least-squares boosting
    %       MAELoss                → L1-boosting (robust, leaf = median)
    %       ClassicHuberLoss       → textbook Huber (Friedman 2001, ESL §10.14)
    %       PseudoHuberLoss        → smooth Huber, Newton-step boosting
    %       MulticlassDevianceLoss → MART K-class (ESL §10.10, Algorithm 10.4)
    %
    %   USAGE:
    %     loss = OceanStats.Loss.Regression.ClassicHuberLoss('Delta', 1.35);
    %     gbm  = OceanStats.Boosting.Algorithms.GradientBoostingMachine(...
    %                'Loss', loss, 'NumTrees', 100, 'LearningRate', 0.1);
    %     gbm.fit(X_train, Y_train);
    %     Y_hat = gbm.predict(X_test);
    %
    %   See also: OceanStats.Loss.Regression.ClassicHuberLoss
    %             OceanStats.Loss.Classification.MulticlassDevianceLoss

    properties
        % Loss  ILoss strategy — defines pseudo-residuals and leaf constants.
        % Type is enforced in the constructor arguments block (ILoss is abstract,
        % so we cannot annotate it here directly without a concrete default).
        Loss

        % NumTrees  M — number of boosting rounds. Default: 100.
        NumTrees double = 100

        % LearningRate  nu — shrinkage applied to each tree. Default: 0.1.
        % Lower values require more trees but generalize better.
        % ISLE memory knob: large nu -> trees are diverse (Boosting);
        %                   small nu -> trees overlap (sequential memorizing).
        LearningRate double = 0.1

        % SubsampleFraction  eta — fraction of training rows used per tree.
        % Sampled WITHOUT replacement (Friedman & Hall 2007).
        %   eta = 1.0  → standard GBM (full dataset, default — backward compatible)
        %   eta = 0.5  → Stochastic GBM / ISLE optimal (halves compute per tree)
        %   eta < 0.5  → maximum diversity (equivalent to RF-style row dropping)
        % ISLE proof: eta=0.5 w/o replacement ≈ eta=1.0 w/ replacement statistically,
        % but costs 50% less compute per boosting round.
        SubsampleFraction double = 1.0

        % MaxDepth  Maximum depth of each base tree. Default: 3.
        % Shallow trees control interactions; depth 1 = additive model (no interactions).
        MaxDepth double = 3

        % MinLeafSize  Minimum samples per leaf. Default: 5.
        MinLeafSize double = 5

        % CandidateGenerator — strategy for generating split candidates per node.
        % Default: ExhaustiveCandidateGenerator (all unique values).
        % Set to QuantileCandidateGenerator(k) for large N to drastically reduce
        % training time with negligible loss in split quality.
        CandidateGenerator = []

        % OutputMode  Execution path selector — mirrors XGBoost's objective routing.
        %   'auto'        — auto-detected from Loss type and K at fit() time (default)
        %   'regression'  — raw score output, no final transform
        %   'binary'      — 1 tree/round, sigmoid applied at predict() → probability
        %   'multiclass'  — K trees/round, softmax applied at predict() → N×K probs
        %
        %   Prefer explicit declaration in production code:
        %     gbm = GBM('Loss', CrossEntropyLoss(), 'OutputMode', 'binary');
        OutputMode string {mustBeMember(OutputMode, ...
            {'auto','regression','binary','multiclass'})} = 'auto'
    end

    properties (SetAccess = private)
        % F0  Initial constant prediction (scalar), set by Loss.initPrediction().
        F0_ double

        % Trees_  Cell array of fitted DecisionTree base learners.
        Trees_ cell

        % TrainLoss_  M×1 vector of training loss per boosting round.
        TrainLoss_ double

        IsFitted_  logical = false

        % Mode_  Resolved execution path (set during fit, read by predict).
        % One of: 'regression' | 'binary' | 'multiclass'
        Mode_      string  = 'regression'

        % Verbose  If true, print progress to console. Default true.
        Verbose    logical = true

        % MaxFeatures  Number of features to consider per split (mtry).
        %   0 or 'all' = all features.
        %   'sqrt'     = floor(sqrt(p)).
        %   number > 0 = exact number of features.
        MaxFeatures
    end

    methods
        function obj = GradientBoostingMachine(options)
            % GRADIENTBOOSTINGMACHINE  Constructor.
            %
            %   gbm = GradientBoostingMachine('Loss', MSELoss(), 'NumTrees', 100, ...)
            arguments
                options.Loss               OceanStats.Loss.ITreeLoss = OceanStats.Loss.Regression.MSELoss()
                options.NumTrees           double {mustBePositive, mustBeInteger} = 100
                options.LearningRate       double {mustBePositive} = 0.1
                options.SubsampleFraction  double {mustBeGreaterThan(options.SubsampleFraction, 0), mustBeLessThanOrEqual(options.SubsampleFraction, 1)} = 1.0
                options.MaxDepth           double {mustBePositive}  = 3
                options.MinLeafSize        double {mustBePositive}  = 5
                options.CandidateGenerator = []
                options.OutputMode string {mustBeMember(options.OutputMode, ...
                    {'auto','regression','binary','multiclass'})} = 'auto'
                options.Verbose            logical = true
                options.MaxFeatures        = 0
            end
            obj.Loss               = options.Loss;
            obj.NumTrees           = options.NumTrees;
            obj.LearningRate       = options.LearningRate;
            obj.SubsampleFraction  = options.SubsampleFraction;
            obj.MaxDepth           = options.MaxDepth;
            obj.MinLeafSize        = options.MinLeafSize;
            obj.CandidateGenerator = options.CandidateGenerator;
            obj.OutputMode         = options.OutputMode;
            obj.Verbose            = options.Verbose;
            obj.MaxFeatures        = options.MaxFeatures;
        end

        function fit(obj, X, Y)
            % FIT  Train the GBM on feature matrix X and targets Y.
            %
            %   gbm.fit(X, Y)
            %
            %   Inputs:
            %     X — N×p double feature matrix
            %     Y — N×1 regression targets OR N×1 integer class labels (1..K)
            %
            %   Automatically detects multiclass when Loss.initPrediction()
            %   returns a K×1 vector with K>1, and switches to MART mode
            %   (Algorithm 10.4): K trees per round, F is N×K.

            N   = size(X, 1);
            M   = obj.NumTrees;
            nu  = obj.LearningRate;
            eta = obj.SubsampleFraction;

            % ---- Step 0: Resolve MaxFeatures & Candidate Generator ----
            p = size(X, 2);
            mtry = obj.resolveMtry_(p);
            
            % If the user provided a custom CandidateGenerator, use it.
            % Otherwise, if mtry < p, use RandomFeatureCandidateGenerator.
            actualGenerator = obj.CandidateGenerator;
            if isempty(actualGenerator) && mtry < p
                actualGenerator = OceanStats.Supervised.Tree.RandomFeatureCandidateGenerator(mtry);
            end

            % ---- Step 1: Initialize F_0 ----
            obj.F0_  = obj.Loss.initPrediction(Y);   % scalar OR K×1
            K        = numel(obj.F0_);               % 1 = regression/binary, >1 = multiclass
            F        = repmat(obj.F0_(:)', N, 1);    % N×1 or N×K

            % ----------------------------------------------------------------
            % ROUTING: Resolve execution mode (binary / regression / multiclass)
            % Mirrors the XGBoost objective routing pattern:
            %   K=1 + classification loss → 'binary'   (1 tree/round, sigmoid at predict)
            %   K=1 + regression loss     → 'regression' (1 tree/round, raw score)
            %   K>1                       → 'multiclass'  (K trees/round, softmax at predict)
            % ----------------------------------------------------------------
            if strcmp(obj.OutputMode, 'auto')
                if K > 1
                    obj.Mode_ = 'multiclass';
                elseif obj.isClassificationLoss_()
                    obj.Mode_ = 'binary';      % 1 tree/round — antisymmetry exploited
                else
                    obj.Mode_ = 'regression';
                end
            else
                obj.Mode_ = obj.OutputMode;    % user-declared (explicit, preferred)
            end
            if obj.Verbose
                fprintf('  OutputMode = %s\n', obj.Mode_);
            end

            obj.Trees_     = cell(M, K);             % M×K — K=1 for regression and binary
            obj.TrainLoss_ = zeros(M, 1);

            % ISLE mode flag — controls log label
            isStochastic = eta < 1.0;
            if isStochastic
                modeLabel = sprintf('Stochastic-GBM/ISLE, eta=%.2f', eta);
            else
                modeLabel = 'Standard-GBM';
            end
            if obj.Verbose
                fprintf('Training GBM [%s] (%s, M=%d, nu=%.2f, depth=%d, K=%d)\n', ...
                    modeLabel, obj.Loss.name(), M, nu, obj.MaxDepth, K);
            end

            for m = 1:M
                % ================================================================
                % ISLE: Row subsampling WITHOUT replacement (eta knob).
                % When eta=1.0: subIdx = all N rows → identical to standard GBM.
                % When eta<1.0: a fresh random subset is drawn each round,
                %   providing geometric diversity across trees (Friedman 2002).
                % F is always updated using ALL N samples after each tree.
                % ================================================================
                [subIdx, X_sub, Y_sub, F_sub] = obj.subsampleRows_(X, Y, F, eta, N);

                % ---- Step 2a: Pseudo-Residuals = -gradient (on subsample) ----
                R = -obj.Loss.gradient(Y_sub, F_sub);  % nSub×1 or nSub×K

                % ---- Steps 2b-2c: Fit one tree per class column ----
                for k = 1:K
                    r_k = R(:, k);   % nSub×1 target for this class

                    tree = OceanStats.Supervised.Tree.DecisionTree( ...
                        'MaxDepth',    obj.MaxDepth, ...
                        'MinLeafSize', obj.MinLeafSize, ...
                        'Criterion',   'regression', ...
                        'CandidateGenerator', actualGenerator);
                    tree.fit(X_sub, r_k);

                    % ---- Step 2c: Leaf Optimization (on subsample) ----
                    [leafIds, nLeaves] = tree.getLeafIds(X_sub);
                    gamma = zeros(nLeaves, 1);
                    for j = 1:nLeaves
                        leafMask = (leafIds == j);
                        lc = obj.Loss.computeLeafConstant(Y_sub, F_sub, leafMask);
                        if K == 1
                            gamma(j) = lc;       % scalar
                        else
                            gamma(j) = lc(k);    % K×1 → scalar for class k
                        end
                    end
                    tree.refitLeafValues(gamma);

                    % ---- Update ALL N scores using full-data tree prediction ----
                    % NOTE: tree was fit on X_sub, but predict() generalises to full X.
                    % This is correct per Friedman (2002) §3: subsample only affects
                    % which rows train the tree, not how F accumulates.
                    obj.Trees_{m, k} = tree;
                    F(:, k) = F(:, k) + nu * tree.predict(X);
                end

                % ---- Track Training Loss (full dataset) ----
                obj.TrainLoss_(m) = obj.Loss.forward(Y, F);

                if obj.Verbose && mod(m, max(1, floor(M/5))) == 0
                    fprintf('  m=%3d  TrainLoss=%.4f\n', m, obj.TrainLoss_(m));
                end
            end

            obj.IsFitted_ = true;
            if obj.Verbose
                fprintf('Done. Final training loss: %.4f\n', obj.TrainLoss_(end));
            end
        end

        function Y_pred = predict(obj, X)
            % PREDICT  Generate predictions for new feature matrix X.
            %
            %   Regression / binary:  Y_pred → N×1 raw scores
            %   Multiclass (K>1):     Y_pred → N×K softmax probabilities
            if ~obj.IsFitted_
                error('GBM:NotFitted', 'Call fit(X, Y) before predict.');
            end
            N      = size(X, 1);
            [M, K] = size(obj.Trees_);
            Y_pred = repmat(obj.F0_(:)', N, 1);   % N×1 or N×K

            for m = 1:M
                for k = 1:K
                    Y_pred(:, k) = Y_pred(:, k) + ...
                        obj.LearningRate * obj.Trees_{m, k}.predict(X);
                end
            end

            % ----------------------------------------------------------------
            % OUTPUT TRANSFORM — mirrors XGBoost routing:
            %   'binary'     → sigmoid(logit)   p = 1 / (1 + exp(-f))
            %   'multicast'  → softmax(logits)  p_k = exp(f_k)/sum(exp(f))
            %   'regression' → raw score (no transform)
            % ----------------------------------------------------------------
            switch obj.Mode_
                case 'binary'
                    Y_pred = 1 ./ (1 + exp(-Y_pred));   % sigmoid — N×1 probability
                case 'multiclass'
                    Z      = Y_pred - max(Y_pred, [], 2);  % log-sum-exp stabilize
                    E      = exp(Z);
                    Y_pred = E ./ sum(E, 2);               % N×K softmax probabilities
                % 'regression' — passthrough, no transform
            end
        end

        function Y_staged = stagedPredict(obj, X)
            % STAGEDPREDICT  Return the cumulative raw prediction at every boosting round.
            %
            %   Returns:
            %     Y_staged : N x M matrix (for regression/binary), where column m
            %                is the raw score F_m(X) at iteration m.
            %     For multiclass, returns an N x K x M tensor.
            if ~obj.IsFitted_
                error('GBM:NotFitted', 'Call fit(X, Y) before predict.');
            end
            N = size(X, 1);
            [M, K] = size(obj.Trees_);
            
            if K == 1
                Y_staged = zeros(N, M);
                curr_F = repmat(obj.F0_(:)', N, 1);
                for m = 1:M
                    curr_F = curr_F + obj.LearningRate * obj.Trees_{m, 1}.predict(X);
                    Y_staged(:, m) = curr_F;
                end
            else
                Y_staged = zeros(N, K, M);
                curr_F = repmat(obj.F0_(:)', N, 1);
                for m = 1:M
                    for k = 1:K
                        curr_F(:, k) = curr_F(:, k) + obj.LearningRate * obj.Trees_{m, k}.predict(X);
                    end
                    Y_staged(:, :, m) = curr_F;
                end
            end
        end


        function L = trainLoss(obj)
            % TRAINLOSS  Return the M×1 training loss curve.
            L = obj.TrainLoss_;
        end

        function fi = featureImportance(obj, p)
            % FEATUREIMPORTANCE  Sum of gain across all splits per feature.
            %
            %   fi = gbm.featureImportance(p)   → p×1 normalized importance scores
            %
            %   Inputs:
            %     p — number of features (so we know the output size)
            %
            %   Works for both regression (M×1 Trees_) and multiclass (M×K Trees_).
            fi = zeros(p, 1);
            [nM, nK] = size(obj.Trees_);
            for m = 1:nM
                for k = 1:nK
                    fi = fi + obj.gatherImportance_(obj.Trees_{m, k}.RootNode, p);
                end
            end
            total = sum(fi);
            if total > 0, fi = fi / total; end
        end
    end

    methods (Access = private)
        function [subIdx, X_sub, Y_sub, F_sub] = subsampleRows_(~, X, Y, F, eta, N)
            % SUBSAMPLEROWS_  Draw a random row subsample WITHOUT replacement.
            %
            %   When eta = 1.0, returns all N rows unchanged (zero overhead).
            %   When eta < 1.0, draws floor(eta*N) unique row indices via
            %   randperm — strictly no-replacement, matching the ISLE paper.
            %
            %   Inputs:
            %     X   — N×p feature matrix
            %     Y   — N×1 or N×K target matrix
            %     F   — N×1 or N×K running score matrix
            %     eta — SubsampleFraction scalar in (0, 1]
            %     N   — number of training samples
            %
            %   Outputs:
            %     subIdx — nSub×1 row index vector
            %     X_sub  — nSub×p
            %     Y_sub  — nSub×1 or nSub×K
            %     F_sub  — nSub×1 or nSub×K
            if eta >= 1.0
                subIdx = (1:N)';
            else
                nSub   = max(1, floor(eta * N));   % at least 1 sample
                subIdx = randperm(N, nSub)';
            end
            X_sub = X(subIdx, :);
            Y_sub = Y(subIdx, :);
            F_sub = F(subIdx, :);
        end

        function fi = gatherImportance_(obj, node, p)
            % Accumulate variance reduction at each internal split.
            fi = zeros(p, 1);
            if node.IsLeaf, return; end
            % node.Split is a SplitResult object with FeatureIndex property
            featJ = node.Split.FeatureIndex;
            if featJ >= 1 && featJ <= p
                fi(featJ) = fi(featJ) + node.Risk;
            end
            for c = 1:length(node.Children)
                fi = fi + obj.gatherImportance_(node.Children{c}, p);
            end
        end

        function tf = isClassificationLoss_(obj)
            % ISCLASSIFICATIONLOSS_  Detect if the injected Loss is a binary classifier.
            %
            %   Used by auto-routing in fit() to distinguish:
            %     regression loss  → raw score output
            %     binary loss      → sigmoid applied at predict()
            %
            %   Strategy: check if the fully qualified class name contains
            %   'Classification' (all binary losses live in +Classification/).
            %   E.g.: 'OceanStats.Loss.Classification.CrossEntropyLoss' → true
            %         'OceanStats.Loss.Regression.MSELoss'               → false
            lossClass = class(obj.Loss);
            tf = contains(lossClass, 'Classification') && ...
                 ~contains(lossClass, 'Multiclass');
            % Multiclass losses are handled via K>1 path, not sigmoid
        end

        function mtry = resolveMtry_(obj, p)
            if ischar(obj.MaxFeatures) || isstring(obj.MaxFeatures)
                val = lower(string(obj.MaxFeatures));
                if val == "sqrt" || val == "auto"
                    mtry = max(1, floor(sqrt(p)));
                elseif val == "all"
                    mtry = p;
                else
                    error('GBM:InvalidMaxFeatures', 'Unknown MaxFeatures string: %s', val);
                end
            elseif isnumeric(obj.MaxFeatures)
                if obj.MaxFeatures <= 0
                    mtry = p;
                elseif obj.MaxFeatures < 1
                    mtry = max(1, floor(obj.MaxFeatures * p));
                else
                    mtry = min(p, floor(obj.MaxFeatures));
                end
            else
                mtry = p;
            end
        end
    end
end
