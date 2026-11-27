classdef DataSplitter
    % DATASPLITTER  Reproducible train/test splitting for ML experiments.
    %
    %   Supports three strategies:
    %     'holdout'    — random N × (1-Ratio) / N × Ratio partition (default)
    %     'kfold'      — K non-overlapping folds, zero overlap guaranteed
    %     'stratified' — K-fold preserving class proportions per fold
    %
    %   USAGE:
    %     % Holdout (default: 80/20 split)
    %     sp    = OceanData.DataArch.DataSplitter();
    %     split = sp.split(X, Y);
    %     Xtrain = X(split.trainIdx, :);
    %     Xtest  = X(split.testIdx,  :);
    %
    %     % K-Fold
    %     sp     = OceanData.DataArch.DataSplitter('Strategy','kfold','K',5);
    %     folds  = sp.split(X, Y);   % 1×5 cell array of structs
    %     for k = 1:5
    %       Xtrain = X(folds{k}.trainIdx, :);
    %       Xtest  = X(folds{k}.testIdx,  :);
    %     end
    %
    %     % Stratified K-Fold
    %     sp    = OceanData.DataArch.DataSplitter('Strategy','stratified','K',5);
    %     folds = sp.split(X, Y);
    %
    %   RETURN FORMAT:
    %     Holdout   →  struct with fields: trainIdx, testIdx (logical column vectors)
    %     K-fold    →  1×K cell array of structs: {trainIdx, testIdx}
    %     Stratified→  1×K cell array of structs: {trainIdx, testIdx}
    %
    %   See also: OceanData.DataArch.FeatureStore

    % ================================================================== %
    properties (SetAccess = private)
        Strategy (1,1) string  = "holdout"  % 'holdout' | 'kfold' | 'stratified'
        K        (1,1) double  = 5          % number of folds (kfold / stratified)
        Ratio    (1,1) double  = 0.8        % train proportion (holdout only)
        Seed     (1,1) double  = 42         % RNG seed for reproducibility
    end

    % ================================================================== %
    methods

        function obj = DataSplitter(opts)
            arguments
                opts.Strategy (1,1) string {mustBeMember(opts.Strategy, ...
                    {'holdout','kfold','stratified'})} = "holdout"
                opts.K        (1,1) double {mustBePositiveInteger}  = 5
                opts.Ratio    (1,1) double {mustBeInRange(opts.Ratio,0,1)} = 0.8
                opts.Seed     (1,1) double = 42
            end
            obj.Strategy = opts.Strategy;
            obj.K        = opts.K;
            obj.Ratio    = opts.Ratio;
            obj.Seed     = opts.Seed;
        end

        % -------------------------------------------------------------- %
        function result = split(obj, X, Y)
            % SPLIT  Partition the dataset according to the chosen strategy.
            %
            %   result = sp.split(X, Y)
            %
            %   X — (N × p) feature matrix
            %   Y — (N × 1) label vector (required only for 'stratified')
            %
            %   Returns:
            %     'holdout'    → struct(trainIdx, testIdx)
            %     'kfold'      → 1×K cell of struct(trainIdx, testIdx)
            %     'stratified' → 1×K cell of struct(trainIdx, testIdx)
            arguments
                obj
                X double
                Y = []
            end
            N   = size(X, 1);
            rng(obj.Seed, 'twister');   % reproducible shuffle

            switch obj.Strategy
                case 'holdout'
                    result = obj.holdoutSplit_(N);
                case 'kfold'
                    result = obj.kfoldSplit_(N);
                case 'stratified'
                    if isempty(Y)
                        error('OceanData:DataArch:DataSplitter:MissingLabels', ...
                            'Y is required for stratified splitting.');
                    end
                    result = obj.stratifiedSplit_(N, Y);
            end
        end

        % -------------------------------------------------------------- %
        function describe(obj)
            fprintf('DataSplitter: strategy="%s"', obj.Strategy);
            switch obj.Strategy
                case 'holdout'
                    fprintf('  train=%.0f%%  test=%.0f%%', ...
                        obj.Ratio*100, (1-obj.Ratio)*100);
                case {'kfold','stratified'}
                    fprintf('  K=%d', obj.K);
            end
            fprintf('  seed=%d\n', obj.Seed);
        end

    end

    % ================================================================== %
    methods (Access = private)

        function s = holdoutSplit_(obj, N)
            % Random holdout split.
            perm     = randperm(N);
            nTrain   = round(N * obj.Ratio);
            trainPerm = perm(1:nTrain);
            testPerm  = perm(nTrain+1:end);

            s.trainIdx = obj.toLogical_(trainPerm, N);
            s.testIdx  = obj.toLogical_(testPerm,  N);
            assert(~any(s.trainIdx & s.testIdx), ...
                'DataSplitter: overlap detected in holdout split.');
        end

        % -------------------------------------------------------------- %
        function folds = kfoldSplit_(obj, N)
            % Non-overlapping K-fold split over a randomly permuted index.
            perm  = randperm(N);
            folds = cell(1, obj.K);

            % Assign each sample to exactly one fold
            foldAssign = obj.assignFolds_(N);

            for k = 1:obj.K
                testPerm  = perm(foldAssign == k);
                trainPerm = perm(foldAssign ~= k);

                folds{k}.testIdx  = obj.toLogical_(testPerm,  N);
                folds{k}.trainIdx = obj.toLogical_(trainPerm, N);

                % Zero-overlap guarantee
                assert(~any(folds{k}.trainIdx & folds{k}.testIdx), ...
                    'DataSplitter: overlap detected in fold %d.', k);
                % Coverage guarantee
                assert(sum(folds{k}.trainIdx) + sum(folds{k}.testIdx) == N, ...
                    'DataSplitter: missing samples in fold %d.', k);
            end
        end

        % -------------------------------------------------------------- %
        function folds = stratifiedSplit_(obj, N, Y)
            % Stratified K-fold: preserves class proportions in each fold.
            classes = unique(Y(:));
            folds   = cell(1, obj.K);

            % Initialise empty fold assignment vector
            foldAssign = zeros(N, 1);

            for c = 1:numel(classes)
                classIdx = find(Y(:) == classes(c));
                nc       = numel(classIdx);
                % Shuffle within class
                classIdx = classIdx(randperm(nc));
                % Assign folds within this class (round-robin)
                localAssign = obj.assignFolds_(nc);
                foldAssign(classIdx) = localAssign;
            end

            for k = 1:obj.K
                testMask  = foldAssign == k;
                trainMask = foldAssign ~= k;

                folds{k}.testIdx  = testMask;
                folds{k}.trainIdx = trainMask;

                % Zero-overlap guarantee
                assert(~any(folds{k}.trainIdx & folds{k}.testIdx), ...
                    'DataSplitter: overlap detected in stratified fold %d.', k);
            end
        end

        % -------------------------------------------------------------- %
        function assign = assignFolds_(obj, N)
            % Assign N samples to K folds as evenly as possible (round-robin).
            assign = mod(0:N-1, obj.K)' + 1;   % 1..K cycling
        end

        % -------------------------------------------------------------- %
        function lv = toLogical_(~, idx, N)
            lv       = false(N, 1);
            lv(idx)  = true;
        end

    end
end

% ====================================================================== %
function mustBePositiveInteger(v)
    if ~(isnumeric(v) && isscalar(v) && v >= 2 && floor(v) == v)
        error('DataSplitter:K must be an integer >= 2.');
    end
end
