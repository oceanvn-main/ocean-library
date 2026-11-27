classdef InterpolationCV < handle
    % INTERPOLATIONCV  K-fold cross-validation tuner for IScatteredInterpolant.
    %
    %   Sweeps a single scalar hyper-parameter and selects the value that
    %   minimises mean CV RMSE on the training data. Does NOT use test data.
    %
    %   Supported tuning modes:
    %     'rbf-epsilon'  : tunes ScatteredRBF Epsilon (Gaussian/MQ shape)
    %     'rbf-lambda'   : tunes ScatteredRBF Lambda  (regularisation)
    %     'knn-k'        : tunes ScatteredKNN K
    %     'spline-lambda': tunes ScatteredSpline Lambda
    %
    %   Usage:
    %     cv = OceanStats.Interpolation.InterpolationCV( ...
    %               'ModelType',  'rbf', ...
    %               'TuneParam',  'rbf-epsilon', ...
    %               'ParamGrid',  logspace(-2, 1, 15), ...
    %               'KFolds',     5, ...
    %               'KernelName', 'gaussian');
    %
    %     cv.fit(X_train, Y_train);
    %     fprintf('Best epsilon: %.4f\n', cv.BestParam);
    %     best_model = cv.BestModel;
    %
    %   Author: Antigravity Code Assistant
    %   Date:   2026-08-18

    properties
        ModelType   (1,1) string = "rbf"
        TuneParam   (1,1) string = "rbf-epsilon"
        ParamGrid   (1,:) double = logspace(-2, 1, 20)
        KFolds      (1,1) double {mustBePositive, mustBeInteger} = 5
        KernelName  (1,1) string = "gaussian"   % for rbf-epsilon / rbf-lambda
        Verbose     (1,1) logical = false
    end

    properties (SetAccess = private)
        BestParam   (1,1) double
        BestModel         % IScatteredInterpolant, fitted on full training set
        CVRmse      (:,1) double    % mean CV RMSE for each ParamGrid value
        CVStd       (:,1) double    % std across folds
    end

    methods
        function obj = InterpolationCV(opts)
            arguments
                opts.ModelType   (1,1) string = "rbf"
                opts.TuneParam   (1,1) string = "rbf-epsilon"
                opts.ParamGrid   (1,:) double = logspace(-2, 1, 20)
                opts.KFolds      (1,1) double = 5
                opts.KernelName  (1,1) string = "gaussian"
                opts.Verbose     (1,1) logical = false
            end
            obj.ModelType  = opts.ModelType;
            obj.TuneParam  = opts.TuneParam;
            obj.ParamGrid  = opts.ParamGrid;
            obj.KFolds     = opts.KFolds;
            obj.KernelName = opts.KernelName;
            obj.Verbose    = opts.Verbose;
        end

        function fit(obj, X, y)
            % FIT  Run K-fold CV sweep and refit the best model on all data.
            %
            %   cv.fit(X_train, Y_train)

            arguments
                obj
                X (:,:) double
                y (:,1) double
            end

            N   = size(X, 1);
            K   = obj.KFolds;
            P   = numel(obj.ParamGrid);

            % --- Build fold indices (stratified random split) ----------------
            idx   = randperm(N);
            foldSz = floor(N / K);
            folds  = cell(K, 1);
            for k = 1:K
                if k < K
                    folds{k} = idx((k-1)*foldSz + 1 : k*foldSz);
                else
                    folds{k} = idx((k-1)*foldSz + 1 : end);
                end
            end

            % --- Sweep parameter grid ----------------------------------------
            rmse_mat = zeros(P, K);

            for p = 1:P
                param = obj.ParamGrid(p);

                for k = 1:K
                    val_idx   = folds{k};
                    train_idx = setdiff(1:N, val_idx);

                    X_tr = X(train_idx, :);  y_tr = y(train_idx);
                    X_va = X(val_idx, :);    y_va = y(val_idx);

                    mdl = obj.buildModel_(param);
                    try
                        mdl.fit(X_tr, y_tr);
                        y_hat = mdl.predict(X_va);
                        rmse_mat(p, k) = sqrt(mean((y_hat - y_va).^2));
                    catch ME
                        % Ill-conditioned system for this parameter — penalise
                        if obj.Verbose
                            fprintf('  [CV] param=%.4g fold=%d failed: %s\n', ...
                                param, k, ME.message);
                        end
                        rmse_mat(p, k) = Inf;
                    end
                end

                if obj.Verbose
                    fprintf('  [CV] param=%.4g  mean RMSE=%.5f\n', ...
                        param, mean(rmse_mat(p, :)));
                end
            end

            obj.CVRmse = mean(rmse_mat, 2);
            obj.CVStd  = std(rmse_mat, 0, 2);

            % --- Select best parameter and refit on full data ----------------
            [~, bestIdx]  = min(obj.CVRmse);
            obj.BestParam = obj.ParamGrid(bestIdx);
            obj.BestModel = obj.buildModel_(obj.BestParam);
            obj.BestModel.fit(X, y);

            if obj.Verbose
                fprintf('[CV] Best param = %.4g  (mean CV RMSE = %.5f)\n', ...
                    obj.BestParam, obj.CVRmse(bestIdx));
            end
        end

        function plotCurve(obj, ax)
            % PLOTCURVE  Plot CV RMSE ± std vs parameter grid (log-scale x).
            %
            %   cv.plotCurve(ax)

            arguments
                obj
                ax = gca()
            end

            semilogx(ax, obj.ParamGrid, obj.CVRmse, 'o-', ...
                'Color', [0.2 0.7 1], 'LineWidth', 1.5, 'MarkerSize', 5); hold(ax, 'on');

            % Std ribbon
            patch(ax, [obj.ParamGrid, fliplr(obj.ParamGrid)], ...
                  [obj.CVRmse' + obj.CVStd', fliplr(obj.CVRmse' - obj.CVStd')], ...
                  [0.2 0.7 1], 'FaceAlpha', 0.15, 'EdgeColor', 'none');

            % Best param marker
            [minVal, bestIdx] = min(obj.CVRmse);
            plot(ax, obj.ParamGrid(bestIdx), minVal, 'r*', ...
                'MarkerSize', 12, 'DisplayName', ...
                sprintf('Best=%.3g', obj.ParamGrid(bestIdx)));

            xlabel(ax, obj.TuneParam, 'FontSize', 11, 'Color', [0.9 0.9 0.9]);
            ylabel(ax, 'CV RMSE', 'FontSize', 11, 'Color', [0.9 0.9 0.9]);
            title(ax, sprintf('%d-Fold CV: %s', obj.KFolds, obj.TuneParam), 'FontSize', 11, 'Color', [0.9 0.9 0.9]);
            legend(ax, {'CV RMSE', '± 1 std', 'Best'}, 'Location', 'north', ...
                'TextColor', [0.9 0.9 0.9], 'Color', [0.09 0.09 0.12], 'EdgeColor', [0.3 0.3 0.35]);
            grid(ax, 'on');
        end

    end

    methods (Access = private)
        function mdl = buildModel_(obj, param)
            % Construct a fresh model with the given parameter value.
            import OceanMath.Interpolation.ScatteredInterpolant
            switch lower(obj.TuneParam)
                case 'rbf-epsilon'
                    mdl = ScatteredInterpolant('Method', 'rbf', 'Kernel', obj.KernelName, 'Epsilon', param);
                case 'rbf-lambda'
                    mdl = ScatteredInterpolant('Method', 'rbf', 'Kernel', obj.KernelName, 'Lambda', param);
                case 'knn-k'
                    mdl = ScatteredInterpolant('Method', 'knn', 'K', round(param));
                case 'spline-lambda'
                    mdl = ScatteredInterpolant('Method', 'spline', 'Lambda', param);
                otherwise
                    error('InterpolationCV:UnknownTuneParam', ...
                        'Unknown TuneParam: "%s".', obj.TuneParam);
            end
        end
    end
end
