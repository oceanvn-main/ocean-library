classdef PredictiveImputer < handle
    % PREDICTIVEIMPUTER  Model-based stochastic missing value imputer.
    %
    %   Implements Predictive Mean Matching (PMM) using either OLS or CART.
    %
    %   ALGORITHM (MICE - Multiple Imputation by Chained Equations):
    %     1. Initial fill: replace NaN with column median (temporary)
    %     2. Iterative chain (MaxIters times):
    %        For each column c that has any NaN:
    %          a. Train model f_c(all_other_cols → col_c) on COMPLETE rows only
    %          b. Predict y_hat = f_c(rows with NaN in col_c)
    %          c. Stochastic fill (PMM):
    %             - Find K nearest observed values to y_hat in col_c
    %             - Randomly sample one → this is the imputed value
    %     3. State captures all p per-column models and donor pools
    %
    %   NO PIPELINE DEPENDENCY:
    %     This class is a pure +OceanStats algorithm. It is completely
    %     independent of +OceanPipeline. Use it standalone or via the
    %     OceanPipeline.Transforms.ModelImputer adapter.
    %
    %   USAGE (standalone):
    %     imp   = OceanStats.Data.PredictiveImputer('ModelType','OLS');
    %     state = imp.fit(X_train);
    %     X_clean = imp.apply(X_train, state);
    %     X_test_clean = imp.apply(X_test, state);
    %
    %   See also: OceanStats.Data.PredictiveImputer.apply
    %             OceanPipeline.Transforms.ModelImputer

    properties (SetAccess = private)
        ModelType  (1,1) string  = "OLS"     % "OLS" or "CART"
        Stochastic (1,1) logical = true       % PMM sampling (true) or raw prediction (false)
        MaxIters   (1,1) double  = 10         % MICE iterations
        K          (1,1) double  = 5          % donor pool size for PMM
        Tol        (1,1) double  = 1e-4       % convergence tolerance
    end

    methods
        function obj = PredictiveImputer(opts)
            arguments
                opts.ModelType  (1,1) string  {mustBeMember(opts.ModelType,  {'OLS','CART'})} = "OLS"
                opts.Stochastic (1,1) logical = true
                opts.MaxIters   (1,1) double  = 10
                opts.K          (1,1) double  = 5
                opts.Tol        (1,1) double  = 1e-4
            end
            obj.ModelType  = opts.ModelType;
            obj.Stochastic = opts.Stochastic;
            obj.MaxIters   = opts.MaxIters;
            obj.K          = opts.K;
            obj.Tol        = opts.Tol;
        end

        % ---------------------------------------------------------------- %
        function state = fit(obj, data)
            % FIT  Run MICE on training data. Learn p per-column models.
            %
            %   state.models     — (1 x p) cell: per-column trained model structs
            %   state.donors     — (1 x p) cell: complete observed values for PMM
            %   state.missCols   — (1 x m) indices of columns that had NaN
            %   state.nCols      — p (for validation in apply)
            %   state.modelType  — char
            arguments
                obj
                data double
            end

            [N, p] = size(data);
            missCols = find(any(isnan(data), 1));

            if isempty(missCols)
                % No NaN — trivial state
                state.models    = {};
                state.donors    = {};
                state.missCols  = [];
                state.nCols     = p;
                state.modelType = char(obj.ModelType);
                return;
            end

            % Step 1: Initial fill with column medians
            X = data;
            for c = missCols
                colVals     = data(~isnan(data(:,c)), c);
                X(isnan(X(:,c)), c) = median(colVals);
            end

            % Step 2: Iterative MICE
            prevX = X;
            for iter = 1:obj.MaxIters
                for c = missCols
                    nanRows = find(isnan(data(:,c)));      % rows to impute
                    okRows  = find(~isnan(data(:,c)));     % complete rows for training

                    if isempty(nanRows), continue; end

                    % Predictor columns (all except c)
                    predCols = setdiff(1:p, c);

                    X_train_c = X(okRows,  predCols);   % complete rows, other cols
                    y_train_c = X(okRows,  c);           % complete rows, target col
                    X_pred_c  = X(nanRows, predCols);   % missing rows, other cols

                    % Train model on complete rows
                    model_c = obj.trainModel_(X_train_c, y_train_c);

                    % Predict for missing rows
                    y_hat = obj.predictModel_(model_c, X_pred_c);

                    % Stochastic fill (PMM) or deterministic
                    if obj.Stochastic
                        y_hat = obj.pmm_(y_hat, y_train_c, obj.K);
                    end

                    X(nanRows, c) = y_hat;
                end

                % Check convergence
                delta = max(abs(X(:) - prevX(:)));
                if delta < obj.Tol
                    break;
                end
                prevX = X;
            end

            % Step 3: Train final models on fully-imputed X (for apply())
            models = cell(1, p);
            donors = cell(1, p);
            for c = missCols
                predCols      = setdiff(1:p, c);
                models{c}     = obj.trainModel_(X(:, predCols), X(:, c));
                donors{c}     = data(~isnan(data(:,c)), c);  % original observed values
            end

            state.models    = models;
            state.donors    = donors;
            state.missCols  = missCols;
            state.nCols     = p;
            state.modelType = char(obj.ModelType);
        end

        % ---------------------------------------------------------------- %
        function output = apply(obj, data, state)
            % APPLY  Impute NaN in data using models learned during fit().
            %
            %   For each missing cell:
            %     1. Predict using the per-column model (from training)
            %     2. If Stochastic=true: sample from K nearest donor values
            %     3. Fill the NaN with the sampled value
            arguments
                obj
                data  double
                state struct
            end
            if size(data, 2) ~= state.nCols
                error('OceanStats:Data:PredictiveImputer:DimMismatch', ...
                    'Expected %d columns, got %d.', state.nCols, size(data,2));
            end

            output = data;

            if isempty(state.missCols)
                return;  % no missing columns — pass through
            end

            p = state.nCols;

            for c = state.missCols
                nanRows = find(isnan(output(:, c)));
                if isempty(nanRows), continue; end

                predCols = setdiff(1:p, c);
                X_pred   = output(nanRows, predCols);

                y_hat = obj.predictModel_(state.models{c}, X_pred);

                if obj.Stochastic
                    y_hat = obj.pmm_(y_hat, state.donors{c}, obj.K);
                end

                output(nanRows, c) = y_hat;
            end
        end

    end

    % ================================================================== %
    methods (Access = private)

        function model = trainModel_(obj, X, y)
            % TRAINMODEL_  Train OLS or CART model. Returns portable struct.
            if size(X, 1) == 0 || size(X, 2) == 0
                model.type = 'null';
                model.mu   = mean(y, 'omitnan');
                return;
            end

            switch obj.ModelType
                case 'OLS'
                    % QR-based least squares: [ones X] * beta = y
                    Xa  = [ones(size(X,1),1), X];
                    beta = Xa \ y;
                    rmse = sqrt(mean((Xa*beta - y).^2));
                    model.type = 'OLS';
                    model.beta = beta;
                    model.rmse = rmse;

                case 'CART'
                    % Fit MATLAB regression tree; store trained object
                    tree = fitrtree(X, y, 'MinLeafSize', max(5, floor(size(X,1)*0.02)));
                    model.type = 'CART';
                    model.tree = tree;
            end
        end

        function y_hat = predictModel_(~, model, X)
            % PREDICTMODEL_  Predict using stored model struct.
            if strcmp(model.type, 'null') || isempty(X)
                y_hat = repmat(model.mu, size(X,1), 1);
                return;
            end

            switch model.type
                case 'OLS'
                    Xa    = [ones(size(X,1),1), X];
                    y_hat = Xa * model.beta;

                case 'CART'
                    y_hat = predict(model.tree, X);
            end
        end

        function y_filled = pmm_(obj, y_hat, donors, K)
            % PMM_  Predictive Mean Matching.
            %   For each predicted value, find the K closest observed
            %   donor values and sample one uniformly at random.
            %   This preserves the marginal distribution of the variable.
            %
            %   y_hat  — (M x 1) predicted values for missing rows
            %   donors — (D x 1) observed values from training column
            %   K      — donor pool size

            K_eff    = min(K, numel(donors));
            y_filled = zeros(size(y_hat));

            for i = 1:numel(y_hat)
                % Compute absolute distance to every donor value
                dists = abs(donors - y_hat(i));

                % Find K nearest donors
                [~, idx] = sort(dists);
                pool     = donors(idx(1:K_eff));

                % Sample one uniformly at random
                y_filled(i) = pool(randi(K_eff));
            end
        end

    end
end
