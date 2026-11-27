classdef KNNStrategy < OceanStats.Base.IModelStrategy
    % KNNSTRATEGY IModelStrategy adapter for KNNClassifier.
    %
    % OVERVIEW:
    %   Exposes KNNClassifier through the IModelStrategy contract so it works
    %   with CrossValidator, BiasVarianceAuditor, ComplexitySweeper, etc.
    %   ModelFactory.create('knn') returns a default instance.
    %
    %   getComplexity() returns K so ComplexitySweeper can sweep K values.
    %
    % USAGE:
    %   strat = OceanStats.Selection.Strategies.KNNStrategy('K', 5);
    %   model = strat.train(X_train, Y_train);
    %   preds = strat.predict(model, X_test);
    %   err   = strat.evaluate(Y_test, preds);
    %
    % Author: Ocean Library
    % Date:   2026-05-12

    properties
        Name        string  = "KNN"
        SupportsAIC logical = false

        K        double = 5
        Metric   string = 'euclidean'
        Weights  string = 'uniform'
        Task     string = 'auto'
        LeafSize double = 10
    end

    methods
        function obj = KNNStrategy(options)
            arguments
                options.K        double = 5
                options.Metric   string = 'euclidean'
                options.Weights  string = 'uniform'
                options.Task     string = 'auto'
                options.LeafSize double = 10
            end
            obj.K        = options.K;
            obj.Metric   = options.Metric;
            obj.Weights  = options.Weights;
            obj.Task     = options.Task;
            obj.LeafSize = options.LeafSize;
            obj.Name     = sprintf("KNN (K=%d, %s, %s)", ...
                obj.K, obj.Metric, obj.Weights);
        end

        function model = train(obj, X, Y, ~)
            knn = OceanStats.Supervised.Neighbors.KNNClassifier( ...
                'K',        obj.K, ...
                'Metric',   obj.Metric, ...
                'Weights',  obj.Weights, ...
                'Task',     obj.Task, ...
                'LeafSize', obj.LeafSize);
            knn.fit(X, Y);
            model = knn;
        end

        function preds = predict(~, model, X)
            % Binary classification: return soft prob of positive class
            if strcmpi(model.Task_, 'classification') && numel(model.Classes) == 2
                P     = model.predictProba(X);
                preds = P(:, 2);
            else
                preds = model.predict(X);
            end
        end

        function err = evaluate(obj, Y_true, Y_pred)
            if strcmpi(obj.Task, 'regression')
                err = mean((Y_true - Y_pred).^2);
            else
                Y_hard = round(Y_pred);
                err    = mean(Y_true ~= Y_hard);
            end
        end

        function k = getComplexity(~, model)
            % K is the complexity parameter for ComplexitySweeper
            k = model.K;
        end
    end
end
