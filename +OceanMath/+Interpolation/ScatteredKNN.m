classdef ScatteredKNN < OceanMath.Interpolation.IScatteredInterpolant
    % SCATTEREDKNN  N-D scattered interpolation via K-Nearest Neighbours.
    %
    %   Wraps OceanStats.Supervised.Neighbors.KNNClassifier in regression mode.
    %   Supports distance-weighted and uniform averaging.
    %
    %   Usage:
    %     knn = OceanMath.Interpolation.ScatteredKNN('K', 5, 'Weights', 'distance');
    %     knn.fit(X, y);
    %     y_hat = knn.predict(Xq);
    %
    %   Author: Antigravity Code Assistant
    %   Date:   2026-08-18

    properties
        K        (1,1) double {mustBePositive, mustBeInteger} = 5
        Weights  (1,1) string {mustBeMember(Weights, ["uniform","distance"])} = "distance"
        Metric   (1,1) string {mustBeMember(Metric,  ["euclidean","manhattan","chebyshev"])} = "euclidean"
    end

    properties (Access = private)
        Model_   % KNNClassifier instance
    end

    methods
        function obj = ScatteredKNN(opts)
            % SCATTEREDKNN Constructor.
            %
            %   obj = ScatteredKNN('K', 3)
            %   obj = ScatteredKNN('K', 7, 'Weights', 'uniform')

            arguments
                opts.K       (1,1) double {mustBePositive, mustBeInteger} = 5
                opts.Weights (1,1) string = "distance"
                opts.Metric  (1,1) string = "euclidean"
            end
            obj.K       = opts.K;
            obj.Weights = opts.Weights;
            obj.Metric  = opts.Metric;
        end

        function name = getName(obj)
            name = sprintf('KNN-K%d-%s', obj.K, obj.Weights);
        end
    end

    methods (Access = protected)

        function fitCore_(obj, X, y)
            % Store training data in a KNNClassifier set to regression mode.
            obj.Model_ = OceanStats.Supervised.Neighbors.KNNClassifier( ...
                'K',       obj.K, ...
                'Weights', char(obj.Weights), ...
                'Metric',  char(obj.Metric), ...
                'Task',    'regression');
            obj.Model_.fit(X, y);
        end

        function y_hat = predictCore_(obj, Xq)
            y_hat = obj.Model_.predict(Xq);
        end

    end

    methods (Static)
        function meta = getMetadata()
            % GETMETADATA  Return pedagogical metadata for the GUI learner application.
            here = fileparts(mfilename('fullpath'));
            raw  = fileread(fullfile(here, 'meta', 'ScatteredKNN.meta.json'));
            meta = jsondecode(raw);
        end
    end
end

