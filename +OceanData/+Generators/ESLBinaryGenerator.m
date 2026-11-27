classdef ESLBinaryGenerator < OceanData.Generators.IDataGenerator
    % ESLBINARYGENERATOR  ESL 13.3.1 binary classification problems.
    %
    % Generates the "easy" and "difficult" simulation problems from ESL
    % Section 13.3.1.  Ten independent features X_j ~ Uniform[0,1].
    %
    %   Easy:      Y = I(X_1 > 0.5)           — hyperplane on feature 1
    %   Difficult: Y = I(sign(prod(X_j-0.5, j=1:3)) > 0) — 3D checkerboard
    %
    % Both problems have Bayes error = 0.
    %
    % USAGE:
    %   gen = OceanData.Generators.ESLBinaryGenerator('N',100,'Problem','easy');
    %   [X, Y] = gen.generate();
    %
    % See also: IDataGenerator, KNNClassifier
    
    properties
        Problem (1,1) string {mustBeMember(Problem, ["easy","difficult"])} = "easy"
        Dims    (1,1) double {mustBePositive, mustBeInteger} = 10
    end
    
    methods
        function obj = ESLBinaryGenerator(opts)
            arguments
                opts.N       (1,1) double = 100
                opts.Problem (1,1) string = "easy"
                opts.Dims    (1,1) double = 10
                opts.Seed    (1,1) double = 0
            end
            obj.N       = opts.N;
            obj.Problem = opts.Problem;
            obj.Dims    = opts.Dims;
            obj.Seed    = opts.Seed;
        end
    end
    
    methods (Access = protected)
        function [X, Y] = generateCore(obj)
            % Generate uniform features
            X = rand(obj.N, obj.Dims);
            
            % Apply decision rule
            if obj.Problem == "easy"
                % Y = I(X_1 > 0.5)
                Y = double(X(:,1) > 0.5);
            else
                % Y = I(sign(prod(X_j - 0.5, j=1:3)) > 0)
                Y = double(sign(prod(X(:,1:3) - 0.5, 2)) > 0);
            end
            
            % Populate metadata
            obj.MetaData.Name        = sprintf('ESL Binary (%s)', obj.Problem);
            obj.MetaData.Problem     = obj.Problem;
            obj.MetaData.BayesError  = 0;
            obj.MetaData.NumFeatures = obj.Dims;
            obj.MetaData.Classes     = [0, 1];
        end
    end
    
    methods
        function tf = hasBayesOracle(~)
            tf = true;
        end
        
        function [y_pred, err, densities] = bayesPredict(obj, X, y_true)
            % BAYESPREDICT  Bayes optimal classifier (error = 0).
            if obj.Problem == "easy"
                y_pred = double(X(:,1) > 0.5);
            else
                y_pred = double(sign(prod(X(:,1:3) - 0.5, 2)) > 0);
            end
            err = mean(y_pred ~= y_true);
            if nargout > 2
                densities = [];
            end
        end
    end
end
