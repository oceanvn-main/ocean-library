classdef ISLEForest < OceanStats.Base.IModelStrategy
    % ISLEFOREST Unified Orchestrator for Importance Sampling Learning Ensembles.
    %
    % OVERVIEW:
    %   A grand unification facade (Friedman & Hall, 2007) that maps a 2D 
    %   hyperparameter space (Shrinkage/nu, SubsampleFraction/eta) to the 
    %   correct underlying ensemble tree builder.
    %
    %   Mathematical mappings:
    %   nu / v = learning rate(memory), eta / n= subsampling
    %   - nu=0, eta=1.0  => Bagging (via RandomForest with mtry=p, bootstrap)
    %   - nu=0, eta<1.0  => Subsampled Bagging (via RandomForest with subsampling)
    %   - nu>0, eta=1.0  => Standard Gradient Boosting Machine (GBM)
    %   - nu>0, eta<1.0  => Stochastic GBM (ISLE optimal region)
    %
    % USAGE:
    %   % ISLE Optimal Stochastic Boosting
    %   forest = OceanStats.Ensemble.ISLEForest('Shrinkage', 0.1, ...
    %                                           'SubsampleFraction', 0.5, ...
    %                                           'NumTrees', 150);
    %   forest.fit(X, Y);
    %   y_pred = forest.predict(X);
    %
    %   % Bagging Approximation
    %   forest = OceanStats.Ensemble.ISLEForest('Shrinkage', 0.0);
    %   forest.fit(X, Y);
    %
    % Author: Ocean Library
    % Date: 2026-05-14

    properties
        Shrinkage          double = 0.1
        SubsampleFraction  double {mustBeGreaterThan(SubsampleFraction, 0), mustBeLessThanOrEqual(SubsampleFraction, 1)} = 0.5
        NumTrees           double {mustBePositive, mustBeInteger} = 100
        MaxFeatures               = 'auto'   % 0/'auto'/'sqrt'/number
        MaxDepth           double {mustBePositive} = 3
        MinLeafSize        double {mustBePositive} = 5
        Loss               % For GBM mode
        
        Verbose            logical = false
    end

    properties (SetAccess = private)
        UnderlyingModel_  % The delegated RandomForest or GradientBoostingMachine
        IsFitted_         logical = false
        Mode_             string  % 'RandomForest' or 'GradientBoostingMachine'
    end

    properties (Constant)
        Name = "ISLE Unified Forest";
        SupportsAIC = false;
    end

    methods
        function obj = ISLEForest(options)
            arguments
                options.Shrinkage         double = 0.1
                options.SubsampleFraction double = 0.5
                options.NumTrees          double = 100
                options.MaxFeatures              = 'auto'
                options.MaxDepth          double = 3
                options.MinLeafSize       double = 5
                options.Loss                     = []
                options.Verbose           logical = false
            end
            
            obj.Shrinkage         = options.Shrinkage;
            obj.SubsampleFraction = options.SubsampleFraction;
            obj.NumTrees          = options.NumTrees;
            obj.MaxFeatures       = options.MaxFeatures;
            obj.MaxDepth          = options.MaxDepth;
            obj.MinLeafSize       = options.MinLeafSize;
            obj.Verbose           = options.Verbose;
            
            if isempty(options.Loss)
                obj.Loss = OceanStats.Loss.Regression.MSELoss();
            else
                obj.Loss = options.Loss;
            end
        end

        % --- IModelStrategy Interface Methods ---
        function model = train(obj, X, Y, ~)
            obj.fit(X, Y);
            model = obj;  % Self-reference to satisfy interface while being stateful
        end

        function preds = predict(obj, ~, X)
            % IModelStrategy signature
            preds = obj.predict_(X);
        end

        function err = evaluate(~, Y_true, Y_pred)
            % Default evaluation (MSE for demo, can be overridden)
            err = mean((Y_true(:) - Y_pred(:)).^2);
        end

        function k = getComplexity(obj, ~)
            k = obj.NumTrees * (2^obj.MaxDepth - 1);
        end
        % ----------------------------------------

        function fit(obj, X, Y)
            % FIT Trains the underlying ensemble based on the knob coordinates.
            if obj.Shrinkage == 0 %No memory
                obj.Mode_ = 'RandomForest';
                if obj.Verbose
                    fprintf('[ISLEForest] Routing to RandomForest (nu=0, eta=%.2f)\n', obj.SubsampleFraction);
                end
                
                % Determine criterion based on loss object if possible
                crit = 'mse';
                if contains(class(obj.Loss), 'Classification')
                    crit = 'gini';
                end
                
                % Translate MaxFeatures
                mtry = obj.MaxFeatures;
                if ischar(mtry) || isstring(mtry)
                    if strcmpi(mtry, 'auto') || strcmpi(mtry, 'sqrt')
                        mtry = 0; % RandomForest handles 0 as auto
                    elseif strcmpi(mtry, 'all')
                        mtry = size(X, 2);
                    end
                end
                
                obj.UnderlyingModel_ = OceanStats.Ensemble.RandomForest( ...
                    'NumTrees',          obj.NumTrees, ...
                    'SubsampleFraction', obj.SubsampleFraction, ...
                    'Mtry',              mtry, ...
                    'MaxDepth',          obj.MaxDepth, ...
                    'MinLeafSize',       obj.MinLeafSize, ...
                    'Criterion',         crit, ...
                    'Verbose',           obj.Verbose);
                
                obj.UnderlyingModel_.fit(X, Y);
                
            else
                obj.Mode_ = 'GradientBoostingMachine';
                if obj.Verbose
                    fprintf('[ISLEForest] Routing to GBM (nu=%.4f, eta=%.2f)\n', ...
                        obj.Shrinkage, obj.SubsampleFraction);
                end
                
                obj.UnderlyingModel_ = OceanStats.Boosting.Algorithms.GradientBoostingMachine( ...
                    'Loss',              obj.Loss, ...
                    'NumTrees',          obj.NumTrees, ...
                    'LearningRate',      obj.Shrinkage, ...
                    'SubsampleFraction', obj.SubsampleFraction, ...
                    'MaxFeatures',       obj.MaxFeatures, ...
                    'MaxDepth',          obj.MaxDepth, ...
                    'MinLeafSize',       obj.MinLeafSize, ...
                    'Verbose',           obj.Verbose);
                
                obj.UnderlyingModel_.fit(X, Y);
            end
            
            obj.IsFitted_ = true;
        end

        function Y_pred = predict_(obj, X)
            % Internal predict logic mapped to both signatures
            if ~obj.IsFitted_
                error('ISLEForest:NotFitted', 'Call fit() before predicting.');
            end
            Y_pred = obj.UnderlyingModel_.predict(X);
        end
        
        function P = predictProba(obj, X)
            % PREDICTPROBA Output class probabilities (Classification only).
            if ~obj.IsFitted_
                error('ISLEForest:NotFitted', 'Call fit() before predicting.');
            end
            
            if strcmp(obj.Mode_, 'RandomForest')
                P = obj.UnderlyingModel_.predictProba(X);
            else
                % GBM naturally returns probabilities if Multiclass/Binary 
                % loss was used, but predict() handles the sigmoid/softmax.
                % We call predict() and assume the GBM Loss routing was correct.
                P = obj.UnderlyingModel_.predict(X);
                % If Binary OutputMode was forced, P might be Nx1.
                % Usually, cross-entropy returns NxK.
            end
        end

        function stats = getKnobs(obj)
            % GETKNOBS Returns the active configuration coordinates.
            stats = struct(...
                'Shrinkage', obj.Shrinkage, ...
                'SubsampleFraction', obj.SubsampleFraction, ...
                'MaxFeatures', obj.MaxFeatures, ...
                'NumTrees', obj.NumTrees, ...
                'Algorithm', obj.Mode_ ...
            );
        end
    end
end
