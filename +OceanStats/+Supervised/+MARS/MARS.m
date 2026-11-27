classdef MARS < handle
    % MARS Multivariate Adaptive Regression Splines
    %
    % Implementation based on ESL Chapter 9.4. Fully strategy-based:
    % no conditional logic inside fit() or predict() — all behaviour
    % is delegated to the four injectable strategies.
    %
    % INJECTABLE STRATEGIES:
    %   Encoder      (IDataEncoder)   — how Y is encoded before fitting
    %   ForwardPass  (IForwardPass)   — how the model grows (RSS or log-likelihood)
    %   BackwardPass (IBackwardPass)  — how the model is pruned
    %
    % USAGE — Standard MARS (regression):
    %   m = OceanStats.Supervised.MARS.MARS();
    %   m.fit(X, Y);
    %
    % USAGE — PolyMARS (multinomial classification):
    %   m = OceanStats.Supervised.MARS.MARS();
    %   m.ForwardPass = OceanStats.Selection.FeatureSelection.PolyMARSForwardSelector();
    %   m.fit(X, Y);
    %
    % USAGE — Swap pruning criterion:
    %   m.BackwardPass.Criterion = MyAICCriterion();
    %
    % USAGE — Manual encoder override:
    %   m.Encoder = OceanStats.Data.IndicatorEncoder();   % force classification
    
    properties
        MaxTerms              (1,1) double = 21
        MaxInteractionDegree  (1,1) double = 1
        PenaltyCost           (1,1) double = 3.0
        CategoricalFeatures   (1,:) double = []
        
        % Core Strategies (all injectable)
        % Encoder is auto-assigned in fit() if not set by the user.
        Encoder      = []  % IDataEncoder — set to IndicatorEncoder or RawEncoder
        ForwardPass  OceanStats.Base.IForwardPass  = OceanStats.Selection.FeatureSelection.ForwardBasisSelector()
        BackwardPass OceanStats.Base.IBackwardPass = OceanStats.Selection.FeatureSelection.GCVBackwardSelector()
        
        % Final Model State (set after fit)
        BasisFunctions cell
        Coefficients   double
    end
    
    methods
        function obj = MARS(varargin)
            if nargin > 0
                for i = 1:2:length(varargin)
                    obj.(varargin{i}) = varargin{i+1};
                end
            end
        end
        
        function fit(obj, X, Y)
            % FIT Encode Y, grow model, prune to optimal size.
            
            opts = struct('MaxTerms',            obj.MaxTerms, ...
                          'MaxInteractionDegree', obj.MaxInteractionDegree, ...
                          'PenaltyCost',          obj.PenaltyCost, ...
                          'CategoricalFeatures',  obj.CategoricalFeatures);
            
            % Auto-detect and assign Encoder if not injected by user
            if isempty(obj.Encoder)
                if OceanStats.Data.IndicatorEncoder.canEncode(Y)
                    obj.Encoder = OceanStats.Data.IndicatorEncoder();
                else
                    obj.Encoder = OceanStats.Data.RawEncoder();
                end
            end
            
            % Encode Y via the strategy (no if-else in MARS)
            Y_enc = obj.Encoder.encode(Y);
            
            % Forward: grow large model
            grownBasis = obj.ForwardPass.build(X, Y_enc, opts);
            
            % Backward: prune to optimal size
            [obj.BasisFunctions, obj.Coefficients, ~] = ...
                obj.BackwardPass.prune(X, Y_enc, grownBasis, opts);
        end
        
        function y_pred = predict(obj, X)
            % PREDICT Return predictions in original output space.
            scores = obj.computeScores(X);
            y_pred = obj.Encoder.decode(scores);
        end
        
        function P = predictProba(obj, X)
            % PREDICTPROBA (N x K) probability matrix via Softmax.
            % Only meaningful for classification encoders.
            if ~obj.Encoder.isClassification()
                error('MARS:predictProba:NotClassification', ...
                    'predictProba requires a classification encoder.');
            end
            Z  = obj.computeScores(X);
            Z  = Z - max(Z, [], 2);
            eZ = exp(Z);
            P  = eZ ./ sum(eZ, 2);
        end
        
        function printModel(obj, featNames)
            fprintf('MARS Model (Interaction Degree: %d)\n', obj.MaxInteractionDegree);
            fprintf('-------------------------------------------\n');
            if isempty(obj.BasisFunctions)
                fprintf('Model not fitted.\n'); return;
            end
            for m = 1:length(obj.BasisFunctions)
                coef = obj.Coefficients(m);
                if nargin > 1
                    bStr = obj.BasisFunctions{m}.toString(featNames);
                else
                    bStr = obj.BasisFunctions{m}.toString();
                end
                if m == 1
                    fprintf('  %7.4f * %s\n', obj.Coefficients(1), bStr);
                else
                    signChar = '+'; if coef < 0; signChar = '-'; coef = abs(coef); end
                    fprintf('%c %7.4f * %s\n', signChar, coef, bStr);
                end
            end
            fprintf('-------------------------------------------\n');
        end
    end
    
    methods (Access = private)
        function scores = computeScores(obj, X)
            % COMPUTESCORES Build basis matrix and compute B*Coefficients.
            N = size(X, 1);
            M = length(obj.BasisFunctions);
            B = zeros(N, M);
            for m = 1:M
                B(:, m) = obj.BasisFunctions{m}.evaluate(X);
            end
            scores = B * obj.Coefficients;
        end
    end
end
