classdef AdditiveModel
    % ADDITIVEMODEL Container for a fitted additive model.
    %
    % THEORY OVERVIEW:
    %   A Generalized Additive Model (GAM) assumes the outcome Y is the sum 
    %   of an intercept and several independent smooth functions:
    %
    %       E[Y] = alpha + f1(x1) + f2(x2) + ... + fp(xp)
    %
    %   This class stores:
    %     1. The global intercept (alpha).
    %     2. The definitions of the curves (f1, f2...) inside 'Terms'.
    %     3. The logic for how to calculate them (Strategies).
    
    properties
        % The global baseline.
        % alpha = mean(Y). All other functions fj are centered around zero.
        Intercept (1,1) double = 0
        
        % The "Brain" of each component.
        % Cell array containing the FITTED objects (e.g., spline knots/coeffs).
        % Terms{1} holds the data for f1, Terms{2} for f2, etc.
        Terms     cell        % (1 x P)
        
        % The "Manual" for each component.
        % Cell array of Strategy objects (e.g., SmoothingSplineStrategy).
        % Knows how to take the data in 'Terms' and produce a prediction.
        Strategies cell       % (1 x P)
        
        % Mapping configuration (Optional).
        % Which column of X corresponds to which term?
        % Default Assumption: Term j uses Column j of X.
        FeatureIndices cell

        % Per-term centering offsets (set by Backfitting.OptimizeCentering).
        % TermOffsets(j) is the mean shift absorbed from function f_j after
        % the deferred centering pass.  predict() subtracts this so that all
        % component functions remain zero-mean for interpretability.
        TermOffsets double = []  % (1 x P), empty = standard mode
    end
    
    methods
        function obj = AdditiveModel(strategies)
            % CONSTRUCTOR
            % Initializes an empty model structure ready to be trained.
            % Input:
            %   strategies: A list of strategy objects defining the model type
            %               (e.g., {'Spline', 'Linear', 'Loess'})
            
            obj.Strategies = strategies;
            obj.Terms = cell(size(strategies));
        end
        
        function fit(obj, X, y, weights)
            % FIT Trains the additive model using the Backfitting algorithm.
            if nargin < 4
                weights = ones(size(X, 1), 1);
            end
            
            % Instantiate standard Gauss-Seidel backfitting solver
            % Using a lower MaxIter (5) and looser Tol (1e-3) to ensure 
            % rapid convergence when fitting 11 parallel models in FDA.
            solver = OceanStats.GAM.Backfitting(5, 1e-3, false);
            solver.Verbose = true; 
            
            % The solver fits the terms and mutates this object's state in place
            solver.fit(obj, X, y, weights);
        end
        
        function y_pred = predict(obj, X)
            % PREDICT Calculates the final output Y for new data X.
            %
            % FORMULA:
            %   y_pred = alpha + f1(x1) + f2(x2) + ...
            %
            % ALGORITHM:
            %   1. Start with the baseline Intercept.
            %   2. Loop through every term j.
            %   3. Calculate the partial effect f_j(x_j).
            %   4. Add it to the running total.
            
            [N, P] = size(X);
            
            % Start with the global average
            y_pred = repmat(obj.Intercept, N, 1);
            
            hasOffsets = ~isempty(obj.TermOffsets) && length(obj.TermOffsets) == length(obj.Terms);

            % Additive Loop
            for j = 1:length(obj.Terms)
                model_j = obj.Terms{j};       % Get the fitted parameters
                strat_j = obj.Strategies{j};  % Get the prediction logic
                
                % Standard 1-to-1 mapping: Function j eats Column j
                if j <= P
                    x_j = X(:, j);
                    
                    % If the term was completely dropped or failed to train, skip it
                    if isempty(model_j)
                        f_j = zeros(N, 1);
                    else
                        % Calculate partial effect: f_j = Strategy.predict(...)
                        f_j = strat_j.predict(model_j, x_j);
                    end

                    % Subtract deferred offset so component is zero-mean
                    if hasOffsets && ~isempty(model_j)
                        f_j = f_j - obj.TermOffsets(j);
                    end

                    y_pred = y_pred + f_j;
                end
            end
        end
        
        function y_terms = predict_terms(obj, X)
             % PREDICT_TERMS Returns matrix of individual component contributions.
             %
             % Output:
             %   Matrix [ f1(x1), f2(x2), ... ]  (each column is zero-mean)
             
             [N, P] = size(X);
             J = length(obj.Terms);
             y_terms = zeros(N, J);
             hasOffsets = ~isempty(obj.TermOffsets) && length(obj.TermOffsets) == J;

             for j = 1:J
                model_j = obj.Terms{j};
                strat_j = obj.Strategies{j};
                
                if j <= P
                    x_j = X(:, j);
                    
                    if isempty(model_j)
                        f_j = zeros(N, 1);
                    else
                        f_j = strat_j.predict(model_j, x_j);
                    end

                    if hasOffsets && ~isempty(model_j)
                        f_j = f_j - obj.TermOffsets(j);
                    end

                    y_terms(:, j) = f_j;
                end
             end
        end
    end
end