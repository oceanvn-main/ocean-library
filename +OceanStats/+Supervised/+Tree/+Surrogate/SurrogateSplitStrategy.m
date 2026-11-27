classdef SurrogateSplitStrategy < OceanStats.Supervised.Tree.Strategy.ISplitStrategy
    % SURROGATESPLITSTRATEGY Decorator that adds surrogate splits to a result.
    %
    % Wraps an existing ISplitStrategy (like CartSplitter).
    % After the inner strategy finds a primary split, this decorator
    % uses SurrogateFinder to find backup splits and attaches them.
    %
    % This keeps the core Splitters (CART, C5.0) clean and unaware of surrogates.
    %
    % Author: Ocean Library System
    % Date: 2026-02-17
    
    properties
        InnerStrategy OceanStats.Supervised.Tree.Strategy.ISplitStrategy
    end
    
    methods
        function obj = SurrogateSplitStrategy(innerStrategy)
            if nargin > 0
                obj.InnerStrategy = innerStrategy;
            end
        end
        
        function result = findSplit(obj, X, Y, params)
            % 1. Delegate to Inner Strategy for Primary Split
            result = obj.InnerStrategy.findSplit(X, Y, params);
            
            % 2. If Primary Split found, look for Surrogates
            if ~result.IsTerminal
                import OceanStats.Supervised.Tree.Surrogate.SurrogateFinder
                import OceanStats.Supervised.Tree.Strategy.SplitResult
                
                % Prepare Data for Surrogate Finding
                % Need X subset where Primary Feature is valid
                % But we passed X, Y to findSplit.
                
                % Primary Split Info
                featIdx = result.FeatureIndex;
                
                % Check if Primary Feature has missing values in THIS node
                x_primary = X(:, featIdx);
                validMask = ~isnan(x_primary); % Basic check, assuming numeric for CART
                % If C5.0 (mixed), we need robust check. 
                % But usually if we are here, we have a split.
                
                if sum(validMask) > 0
                    % Extract training data for surrogates
                    X_surr = X(validMask, :);
                    
                    % Calculate "Truth" direction from Primary Split
                    % We need to know how the Primary Split routes these valid rows.
                    % Result object has a 'route' method or Threshold property.
                    % We can use 'result.route' but it implies looping.
                    % Vectorized check is better.
                    
                    % We assume Binary Split for now (SurrogateFinder is binary-oriented)
                    % or we use the specific properties of Result.
                    
                    if ~isnan(result.Threshold)
                        % Binary: <= Threshold is Left(1), Else Right(0)
                         y_prime = double(x_primary(validMask) <= result.Threshold);
                         
                         % Find Surrogates
                         sStructs = SurrogateFinder.findSurrogates(X_surr, y_prime, featIdx);
                         
                         % Convert to SplitResult objects and Attach
                         for k = 1:length(sStructs)
                            s = sStructs(k);
                            sr = SplitResult(false);
                            sr.FeatureIndex = s.FeatureIndex;
                            sr.Score = s.Score;
                            sr.Description = sprintf("Surrogate Feat %d (Score %.2f)", ...
                                s.FeatureIndex, s.Score);
                            
                            if s.GoLeft
                                sr.Threshold = s.Threshold;
                                sr.Description = sr.Description + sprintf(" <= %.4f", s.Threshold);
                            else
                                % Inverted Logic
                                t = s.Threshold;
                                sr.Threshold = NaN; 
                                sr.RouteFn = @(x) 2 - double(x <= t); 
                                sr.Description = sr.Description + sprintf(" > %.4f [Inverted]", s.Threshold);
                            end
                            
                            % Append
                            result.Surrogates = [result.Surrogates, sr];
                         end
                    end
                end
            end
        end
    end
end
