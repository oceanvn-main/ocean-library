classdef C50Splitter < OceanStats.Supervised.Tree.Strategy.ISplitStrategy
    % C50SPLITTER Implements C5.0 splitting strategy.
    %
    % Logic:
    %   - Continuous Features: Binary split using Gain Ratio.
    %   - Categorical Features: Multi-way split (one child per category).
    %   - Criterion: Uses Information Gain Ratio instead of Gini.
    
    properties
        Optimizer % Optional: and C5.0 usually just loops exhaustive points
        GainEvaluator % GainRatioEvaluator instance
    end
    
    methods
        function obj = C50Splitter(optimizer)
            if nargin > 0
                obj.Optimizer = optimizer;
            end
            import OceanStats.Supervised.Tree.Criterion.GainRatioEvaluator
            obj.GainEvaluator = GainRatioEvaluator();
        end
        
        function result = findSplit(obj, X, Y, params)
            % FINDSPLIT Loops through features to find the best Gain Ratio split.
            import OceanStats.Supervised.Tree.Strategy.SplitResult
            import OceanCS.Parallel.Scheduler
            
            [N, P] = size(X);
            
            % Heuristic Check: N*log(N) per feature (sorting in continuous)
            % Total Cost ~ P * N * log(N)
            costPerTask = N * log2(N); 
            payloadSize = N * 8; % Bytes (double)
            
            useParallel = Scheduler.shouldParallelize(P, costPerTask, payloadSize);
            
            if useParallel
                 % --- Parallel Execution ---
                 % Preallocate results to avoid race conditions
                 ratios = -Inf(1, P);
                 results = cell(1, P);
                 
                 % Localize evaluator for broadcast
                 localEvaluator = obj.GainEvaluator;
                 
                 parfor j = 1:P
                     x_col = X(:, j);
                     % Re-instantiate trivial result to be safe in parfor
                     res = SplitResult(true);
                     r = -Inf;
                     
                     if iscategorical(x_col) || isstring(x_col) || iscellstr(x_col)
                         % Categorical
                         [r, res] = OceanStats.Supervised.Tree.Strategy.C50Splitter.evaluateCategorical_Static(...
                             x_col, Y, j, params.MinLeafSize, localEvaluator);
                     else
                         % Continuous
                         [r, res] = OceanStats.Supervised.Tree.Strategy.C50Splitter.evaluateContinuous_Static(...
                             x_col, Y, j, params, localEvaluator);
                     end
                     
                     ratios(j) = r;
                     results{j} = res;
                 end
                 
                 % Reduction
                 [bestRatio, idx] = max(ratios);
                 if isinf(bestRatio)
                     result = SplitResult(true);
                 else
                     result = results{idx};
                     result.Gain = bestRatio;
                 end
                 
            else
                % --- Serial Execution (Original) ---
                bestRatio = -Inf;
                bestResult = SplitResult(true);

                for j = 1:P
                    x_col = X(:, j);

                    if iscategorical(x_col) || isstring(x_col) || iscellstr(x_col)
                        [ratio, res] = obj.evaluateCategorical(x_col, Y, j, params.MinLeafSize);
                    else
                        [ratio, res] = obj.evaluateContinuous(x_col, Y, j, params);
                    end

                    if ratio > bestRatio
                        bestRatio = ratio;
                        bestResult = res;
                        bestResult.Gain = ratio;
                    end
                end
                result = bestResult;
            end
        end
    end
    
    methods (Static)
        function [ratio, result] = evaluateCategorical_Static(x, y, featIdx, minLeaf, gainEvaluator)
            import OceanStats.Supervised.Tree.Strategy.SplitResult
            ratio = -Inf;
            result = SplitResult(true);
            
            % 1. Find unique categories
            if ischar(x), x = string(x); end
            cats = unique(x);
            cats(ismissing(cats)) = []; 
            
            if length(cats) < 2, return; end
            
            % 2. Partition data
            yChildren = cell(1, length(cats));
            counts = zeros(1, length(cats));
            for i = 1:length(cats)
                mask = (x == cats(i));
                yChildren{i} = y(mask);
                counts(i) = sum(mask);
            end
            
            if any(counts < minLeaf), return; end
            
            % 3. Calculate Gain Ratio
            [ratio, ~, ~] = gainEvaluator.compute(y, yChildren);
            
            % 4. Create Result
            result = SplitResult(false);
            result.FeatureIndex = featIdx;
            result.Categories = cellstr(string(cats));
            result.Description = sprintf('Categorical Split on Feature %d (%d levels)', featIdx, length(cats));
            
            map = containers.Map('KeyType', 'char', 'ValueType', 'double');
            for i = 1:length(cats)
                map(char(string(cats(i)))) = i;
            end
            result.CategoryMap = map;
        end
        
        function [ratio, result] = evaluateContinuous_Static(x, y, featIdx, params, gainEvaluator)
            import OceanStats.Supervised.Tree.Strategy.SplitResult
            ratio = -Inf;
            result = SplitResult(true);
            
            candidates = params.CandidateGenerator.generate(x, y, params.MinLeafSize);
            
            for i = 1:length(candidates)
                t = candidates(i).Threshold;
                mask = x <= t;
                yChildren = {y(mask), y(~mask)};
                
                [r, ~, ~] = gainEvaluator.compute(y, yChildren);
                
                if r > ratio
                    ratio = r;
                    result = SplitResult(false);
                    result.Description = sprintf('Feature %d <= %.4f', featIdx, t);
                    result.FeatureIndex = featIdx;
                    result.Threshold = t;
                end
            end
        end
    end
    
    methods (Access = private)
        % Wrappers for serial loop (legacy support or ease of use)
        function [ratio, result] = evaluateCategorical(obj, x, y, featIdx, minLeaf)
            [ratio, result] = OceanStats.Supervised.Tree.Strategy.C50Splitter.evaluateCategorical_Static(...
                x, y, featIdx, minLeaf, obj.GainEvaluator);
        end
        
        function [ratio, result] = evaluateContinuous(obj, x, y, featIdx, params)
             [ratio, result] = OceanStats.Supervised.Tree.Strategy.C50Splitter.evaluateContinuous_Static(...
                x, y, featIdx, params, obj.GainEvaluator);
        end
    end
end
