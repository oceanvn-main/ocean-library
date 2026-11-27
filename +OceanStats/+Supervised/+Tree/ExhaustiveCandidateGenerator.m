classdef ExhaustiveCandidateGenerator < OceanStats.Supervised.Tree.ICandidateGenerator
    % EXHAUSTIVECANDIDATEGENERATOR Generates all valid midpoint splits.
    %
    % Standard CART logic: For each continuous feature, unique values are 
    % sorted and midpoints between adjacent values are tested as thresholds.
    %
    % Author: Ocean Library System
    % Date: 2026-02-17

    methods
        function candidates = generate(~, X, ~, minLeaf)
            % GENERATE Creates all valid midpoint split points using Linear Scan optimization.
            
            candidates = struct('FeatureIndex', {}, 'Threshold', {});
            [N, P] = size(X);

            for j = 1:P
                x_col = X(:, j);
                
                % 1. Filter out NaNs and sort
                validIdx = ~isnan(x_col);
                x_valid = sort(x_col(validIdx));
                nValid = length(x_valid);
                
                if nValid < (2 * minLeaf)
                    continue;
                end

                % 2. Linear Scan for thresholds
                % Since x is sorted, we can check minLeaf constraints by index.
                % Left child has 'i' samples, Right has 'nValid - i' samples.
                
                % Start at minLeaf, end at nValid - minLeaf
                for i = minLeaf : (nValid - minLeaf)
                    % Only suggest a split between different values
                    if x_valid(i) < x_valid(i+1)
                        t = (x_valid(i) + x_valid(i+1)) / 2;
                        
                        cand.FeatureIndex = j;
                        cand.Threshold = t;
                        candidates(end+1) = cand; %#ok<AGROW>
                    end
                end
            end
        end
    end
end
