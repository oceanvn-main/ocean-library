classdef GainRatioEvaluator < handle
    % GAINRATIOEVALUATOR Helper for computing C5.0 split criterion.
    %
    % C5.0 uses Information Gain Ratio:
    %   GainRatio(S) = InformationGain(S) / SplitInfo(S)
    %
    % This prevents favoring features with many partitions (high SplitInfo).
    
    methods
        function [ratio, gain, splitInfo] = compute(~, yParent, yChildren)
            % COMPUTE Calculates Gain Ratio for a split.
            %
            % Args:
            %   yParent: Target values in the parent node.
            %   yChildren: Cell array of target values for each child node.
            
            import OceanMath.Information.Entropy
            
            nTotal = length(yParent);
            if nTotal == 0, ratio = 0; return; end
            
            % 1. Compute Parent Entropy (in bits)
            parentEntropy = Entropy(yParent, 'Base', 2);
            
            % 2. Compute Weighted Child Entropy & SplitInfo
            childEntropy = 0;
            childWeights = zeros(1, length(yChildren));
            
            for i = 1:length(yChildren)
                yChild = yChildren{i};
                nChild = length(yChild);
                if nChild == 0, continue; end
                
                weight = nChild / nTotal;
                childWeights(i) = weight;
                childEntropy = childEntropy + weight * Entropy(yChild, 'Base', 2);
            end
            
            % 3. Calculate Information Gain
            gain = parentEntropy - childEntropy;
            
            % 4. Compute SplitInfo (Intrinsic Value)
            % This is exactly the entropy of the partition sizes.
            % We pass the weights directly (sum to 1) to Entropy().
            splitInfo = Entropy(childWeights, 'Base', 2);
            
            % 5. Calculate Gain Ratio
            % Convention: ratio = gain / splitInfo. If splitInfo near 0, ratio = 0.
            if splitInfo < 1e-10
                ratio = 0;
            else
                ratio = gain / splitInfo;
            end
        end
    end
end
