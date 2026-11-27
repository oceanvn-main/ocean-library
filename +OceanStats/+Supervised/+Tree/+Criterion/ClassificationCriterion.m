classdef ClassificationCriterion < OceanStats.Supervised.Tree.Criterion.INodeCriterion
    % CLASSIFICATIONCRITERION Node criterion for classification trees.
    %
    % Node value: majority class
    % Node risk:  misclassification count = (1 - max(p_k)) * N
    % Split eval: Uses TreeSplitProblem with a weighted-impurity closure.
    % CV error:   number of misclassified samples
    %
    % Author: Ocean Library System
    % Date: 2026-02-15

    properties
        ImpurityMeasure  % IImpurityMeasure instance (Gini, CrossEntropy, etc.)
        Classes          % Unique sorted class labels (discovered during fit)
        LossMatrix       % K x K matrix. L(i,j) = Cost of predicting i when true is j.
        UseAlteredPriors % Logical. If true, re-weight priors for splitting when LossMatrix is present.
    end

    methods
        function obj = ClassificationCriterion(impurityMeasure, options)
            % CLASSIFICATIONCRITERION Constructor.
            %
            % Args:
            %   impurityMeasure (IImpurityMeasure): Impurity criterion.
            %       Default: GiniIndex.
            %   options.LossMatrix (K x K): Optional cost matrix.
            %   options.UseAlteredPriors (logical): Re-weight priors? Default: true if LossMatrix set.
            arguments
                impurityMeasure = OceanStats.Supervised.Tree.Impurity.GiniIndex()
                options.LossMatrix = []
                options.UseAlteredPriors = []
            end
            obj.ImpurityMeasure = impurityMeasure;
            obj.LossMatrix = options.LossMatrix;
            
            if isempty(options.UseAlteredPriors)
                obj.UseAlteredPriors = ~isempty(obj.LossMatrix);
            else
                obj.UseAlteredPriors = options.UseAlteredPriors;
            end
        end

        function stats = computeNodeStats(obj, Y)
            % COMPUTENODESTATS Returns node value and risk.
            %
            % Empirical Bayes Rule:
            %   k(m) = arg min_k sum_l L(k, l) * p(l|m)
            % Expected Loss:
            %   R(m) = sum_l L(k(m), l) * p(l|m) * N_m

            N = length(Y);

            % Discover classes on first call
            if isempty(obj.Classes)
                obj.Classes = unique(Y(:))';
            end

            K = length(obj.Classes);
            %Count Samples Per Class
            classCounts = zeros(1, K);
            for k = 1:K
                classCounts(k) = sum(Y == obj.Classes(k));
            end

            classProbs = classCounts / max(N, 1);
            
            if isempty(obj.LossMatrix)
                % Standard Majority Vote
                [~, majIdx] = max(classCounts);
                nodeValue = obj.Classes(majIdx);
                
                % Risk = misclassification count = (1 - max(p)) * N
                miscRate = 1 - max(classProbs);
                risk = miscRate * N;
            else
                % Cost-Sensitive Assignment
                % Expected Risk for predicting class k: E[L(k, .)] = sum_j L(k, j) * p_j
                % LossMatrix L(k, j) -> Row k, Col j
                
                % We need L to be compatible with obj.Classes indices.
                % Assumption: LossMatrix is K x K and corresponds to sorted obj.Classes.
                
                L = obj.LossMatrix;
                if size(L,1) ~= K || size(L,2) ~= K
                     % Fallback if matrix size mismatch (e.g. strict subset of classes in node vs global)
                     % For now, assume consistent global classes.
                     warning('ClassificationCriterion:LossMatrixSizeMismatch', ...
                         'LossMatrix size does not match number of classes in node.');
                end
                
                expectedRisks = L * classProbs(:); % (KxK) * (Kx1) = Kx1 vector of risks
                [minRisk, bestIdx] = min(expectedRisks);
                
                nodeValue = obj.Classes(bestIdx);
                risk = minRisk * N; % Scale by N for total node risk
            end

            stats = struct(...
                'Value', nodeValue, ...
                'Risk', risk, ...
                'ClassProbs', classProbs, ...
                'ClassCounts', classCounts);
        end

        function tf = isPure(obj, Y)
            % ISPURE True if impurity is negligible.
            if isempty(obj.Classes)
                obj.Classes = unique(Y(:))';
            end
            K = length(obj.Classes);
            counts = zeros(1, K);
            for k = 1:K
                counts(k) = sum(Y == obj.Classes(k));
            end
            
            % If using Altered Priors, we should technically check if the *altered* distribution is pure?
            % Or just stick to raw purity. Standard trees stop if raw node is pure.
            tf = obj.ImpurityMeasure.compute(counts) < 1e-10;
        end

        function problem = createSplitProblem(obj, X, Y, minLeaf, candidateGenerator)
            % CREATESPLITPROBLEM Uses unified TreeSplitProblem with impurity closure.
            arguments
                obj
                X
                Y
                minLeaf
                candidateGenerator = []
            end
            classes = obj.Classes;
            imp = obj.ImpurityMeasure;
            
            % Altered Priors Logic
            useAltered = obj.UseAlteredPriors && ~isempty(obj.LossMatrix);
            
            if useAltered
               L = obj.LossMatrix;
               classWeights = sum(L, 1); 
               if sum(classWeights) > 0
                   classWeights = classWeights / sum(classWeights);
               else
                   classWeights = ones(1, length(classes));
               end
            else
               classWeights = ones(1, length(classes)); 
            end
            evalFn = @(yL, yR) weightedImpurity(yL, yR, classWeights);
            incFn = @(y_sorted) obj.incrementalImpurity(y_sorted, classWeights);
            problem = OceanStats.Supervised.Tree.TreeSplitProblem(X, Y, minLeaf, evalFn, candidateGenerator, incFn);

            function score = weightedImpurity(yLeft, yRight, w)
                nL = length(yLeft);  nR = length(yRight);  N = nL + nR;
                K = length(classes);
                countsL = zeros(1, K);  countsR = zeros(1, K);
                
                for kk = 1:K
                    c_L = sum(yLeft  == classes(kk));
                    c_R = sum(yRight == classes(kk));
                    
                    % Apply Altered Priors / Weights directly to counts passed to impurity
                    if useAltered                        
                        countsL(kk) = c_L * w(kk);
                        countsR(kk) = c_R * w(kk);
                    else
                        countsL(kk) = c_L;
                        countsR(kk) = c_R;
                    end
                end
                
                % Compute Impurity on potentially re-weighted counts
                % Note: compute() internally normalizes by sum(classCounts).
                
                impL = imp.compute(countsL);
                impR = imp.compute(countsR);
                
                % Weight the child nodes by their (possibly re-weighted) proportions?
                % Standard CART: p_L * i(t_L) + p_R * i(t_R)
                % If we used altered priors, p_L and p_R should also reflect that.
                
                if useAltered
                    sumL = sum(countsL); sumR = sum(countsR); sumTot = sumL + sumR;
                    if sumTot == 0, score = 0; return; end
                    pL = sumL / sumTot;
                    pR = sumR / sumTot;
                else
                     pL = nL / N;
                     pR = nR / N;
                end
                
                score = pL * impL + pR * impR;
            end
        end

        function err = cvError(obj, Y_true, Y_pred)
            % CVERROR 
            % Returns: Total Loss if LossMatrix present, else Misclassification count.
            if isempty(obj.LossMatrix)
                err = sum(Y_true ~= Y_pred);
            else
                % Sum of costs L(y_pred, y_true)
                % Y_true, Y_pred are vectors.
                % Map classes to indices.
                L = obj.LossMatrix;
                classes = obj.Classes;
                
                if isempty(classes)
                    classes = unique(Y_true); 
                end
                
                map = containers.Map(num2cell(classes), 1:length(classes));
                
                totalCost = 0;
                for i = 1:length(Y_true)
                    tIdx = map(Y_true(i));
                    pIdx = map(Y_pred(i));
                    totalCost = totalCost + L(pIdx, tIdx);
                end
                err = totalCost;
            end
        end
        
        function scores = incrementalImpurity(obj, y_sorted, w)
            % INCREMENTALIMPURITY Vectorized O(N) evaluation of all split points
            N = length(y_sorted);
            K = length(obj.Classes);
            
            % 1. Create one-hot indicator matrix (N x K) and apply weights
            I = zeros(N, K);
            for k = 1:K
                I(:, k) = (y_sorted == obj.Classes(k)) * w(k);
            end
            
            % 2. Cumulative sums to get left and right counts for all splits
            cumL = cumsum(I, 1);
            tot = cumL(end, :);
            cumR = bsxfun(@minus, tot, cumL);
            
            % We only care about splits 1 to N-1
            cumL = cumL(1:end-1, :);
            cumR = cumR(1:end-1, :);
            
            % Node sizes
            if obj.UseAlteredPriors && ~isempty(obj.LossMatrix)
                % If priorities are altered, the denominator is the sum of weighted counts
                NL = sum(cumL, 2);
                NR = sum(cumR, 2);
            else
                % Standard node sizes
                NL = (1:N-1)';
                NR = (N-1:-1:1)';
            end
            
            % To avoid division by zero
            NL = max(NL, eps);
            NR = max(NR, eps);
            
            % Normalize counts to probabilities
            pL = bsxfun(@rdivide, cumL, NL);
            pR = bsxfun(@rdivide, cumR, NR);
            
            % 3. Calculate Vectorized Impurity (delegated to strategy)
            impL = obj.ImpurityMeasure.compute(pL);
            impR = obj.ImpurityMeasure.compute(pR);
            
            % 4. Combine impurities weighted by node sizes
            if obj.UseAlteredPriors && ~isempty(obj.LossMatrix)
                totWeight = sum(tot);
                scores = (NL ./ totWeight) .* impL + (NR ./ totWeight) .* impR;
            else
                scores = (NL ./ N) .* impL + (NR ./ N) .* impR;
            end
        end
    end
end

