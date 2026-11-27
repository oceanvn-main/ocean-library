classdef ForwardBasisSelector < OceanStats.Base.IForwardPass
    % FORWARDBASISSELECTOR Greedy forward stepwise selection for additive models.
    %
    % Evaluates all possible combinations of parent basis functions,
    % features, and knots to find the pair of reflected hinges that yields
    % the greatest drop in Residual Sum of Squares (RSS).
    
    methods
        function basisList = build(obj, X, Y, options)
            import OceanStats.Supervised.MARS.Basis.*
            [N, P] = size(X);
            
            % Initial Model Set M contains only h_0(X) = 1
            basisList = { ConstantBasis() };
            
            % Initial Basis Matrix and Fast Linear Algebra State
            B = ones(N, 1);
            orthoStrategy = OceanMath.LinearAlgebra.Orthogonalization.HouseholderReflection();
            iqr = OceanMath.LinearAlgebra.IncrementalQR(B, orthoStrategy);
            
            % Initial Residuals (calculated once here)
            [~, ~, base_res] = iqr.recalculate(Y);
            
            fprintf('MARS Forward Pass: Growing up to %d terms...\n', options.MaxTerms);
            
            % Main Forward Loop: Add pairs of reflected hinges
            % MaxTerms includes the intercept, and we add 2 terms at a time.
            while length(basisList) < options.MaxTerms - 1
                bestErr = inf;
                bestNew1 = [];
                bestNew2 = [];
                
                % Iterate over every parent in M
                for m = 1:length(basisList)
                    parent = basisList{m};
                    
                    % 1. Enforce Interaction Degree Limit
                    if parent.getDegree() >= options.MaxInteractionDegree
                        continue; 
                    end
                    
                    parent_eval = B(:, m);
                    
                    % 2. Iterate over every feature
                    for j = 1:P
                        x_j = X(:, j);
                        
                        % 3. Check if feature is Categorical
                        if isfield(options, 'CategoricalFeatures') && ismember(j, options.CategoricalFeatures)
                            unique_vals = unique(x_j);
                            K = length(unique_vals);
                            if K > 15
                                warning('MARS:TooManyCategories', 'Feature %d has %d categories, exceeding limit of 15. Ignoring to prevent combinatorial explosion.', j, K);
                                continue;
                            end
                            if K < 2
                                continue;
                            end
                            
                            % Generate all 2^(K-1) - 1 binary partitions
                            % (Fixing the first unique value to one side prevents duplicate mirror pairs)
                            num_partitions = 2^(K-1) - 1;
                            for p_idx = 1:num_partitions
                                bin_str = dec2bin(p_idx, K);
                                subset1 = []; subset2 = [];
                                for k = 1:K
                                    if bin_str(k) == '1'
                                        subset1 = [subset1, unique_vals(k)];
                                    else
                                        subset2 = [subset2, unique_vals(k)];
                                    end
                                end
                                
                                % Fast vector evaluation
                                h1_val = parent_eval .* double(ismember(x_j, subset1));
                                h2_val = parent_eval .* double(ismember(x_j, subset2));
                                
                                % Prevent adding degenerate/empty basis functions
                                if sum(h1_val > 0) < 5 || sum(h2_val > 0) < 5
                                    continue;
                                end
                                
                                % Fast OLS testing bypasses beta estimation entirely
                                [rss1, q_new1] = iqr.fastResidualUpdate(h1_val, base_res);
                                if isinf(rss1); continue; end
                                
                                res_temp = base_res - q_new1 * (q_new1' * base_res);
                                [rss, ~] = orthoStrategy.fastProject([iqr.Q, q_new1], h2_val, res_temp);
                                
                                % Record if it's the best drop in error
                                if rss < bestErr || (isinf(rss) && rss1 < bestErr)
                                    % If h2 is redundant (always true for binary partitions of parent),
                                    % we use the rss from h1 only.
                                    if isinf(rss)
                                        actualRSS = rss1;
                                    else
                                        actualRSS = rss;
                                    end
                                    
                                    if actualRSS < bestErr
                                        bestErr = actualRSS;
                                        base1 = CategoricalBasis(j, subset1);
                                        base2 = CategoricalBasis(j, subset2);
                                        if isa(parent, 'OceanStats.Supervised.MARS.Basis.ConstantBasis')
                                            bestNew1 = base1;
                                            bestNew2 = base2;
                                        else
                                            bestNew1 = ProductBasis(parent, base1);
                                            bestNew2 = ProductBasis(parent, base2);
                                        end
                                    end
                                end
                            end
                            continue; % Skip continuous evaluation for this categorical feature
                        end
                        
                        % 3. Subsample Knots to keep computation tractable (Continuous)
                        unique_vals = unique(x_j);
                        if length(unique_vals) > 50
                            t_cands = quantile(unique_vals, linspace(0.05, 0.95, 50));
                        else
                            t_cands = unique_vals;
                        end
                        
                        for i = 1:length(t_cands)
                            t = t_cands(i);
                            
                            % Fast vector evaluation of candidate pairs
                            h1_val = parent_eval .* max(x_j - t, 0);
                            h2_val = parent_eval .* max(t - x_j, 0);
                            
                            % Prevent adding degenerate/empty basis functions
                            if sum(h1_val > 0) < 5 || sum(h2_val > 0) < 5
                                continue;
                            end
                            
                            % Fast OLS testing bypasses beta estimation entirely
                            % and orthogonalizes the test columns incrementally.
                            [rss1, q_new1] = iqr.fastResidualUpdate(h1_val, base_res);
                            
                            if isinf(rss1)
                                continue;
                            end
                            
                            % Temporarily update the residual and Q matrix for the second hinge
                            res_temp = base_res - q_new1 * (q_new1' * base_res);
                            
                            % We have to temporarily append q_new1 to Q for the second step.
                            % But to keep iqr decoupled, we just pass the temporarily expanded Q
                            % to the strategy directly for the second check
                            [rss, ~] = orthoStrategy.fastProject([iqr.Q, q_new1], h2_val, res_temp);
                            
                            % Record if it's the best drop in error
                            if rss < bestErr
                                bestErr = rss;
                                hinge1 = HingeBasis(j, t, 1);
                                hinge2 = HingeBasis(j, t, -1);
                                
                                % If parent is Constant(1), output is just the Hinge
                                if isa(parent, 'OceanStats.Supervised.MARS.Basis.ConstantBasis')
                                    bestNew1 = hinge1;
                                    bestNew2 = hinge2;
                                else
                                    bestNew1 = ProductBasis(parent, hinge1);
                                    bestNew2 = ProductBasis(parent, hinge2);
                                end
                            end
                        end
                    end
                end
                
                % Stop early if no improvement found (e.g. constraints blocked)
                if isempty(bestNew1)
                    break;
                end
                
                % Commit the best pair to M
                basisList{end+1} = bestNew1;
                basisList{end+1} = bestNew2;
                
                % Update Fast Matrix State
                V_add = [bestNew1.evaluate(X), bestNew2.evaluate(X)];
                [~, ~, base_res] = iqr.addColumns(V_add, Y);
                B = [B, V_add];
                
                fprintf('  Added terms %d & %d. Current RSS: %.4f\n', ...
                    length(basisList)-1, length(basisList), bestErr);
            end
        end
    end
end
