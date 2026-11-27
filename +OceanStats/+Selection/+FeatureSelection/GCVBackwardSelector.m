classdef GCVBackwardSelector < OceanStats.Base.IBackwardPass
    % GCVBACKWARDSELECTOR Backward deletion with a pluggable pruning criterion.
    %
    % Prunes the oversized MARS model one term at a time, selecting the
    % optimal model size according to the injected Criterion strategy.
    %
    % Default: GCVCriterion (Friedman 1991). Swap with AICCriterion etc.
    
    properties
        Criterion OceanStats.Base.IPruningCriterion = ...
            OceanStats.Selection.Criteria.GCVCriterion()
    end
    
    methods
        function [bestBasisList, bestCoefs, history] = prune(obj, X, Y, basisList, options)
            [N, ~] = size(X);
            
            numTerms = length(basisList);
            fprintf('MARS Backward Pass: Pruning %d terms...\n', numTerms);
            
            % Initial Full Evaluation Matrix
            B_full = zeros(N, numTerms);
            for m = 1:numTerms
                B_full(:, m) = basisList{m}.evaluate(X);
            end
            
            % Setup active set / history
            active_idx = 1:numTerms; % index 1 is always the intercept
            history(numTerms) = struct('BasisIndices', [], 'RSS', [], 'GCV', [], 'Coefs', []);
            
            warning('off', 'MATLAB:rankDeficientMatrix');
            
            % Evaluate the full model to seed history
            beta = B_full \ Y;
            rss = sum((Y - B_full * beta).^2, 'all');
            history(numTerms).BasisIndices = active_idx;
            history(numTerms).RSS = rss;
            history(numTerms).GCV = obj.Criterion.evaluate(rss, N, numTerms, options);
            history(numTerms).Coefs = beta;
            
            % Loop down to just the intercept
            for step = numTerms:-1:2
                best_drop_idx = -1;
                best_new_rss  = inf;
                best_new_beta = [];
                
                % Try removing each term EXCEPT the intercept (index 1 in active_idx)
                for i = 2:length(active_idx)
                    test_idx = active_idx;
                    test_idx(i) = []; % remove one term
                    
                    B_test = B_full(:, test_idx);
                    beta_test = B_test \ Y;
                    rss_test = sum((Y - B_test * beta_test).^2, 'all');
                    
                    if rss_test < best_new_rss
                        best_new_rss = rss_test;
                        best_drop_idx = i;
                        best_new_beta = beta_test;
                    end
                end
                
                % Formally remove the worst contributor
                active_idx(best_drop_idx) = [];
                
                % Save state to history array
                prev_step = step - 1;
                history(prev_step).BasisIndices = active_idx;
                history(prev_step).RSS = best_new_rss;
                % GCV penalizes the number of remaining terms
                history(prev_step).GCV = obj.Criterion.evaluate(best_new_rss, N, prev_step, options);
                history(prev_step).Coefs = best_new_beta;
            end
            
            warning('on', 'MATLAB:rankDeficientMatrix');
            
            % Find optimal model minimizing GCV
            all_gcvs = [history.GCV];
            [~, optimal_len] = min(all_gcvs);
            
            optimal_state = history(optimal_len);
            
            % Reconstruct final list
            bestBasisList = basisList(optimal_state.BasisIndices);
            bestCoefs = optimal_state.Coefs;
            
            fprintf('  Optimal model selected: %d terms (GCV = %.4f)\n', optimal_len, optimal_state.GCV);
        end
        
    end
end
