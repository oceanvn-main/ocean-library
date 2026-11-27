classdef Bootstrap632 < OceanStats.Base.IValidator
    % BOOTSTRAP632 Validates using the .632+ Bootstrap Estimator.
    %
    % OVERVIEW:
    %   estimates Err = 0.368 * Apparent + 0.632 * Err(1)
    
    properties
        Name string = ".632+ Bootstrap"
        NumIters (1, 1) double = 50
    end
    
    methods
        function obj = Bootstrap632(iters)
            if nargin > 0, obj.NumIters = iters; end
        end
        
        function result = validate(obj, selector, strategy, X, Y)
            [N, P] = size(X);
            B = obj.NumIters;
            
            fprintf('   > %s (B=%d)...\n', obj.Name, B);

            % 1. Apparent Error (Training Error)
            % We assume the global search has been run or we run it?
            % Ideally we shouldn't re-run global search if passed in, 
            % but Validator interface assumes independent execution.
            % Let's run it to be safe and consistent.
            global_hist = selector.runExhaustiveSearch(X, Y, strategy);
            apparent_err = [global_hist.Error]' / N;
            
            % 2. OOB Error Accumulation
            oob_err_sum = zeros(N, P+1);
            oob_counts = zeros(N, 1);
            
            for b = 1:B
                % Resample
                boot_idx = randsample(N, N, true);
                oob_idx = setdiff(1:N, boot_idx);
                
                if isempty(oob_idx), continue; end
                oob_counts(oob_idx) = oob_counts(oob_idx) + 1;
                
                % Train on Bootstrap Sample
                boot_hist = selector.runExhaustiveSearch(X(boot_idx,:), Y(boot_idx), strategy);
                
                % Evaluate on OOB
                X_oob = X(oob_idx, :);
                Y_oob = Y(oob_idx);
                
                for k = 1:(P+1)
                    m = boot_hist(k);
                    if m.Size == 0
                        X_sub = zeros(length(oob_idx), 0);
                    else
                        X_sub = X_oob(:, m.Mask);
                    end
                    y_pred = strategy.predict(m.Model, X_sub);
                    
                    vec_err = (Y_oob - y_pred).^2;
                    
                    % Accumulate
                    for j = 1:length(oob_idx)
                        oid = oob_idx(j);
                        oob_err_sum(oid, k) = oob_err_sum(oid, k) + vec_err(j);
                    end
                end
            end
            
            % 3. Compute Err(1)
            valid = oob_counts > 0;
            err_1_obs = zeros(N, P+1);
            err_1_obs(valid,:) = oob_err_sum(valid,:) ./ oob_counts(valid);
            err_1 = mean(err_1_obs, 1)';
            
            % 4. Compute .632
            err_632 = 0.368 * apparent_err + 0.632 * err_1;
            
            result = struct();
            result.Error = err_632;
            result.Apparent = apparent_err;
            result.OOB = err_1;
        end
    end
end
