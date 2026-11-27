classdef ComplexityAudit
    % COMPLEXITYAUDIT A rigorous engine for Bias-Variance trade-off analysis.
    %
    %   This component automates the comparison between AIC (Likelihood-based) 
    %   and Cp (Prediction Risk-based) for classification models.
    %
    %   -----------------------------------------------------------------------
    %   THEORETICAL FOUNDATIONS
    %   -----------------------------------------------------------------------
    %
    %   1. DEVIANCE (D)
    %      Measure of lack-of-fit. Defined as twice the difference between the
    %      log-likelihood of the Saturated Model (Perfect Fit) and the current model.
    %
    %      Formula:  D = -2 * ln( Likelihood_model )
    %
    %   2. AIC (Akaike Information Criterion)
    %      Estimates the relative information lost by the model. It balances
    %      fit (Deviance) against complexity (k parameters).
    %
    %      Formula:  AIC = D + 2k
    %                (Lower is better. Penalizes 2 units per parameter)
    %
    %   3. TOTAL RSS (Residual Sum of Squares)
    %      Adapted for Classification. We treat probability errors as geometric distances.
    %
    %      Formula:  RSS = Sum_over_samples( Sum_over_classes( (y_true - y_prob)^2 ) )
    %
    %   4. MALLOWS' CP
    %      Estimates the Mean Squared Prediction Error. It requires an estimate of
    %      the "True Noise Variance" (Sigma^2), which we derive from the most
    %      complex model (Saturated Model).
    %
    %      Formula:  Cp = (RSS_d / Sigma^2) - N + 2k
    %                (Values close to k suggest an unbiased model)
    %   -----------------------------------------------------------------------
    
    methods (Static)
        
        function report = execute(X, Y, options)
            % EXECUTE Runs the complexity analysis pipeline.
            %
            % Inputs:
            %   X (double) : Feature Matrix (N x Features)
            %   Y (double) : Label Vector (N x 1)
            %   options    : Name-Value pairs for configuration.
            %
            % Output:
            %   report     : Struct containing 'Table' (Results), 'Best' models,
            %                and 'Data' used for plotting.
            
            arguments
                X (:, :) double
                Y (:, 1) double
                options.MaxLevel (1, 1) double {mustBeInteger, mustBePositive} = 6
                
                options.TargetClasses (1, :) double = [] 
                options.SigmaSq (1, 1) double = NaN 
                options.Strategy (1, 1) struct = ...
                    OceanStats.Selection.ComplexityAudit.getLogisticStrategy()
            end
            
            % --- 1. Data Filtering & Prep ---
            if ~isempty(options.TargetClasses)
                idx_keep = ismember(Y, options.TargetClasses);
                X = X(idx_keep, :);
                Y = Y(idx_keep);
            end
            
            % Remap Y to 1..K (Standardizes logic for any class labels)
            [~, ~, Y_idx] = unique(Y);
            classes = unique(Y);
            N = size(X, 1);
            K = length(classes);
            
            % --- 2. PCA Projection ---
            % We strictly project to 2D for consistent visualization.
            % If X is already 2D, this is a no-op.
            if size(X, 2) > 2
                [~, score, ~] = pca(X);
                X_input = score(:, 1:2);
            else
                X_input = X;
            end
            
            Strategy = options.Strategy;
            
            % --- 3. Noise Estimation (Sigma^2) ---
            % Cp requires the variance of the "True Noise".
            % If not provided (Theoretical), we estimate it empirically using
            % the "Saturated Model" (The most complex model we are willing to build).
            if ~isnan(options.SigmaSq)
                sigma_sq_used = options.SigmaSq;
            else
                % A. Train Saturated Model
                warning('Mallows Cp conditions are violated here. We donot have true variance, so we use unbiased estimated sigma_hat^2 instead.');
                model_sat = Strategy.TrainFcn(X_input, Y_idx, options.MaxLevel);
                k_sat     = Strategy.CountParamsFcn(model_sat);
                P_sat     = Strategy.PredictFcn(model_sat, X_input, options.MaxLevel);
                
                % B. Calculate Residuals of Saturated Model
                RSS_sat = OceanStats.Selection.ComplexityAudit.calcTotalRSS(P_sat, Y_idx, K);
                
                % C. Unbiased Estimator (RSS / DoF)
                sigma_sq_used = RSS_sat / (N - k_sat); 
            end
            
            % --- 4. The Selection Loop ---
            results = table('Size', [options.MaxLevel, 3], ...
                            'VariableTypes', {'double', 'double', 'double'}, ...
                            'VariableNames', {'Level', 'AIC', 'Cp'});
            models = cell(options.MaxLevel, 1);
            
            for level = 1:options.MaxLevel
                % A. Train Model (Strategy handles feature expansion)
                mdl = Strategy.TrainFcn(X_input, Y_idx, level);
                models{level} = mdl;
                k = Strategy.CountParamsFcn(mdl);
                
                % B. CRITERION 1: AIC
                % Formula: AIC = Deviance + 2k
                dev = Strategy.GetDevianceFcn(mdl, X_input, Y_idx, level);
                results.AIC(level) = dev + 2 * k;
                
                % C. CRITERION 2: Cp
                % Formula: Cp = (RSS / sigma^2) - N + 2k
                P_pred = Strategy.PredictFcn(mdl, X_input, level);
                RSS_d  = OceanStats.Selection.ComplexityAudit.calcTotalRSS(P_pred, Y_idx, K);
                results.Cp(level) = OceanStats.ModelSelection.Criteria.mallows(RSS_d, sigma_sq_used, N, k);
                
                results.Level(level) = level;
            end
            
            % --- 5. Package Results ---
            [~, idx_aic] = min(results.AIC);
            [~, idx_cp]  = min(results.Cp);
            
            report = struct();
            report.Table = results;
            report.Strategy = Strategy;
            report.Best.AIC = struct('Level', results.Level(idx_aic), 'Model', models{idx_aic});
            report.Best.Cp  = struct('Level', results.Level(idx_cp),  'Model', models{idx_cp});
            report.Data = struct('X', X_input, 'Y_idx', Y_idx, 'Classes', classes);
            report.Meta = struct('N', N, 'K', K, 'SigmaSq', sigma_sq_used);
        end
        
        function plotComparison(report, vizEngine, options)
            % PLOTCOMPARISON Generates the 4-panel diagnostic figure.
            %
            % Visualizes:
            %   1. Decision Terraces (Where the model draws the line).
            %   2. Uncertainty Topology (Where the model is confused).
            
            arguments
                report (1, 1) struct
                vizEngine (1, 1) % Expecting OceanViz.Core object
                options.Colors (:, 3) double = [] % Optional: Custom RGB matrix
            end
            
            X = report.Data.X;
            Y_idx = report.Data.Y_idx;
            Strategy = report.Strategy;
            K = report.Meta.K;
            classes = report.Data.Classes;
            
            % Default Colors: Use lines() if none provided
            if isempty(options.Colors)
                cls_colors = lines(K);
            else
                cls_colors = options.Colors;
            end
            
            % 1. Grid Setup (80x80 resolution for smooth surfaces)
            [gx, gy] = meshgrid(linspace(min(X(:,1)), max(X(:,1)), 80), ...
                                linspace(min(X(:,2)), max(X(:,2)), 80));
            
            grid_struct = struct('gx', gx, 'gy', gy, 'X', X, 'Y_idx', Y_idx, ...
                                 'classes', classes, 'K', K, 'colors', cls_colors);

            % --- FIGURE 1: Decision Terraces ---
            vizEngine.createFigure('Audit: Decision Boundaries');
            t1 = tiledlayout(1, 2, 'TileSpacing', 'compact');
            
            % Plot AIC Winner
            lvl_aic = report.Best.AIC.Level;
            OceanStats.Selection.ComplexityAudit.renderTile(nexttile(t1), vizEngine, ...
                grid_struct, Strategy, lvl_aic, report.Best.AIC.Model, ...
                sprintf('AIC Winner (Level %d)', lvl_aic), "Decision");
            
            % Plot Cp Winner
            lvl_cp = report.Best.Cp.Level;
            OceanStats.Selection.ComplexityAudit.renderTile(nexttile(t1), vizEngine, ...
                grid_struct, Strategy, lvl_cp, report.Best.Cp.Model, ...
                sprintf('Cp Winner (Level %d)', lvl_cp), "Decision");

            % --- FIGURE 2: Uncertainty Topology ---
            vizEngine.createFigure('Audit: Uncertainty Topology');
            t2 = tiledlayout(1, 2, 'TileSpacing', 'compact');
            
            % Plot AIC Entropy
            OceanStats.Selection.ComplexityAudit.renderTile(nexttile(t2), vizEngine, ...
                grid_struct, Strategy, lvl_aic, report.Best.AIC.Model, ...
                sprintf('AIC Entropy (Level %d)', lvl_aic), "Entropy");
                
            % Plot Cp Entropy
            OceanStats.Selection.ComplexityAudit.renderTile(nexttile(t2), vizEngine, ...
                grid_struct, Strategy, lvl_cp, report.Best.Cp.Model, ...
                sprintf('Cp Entropy (Level %d)', lvl_cp), "Entropy");
        end

        % -----------------------------------------------------------------
        % STRATEGY FACTORY (ADAPTER PATTERN)
        % -----------------------------------------------------------------
        function s = getLogisticStrategy()
            %GETLOGISTICSTRATEGY Returns the recipe for Polynomial Logistic
            %Regression.
            %We use 'mnrfit' (Multinomial Logistic Regression) and
            %'mnrval'.
            % Level = Poynomial Degree.
            s.Name = "Polynomial Logistic Regression";
            s.TrainFcn = @(X, Y, level)mnrfit(OceanStats.Data.polyFeatures(X, level), Y);

            s.PredictFcn = @(model, X, level) mnrval(model, OceanStats.Data.polyFeatures(X, level));

            s.CountParamsFcn = @(model) numel(model); 

            %Deviance Calculation: -2*Loglikelihood
            s.GetDevianceFcn = @(model, X, Y, level) ...
                OceanStats.Selection.ComplexityAudit.calcDevianceLogistic(...
                model, OceanStats.Data.polyFeatures(X, level), Y);
        end

    end
    methods (Static, Access = private)

        % --- RENDERING CORE ---
        function renderTile(ax, viz, grid, Strategy, level, model, title_txt, mode)
            %Helper to render a single visualization tile.

            %1. Predict on Grid
            X_grid = [grid.gx(:), grid.gy(:)];
            probs = Strategy.PredictFcn(model, X_grid, level);

            if mode == "Decision"
                [~, Y_pred] = max(probs, [], 2); %Winner-Takes-All
                Z = reshape(Y_pred, size(grid.gx)); %Reshape back to 2D grid
                z_label = "Class ID";
                alpha_val = 0.6 %semi-transparent


            else 
                % Formula: H(x) = - Σ p(x) * ln(p(x))
                epsilon = 1e-10; %Stabability to prevent ln(0) = -Inf
                H_vals = -sum(probs .* log(probs + epsilon), 2);
                Z = reshape(H_vals, size(grid.gx));
                z_label = "Entropy";
                alpha_val = 0.8;

            end

            % 2. Rendering Surface
            data_s = struct('X', grid.gx, 'Y', grid.gy, 'Z', Z, 'RawX', [], 'RawY', [], 'RawZ', []);
            viz.plot(ax, "Surface", data_s, ...
                'Title', title_txt, 'XLabel', 'PC1', 'YLabel', 'PC2', 'ZLabel', z_label, ...
                'ShowContour', (mode=="Entropy"), 'FaceAlpha', alpha_val, 'ShowScatter', false);
            
            hold(ax, 'on');

            % 3. Calculate Real Point Heights (For 3D scatter)
            if mode == "Entropy"
                %Calculate Entropy for actual data points
                probs_real = Strategy.PredictFcn(model, grid.X, level);
                real_Z = -sum(probs_real .* log(probs_real + 1e-10), 2);
            end

            % 4. Render Scatter Points ( Z-lift technique)
            h_leg = gobjects(grid.K, 1);
            for k = 1:grid.K
                idx = (grid.Y_idx == k); %Filter for class K
                if mode == "Decision"
                    % Lift: Place point at (ClassID + 0.1)
                    z_pos = ones(sum(idx), 1)*k + 0.1;

                else
                    % Lift: Place point at (Entropy + 0.05)
                    z_pos = real_Z(idx) + 0.05;
                end
                h_leg(k) = scatter3(ax, grid.X(idx,1), grid.X(idx,2), z_pos, ...
                    25, grid.colors(k,:), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.1);
                    
            end
            %5. Final Decoration
            if mode == "Decision"
                legend(ax, h_leg, string(grid.classes), 'Location', 'best', ...
                    'Color', 'w', 'Box', 'on');
                zticks(ax, 1:grid.K); 
                zticklabels(ax, string(grid.classes));
            else
                colormap(ax, 'jet'); 
                c = colorbar(ax); c.Label.String = 'Bits';
            end
            view(ax, [25, 50]); 
            hold(ax, 'off');
        end

        % --- MATH HELPERS ---
        
        function RSS = calcTotalRSS(probs, Y_idx, K)
            % Computes Total Residual Sum of Squares (RSS).
            % RSS = Sum( (y_true - y_pred)^2 ) across all K classes.
            RSS = 0;
            for k = 1:K
                Y_binary_k = double(Y_idx == k); % One-hot vector for class k
                RSS = RSS + sum((Y_binary_k - probs(:, k)).^2);
            end
        end
        
        function dev = calcDevianceLogistic(B, X_poly, Y_idx)
            % Computes Deviance for Multinomial Logistic Regression.
            % Deviance = -2 * LogLikelihood
            
            pi_hat = mnrval(B, X_poly);
            
            % Extract probability of the TRUE class for each sample
            idx_lin = sub2ind(size(pi_hat), (1:length(Y_idx))', Y_idx);
            p_correct = pi_hat(idx_lin);
            
            epsilon = 1e-10; % Log stability
            dev = -2 * sum(log(p_correct + epsilon));
        end
    end
end