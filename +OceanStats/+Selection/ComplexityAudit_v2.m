classdef ComplexityAudit_v2
    % COMPLEXITYAUDIT_V2 (Moves to +OceanStats/+ModelSelection/+Audits)
    % A rigorous engine for Bias-Variance trade-off analysis.
    %   Refactored 2026-02-04: Uses Strategy Pattern for Visualization.
    
    methods (Static)
        
        function report = execute(X, Y, userOpts)
            % EXECUTE Runs the complexity analysis pipeline.
            % (Math Engine remains unchanged)
            
            arguments
                X (:, :) double
                Y (:, 1) double
                userOpts (1, 1) struct = struct()
            end
            
            % 1. VALIDATION
            requiredFields = ["Task", "MaxLevel"];
            for f = requiredFields
                if ~isfield(userOpts, f)
                    error('OceanStats:MissingConfig', ...
                        'Configuration field "%s" is missing. Please define it explicitly.', f);
                end
            end
            
            % 2. MERGE DEFAULTS
            options = userOpts;
            if ~isfield(options, 'UseParallel'), options.UseParallel = true; end
            if ~isfield(options, 'SigmaSq'),     options.SigmaSq = NaN;      end
            if ~isfield(options, 'Strategy'),    options.Strategy = struct(); end
            if ~isfield(options, 'BootstrapIters'), options.BootstrapIters = 0; end
            
            if ~ismember(options.Task, ["Regression", "Classification"])
                error('OceanStats:InvalidTask', 'Task must be "Regression" or "Classification".');
            end
            
            % 3. STRATEGY RETRIEVAL
            Strategy = options.Strategy;
            if isempty(fieldnames(Strategy))
                if options.Task == "Classification"
                    Strategy = OceanStats.ModelSelection.Audits.ComplexityAudit_v2.getLogisticStrategy();
                else
                    Strategy = OceanStats.ModelSelection.Audits.ComplexityAudit_v2.getRegressionStrategy();
                end
            end
            validateStrategy(Strategy);
            
            % 4. DATA PREP
            [N, ~] = size(X);
            if options.Task == "Classification"
                [~, ~, Y_internal] = unique(Y);
                classes = unique(Y);
                K = length(classes);
            else
                Y_internal = Y;
                classes = [];
                K = 1;
            end
            
            % 5. NOISE ESTIMATION
            if ~isnan(options.SigmaSq)
                sigma_sq_used = options.SigmaSq;
            else
                if options.UseParallel
                     mdl_sat = feval(Strategy.TrainFcn, X, Y_internal, options.MaxLevel);
                else
                     mdl_sat = Strategy.TrainFcn(X, Y_internal, options.MaxLevel);
                end
                k_sat = Strategy.CountParamsFcn(mdl_sat); 
                
                if options.Task == "Regression"
                    if options.UseParallel
                         y_hat_sat = feval(Strategy.PredictFcn, mdl_sat, X, options.MaxLevel);
                    else
                         y_hat_sat = Strategy.PredictFcn(mdl_sat, X, options.MaxLevel);
                    end
                    rss_sat = sum((Y_internal - y_hat_sat).^2);
                    denom = max(1, N - k_sat);
                    sigma_sq_used = rss_sat / denom;
                else
                    sigma_sq_used = 1; 
                end
            end
            
            % 6. SELECTION LOOP
            levels = 1:options.MaxLevel;
            aic_vals = zeros(options.MaxLevel, 1);
            bic_vals = zeros(options.MaxLevel, 1);
            cp_vals  = zeros(options.MaxLevel, 1);
            boot_vals = nan(options.MaxLevel, 1);
            models   = cell(options.MaxLevel, 1);
            
            TrainFcn = Strategy.TrainFcn; PredictFcn = Strategy.PredictFcn;
            CountFcn = Strategy.CountParamsFcn; DevianceFcn = Strategy.GetDevianceFcn;
            
            doBoot = options.BootstrapIters > 0;
            nBoot = options.BootstrapIters;
            isBigJob = (N > 1000) || (options.MaxLevel > 10) || doBoot;
            use_par = options.UseParallel && isBigJob;
            
            if use_par
                parfor lvl = 1:options.MaxLevel
                    warning('off', 'stats:mnrfit:IterLimit');
                    
                    mdl = feval(TrainFcn, X, Y_internal, lvl);
                    k = feval(CountFcn, mdl);
                    
                    if options.Task == "Classification"
                        dev = feval(DevianceFcn, mdl, X, Y_internal, lvl);
                        logL = -0.5 * dev; 
                        aic_vals(lvl) = OceanStats.ModelSelection.Criteria.aic(logL, k, 'Variant', 'Standard');
                        bic_vals(lvl) = OceanStats.ModelSelection.Criteria.bic(logL, k, 'N', N, 'Type', 'LogLik');
                        cp_vals(lvl) = NaN;
                    else
                        y_pred = feval(PredictFcn, mdl, X, lvl);
                        rss = sum((Y_internal - y_pred).^2);
                        aic_vals(lvl) = OceanStats.ModelSelection.Criteria.aic(rss, k, 'N', N, 'Variant', 'AICc');
                        bic_vals(lvl) = OceanStats.ModelSelection.Criteria.bic(rss, k, 'N', N, 'Type', 'RSS');
                        cp_vals(lvl)  = OceanStats.ModelSelection.Criteria.mallows(rss, sigma_sq_used, N, k);
                    end
                    models{lvl} = mdl;
                    
                    if doBoot
                        boot_vals(lvl) = runBootstrapInternal(X, Y_internal, lvl, nBoot, TrainFcn, PredictFcn, options.Task);
                    end
                    warning('on', 'stats:mnrfit:IterLimit');
                end
            else
                for lvl = 1:options.MaxLevel
                    warning('off', 'stats:mnrfit:IterLimit');
                    mdl = TrainFcn(X, Y_internal, lvl);
                    models{lvl} = mdl;
                    k = CountFcn(mdl);
                    
                    if options.Task == "Classification"
                        dev = DevianceFcn(mdl, X, Y_internal, lvl);
                        logL = -0.5 * dev; 
                        aic_vals(lvl) = OceanStats.ModelSelection.Criteria.aic(logL, k, 'Variant', 'Standard');
                        bic_vals(lvl) = OceanStats.ModelSelection.Criteria.bic(logL, k, 'N', N, 'Type', 'LogLik');
                        cp_vals(lvl) = NaN;
                    else
                        y_pred = PredictFcn(mdl, X, lvl);
                        rss = sum((Y_internal - y_pred).^2);
                        aic_vals(lvl) = OceanStats.ModelSelection.Criteria.aic(rss, k, 'N', N, 'Variant', 'AICc');
                        bic_vals(lvl) = OceanStats.ModelSelection.Criteria.bic(rss, k, 'N', N, 'Type', 'RSS');
                        cp_vals(lvl)  = OceanStats.ModelSelection.Criteria.mallows(rss, sigma_sq_used, N, k);
                    end
                    
                    if doBoot
                        boot_vals(lvl) = runBootstrapInternal(X, Y_internal, lvl, nBoot, TrainFcn, PredictFcn, options.Task);
                    end
                    warning('on', 'stats:mnrfit:IterLimit');
                end
            end
            
            % 7. PACKAGE RESULTS
            dataVars = {levels', aic_vals, bic_vals, cp_vals};
            varNames = {'Level', 'AIC', 'BIC', 'Cp'};
            if doBoot
                dataVars{end+1} = boot_vals; varNames{end+1} = 'Bootstrap';
            end
            results = table(dataVars{:}, 'VariableNames', varNames);
            
            report = struct();
            report.Best = struct();
            
            function addBest(key, vec)
                if all(isnan(vec)), return; end
                [~, idx] = min(vec);
                report.Best.(key) = struct('Level', results.Level(idx), 'Model', models{idx});
            end
            
            addBest('AIC', aic_vals);
            addBest('BIC', bic_vals);
            addBest('Cp',  cp_vals);
            if doBoot, addBest('Bootstrap', boot_vals); end

            report.Table = results;
            report.Strategy = Strategy;
            report.Task = options.Task;
            report.Data = struct('X', X, 'Y', Y_internal, 'Classes', classes);
            report.Meta = struct('N', N, 'K', K, 'SigmaSq', sigma_sq_used);
        end
        
        function plotComparison(report, vizEngine, compareKeys, options)
            % PLOTCOMPARISON Refactored to use OceanViz.VizFactory.
            arguments
                report (1, 1) struct
                vizEngine (1, 1) 
                compareKeys (1, :) string = ["AIC", "BIC"] 
                options.Colors (:, 3) double = [] 
                options.GroundTruthFcn = [] 
                options.ShowGradient (1, 1) logical = true
            end
            
            Theme = vizEngine.Colors;
            
            % 1. METRICS PANEL
            vizEngine.createFigure(sprintf('Audit: %s Metrics', report.Task));
            ax = gca;
            renderMetricsPanel(vizEngine, ax, report.Table, compareKeys, Theme);
            
            % 2. MODEL VISUALIZATION PANEL (The New Architecture)
            vizEngine.createFigure(['Audit: ' char(strjoin(compareKeys, ' vs '))]);
            % Changed 'compact' to 'loose' to separate the plots
            t = tiledlayout(1, length(compareKeys), 'TileSpacing', 'loose', 'Padding', 'loose');
            
            for key = compareKeys
                if ~isfield(report.Best, key)
                    warning('Metric %s not found. Skipping.', key);
                    continue;
                end
                
                % A. Prepare the "Render Context" (Standardized Data Struct)
                bestRun = report.Best.(key);
                plotData = struct();
                plotData.X = report.Data.X;
                plotData.Y = report.Data.Y; % Internal Numeric Y
                plotData.Model = bestRun.Model;
                plotData.Level = bestRun.Level;
                plotData.Strategy = report.Strategy;
                plotData.Classes = report.Data.Classes;
                plotData.Meta = report.Meta;
                
                % B. Configure Options
                plotOpts = options;
                plotOpts.Title = sprintf('%s Winner (d=%d)', key, bestRun.Level);
                
                % C. THE MAGIC: Ask Factory for a Plotter
                axTile = nexttile(t);
                plotter = OceanViz.VizFactory.getStrategy(plotData);
                
                % D. Execute Draw
                plotter.draw(axTile, plotData, plotOpts, Theme);
            end
        end
        
        % -----------------------------------------------------------------
        % MODEL STRATEGIES
        % -----------------------------------------------------------------
        function s = getLogisticStrategy()
            s.Name = "Polynomial Logistic Regression";
            s.TrainFcn = @(X, Y, level) mnrfit(OceanStats.Data.polyFeatures(X, level), Y);
            s.PredictFcn = @(model, X, level) mnrval(model, OceanStats.Data.polyFeatures(X, level));
            s.CountParamsFcn = @(model) numel(model); 
            s.GetDevianceFcn = @(model, X, Y, level) ...
                calcDevianceLogistic(model, OceanStats.Data.polyFeatures(X, level), Y);
        end
        
        function s = getRegressionStrategy()
            s.Name = "Polynomial OLS Regression";
            s.TrainFcn = @(X, Y, level) OceanStats.Data.polyFeatures(X, level) \ Y;
            s.PredictFcn = @(model, X, level) OceanStats.Data.polyFeatures(X, level) * model;
            s.CountParamsFcn = @(model) numel(model);
            s.GetDevianceFcn = @(m,x,y,l) NaN; 
        end
    end
end 

% =========================================================================
% HELPER FUNCTIONS (Math Only - No Viz)
% =========================================================================
function err = runBootstrapInternal(X, Y, lvl, B, TrainFcn, PredictFcn, Task)
    [N, ~] = size(X);
    mdl_full = feval(TrainFcn, X, Y, lvl);
    if Task == "Regression"
        y_hat = feval(PredictFcn, mdl_full, X, lvl);
        apparent = mean((Y - y_hat).^2);
    else
        probs = feval(PredictFcn, mdl_full, X, lvl);
        [~, y_hat] = max(probs, [], 2);
        apparent = mean(Y ~= y_hat); 
    end
    
    oob_accum = 0; valid_iters = 0;
    for b = 1:B
        idx = randsample(N, N, true);
        oob_idx = setdiff(1:N, idx);
        if isempty(oob_idx), continue; end
        X_boot = X(idx, :); Y_boot = Y(idx);
        X_oob  = X(oob_idx, :); Y_oob  = Y(oob_idx);
        try
            mdl_b = feval(TrainFcn, X_boot, Y_boot, lvl);
            if Task == "Regression"
                pred_b = feval(PredictFcn, mdl_b, X_oob, lvl);
                err_b = mean((Y_oob - pred_b).^2);
            else
                probs_b = feval(PredictFcn, mdl_b, X_oob, lvl);
                [~, pred_b] = max(probs_b, [], 2);
                err_b = mean(Y_oob ~= pred_b);
            end
            oob_accum = oob_accum + err_b;
            valid_iters = valid_iters + 1;
        catch
        end
    end
    if valid_iters > 0, oob_err = oob_accum / valid_iters; err = 0.368 * apparent + 0.632 * oob_err; else, err = NaN; end
end

function validateStrategy(s)
    reqFields = ["TrainFcn", "PredictFcn", "CountParamsFcn", "GetDevianceFcn"];
    for f = reqFields
        if ~isfield(s, f) || ~isa(s.(f), 'function_handle')
            error("OceanStats:InvalidStrategy", "Strategy struct must contain function handle: '%s'", f);
        end
    end
end

function dev = calcDevianceLogistic(B, X_poly, Y_idx)
    pi_hat = mnrval(B, X_poly);
    idx_lin = sub2ind(size(pi_hat), (1:length(Y_idx))', Y_idx);
    p_correct = pi_hat(idx_lin);
    dev = -2 * sum(log(p_correct + 1e-10));
end

function renderMetricsPanel(viz, ax, tbl, keys, Theme)
    hold(ax, 'on');
    colors = viz.getColorSequence(length(keys));
    for i = 1:length(keys)
        k = keys(i);
        if ~ismember(k, tbl.Properties.VariableNames), continue; end
        y_vals = tbl.(k);
        plot(ax, tbl.Level, y_vals, '-o', 'LineWidth', 2, 'Color', colors(i,:), 'MarkerFaceColor', Theme.Background, 'DisplayName', k);
    end
    title(ax, 'Model Selection Criteria', 'Interpreter', 'latex');
    xlabel(ax, 'Complexity Level', 'Interpreter', 'latex');
    ylabel(ax, 'Score (Lower is Better)', 'Interpreter', 'latex');
    viz.styleAxes(ax);
    legend(ax, 'Location', 'best', 'Interpreter', 'latex');
    hold(ax, 'off');
end
% classdef ComplexityAudit_v2
%     % COMPLEXITYAUDIT_V2 A rigorous engine for Bias-Variance trade-off analysis.
%     %   Integrated with OceanViz.Core for dynamic, theme-aware visualization.
% 
%     methods (Static)
% 
%         function report = execute(X, Y, userOpts)
%             % EXECUTE Runs the complexity analysis pipeline.
%             % INPUTS:
%             %   X, Y: Data
%             %   userOpts: (Optional) Configuration structure
% 
%             arguments
%                 X (:, :) double
%                 Y (:, 1) double
%                 userOpts (1, 1) struct = struct()
%             end
% 
%             % 1. VALIDATION
%             requiredFields = ["Task", "MaxLevel"];
%             for f = requiredFields
%                 if ~isfield(userOpts, f)
%                     error('OceanStats:MissingConfig', ...
%                         'Configuration field "%s" is missing. Please define it explicitly.', f);
%                 end
%             end
% 
%             % 2. MERGE DEFAULTS
%             options = userOpts;
%             if ~isfield(options, 'UseParallel'), options.UseParallel = true; end
%             if ~isfield(options, 'SigmaSq'),     options.SigmaSq = NaN;      end
%             if ~isfield(options, 'Strategy'),    options.Strategy = struct(); end
% 
%             if ~ismember(options.Task, ["Regression", "Classification"])
%                 error('OceanStats:InvalidTask', 'Task must be "Regression" or "Classification".');
%             end
% 
%             % -----------------------------------------------------------
%             % ENGINE START
%             % -----------------------------------------------------------
% 
%             % 3. Strategy Retrieval
%             Strategy = options.Strategy;
%             if isempty(fieldnames(Strategy))
%                 if options.Task == "Classification"
%                     Strategy = OceanStats.Selection.ComplexityAudit_v2.getLogisticStrategy();
%                 else
%                     Strategy = OceanStats.Selection.ComplexityAudit_v2.getRegressionStrategy();
%                 end
%             end
%             validateStrategy(Strategy);
% 
%             % 4. Data Filtering & Prep
%             [N, ~] = size(X);
%             if options.Task == "Classification"
%                 [~, ~, Y_internal] = unique(Y);
%                 classes = unique(Y);
%                 K = length(classes);
%             else
%                 Y_internal = Y;
%                 classes = [];
%                 K = 1;
%             end
% 
%             % 5. Noise Estimation (Sigma^2)
%             if ~isnan(options.SigmaSq)
%                 sigma_sq_used = options.SigmaSq;
%             else
%                 % Estimate from Saturated Model
%                 if options.UseParallel
%                      mdl_sat = feval(Strategy.TrainFcn, X, Y_internal, options.MaxLevel);
%                 else
%                      mdl_sat = Strategy.TrainFcn(X, Y_internal, options.MaxLevel);
%                 end
% 
%                 k_sat = Strategy.CountParamsFcn(mdl_sat); 
% 
%                 if options.Task == "Regression"
%                     if options.UseParallel
%                          y_hat_sat = feval(Strategy.PredictFcn, mdl_sat, X, options.MaxLevel);
%                     else
%                          y_hat_sat = Strategy.PredictFcn(mdl_sat, X, options.MaxLevel);
%                     end
%                     rss_sat = sum((Y_internal - y_hat_sat).^2);
%                     denom = max(1, N - k_sat);
%                     sigma_sq_used = rss_sat / denom;
%                 else
%                     sigma_sq_used = 1; 
%                 end
%             end
% 
%             % 6. The Selection Loop
%             levels = 1:options.MaxLevel;
%             aic_vals = zeros(options.MaxLevel, 1);
%             bic_vals = zeros(options.MaxLevel, 1);
%             cp_vals  = zeros(options.MaxLevel, 1);
%             models   = cell(options.MaxLevel, 1);
% 
%             TrainFcn = Strategy.TrainFcn; PredictFcn = Strategy.PredictFcn;
%             CountFcn = Strategy.CountParamsFcn; DevianceFcn = Strategy.GetDevianceFcn;
% 
%             isBigJob = (N > 1000) || (options.MaxLevel > 10);
%             use_par = options.UseParallel && isBigJob;
% 
%             if use_par
%                 parfor lvl = 1:options.MaxLevel
%                     % Suppress iteration limit warnings during grid search
%                     warning('off', 'stats:mnrfit:IterLimit');
% 
%                     mdl = feval(TrainFcn, X, Y_internal, lvl);
%                     k = feval(CountFcn, mdl);
% 
%                     if options.Task == "Classification"
%                         dev = feval(DevianceFcn, mdl, X, Y_internal, lvl);
%                         logL = -0.5 * dev; 
%                         aic_vals(lvl) = OceanStats.ModelSelection.aic(logL, k, 'Variant', 'Standard');
%                         bic_vals(lvl) = OceanStats.ModelSelection.bic(logL, k, 'N', N, 'Type', 'LogLik');
%                         cp_vals(lvl)  = NaN;
%                     else
%                         y_pred = feval(PredictFcn, mdl, X, lvl);
%                         rss = sum((Y_internal - y_pred).^2);
%                         aic_vals(lvl) = OceanStats.ModelSelection.aic(rss, k, 'N', N, 'Variant', 'AICc');
%                         bic_vals(lvl) = OceanStats.ModelSelection.bic(rss, k, 'N', N, 'Type', 'RSS');
%                         cp_vals(lvl)  = OceanStats.ModelSelection.mallows(rss, sigma_sq_used, N, k);
%                     end
%                     models{lvl} = mdl;
%                 end
%                 warning('on', 'stats:mnrfit:IterLimit'); % Restore warnings
%             else
%                 for lvl = 1:options.MaxLevel
%                     warning('off', 'stats:mnrfit:IterLimit');
%                     mdl = TrainFcn(X, Y_internal, lvl);
%                     warning('on', 'stats:mnrfit:IterLimit');
% 
%                     models{lvl} = mdl;
%                     k = CountFcn(mdl);
% 
%                     if options.Task == "Classification"
%                         dev = DevianceFcn(mdl, X, Y_internal, lvl);
%                         logL = -0.5 * dev; 
%                         aic_vals(lvl) = OceanStats.ModelSelection.aic(logL, k, 'Variant', 'Standard');
%                         bic_vals(lvl) = OceanStats.ModelSelection.bic(logL, k, 'N', N, 'Type', 'LogLik');
%                         cp_vals(lvl) = NaN;
%                     else
%                         y_pred = PredictFcn(mdl, X, lvl);
%                         rss = sum((Y_internal - y_pred).^2);
%                         aic_vals(lvl) = OceanStats.ModelSelection.aic(rss, k, 'N', N, 'Variant', 'AICc');
%                         bic_vals(lvl) = OceanStats.ModelSelection.bic(rss, k, 'N', N, 'Type', 'RSS');
%                         cp_vals(lvl)  = OceanStats.ModelSelection.mallows(rss, sigma_sq_used, N, k);
%                     end
%                 end
%             end
% 
%             % 7. Package Results
%             results = table(levels', aic_vals, bic_vals, cp_vals, 'VariableNames', {'Level', 'AIC', 'BIC', 'Cp'});
% 
%             [~, idx_aic] = min(results.AIC);
%             [~, idx_bic] = min(results.BIC); 
% 
%             % Only compute Cp winner if applicable
%             if all(isnan(results.Cp))
%                  idx_cp = 1; % Dummy
%             else
%                  [~, idx_cp] = min(results.Cp);
%             end
% 
%             report = struct();
%             report.Table = results;
%             report.Strategy = Strategy;
%             report.Task = options.Task;
%             report.Best.AIC = struct('Level', results.Level(idx_aic), 'Model', models{idx_aic});
%             report.Best.BIC = struct('Level', results.Level(idx_bic), 'Model', models{idx_bic});
%             if report.Task == "Regression"
%                  report.Best.Cp = struct('Level', results.Level(idx_cp), 'Model', models{idx_cp});
%             end
%             report.Data = struct('X', X, 'Y', Y_internal, 'Classes', classes);
%             report.Meta = struct('N', N, 'K', K, 'SigmaSq', sigma_sq_used);
%         end
% 
%         function plotComparison(report, vizEngine, compareKeys, options)
%             % PLOTCOMPARISON Generates side-by-side diagnostic figures.
% 
%             arguments
%                 report (1, 1) struct
%                 vizEngine (1, 1) 
%                 compareKeys (1, :) string = ["AIC", "BIC"] % Default Comparison
%                 options.Colors (:, 3) double = [] 
%                 options.GroundTruthFcn = [] 
%                 options.ShowGradient (1, 1) logical = true
%             end
% 
%             Theme = vizEngine.Colors;
% 
%             % 1. Prepare Models to Visualize based on input keys
%             modelsToViz = struct();
%             validKeys = [];
%             for key = compareKeys
%                 if isfield(report.Best, key)
%                     modelsToViz.(key) = report.Best.(key);
%                     validKeys = [validKeys, key]; %#ok<AGROW>
%                 else
%                     warning('OceanStats:MetricNotFound', 'Metric "%s" not found in report. Skipping.', key);
%                 end
%             end
%             if isempty(fieldnames(modelsToViz)), error("No valid models found to visualize."); end
% 
%             % 2. Common Panel: Bias-Variance Metrics
%             vizEngine.createFigure(sprintf('Audit: %s Metrics', report.Task));
%             ax = gca;
%             labels = ["$AIC$ (Prediction)", "$BIC$ (Consistency)"];
%             if report.Task == "Regression", labels(1) = "$AIC_c$ (Prediction)"; end
%             renderMetricsPanel(vizEngine, ax, report.Table, labels);
% 
%             % 3. Specific Panels
%             if report.Task == "Classification"
%                 % FIX: Pass 'options' struct directly, it contains .Colors
%                 plotClassification(report, vizEngine, modelsToViz, validKeys, options, Theme);
%             else
%                 plotRegression(report, vizEngine, modelsToViz, validKeys, options, Theme);
%             end
%         end
% 
%         % -----------------------------------------------------------------
%         % STRATEGY FACTORY (Public Static)
%         % -----------------------------------------------------------------
%         function s = getLogisticStrategy()
%             s.Name = "Polynomial Logistic Regression";
%             s.TrainFcn = @(X, Y, level) mnrfit(OceanStats.Data.polyFeatures(X, level), Y);
%             s.PredictFcn = @(model, X, level) mnrval(model, OceanStats.Data.polyFeatures(X, level));
%             s.CountParamsFcn = @(model) numel(model); 
%             s.GetDevianceFcn = @(model, X, Y, level) ...
%                 calcDevianceLogistic(model, OceanStats.Data.polyFeatures(X, level), Y);
%         end
% 
%         function s = getRegressionStrategy()
%             s.Name = "Polynomial OLS Regression";
%             s.TrainFcn = @(X, Y, level) OceanStats.Data.polyFeatures(X, level) \ Y;
%             s.PredictFcn = @(model, X, level) OceanStats.Data.polyFeatures(X, level) * model;
%             s.CountParamsFcn = @(model) numel(model);
%             s.GetDevianceFcn = @(m,x,y,l) NaN; 
%         end
%     end
% end 
% 
% % =========================================================================
% % LOCAL FUNCTIONS
% % =========================================================================
% function validateStrategy(s)
%     reqFields = ["TrainFcn", "PredictFcn", "CountParamsFcn", "GetDevianceFcn"];
%     for f = reqFields
%         if ~isfield(s, f) || ~isa(s.(f), 'function_handle')
%             error("OceanStats:InvalidStrategy", "Strategy struct must contain function handle: '%s'", f);
%         end
%     end
% end
% function renderMetricsPanel(viz, ax, tbl, labels)
%     data = struct('X', tbl.Level, 'Y1', tbl.AIC, 'Y2', tbl.BIC);
%     viz.plot(ax, 'DualMetrics', data, ...
%         'Title', 'Bias-Variance Tradeoff', 'XLabel', 'Complexity Level', ...
%         'YLabel', labels(1), 'YLabelRight', labels(2), 'LegendLabels', labels);
% end
% function gridData = generateGrid(X)
%     gridData = struct();
%     if size(X, 2) > 2
%         [coeff, score] = pca(X);
%         gridData.X_vis = score(:, 1:2);
%         gridData.coeff = coeff;
%     else
%         gridData.X_vis = X;
%         gridData.coeff = eye(2);
%     end
%     res = 40; 
%     [gridData.gx, gridData.gy] = meshgrid(...
%         linspace(min(gridData.X_vis(:,1)), max(gridData.X_vis(:,1)), res), ...
%         linspace(min(gridData.X_vis(:,2)), max(gridData.X_vis(:,2)), res));
% end
% function plotRegression(report, viz, modelsToViz, keys, options, Theme)
%     Strategy = report.Strategy;
%     X = report.Data.X; Y = report.Data.Y; 
%     numPlots = numel(keys);
%     gridData = generateGrid(X);
%     if size(X,2) > 2, X_in = [gridData.gx(:) gridData.gy(:)] * gridData.coeff(:,1:2)' + mean(X);
%     else, X_in = [gridData.gx(:) gridData.gy(:)]; end
%     scatterData = struct('X', gridData.X_vis(:,1), 'Y', gridData.X_vis(:,2), 'Z', Y);
%     viz.createFigure('Audit: Regression Comparison');
%     t = tiledlayout(1, numPlots, 'TileSpacing', 'compact', 'Padding', 'compact');
%     for i = 1:numPlots
%         key = keys(i); mdlStruct = modelsToViz.(key); ax = nexttile(t);
%         if size(X, 2) == 1
%             [X_sort, ~] = sort(X); Y_pred = Strategy.PredictFcn(mdlStruct.Model, X_sort, mdlStruct.Level);
%             scatter(ax, X, Y, 25, 'MarkerEdgeColor', 'none', 'MarkerFaceColor', Theme.Scatter, 'MarkerFaceAlpha', 0.5, 'DisplayName', 'Data'); hold(ax, 'on');
%             plot(ax, X_sort, Y_pred, 'LineWidth', 2, 'Color', Theme.Primary, 'DisplayName', 'Best Fit'); 
%             if ~isempty(options.GroundTruthFcn), plot(ax, X_sort, options.GroundTruthFcn(X_sort), '--', 'Color', Theme.Secondary, 'LineWidth', 1.5, 'DisplayName', 'Truth'); end
%             title(ax, sprintf('%s Winner (d=%d)', key, mdlStruct.Level), 'Interpreter', 'latex'); viz.styleAxes(ax); legend(ax, 'Location', 'best');
%         else
%             Z_surf = reshape(Strategy.PredictFcn(mdlStruct.Model, X_in, mdlStruct.Level), size(gridData.gx));
%             opts = struct('Mode', 'Continuous', 'ZLabel', 'Predicted Y', 'Title', sprintf('%s Winner (d=%d)', key, mdlStruct.Level), 'GroundTruthFcn', options.GroundTruthFcn, 'Theme', Theme, 'ShowGradient', options.ShowGradient);
%             renderManifold(viz, ax, gridData, Z_surf, scatterData, opts);
%         end
%     end
% end
% function plotClassification(report, viz, modelsToViz, keys, options, Theme)
%     Strategy = report.Strategy;
%     X = report.Data.X; Y_idx = report.Data.Y; 
%     numPlots = numel(keys);
% 
%     customColors = options.Colors;
%     if isempty(customColors), customColors = lines(report.Meta.K); end
% 
%     gridData = generateGrid(X);
%     if size(X,2) > 2, X_in = [gridData.gx(:) gridData.gy(:)] * gridData.coeff(:,1:2)' + mean(X);
%     else,            X_in = [gridData.gx(:) gridData.gy(:)]; end
% 
%     % --- FIGURE 1: Decision Boundaries ---
%     viz.createFigure('Audit: Decision Boundaries');
%     t1 = tiledlayout(1, numPlots, 'TileSpacing', 'compact', 'Padding', 'compact');
% 
%     for i = 1:numPlots
%         key = keys(i);
%         mdlStruct = modelsToViz.(key);
%         ax = nexttile(t1);
%         probs = Strategy.PredictFcn(mdlStruct.Model, X_in, mdlStruct.Level);
%         [~, Y_pred] = max(probs, [], 2);
%         Z_dec = reshape(Y_pred, size(gridData.gx));
%         scatA = struct('X', gridData.X_vis(:,1), 'Y', gridData.X_vis(:,2), 'Z', Y_idx + 0.1, 'Color', Y_idx, 'Map', customColors);
%         optsA = struct('Mode', 'Discrete', 'ZLabel', 'Class', 'Title', sprintf('%s Winner (d=%d)', key, mdlStruct.Level), 'Classes', report.Data.Classes, 'Theme', Theme, 'ShowGradient', false); 
%         renderManifold(viz, ax, gridData, Z_dec, scatA, optsA);
%     end
% 
%     % --- FIGURE 2: Uncertainty Topology ---
%     viz.createFigure('Audit: Uncertainty Topology');
%     t2 = tiledlayout(1, numPlots, 'TileSpacing', 'compact', 'Padding', 'compact');
%     epsilon = 1e-10;
%     for i = 1:numPlots
%         key = keys(i);
%         mdlStruct = modelsToViz.(key);
%         ax = nexttile(t2);
%         probs = Strategy.PredictFcn(mdlStruct.Model, X_in, mdlStruct.Level);
%         H_vals = -sum(probs .* log(probs + epsilon), 2); 
%         Z_ent = reshape(H_vals, size(gridData.gx));
%         probs_real = Strategy.PredictFcn(mdlStruct.Model, X, mdlStruct.Level);
%         H_real = -sum(probs_real .* log(probs_real + epsilon), 2);
%         scatB = struct('X', gridData.X_vis(:,1), 'Y', gridData.X_vis(:,2), 'Z', H_real + 0.05, 'Color', Y_idx, 'Map', customColors);
%         optsB = struct('Mode', 'Entropy', 'ZLabel', 'Entropy (Bits)', 'Title', sprintf('%s Entropy (d=%d)', key, mdlStruct.Level), 'Theme', Theme, 'ShowGradient', options.ShowGradient, 'Classes', report.Data.Classes); 
%         renderManifold(viz, ax, gridData, Z_ent, scatB, optsB);
%     end
% end
% function renderManifold(viz, ax, gridData, Z, scatterData, opts)
%     hold(ax, 'on');
%     handles = [];
%     hSurf = layerSurface(ax, gridData, Z, opts);
%     if isvalid(hSurf), handles = [handles; hSurf]; end
%     layerVectorFields(ax, gridData, Z, opts);
%     hTruth = layerGroundTruth(ax, gridData, opts);
%     if ~isempty(hTruth) && isvalid(hTruth), handles = [handles; hTruth]; end
%     [hScatter, hResid] = layerScatter(ax, gridData, Z, scatterData, opts);
%     if ~isempty(hScatter), handles = [handles; hScatter]; end
%     viz.styleAxes(ax); 
%     view(ax, [-30, 30]); 
%     rotate3d(ax, 'on');
%     zlabel(ax, opts.ZLabel, 'Interpreter', 'latex');
%     title(ax, opts.Title, 'Interpreter', 'latex');
%     if ~isempty(handles)
%         bgCol = opts.Theme.Background;
%         if iscolumn(bgCol), bgCol = bgCol'; end
%         lgd = legend(ax, handles, 'Location', 'best', 'Interpreter', 'latex', 'TextColor', opts.Theme.Text, 'EdgeColor', 'none', 'Color', [bgCol 0.8]);
%     end
%     hold(ax, 'off');
% end
% function s = layerSurface(ax, gridData, Z, opts)
%     s = surf(ax, gridData.gx, gridData.gy, Z);
%     faceCol = opts.Theme.Primary; 
%     set(s, 'EdgeColor', 'none', 'FaceColor', faceCol, 'FaceAlpha', 0.7, 'DisplayName', 'Model Surface');
%     camlight(ax, 'headlight'); lighting(ax, 'gouraud');
% end
% function layerVectorFields(ax, gridData, Z, opts)
%     if strcmp(opts.Mode, 'Discrete'), return; end
%     if isfield(opts, 'ShowGradient') && ~opts.ShowGradient, return; end
%     [Fx, Fy] = gradient(Z);
%     step = max(1, floor(size(gridData.gx, 1) / 15));
%     idx_x = 1:step:size(gridData.gx, 2); idx_y = 1:step:size(gridData.gx, 1);
%     arrowColor = opts.Theme.Text; 
%     drawQuiver(ax, gridData.gx, gridData.gy, Z, Fx, Fy, idx_x, idx_y, 'Gradient', arrowColor);
% end
% function drawQuiver(ax, X, Y, Z, U, V, idx_x, idx_y, name, color)
%     X_q = X(idx_y, idx_x); Y_q = Y(idx_y, idx_x); Z_q = Z(idx_y, idx_x);
%     U_q = U(idx_y, idx_x); V_q = V(idx_y, idx_x); W_q = zeros(size(U_q));
%     quiver3(ax, X_q, Y_q, Z_q, U_q, V_q, W_q, 1.5, 'Color', color, 'LineWidth', 1, 'DisplayName', name);
% end
% function m = layerGroundTruth(ax, gridData, opts)
%     m = [];
%     if isfield(opts, 'GroundTruthFcn') && ~isempty(opts.GroundTruthFcn)
%         if size(gridData.coeff, 2) > 1, X_in = [gridData.gx(:) gridData.gy(:)] * gridData.coeff(:,1:2)' + mean(gridData.X_vis); 
%         else, X_in = [gridData.gx(:) gridData.gy(:)]; end
%         try
%             Z_true = reshape(opts.GroundTruthFcn(X_in), size(gridData.gx));
%             edgeCol = opts.Theme.Secondary;
%             m = mesh(ax, gridData.gx, gridData.gy, Z_true, 'EdgeColor', edgeCol, 'FaceAlpha', 0, 'LineWidth', 1.5, 'DisplayName', 'Ground Truth');
%         catch, warning('Ground Truth Plot Failed: Function mismatch.'); end
%     end
% end
% function [handles, resid_handle] = layerScatter(ax, gridData, Z, scatterData, opts)
%     handles = []; resid_handle = [];
%     textColor = opts.Theme.Text; if iscolumn(textColor), textColor = textColor'; end
%     if strcmp(opts.Mode, 'Discrete') || strcmp(opts.Mode, 'Entropy')
%         classes = unique(scatterData.Color); nClasses = length(classes); cMap = scatterData.Map; 
%         for i = 1:nClasses
%             clsID = classes(i); idx = (scatterData.Color == clsID);
%             if isfield(opts, 'Classes') && clsID <= length(opts.Classes), lbl = string(opts.Classes(clsID)); else, lbl = sprintf('Class %d', clsID); end
%             if strcmp(opts.Mode, 'Entropy'), edgeCol = [1 1 1]; else, edgeCol = textColor; end
%             h = scatter3(ax, scatterData.X(idx), scatterData.Y(idx), scatterData.Z(idx), 25, cMap(clsID,:), 'filled', 'MarkerEdgeColor', edgeCol, 'DisplayName', lbl);
%             handles = [handles; h];
%         end
%         if strcmp(opts.Mode, 'Entropy')
%              colormap(ax, 'jet'); c = colorbar(ax); c.Label.String = 'Entropy (Bits)'; c.Label.Interpreter = 'latex'; c.Label.Color = opts.Theme.Text; c.Color = opts.Theme.Text;
%         else
%              zticks(ax, 1:nClasses); if isfield(opts, 'Classes'), zticklabels(ax, string(opts.Classes)); end
%              colormap(ax, cMap);
%         end
%     else
%         scatterColor = opts.Theme.Scatter; if iscolumn(scatterColor), scatterColor = scatterColor'; end
%         h = scatter3(ax, scatterData.X, scatterData.Y, scatterData.Z, 25, 'MarkerEdgeColor', textColor, 'MarkerFaceColor', scatterColor, 'MarkerFaceAlpha', 0.6, 'DisplayName', 'Data');
%         handles = [handles; h];
%         Z_interp = interp2(gridData.gx, gridData.gy, Z, scatterData.X, scatterData.Y);
%         lineColor = [textColor, 0.3]; 
%         resid_handle = plot3(ax, [scatterData.X, scatterData.X]', [scatterData.Y, scatterData.Y]', [scatterData.Z, Z_interp]', 'Color', lineColor, 'DisplayName', 'Residuals');
%         axis(ax, 'vis3d'); 
%     end
% end
% function dev = calcDevianceLogistic(B, X_poly, Y_idx)
%     pi_hat = mnrval(B, X_poly);
%     idx_lin = sub2ind(size(pi_hat), (1:length(Y_idx))', Y_idx);
%     p_correct = pi_hat(idx_lin);
%     dev = -2 * sum(log(p_correct + 1e-10));
% end