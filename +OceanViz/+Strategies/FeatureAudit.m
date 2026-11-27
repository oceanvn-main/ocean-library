classdef FeatureAudit < handle
    % OCEANVIZ.STRATEGIES.FEATUREAUDIT
    % Strategy for comparative analysis of Error vs. Complexity.
    %
    % OVERVIEW:
    %   This class implements a Dual-Axis visualization strategy. It separates:
    %     1. LEFT AXIS: Prediction Error (MSE) - Lower is better.
    %     2. RIGHT AXIS: Information Criteria (AIC/BIC) - Lower is better (negative).
    %   This separation is crucial because AIC/BIC values (e.g., -60) have a
    %   vastly different scale than MSE (e.g., 0.5), and plotting them together
    %   on one axis would flatten the error curves.
    
    properties
        Parent % Reference to the OceanViz.Core engine (Dependency Injection)
    end
    
    methods
        function plot(obj, ax, data, options)
            % PLOT Executes the dual-axis rendering pipeline.
            %
            % PROCESS:
            %   1. Unpack Data & Theme Colors.
            %   2. Generate Robust Color Palette (handling Dark/Light modes).
            %   3. Render Left Axis (Training, CV, Bootstrap).
            %   4. Render Right Axis (AIC, BIC) if requested.
            %   5. Apply Scientific Formatting (LaTeX, Grids, Integer Ticks).
            
            arguments
                obj
                ax (1,1) matlab.graphics.axis.Axes
                data (1,1) struct
                options struct 
            end
            
            % =================================================================
            % 1. CONTEXT & COLOR SETUP
            % =================================================================
            tbl = data.Table;
            colors = obj.Parent.Colors;
            
            % ROBUSTNESS: Ensure colors are Row Vectors [1x3].
            % MATLAB errors if you try to concatenate a Column [3x1] with a scalar.
            cText = obj.ensureRow(colors.Text);
            cBack = obj.ensureRow(colors.Background);
            
            % Standard Error Metric Colors (Fallback to grey/blue/orange if missing)
            if isfield(colors, 'Tertiary'), cBias = obj.ensureRow(colors.Tertiary); else, cBias = [0.5 0.5 0.5]; end
            if isfield(colors, 'Primary'),  cVar  = obj.ensureRow(colors.Primary);  else, cVar  = [0 0.45 0.74]; end
            if isfield(colors, 'Secondary'),cGen  = obj.ensureRow(colors.Secondary);else, cGen  = [0.85 0.33 0.1]; end
            
            % CALCULATED COLORS: Complexity (Teal & Purple)
            % We hardcode these base tones for contrast against the standard palette,
            % but mathematically adjust them based on the background brightness.
            cAIC = [0.0 0.6 0.5]; % Teal
            cBIC = [0.5 0.2 0.6]; % Purple
            
            % Dark Mode Check: If background is dark (<0.5), lighten these colors
            if mean(cBack) < 0.5 
                cAIC = cAIC + 0.2; cAIC(cAIC>1)=1;
                cBIC = cBIC + 0.3; cBIC(cBIC>1)=1;
            end

            % =================================================================
            % 2. LEFT AXIS: PREDICTION ERROR (MSE)
            % =================================================================
            % This axis tracks the "Physical" error of the model.
            yyaxis(ax, 'left');
            hold(ax, 'on');
            
            % A. Training Error (Bias)
            % Plot as dashed/thinner to indicate it is a theoretical baseline,
            % not a metric for selection (prevents overfitting interpretation).
            plot(ax, tbl.K, tbl.TrainErr, '--o', ...
                'LineWidth', 1.2, 'MarkerSize', 5, 'Color', cBias, ...
                'MarkerFaceColor', cBack, 'DisplayName', 'Training Error (Bias)');
                
            % B. Cross-Validation (Variance)
            % Standard solid line. Supports multiple CV columns if present.
            cv_cols = find(startsWith(tbl.Properties.VariableNames, 'CV_'));
            for i = 1:length(cv_cols)
                col_name = tbl.Properties.VariableNames{cv_cols(i)};
                clean_name = strrep(col_name, '_', '-Fold ');
                plot(ax, tbl.K, tbl.(col_name), '-s', ...
                    'LineWidth', 1.5, 'MarkerSize', 6, 'Color', cVar, ...
                    'MarkerFaceColor', cBack, 'DisplayName', clean_name);
            end
            
            % C. Bootstrap (Generalization)
            % The "Gold Standard" estimate. Plotted thicker/distinctive.
            if ismember('Boot632', tbl.Properties.VariableNames)
                plot(ax, tbl.K, tbl.Boot632, '-d', ...
                    'LineWidth', 2.0, 'MarkerSize', 7, 'Color', cGen, ...
                    'MarkerFaceColor', cGen, 'DisplayName', '.632+ Bootstrap (Estimate)');
                    
                % ANNOTATION: Vertical line at the optimal K
                [~, idx] = min(tbl.Boot632);
                best_k = tbl.K(idx);
                % Use transparent text color for the line
                xline(ax, best_k, '--', 'Color', [cText 0.3], ...
                    'LineWidth', 1.0, 'HandleVisibility', 'off'); 
            end
            
            ylabel(ax, options.YLabel, 'Interpreter', 'latex', 'Color', cText);
            ax.YColor = cText; % Match axis ticks to text color
            
            % =================================================================
            % 3. RIGHT AXIS: INFORMATION CRITERIA (AIC/BIC)
            % =================================================================
            % This axis tracks "Model Parsimony" (Goodness of fit vs. Complexity).
            hasAIC = ismember('AIC', tbl.Properties.VariableNames);
            hasBIC = ismember('BIC', tbl.Properties.VariableNames);
            
            % LOGIC: Only show if data exists AND user didn't disable it via options
            if isfield(options, 'ShowComplexity')
                showComp = options.ShowComplexity;
            else
                showComp = true; 
            end
            
            if showComp && (hasAIC || hasBIC)
                yyaxis(ax, 'right');
                
                if hasAIC
                    plot(ax, tbl.K, tbl.AIC, ':^', 'LineWidth', 1.5, 'MarkerSize', 6, ...
                        'Color', cAIC, 'MarkerFaceColor', cBack, 'DisplayName', 'AIC (Complexity)');
                end
                
                if hasBIC
                    plot(ax, tbl.K, tbl.BIC, '-.v', 'LineWidth', 1.5, 'MarkerSize', 6, ...
                        'Color', cBIC, 'MarkerFaceColor', cBack, 'DisplayName', 'BIC (Conservative)');
                end
                
                ylabel(ax, 'Information Criteria', 'Interpreter', 'latex', 'Color', cText);
                ax.YColor = cText; % Keep axis neutral, lines are colored
            else
                % HANDLE THE "OFF" CASE:
                % If we just used "axis off", the right spine of the plot box
                % would disappear, looking unprofessional.
                yyaxis(ax, 'right');
                cla(ax); % Clear any existing plots
                
                % STRATEGY: Hide the content, keep the structure.
                set(ax, 'YTick', []);     % Remove numbers/ticks
                set(ax, 'YColor', cText); % Keep the spine visible (colored like text)
                ylabel(ax, '');           % Remove label
                
                yyaxis(ax, 'left');       % IMPORTANT: Return focus to left
            end
            
            % =================================================================
            % 4. FINAL FORMATTING
            % =================================================================
            title(ax, options.Title, 'Interpreter', 'latex', 'Color', cText);
            xlabel(ax, options.XLabel, 'Interpreter', 'latex', 'Color', cText);
            
            % Grid: Only applied to the active axis (Left)
            grid(ax, 'on');
            if isfield(colors, 'Grid'), ax.GridColor = colors.Grid; end
            ax.GridAlpha = 0.25;
            
            % X-Axis: Ensure integer ticks (since K is discrete count)
            xticks(ax, unique(floor(tbl.K)));
            xlim(ax, [min(tbl.K)-0.2, max(tbl.K)+0.2]);
            
            % Force the Box to be closed (Top and Right spines visible)
            set(ax, 'Box', 'on');
            
            % Unified Legend: Captures lines from BOTH axes automatically
            lgd = legend(ax, 'Location', 'best');
            set(lgd, 'Interpreter', 'latex', 'TextColor', cText, ...
                'EdgeColor', 'none', 'Color', [cBack 0.85]); % Semi-transparent back
            
            hold(ax, 'off');
        end
        
        function vec = ensureRow(~, vec)
            % Helper: Forces column vectors to row vectors [1xN]
            % Essential for RGBA color concatenation: [r g b] + alpha
            if iscolumn(vec), vec = vec'; end
        end
    end
end