classdef ActivationFlowStrategy < OceanViz.Strategies.BaseStrategy
    % ACTIVATIONFLOWSTRATEGY  Bar chart of per-layer activation health.
    %
    %   Renders a horizontal bar chart showing the mean absolute activation
    %   magnitude and health state of each layer in a neural network,
    %   based on a ForwardPassResult struct from ForwardPassEngine.
    %
    %   COLOR ENCODING:
    %     Blue  — healthy (good mean, good variance)
    %     Gray  — dead zone (relu layer with mean < DeadThreshold)
    %     Red   — saturated (sigmoid/tanh layer with high saturation rate)
    %     Amber — low variance (potential collapse)
    %
    %   Usage:
    %     engine = OceanDL.Simulation.ForwardPassEngine();
    %     result = engine.run(graph, X);
    %     strat  = OceanViz.Strategies.ActivationFlowStrategy();
    %     strat.plot(ax, result, struct());
    %
    % See also: OceanDL.Simulation.ForwardPassEngine
    %           OceanViz.Strategies.NetworkGraphStrategy

    properties (Constant, Access = private)
        COLOR_HEALTHY    = [0.25 0.60 1.00]  % ocean blue
        COLOR_DEAD       = [0.45 0.48 0.52]  % dim gray
        COLOR_SATURATED  = [0.90 0.30 0.25]  % warning red
        COLOR_COLLAPSED  = [0.85 0.65 0.10]  % amber warning
        COLOR_BG         = [0.08 0.08 0.10]
        COLOR_TEXT       = [0.85 0.85 0.90]
        COLOR_DIM        = [0.50 0.53 0.58]
        SAT_THRESHOLD    = 0.30   % >30% saturated → warn red
        STD_THRESHOLD    = 0.05   % std < 0.05 → warn amber (collapsed)
    end

    methods
        function plot(obj, ax, result, options) %#ok<INUSL,INUSD>
            % PLOT  Render the activation flow bar chart.
            %
            % INPUTS:
            %   ax     - MATLAB axes or UIAxes handle
            %   result - ForwardPassResult struct from ForwardPassEngine.run()
            %   options - (optional) struct (reserved for future use)

            if nargin < 4, options = struct(); end

            thm = obj.extractColors_(options);

            cla(ax, 'reset');
            hold(ax, 'on');

            nL     = result.NumLayers;
            means  = result.LayerMeans;
            stds   = result.LayerStd;
            satR   = result.SaturationRate;
            dead   = result.DeadNeuronMask;
            names  = result.LayerNames;
            acts   = result.Activations;

            maxMean = max(means);
            if maxMean < 1e-12, maxMean = 1; end

            yPositions = nL:-1:1;  % top = input, bottom = output

            for li = 1:nL
                y     = yPositions(li);
                val   = means(li);
                barW  = val / maxMean;  % normalized [0,1]
                color = obj.pickColor_(li, acts, satR, stds, dead, thm);

                % Draw bar
                rectangle(ax, ...
                    'Position',        [0, y - 0.35, barW, 0.70], ...
                    'FaceColor',       color, ...
                    'EdgeColor',       'none');

                % Draw background track
                rectangle(ax, ...
                    'Position',        [0, y - 0.35, 1.0, 0.70], ...
                    'FaceColor',       'none', ...
                    'EdgeColor',       thm.Dim * 0.5, ...
                    'LineWidth',       0.5);

                % Layer name label (left)
                text(ax, -0.02, y, names{li}, ...
                    'Color',               thm.Text, ...
                    'FontSize',            9, ...
                    'HorizontalAlignment', 'right', ...
                    'VerticalAlignment',   'middle', ...
                    'Interpreter',         'none');

                % Value label (right)
                text(ax, 1.03, y, ...
                    sprintf('%.2f ± %.2f', val, stds(li)), ...
                    'Color',               thm.Dim, ...
                    'FontSize',            8, ...
                    'HorizontalAlignment', 'left', ...
                    'VerticalAlignment',   'middle');

                % Warning badge
                badge = obj.buildBadge_(li, acts, satR, stds, dead);
                if ~isempty(badge)
                    text(ax, barW + 0.02, y, badge, ...
                        'Color',             color, ...
                        'FontSize',          8, ...
                        'FontWeight',        'bold', ...
                        'HorizontalAlignment','left', ...
                        'VerticalAlignment', 'middle');
                end
            end

            % Axes styling
            ax.Color       = thm.Bg;
            ax.XColor      = 'none';
            ax.YColor      = 'none';
            ax.Box         = 'off';
            ax.XLim        = [-0.35, 1.45];
            ax.YLim        = [0.3, nL + 0.7];
            ax.DataAspectRatioMode = 'auto';

            title(ax, 'Activation Flow', ...
                'Color',      thm.Text, ...
                'FontSize',   10, ...
                'FontWeight', 'bold');

            hold(ax, 'off');
        end
    end

    methods (Access = private)

        function c = extractColors_(obj, options)
            if isfield(options, 'Colors') && ~isempty(options.Colors)
                cObj = options.Colors;
                c.Healthy   = cObj.Primary;
                c.Saturated = cObj.Secondary;
                c.Collapsed = cObj.Boundary;
                c.Dead      = cObj.Text * 0.4 + cObj.Background * 0.6;
                c.Bg        = cObj.Background;
                c.Text      = cObj.Text;
                c.Dim       = cObj.Text * 0.6 + cObj.Background * 0.4;
            else
                c.Healthy   = obj.COLOR_HEALTHY;
                c.Saturated = obj.COLOR_SATURATED;
                c.Collapsed = obj.COLOR_COLLAPSED;
                c.Dead      = obj.COLOR_DEAD;
                c.Bg        = obj.COLOR_BG;
                c.Text      = obj.COLOR_TEXT;
                c.Dim       = obj.COLOR_DIM;
            end
        end

        function color = pickColor_(obj, li, acts, satR, stds, dead, thm)
            act   = lower(acts{li});
            dMask = dead{li};

            % Dead zone: relu with all neurons dead
            if strcmp(act, 'relu') && ~isempty(dMask) && all(dMask)
                color = thm.Dead;
                return;
            end

            % Saturated: sigmoid/tanh with high saturation rate
            if ismember(act, {'sigmoid','tanh'}) && satR(li) > obj.SAT_THRESHOLD
                color = thm.Saturated;
                return;
            end

            % Collapsed: variance vanishing
            if stds(li) < obj.STD_THRESHOLD
                color = thm.Collapsed;
                return;
            end

            % Healthy
            color = thm.Healthy;
        end

        function badge = buildBadge_(obj, li, acts, satR, stds, dead)
            badge = '';
            act   = lower(acts{li});
            dMask = dead{li};
            m     = stds(li);

            if strcmp(act, 'relu') && ~isempty(dMask) && all(dMask)
                % All ReLU neurons at zero — may be zero input or dead neurons
                badge = char(9744) + " all-ReLU-zero";  % ☐ symbol
            elseif ismember(act, {'sigmoid','tanh'}) && satR(li) > obj.SAT_THRESHOLD
                badge = sprintf('^ sat %.0f%%', satR(li)*100);
            elseif m < obj.STD_THRESHOLD
                if m < 1e-9
                    badge = '-- idle (zero input)';
                else
                    badge = sprintf('~ low variance (%.3f)', m);
                end
            end
        end
    end
end
