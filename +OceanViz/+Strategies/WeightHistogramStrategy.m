classdef WeightHistogramStrategy < OceanViz.Strategies.BaseStrategy
    % WEIGHTHISTOGRAMSTRATEGY  Bar chart of weight distributions per layer.
    %
    %   Renders a histogram of edge weight values for each layer-to-layer
    %   connection pair. Optionally overlays a second (snapshot) distribution
    %   for before/after comparison.
    %
    %   COLOR CODING (std-based health):
    %     Blue   — healthy spread (He/Glorot range)
    %     Amber  — collapsed (std < 0.05 → vanishing gradient risk)
    %     Red    — exploding (std > 3.0 → exploding gradient risk)
    %
    %   Usage (single distribution):
    %     data  = graph.getWeightHistData();
    %     strat = OceanViz.Strategies.WeightHistogramStrategy();
    %     strat.plot(ax, data, struct());
    %
    %   Usage (comparison with snapshot):
    %     snap  = graph.saveSnapshot();
    %     data  = graph.getWeightHistData('Snapshot', snap);
    %     strat.plot(ax, data, struct());
    %
    % See also: OceanDL.Graph.NNGraph
    %           OceanDL.Graph.NNGraphMemento

    properties (Constant, Access = private)
        COLOR_HEALTHY   = [0.25 0.60 1.00]   % ocean blue
        COLOR_COLLAPSED = [0.85 0.65 0.10]   % amber
        COLOR_EXPLODING = [0.90 0.30 0.25]   % warning red
        COLOR_SNAP      = [0.60 0.90 0.60]   % snapshot overlay (green)
        COLOR_BG        = [0.08 0.08 0.10]
        COLOR_TEXT      = [0.85 0.85 0.90]
        COLOR_DIM       = [0.50 0.53 0.58]
        N_BINS          = 20
        STD_COLLAPSE    = 0.05
        STD_EXPLODE     = 3.00
    end

    methods
        function plot(obj, ax, data, options) %#ok<INUSL,INUSD>
            % PLOT  Render weight histograms for all layer pairs.
            %
            % INPUTS:
            %   ax      - MATLAB axes or UIAxes
            %   data    - struct from NNGraph.getWeightHistData()
            %             .NumPairs   (1x1) number of layer pairs
            %             .Labels     cell of 'L1→L2' strings
            %             .Weights    cell of flat weight vectors per pair
            %             .SnapWeights (optional) snapshot weights per pair
            %   options - struct (reserved)

            if nargin < 4, options = struct(); end
            hasSnap = isfield(data, 'SnapWeights') && ~isempty(data.SnapWeights);

            thm = obj.extractColors_(options);

            cla(ax, 'reset');
            hold(ax, 'on');

            nP = data.NumPairs;
            if nP == 0
                text(ax, 0.5, 0.5, 'No weight pairs to display.', ...
                    'Color', thm.Dim, 'HorizontalAlignment', 'center', ...
                    'Units', 'normalized');
                hold(ax, 'off');
                return;
            end

            % Compute global weight range for shared x-axis
            allW = [];
            for p = 1:nP, allW = [allW; data.Weights{p}(:)]; end %#ok<AGROW>
            if hasSnap
                for p = 1:nP, allW = [allW; data.SnapWeights{p}(:)]; end %#ok<AGROW>
            end
            wMin = min(allW); wMax = max(allW);
            if wMin == wMax, wMin = wMin - 1; wMax = wMax + 1; end
            edges = linspace(wMin, wMax, obj.N_BINS + 1);

            % Y offset per pair (stacked rows)
            rowH = 1.0;
            yBase = (nP - 1) * rowH;

            for p = 1:nP
                y0   = yBase - (p - 1) * rowH;
                W    = data.Weights{p}(:);
                colr = obj.pickColor_(W, thm);

                % Bar histogram (normalized to row height)
                [cnts, ~] = histcounts(W, edges);
                maxCnt    = max(cnts);
                if maxCnt < 1, maxCnt = 1; end
                barH = (cnts / maxCnt) * (rowH * 0.80);

                for b = 1:numel(barH)
                    x0 = edges(b); x1 = edges(b+1);
                    if barH(b) > 0
                        rectangle(ax, 'Position', [x0, y0, x1-x0, barH(b)], ...
                            'FaceColor', colr, 'EdgeColor', 'none');
                    end
                end

                % Snapshot overlay
                if hasSnap
                    Ws = data.SnapWeights{p}(:);
                    [cntsS, ~] = histcounts(Ws, edges);
                    maxCntS    = max(cntsS);
                    if maxCntS < 1, maxCntS = 1; end
                    barHS = (cntsS / maxCntS) * (rowH * 0.80);

                    for b = 1:numel(barHS)
                        x0 = edges(b); x1 = edges(b+1);
                        if barHS(b) > 0
                            rectangle(ax, 'Position', [x0, y0, x1-x0, barHS(b)], ...
                                'FaceColor', thm.Snap, ...
                                'EdgeColor', 'none', 'FaceAlpha', 0.35);
                        end
                    end
                end

                % Stats label
                mu  = mean(W); sg = std(W);
                txt = sprintf('%s   μ=%.2f  σ=%.2f', data.Labels{p}, mu, sg);
                text(ax, wMin, y0 + rowH * 0.88, txt, ...
                    'Color', thm.Text, 'FontSize', 8.5, ...
                    'VerticalAlignment', 'top', 'Interpreter', 'none');

                % Health badge
                badge = '';
                if sg < obj.STD_COLLAPSE,  badge = '⚠ collapse'; end
                if sg > obj.STD_EXPLODE,   badge = '⚠ explode';  end
                if ~isempty(badge)
                    text(ax, wMax, y0 + rowH * 0.88, badge, ...
                        'Color', colr, 'FontSize', 8, 'FontWeight', 'bold', ...
                        'HorizontalAlignment', 'right', ...
                        'VerticalAlignment',   'top');
                end
            end

            % Legend
            if hasSnap
                text(ax, wMin, yBase + rowH * 0.95, 'Current', ...
                    'Color', thm.Healthy, 'FontSize', 8, 'FontWeight', 'bold');
                text(ax, wMin + (wMax-wMin)*0.15, yBase + rowH * 0.95, 'Snapshot', ...
                    'Color', thm.Snap, 'FontSize', 8, 'FontWeight', 'bold');
            end

            % Axes styling
            ax.Color  = thm.Bg;
            ax.XColor = thm.Dim;
            ax.YColor = 'none';
            ax.Box    = 'off';
            ax.XLim   = [wMin - 0.05*(wMax-wMin), wMax + 0.15*(wMax-wMin)];
            ax.YLim   = [-0.1, yBase + rowH * 1.05];

            title(ax, 'Weight Distributions', ...
                'Color', thm.Text, 'FontSize', 10, 'FontWeight', 'bold');

            hold(ax, 'off');
        end
    end

    methods (Access = private)
        function c = extractColors_(obj, options)
            if isfield(options, 'Colors') && ~isempty(options.Colors)
                thmObj = options.Colors;
                c.Healthy   = thmObj.Primary;
                c.Collapsed = thmObj.Boundary;
                c.Exploding = thmObj.Secondary;
                c.Snap      = min(thmObj.Background + 0.35, 1.0);
                c.Bg        = thmObj.Background;
                c.Text      = thmObj.Text;
                c.Dim       = thmObj.Text * 0.6 + thmObj.Background * 0.4;
            else
                c.Healthy   = obj.COLOR_HEALTHY;
                c.Collapsed = obj.COLOR_COLLAPSED;
                c.Exploding = obj.COLOR_EXPLODING;
                c.Snap      = obj.COLOR_SNAP;
                c.Bg        = obj.COLOR_BG;
                c.Text      = obj.COLOR_TEXT;
                c.Dim       = obj.COLOR_DIM;
            end
        end

        function color = pickColor_(obj, W, thm)
            sg = std(W(:));
            if sg < obj.STD_COLLAPSE
                color = thm.Collapsed;
            elseif sg > obj.STD_EXPLODE
                color = thm.Exploding;
            else
                color = thm.Healthy;
            end
        end
    end
end
