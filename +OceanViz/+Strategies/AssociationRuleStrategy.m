classdef AssociationRuleStrategy < OceanViz.Strategies.BaseStrategy
    % ASSOCIATIONRULESTRATEGY  Scatter plot of association rules (confidence vs lift).
    %
    %   Standard Market Basket Analysis visualization:
    %     X-axis: Confidence (rule reliability)
    %     Y-axis: Lift (how much better than random)
    %     Dot size: proportional to support (rule coverage)
    %     Color: by itemset size k (pair, triple, etc.)
    %     Reference line: Lift = 1.0 (random baseline)
    %
    %   DATA STRUCT:
    %     data.confidence      — double vector
    %     data.lift            — double vector
    %     data.support         — double vector
    %     data.itemsetSize     — double vector (for color grouping)
    %     data.labels          — cell array of strings (for annotation)
    %     data.nAnnotate       — (optional) number of top rules to label (default: 5)
    %
    %   See also: RuleGenerator, RuleFilter, OceanViz.Core

    methods
        function plot(obj, ax, data, options)
            
            conf = data.confidence;
            lift = data.lift;
            sup  = data.support;
            kSizes = data.itemsetSize;
            nPts = numel(conf);
            
            if isfield(data, 'labels')
                labels = data.labels;
            else
                labels = {};
            end
            
            if isfield(data, 'nAnnotate')
                nAnnotate = data.nAnnotate;
            else
                nAnnotate = 5;
            end
            
            % --- Color by itemset size ---
            uniqueK = unique(kSizes);
            nGroups = numel(uniqueK);
            palette = obj.Parent.getColorSequence(nGroups);
            
            % --- Dot size proportional to support ---
            supNorm = (sup - min(sup)) / max(max(sup) - min(sup), 1e-10);
            dotSizes = 60 + supNorm * 400;
            
            hold(ax, 'on');
            
            % --- Glow layer: large faint circles behind dots for depth ---
            for g = 1:nGroups
                kVal = uniqueK(g);
                mask = (kSizes == kVal);
                scatter(ax, conf(mask), lift(mask), dotSizes(mask) * 2.5, ...
                    'MarkerFaceColor', palette(g, :), ...
                    'MarkerEdgeColor', 'none', ...
                    'MarkerFaceAlpha', 0.08, ...
                    'HandleVisibility', 'off');
            end
            
            % --- Main dot layer ---
            legendEntries = gobjects(nGroups, 1);
            for g = 1:nGroups
                kVal = uniqueK(g);
                mask = (kSizes == kVal);
                
                legendEntries(g) = scatter(ax, ...
                    conf(mask), lift(mask), dotSizes(mask), ...
                    'MarkerFaceColor', palette(g, :), ...
                    'MarkerEdgeColor', min(palette(g, :) * 0.5 + 0.1, 1), ...
                    'MarkerFaceAlpha', 0.75, ...
                    'MarkerEdgeAlpha', 0.9, ...
                    'LineWidth', 1.0, ...
                    'DisplayName', sprintf('k = %d', kVal));
            end
            
            % --- Reference line: Lift = 1.0 ---
            xRange = [min(conf) - 0.02, max(conf) + 0.02];
            plot(ax, xRange, [1.0, 1.0], ':', ...
                'Color', [obj.Parent.Colors.Text, 0.35], ...
                'LineWidth', 1.2, ...
                'HandleVisibility', 'off');
            
            text(ax, xRange(2), 1.02, 'random baseline', ...
                'Color', obj.Parent.Colors.Text * 0.5, ...
                'FontSize', 8, 'FontAngle', 'italic', ...
                'VerticalAlignment', 'bottom', ...
                'HorizontalAlignment', 'right');
            
            % --- Smart annotation: top rules by lift, avoid overlap ---
            if ~isempty(labels) && nAnnotate > 0
                [sortedLift, topIdx] = sort(lift, 'descend');
                nLabel = min(nAnnotate, numel(topIdx));
                
                % Track placed label positions to avoid overlap
                placedX = [];
                placedY = [];
                minSepX = (xRange(2) - xRange(1)) * 0.12;
                minSepY = (max(lift) - min(lift)) * 0.08;
                
                for i = 1:nLabel
                    idx = topIdx(i);
                    lx = conf(idx);
                    ly = lift(idx);
                    
                    % Check if too close to an existing label
                    tooClose = false;
                    for p = 1:numel(placedX)
                        if abs(lx - placedX(p)) < minSepX && abs(ly - placedY(p)) < minSepY
                            tooClose = true;
                            break;
                        end
                    end
                    if tooClose, continue; end
                    
                    labelStr = labels{idx};
                    if numel(labelStr) > 35
                        labelStr = [labelStr(1:32), '...'];
                    end
                    
                    % Offset the text with a leader line
                    offX = (xRange(2) - xRange(1)) * 0.03;
                    offY = (max(lift) - min(lift)) * 0.05;
                    
                    % Alternate offset direction based on position
                    if lx > median(conf)
                        offX = -offX;
                        hAlign = 'right';
                    else
                        hAlign = 'left';
                    end
                    
                    % Leader line
                    plot(ax, [lx, lx + offX], [ly, ly + offY], '-', ...
                        'Color', [obj.Parent.Colors.Text, 0.3], ...
                        'LineWidth', 0.8, ...
                        'HandleVisibility', 'off');
                    
                    % Label
                    text(ax, lx + offX, ly + offY, ['  ', labelStr], ...
                        'Color', obj.Parent.Colors.Text * 0.9, ...
                        'FontSize', 8.5, ...
                        'FontWeight', 'bold', ...
                        'VerticalAlignment', 'bottom', ...
                        'HorizontalAlignment', hAlign, ...
                        'Interpreter', 'none', ...
                        'BackgroundColor', [obj.Parent.Colors.Background, 0.7], ...
                        'Margin', 2);
                    
                    placedX(end+1) = lx;
                    placedY(end+1) = ly;
                end
            end
            
            hold(ax, 'off');
            
            % --- Formatting ---
            title(ax, options.Title, ...
                'Color', obj.Parent.Colors.Text, ...
                'FontSize', 15, 'FontWeight', 'bold');
            xlabel(ax, options.XLabel, ...
                'Color', obj.Parent.Colors.Text, 'FontSize', 12);
            ylabel(ax, options.YLabel, ...
                'Color', obj.Parent.Colors.Text, 'FontSize', 12);
            
            % Axis padding
            yRange = [min(lift) * 0.95, max(lift) * 1.08];
            xlim(ax, xRange);
            ylim(ax, yRange);
            
            if options.ShowLegend
                lg = legend(ax, legendEntries, 'Location', 'northwest', ...
                    'TextColor', obj.Parent.Colors.Text, ...
                    'Color', [obj.Parent.Colors.Background * 1.2, 0.85], ...
                    'EdgeColor', obj.Parent.Colors.Grid, ...
                    'FontSize', 9);
                title(lg, 'Itemset Size');
            end
        end
    end
end
