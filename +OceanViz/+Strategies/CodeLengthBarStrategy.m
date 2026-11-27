classdef CodeLengthBarStrategy < OceanViz.Strategies.BaseStrategy
    % CODELENGTHBARSTRATEGY  Bar chart: actual vs optimal codeword lengths.
    %
    % Produces a grouped bar chart for each symbol showing:
    %   Blue bar   — Actual codeword length l_k (Huffman or any coder)
    %   Orange bar — Optimal (Shannon) length -log2(p_k)
    % Plus a red dashed horizontal line at H(X) (source entropy).
    %
    % Registration: 'CodeLengthBar' in OceanViz.Core.plot() switch block.
    %
    % Input data struct fields:
    %   data.table        (CodeTable, required)  source code table to visualize
    %   data.title        (string,  default '')  axes title override
    %   data.showEntropy  (logical, default true) show H(X) horizontal line
    %   data.showLegend   (logical, default true) show legend
    %
    % Usage:
    %   coder = OceanMath.Information.Coding.HuffmanCoder();
    %   table = coder.encode({'a','b','c','d'}, [0.4, 0.3, 0.2, 0.1]);
    %   viz = OceanViz.Core('Theme', 'Dark');
    %   viz.plot(ax, 'CodeLengthBar', struct('table', table));
    %
    % See also: OceanMath.Information.Coding.CodeTable
    %           OceanViz.Strategies.BaseStrategy

    % ------------------------------------------------------------------ %
    methods
        function plot(obj, ax, data, options)
            % PLOT  Render the code length comparison bar chart.
            %
            % INPUTS:
            %   ax      - MATLAB axes handle
            %   data    - struct with data.table (CodeTable)
            %   options - struct from OceanViz.Core

            table = data.table;
            if ~isa(table, 'OceanMath.Information.Coding.CodeTable')
                error('OceanViz:CodeLengthBarStrategy:BadInput', ...
                    'data.table must be a CodeTable. Got: %s.', class(table));
            end

            showEntropy = obj.getField_(data, 'showEntropy', true);
            showLegend  = obj.getField_(data, 'showLegend',  true);
            titleStr    = obj.getField_(data, 'title', '');
            if isfield(options, 'Title') && ~isempty(char(options.Title))
                titleStr = char(options.Title);
            end

            % ---- Theme colors ---------------------------------------- %
            colors = obj.getColors_();

            % ---- Data ------------------------------------------------ %
            K        = table.getSize();
            L_actual = table.getLengths();        % (1 x K) integers
            L_opt    = table.getOptimalLengths(); % (1 x K) fractional
            H        = table.entropyBits();
            L_avg    = table.expectedLength();
            eta      = table.efficiency();
            syms     = table.getSymbols();

            % ---- Grouped bar plot ------------------------------------ %
            hold(ax, 'on');
            barData = [L_actual(:), L_opt(:)];    % (K x 2)
            bh = bar(ax, barData, 'grouped');

            % Apply theme colors
            bh(1).FaceColor = colors.actual;
            bh(1).EdgeColor = 'none';
            bh(1).FaceAlpha = 0.88;
            bh(2).FaceColor = colors.optimal;
            bh(2).EdgeColor = 'none';
            bh(2).FaceAlpha = 0.65;

            % ---- Entropy horizontal line ----------------------------- %
            if showEntropy
                yline(ax, H, '--', ...
                    'Color', colors.entropy, 'LineWidth', 1.6, ...
                    'Label', sprintf('H(X) = %.3f bits', H), ...
                    'LabelHorizontalAlignment', 'right', ...
                    'Interpreter', 'none', ...
                    'FontSize', 8);
                yline(ax, L_avg, ':', ...
                    'Color', colors.lavg, 'LineWidth', 1.4, ...
                    'Label', sprintf('L_{avg} = %.3f', L_avg), ...
                    'LabelHorizontalAlignment', 'left', ...
                    'Interpreter', 'none', ...
                    'FontSize', 8);
            end

            % ---- Axes labels / ticks --------------------------------- %
            allProbs  = table.getProbabilities();
            symLabels = cell(1, K);
            for k = 1:K
                s = syms{k};
                if ~ischar(s), s = num2str(s); end
                symLabels{k} = sprintf('%s (%.3g)', s, allProbs(k));
            end
            ax.XTick      = 1:K;
            ax.XTickLabel = symLabels;
            ax.TickLabelInterpreter = 'none';
            xtickangle(ax, 30);
            xlabel(ax, 'Symbol (probability)', 'Color', colors.axisLabel, 'FontSize', 10);
            ylabel(ax, 'Codeword length (bits)', 'Color', colors.axisLabel, 'FontSize', 10);

            % ---- Legend --------------------------------------------- %
            if showLegend
                legend(ax, bh, {'Actual l_k', 'Optimal -log_2(p_k)'}, ...
                    'TextColor', colors.axisLabel, ...
                    'Color', 'none', ...
                    'EdgeColor', colors.axisLabel, ...
                    'FontSize', 9, ...
                    'Location', 'northeast', ...
                    'Interpreter', 'none');
            end

            % ---- Summary annotation ---------------------------------- %
            xlim_v = xlim(ax);
            ylim_v = ylim(ax);
            ann = sprintf('eta = %.1f%%  |  R = %.3f  |  K = %.5f', ...
                eta*100, table.redundancy(), table.kraftSum());
            text(ax, xlim_v(1) + diff(xlim_v)*0.01, ylim_v(2)*0.97, ann, ...
                'Color', colors.annotation, 'FontSize', 8, ...
                'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
                'Interpreter', 'none');

            % ---- Title ---------------------------------------------- %
            if isempty(titleStr)
                titleStr = sprintf('Code Lengths: %s — %d symbols', ...
                    table.CoderName_, K);
            end
            title(ax, titleStr, 'Color', colors.title, ...
                'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');

            hold(ax, 'off');
        end
    end

    % ------------------------------------------------------------------ %
    methods (Access = private)
        function colors = getColors_(obj)
            if isempty(obj.Parent)
                colors.actual      = [0.20, 0.55, 1.00];
                colors.optimal     = [1.00, 0.55, 0.10];
                colors.entropy     = [1.00, 0.30, 0.30];
                colors.lavg        = [0.40, 0.90, 0.60];
                colors.axisLabel   = [0.85, 0.85, 0.90];
                colors.annotation  = [0.70, 0.70, 0.75];
                colors.title       = [0.95, 0.95, 1.00];
            else
                C = obj.Parent.Colors;
                colors.actual      = C.Primary;
                colors.optimal     = C.Secondary;
                colors.entropy     = C.Boundary;
                colors.lavg        = [0.40, 0.90, 0.60];
                colors.axisLabel   = C.Text;
                colors.annotation  = C.Text .* 0.75;
                colors.title       = C.Text;
            end
        end

        function v = getField_(~, s, field, default)
            if isfield(s, field)
                v = s.(field);
            else
                v = default;
            end
        end
    end
end
