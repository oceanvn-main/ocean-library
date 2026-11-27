classdef SpeechVizUtils
    % SPEECHVIZUTILS  Shared helpers of the +OceanViz/+Strategies/+Speech family.
    %
    %   All helpers take the theme colours from the strategy's Parent
    %   (OceanViz.Core), so every speech plot follows the active theme. Labels use
    %   Interpreter 'none' (token names such as "<blank>" or "|" are literal text;
    %   OceanViz.Core.styleAxes sets a LaTeX tick interpreter that would mangle them).
    %   Nothing here computes speech quantities: strategies only draw what the data
    %   struct holds (display transforms such as dB conversion are documented where used).

    methods (Static)
        function c = colors(strategy)
            % COLORS  Theme colours of the strategy's Parent engine (error if unset).
            if isempty(strategy.Parent) || ~isa(strategy.Parent, 'OceanViz.Core')
                error('OceanViz:Strategies:Speech:NoParent', ...
                    'Speech strategies need Parent = an OceanViz.Core (use OceanViz.Core.plot).');
            end
            c = strategy.Parent.Colors;
        end

        function requireFields(data, names, strategyName)
            % REQUIREFIELDS  Error listing every missing data field.
            missing = names(~isfield(data, cellstr(names)));
            if ~isempty(missing)
                error('OceanViz:Strategies:Speech:MissingField', '%s data needs field(s): %s', ...
                    strategyName, strjoin(missing, ', '));
            end
        end

        function t = pickTitle(options, data, fallback)
            % PICKTITLE  options.Title unless it is the Core default "Analysis", then data.Title, then FALLBACK.
            t = string(fallback);
            if isfield(data, 'Title') && strlength(string(data.Title)) > 0
                t = string(data.Title);
            end
            if isfield(options, 'Title') && string(options.Title) ~= "Analysis" && strlength(string(options.Title)) > 0
                t = string(options.Title);
            end
        end

        function label(ax, c, titleText, xText, yText)
            % LABEL  Themed title / axis labels, literal (non-TeX) text.
            ax.TickLabelInterpreter = 'none';
            title(ax, titleText, 'Color', c.Text, 'FontSize', 11, 'FontWeight', 'bold', 'Interpreter', 'none');
            xlabel(ax, xText, 'Color', c.Text, 'Interpreter', 'none');
            ylabel(ax, yText, 'Color', c.Text, 'Interpreter', 'none');
        end

        function lg = legend(ax, c, varargin)
            % LEGEND  Legend styled with the theme background / text / grid colours.
            lg = legend(ax, varargin{:});
            lg.TextColor = c.Text;
            lg.Color = c.Background;
            lg.EdgeColor = c.Grid;
            lg.Interpreter = 'none';
            lg.AutoUpdate = 'off';
        end

        function cb = colorbar(ax, c, labelText)
            % COLORBAR  Themed colour bar (the legend of an image plot) with a unit label.
            cb = colorbar(ax);
            cb.Color = c.Text;
            cb.TickLabelInterpreter = 'none';
            cb.Label.String = labelText;
            cb.Label.Color = c.Text;
            cb.Label.Interpreter = 'none';
        end

        function [x, xText] = timeAxis(frameTimes, T)
            % TIMEAXIS  Frame-centre times in seconds when given, else the frame index.
            if ~isempty(frameTimes) && numel(frameTimes) == T && all(isfinite(frameTimes))
                x = reshape(frameTimes, 1, []);
                xText = "Time (s), frame centres";
            else
                x = 1:T;
                xText = "Frame index";
            end
        end

        function s = valueOr(data, name, fallback)
            if isfield(data, name) && ~isempty(data.(name))
                s = data.(name);
            else
                s = fallback;
            end
        end
    end
end
