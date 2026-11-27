classdef CandlestickStrategy < OceanViz.Strategies.BaseStrategy
    % CANDLESTICKSTRATEGY  Binance-style 2-panel Candlestick + Volume chart strategy.
    %
    % OceanViz strategy wrapper around OceanViz.Charts.CandlestickChart.
    % Delegates rendering to the performance-optimized vectorized graphics engine.
    %
    % See also: OceanViz.Core, OceanViz.Charts.CandlestickChart

    methods
        function plot(obj, ax, data, options)
            arguments
                obj
                ax
                data
                options struct = struct()
            end

            colors = obj.Parent.Colors;

            % Delegate to performance-optimized CandlestickChart renderer
            OceanViz.Charts.CandlestickChart.plot(ax, data, ...
                'Theme', obj.Parent.ThemeName, ...
                'Colors', colors);
        end
    end
end
