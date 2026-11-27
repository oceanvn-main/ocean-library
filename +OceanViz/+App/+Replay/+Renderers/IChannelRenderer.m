classdef (Abstract) IChannelRenderer < handle
    % ICHANNELRENDERER  Draws ONE stored value of a Replay channel on a uiaxes.
    %   Replay Studio picks the renderer by the channel's KIND (Registry), so
    %   any recording opens without domain code. Renderers never compute new
    %   values: they only draw what was stored.
    %
    %   tf = canRender(info)                  info = OceanReplay.Core.ChannelInfo / ViewInfo
    %   draw(ax, value, info, ctx)            ctx: struct with Theme (OceanViz colours),
    %                                         Recording, Coords (the value's stored coordinates)
    %
    %   Drawing pattern: prepare(ax) -> plot -> finish(ax, theme, showAxes). Styling
    %   comes LAST because high-level plot calls (image, plot, ...) reset axes
    %   properties when the axes is not held.

    methods (Abstract)
        tf = canRender(obj, info)
        draw(obj, ax, value, info, ctx)
    end

    methods (Static)
        function prepare(ax)
            cla(ax); legend(ax, 'off'); colorbar(ax, 'off'); hold(ax, 'off'); ax.Tag = '';
        end

        function finish(ax, theme, showAxes)
            % Theme the axes (OceanViz colours); axes lines/ticks only for plots.
            ax.Color = theme.Background;
            ax.XColor = theme.Text; ax.YColor = theme.Text; ax.GridColor = theme.Grid;
            ax.Title.Color = theme.Text;
            ax.FontSize = 9;
            if ~showAxes
                ax.XColor = 'none'; ax.YColor = 'none'; ax.XTick = []; ax.YTick = [];
            end
            ax.Toolbar.Visible = 'off';
            disableDefaultInteractivity(ax);
        end

        function styleLegend(lg, theme)
            lg.TextColor = theme.Text; lg.Color = 'none'; lg.EdgeColor = 'none'; lg.FontSize = 8;
            lg.Interpreter = 'none';                      % symbols as written (eps_phi, not TeX subscripts)
        end
    end
end
