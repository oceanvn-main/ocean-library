classdef SpeechFrameStrategy < OceanViz.Strategies.BaseStrategy
    % SPEECHFRAMESTRATEGY  One analysis frame x_t[n], the analysis window w[n] and the
    %   windowed frame w[n] x_t[n] over the sample index within the frame.
    %
    %   viz.plot(ax, "SpeechFrame", data)
    %
    %   DATA CONTRACT (column t of the "Frames + window" artifact of SpeechProcessAdapter):
    %     data.Frame        L x 1 raw frame samples x_t[n], n = 0..L-1
    %     data.Window       L x 1 analysis window w[n] (as used by the front end)
    %     data.SampleRate   f_s in Hz (x axis in ms within the frame)
    %     data.FrameIndex   (optional) t, shown in the title
    %     data.FrameStart   (optional) 0-based first sample s_t, shown in the title
    %     data.WindowType   (optional) window name, shown in the legend
    %     data.Title        (optional)
    %   The window is drawn scaled by max|x_t| so its shape is visible on the same axis
    %   (the scale factor is stated in the legend). Legend: the three curves.

    methods
        function plot(obj, ax, data, options)
            arguments
                obj
                ax
                data struct
                options struct = struct()
            end
            U = OceanViz.Strategies.Speech.SpeechVizUtils;
            U.requireFields(data, ["Frame", "Window", "SampleRate"], "SpeechFrame");
            c = U.colors(obj);
            x = double(data.Frame(:));
            w = double(data.Window(:));
            if numel(x) ~= numel(w)
                error('OceanViz:Strategies:Speech:SpeechFrameStrategy:Length', ...
                    'Frame has %d samples, window %d.', numel(x), numel(w));
            end
            L = numel(x);
            tms = (0:L-1)' / double(data.SampleRate) * 1000;
            scale = max(abs(x));
            if scale == 0, scale = 1; end
            pal = obj.Parent.getColorSequence(3);
            hold(ax, 'on');
            plot(ax, tms, x, 'Color', pal(1, :), 'LineWidth', 0.9, 'DisplayName', "Raw frame x_t[n]");
            wname = string(U.valueOr(data, 'WindowType', "window"));
            plot(ax, tms, w * scale, '--', 'Color', pal(2, :), 'LineWidth', 1.2, ...
                'DisplayName', sprintf("%s window w[n] x %.3g (= max|x_t|)", wname, scale));
            plot(ax, tms, w .* x, 'Color', pal(3, :), 'LineWidth', 1.3, 'DisplayName', "Windowed w[n] x_t[n]");
            hold(ax, 'off');
            if L > 1, xlim(ax, [0, tms(end)]); end
            ttl = "Analysis frame";
            if isfield(data, 'FrameIndex'), ttl = ttl + sprintf(" t = %d", data.FrameIndex); end
            if isfield(data, 'FrameStart')
                ttl = ttl + sprintf(" (s_t = %d, %.3f s)", data.FrameStart, data.FrameStart / double(data.SampleRate));
            end
            U.label(ax, c, U.pickTitle(options, data, ttl), sprintf("Time within frame (ms), L = %d samples", L), ...
                "Amplitude (full scale)");
            if ~isfield(options, 'ShowLegend') || options.ShowLegend
                U.legend(ax, c, 'Location', 'southoutside', 'Orientation', 'horizontal');
            end
        end
    end
end
