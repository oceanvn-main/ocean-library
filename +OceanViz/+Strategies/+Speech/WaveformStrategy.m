classdef WaveformStrategy < OceanViz.Strategies.BaseStrategy
    % WAVEFORMSTRATEGY  Sampled speech waveform x[n] against time t = n / f_s.
    %
    %   viz.plot(ax, "SpeechWaveform", data)
    %
    %   DATA CONTRACT (the "Waveform" artifact of SpeechProcessAdapter has this shape):
    %     data.Samples      S x C double, one column per channel (audioread scale, |x| <= 1)
    %     data.SampleRate   f_s in Hz
    %     data.Source       (optional) source path, shown in the title
    %     data.FrameStarts  (optional) 1 x T 0-based first sample of each analysis frame
    %     data.WindowLength (optional) frame length L in samples; with FrameStarts the
    %                       first analysis frame [s_1, s_1 + L) is shaded
    %     data.Title        (optional)
    %   Axes: x "Time (s)", y "Amplitude (full scale)". Legend: one entry per channel.

    methods
        function plot(obj, ax, data, options)
            arguments
                obj
                ax
                data struct
                options struct = struct()
            end
            U = OceanViz.Strategies.Speech.SpeechVizUtils;
            U.requireFields(data, ["Samples", "SampleRate"], "SpeechWaveform");
            c = U.colors(obj);
            x = double(data.Samples);
            fs = double(data.SampleRate);
            [S, C] = size(x);
            t = (0:S-1)' / fs;
            hold(ax, 'on');
            pal = obj.Parent.getColorSequence(max(C, 1));
            if isfield(data, 'FrameStarts') && isfield(data, 'WindowLength') && ~isempty(data.FrameStarts)
                s0 = data.FrameStarts(1) / fs;
                s1 = (data.FrameStarts(1) + data.WindowLength) / fs;
                lim = max(abs(x(:))) + eps;
                patch(ax, [s0 s1 s1 s0], [-lim -lim lim lim], c.Boundary, 'FaceAlpha', 0.15, ...
                    'EdgeColor', 'none', 'DisplayName', sprintf('First analysis frame (L = %d samples)', data.WindowLength));
            end
            for ch = 1:C
                name = "Waveform";
                if C > 1, name = sprintf("Channel %d", ch); end
                plot(ax, t, x(:, ch), 'Color', pal(ch, :), 'LineWidth', 0.8, 'DisplayName', name);
            end
            hold(ax, 'off');
            if S > 1, xlim(ax, [0, t(end)]); end
            ttl = U.pickTitle(options, data, sprintf("Waveform (%g Hz, %.3f s)", fs, S / fs));
            src = string(U.valueOr(data, 'Source', ""));
            if src ~= "" && ~(isfield(options, 'Title') && string(options.Title) ~= "Analysis")
                ttl = ttl + " - " + src;
            end
            U.label(ax, c, ttl, "Time (s)", "Amplitude (full scale)");
            if ~isfield(options, 'ShowLegend') || options.ShowLegend
                U.legend(ax, c, 'Location', 'northeast');
            end
        end
    end
end
