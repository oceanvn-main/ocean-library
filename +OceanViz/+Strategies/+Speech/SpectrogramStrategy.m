classdef SpectrogramStrategy < OceanViz.Strategies.BaseStrategy
    % SPECTROGRAMSTRATEGY  Short-time power spectrum P_t[k] as a time-frequency image in dB.
    %
    %   viz.plot(ax, "Spectrogram", data)
    %
    %   DATA CONTRACT (the "Power spectrum" artifact of SpeechProcessAdapter):
    %     data.Values          K x T non-negative power |X_t[k]|^2 (linear, not log)
    %     data.FrequencyHz     K x 1 bin frequencies f_k = k f_s / NFFT (Hz)
    %     data.FrameTimes      1 x T frame-centre times (s); [] -> frame index
    %     data.Units           (optional) unit of Values, e.g. "power |X[k]|^2"
    %     data.DynamicRangeDB  (optional, default 80) display range below the maximum
    %     data.Title           (optional)
    %
    %   DISPLAY TRANSFORM (stated on the colour bar): D = 10 log10(max(P, P_max 10^(-R/10)))
    %   with R = DynamicRangeDB, i.e. decibels of the given unit, floored R dB below the
    %   largest value so silent frames do not dominate the colour scale.
    %   Axes: x time (s), y "Frequency (Hz)"; the colour bar is the legend.

    methods
        function plot(obj, ax, data, options)
            arguments
                obj
                ax
                data struct
                options struct = struct()
            end
            U = OceanViz.Strategies.Speech.SpeechVizUtils;
            U.requireFields(data, ["Values", "FrequencyHz", "FrameTimes"], "Spectrogram");
            c = U.colors(obj);
            P = double(data.Values);
            if any(P(:) < 0)
                error('OceanViz:Strategies:Speech:SpectrogramStrategy:NegativePower', ...
                    'Spectrogram expects linear (non-negative) power; plot log features with "SpeechFeatures".');
            end
            [K, T] = size(P);
            f = reshape(double(data.FrequencyHz), [], 1);
            if numel(f) ~= K
                error('OceanViz:Strategies:Speech:SpectrogramStrategy:FrequencySize', ...
                    '%d frequencies for %d bins.', numel(f), K);
            end
            R = U.valueOr(data, 'DynamicRangeDB', 80);
            pmax = max(P(:));
            if pmax <= 0, pmax = 1; end
            D = 10 * log10(max(P, pmax * 10^(-R / 10)));
            [x, xText] = U.timeAxis(U.valueOr(data, 'FrameTimes', []), T);
            imagesc(ax, x, f, D);
            set(ax, 'YDir', 'normal');
            colormap(ax, parula(256));
            if T > 1, xlim(ax, [x(1), x(end)] + [-0.5 0.5] * (x(2) - x(1))); end
            ylim(ax, [f(1), f(end)]);
            units = string(U.valueOr(data, 'Units', "power"));
            U.label(ax, c, U.pickTitle(options, data, "Power spectrogram"), xText, "Frequency (Hz)");
            U.colorbar(ax, c, sprintf("10 log10(%s) (dB, floor %g dB below max)", units, R));
        end
    end
end
