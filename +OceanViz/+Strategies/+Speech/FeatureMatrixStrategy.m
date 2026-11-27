classdef FeatureMatrixStrategy < OceanViz.Strategies.BaseStrategy
    % FEATUREMATRIXSTRATEGY  Any F x T speech feature matrix (log-mel, MFCC, normalised
    %   input, encoder states, logits) as an image over time.
    %
    %   viz.plot(ax, "SpeechFeatures", data)
    %   data = OceanViz.Strategies.Speech.FeatureMatrixStrategy.fromFeatureSequence(fs)
    %
    %   DATA CONTRACT (the "Log-mel", "Model features", "Normalised input",
    %   "Encoder j" and "Logits" artifacts of SpeechProcessAdapter have this shape):
    %     data.Values       F x T values (row = feature bin / unit / class, column = frame)
    %     data.FrameTimes   1 x T frame-centre times (s); [] -> frame index
    %     data.FrequencyHz  (optional) F x 1 bin centre frequencies (Hz); when given the
    %                       y axis is the bin index with centre frequencies on the ticks
    %     data.AxisNames    (optional) 1 x 2, e.g. ["mel","time"], ["cepstral","time"]
    %     data.Units        (optional) unit of Values (colour-bar label)
    %     data.Tokens       (optional) 1 x F row names (e.g. vocabulary tokens for logits)
    %     data.Title        (optional)
    %   Values are drawn as given (no rescaling); the colour bar is the legend.

    methods
        function plot(obj, ax, data, options)
            arguments
                obj
                ax
                data struct
                options struct = struct()
            end
            U = OceanViz.Strategies.Speech.SpeechVizUtils;
            U.requireFields(data, "Values", "SpeechFeatures");
            c = U.colors(obj);
            V = double(data.Values);
            [F, T] = size(V);
            [x, xText] = U.timeAxis(U.valueOr(data, 'FrameTimes', []), T);
            imagesc(ax, x, 1:F, V);
            set(ax, 'YDir', 'normal');
            colormap(ax, parula(256));
            if T > 1, xlim(ax, [x(1), x(end)] + [-0.5 0.5] * (x(2) - x(1))); end
            ylim(ax, [0.5, F + 0.5]);
            names = string(U.valueOr(data, 'AxisNames', ["feature", "time"]));
            fHz = reshape(double(U.valueOr(data, 'FrequencyHz', [])), [], 1);
            tokens = string(U.valueOr(data, 'Tokens', strings(1, 0)));
            yText = names(1) + " index";
            if numel(tokens) == F
                ax.YTick = 1:F;
                ax.YTickLabel = cellstr(tokens);
                yText = names(1);
            elseif numel(fHz) == F
                ticks = unique(round(linspace(1, F, min(F, 6))));
                ax.YTick = ticks;
                ax.YTickLabel = compose("%d: %.0f Hz", ticks(:), fHz(ticks));
                yText = names(1) + " band (index: centre frequency, Hz)";
            end
            units = string(U.valueOr(data, 'Units', "value"));
            U.label(ax, c, U.pickTitle(options, data, "Feature matrix (" + names(1) + " x " + names(2) + ")"), xText, yText);
            U.colorbar(ax, c, units);
        end
    end

    methods (Static)
        function data = fromFeatureSequence(fs, titleText)
            % FROMFEATURESEQUENCE  Data struct of an OceanNLP.Speech.Core.FeatureSequence.
            arguments
                fs (1,1) OceanNLP.Speech.Core.FeatureSequence
                titleText (1,1) string = ""
            end
            data = struct('Values', fs.Values, 'FrameTimes', fs.FrameTimes, 'FrequencyHz', fs.FrequencyHz, ...
                'AxisNames', fs.AxisNames, 'Units', fs.Units, 'Title', titleText);
        end
    end
end
