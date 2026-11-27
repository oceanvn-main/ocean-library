classdef PitchContourStrategy < OceanViz.Strategies.BaseStrategy
    % PITCHCONTOURSTRATEGY  Fundamental frequency F0 (voiced frames) and voicing strength over time.
    %
    %   viz.plot(ax, "PitchContour", data)
    %   data = OceanViz.Strategies.Speech.PitchContourStrategy.fromFeatureSequence(fs)
    %          (fs = OceanNLP.Speech.Features.PitchContour(...).forward(waveform))
    %
    %   DATA CONTRACT:
    %     data.F0Hz        1 x T F0 estimate in Hz, 0 = unvoiced (not drawn)
    %     data.Voicing     1 x T voicing strength in [0, 1] (YIN: 1 - d'(tau*))
    %     data.FrameTimes  1 x T frame-centre times (s); [] -> frame index
    %     data.VoicingThreshold (optional) YIN threshold theta; the frame is voiced when
    %                      d'(tau*) < theta, i.e. strength > 1 - theta (dashed line)
    %     data.Title       (optional)
    %   Axes: x time (s); left y "F0 (Hz)"; right y "Voicing strength (0-1)".

    methods
        function plot(obj, ax, data, options)
            arguments
                obj
                ax
                data struct
                options struct = struct()
            end
            U = OceanViz.Strategies.Speech.SpeechVizUtils;
            U.requireFields(data, ["F0Hz", "Voicing"], "PitchContour");
            c = U.colors(obj);
            f0 = reshape(double(data.F0Hz), 1, []);
            v = reshape(double(data.Voicing), 1, []);
            T = numel(f0);
            if numel(v) ~= T
                error('OceanViz:Strategies:Speech:PitchContourStrategy:Size', 'F0Hz and Voicing differ in length.');
            end
            [x, xText] = U.timeAxis(U.valueOr(data, 'FrameTimes', []), T);
            voiced = f0 > 0;

            yyaxis(ax, 'left');
            hF0 = plot(ax, x(voiced), f0(voiced), 'o', 'Color', c.Primary, 'MarkerFaceColor', c.Primary, ...
                'MarkerSize', 4, 'LineStyle', 'none', 'DisplayName', "F0 (voiced frames)");
            ylabel(ax, "F0 (Hz)", 'Color', c.Text, 'Interpreter', 'none');
            if any(voiced)
                ylim(ax, [0, max(f0(voiced)) * 1.15]);
            end
            ax.YAxis(1).Color = c.Primary;

            yyaxis(ax, 'right');
            hold(ax, 'on');
            hV = plot(ax, x, v, '-', 'Color', c.Secondary, 'LineWidth', 1.2, 'DisplayName', "Voicing strength");
            handles = [hF0, hV];
            if isfield(data, 'VoicingThreshold') && ~isempty(data.VoicingThreshold)
                hT = yline(ax, 1 - data.VoicingThreshold, '--', 'Color', c.Boundary, 'LineWidth', 1, ...
                    'DisplayName', sprintf("Voicing boundary 1 - theta = %.2f", 1 - data.VoicingThreshold));
                handles(end+1) = hT;
            end
            hold(ax, 'off');
            ylim(ax, [0, 1.05]);
            ylabel(ax, "Voicing strength (0-1)", 'Color', c.Text, 'Interpreter', 'none');
            ax.YAxis(2).Color = c.Secondary;
            if T > 1, xlim(ax, [x(1), x(end)]); end

            ax.TickLabelInterpreter = 'none';
            title(ax, U.pickTitle(options, data, sprintf("Pitch contour (%d of %d frames voiced)", nnz(voiced), T)), ...
                'Color', c.Text, 'FontSize', 11, 'FontWeight', 'bold', 'Interpreter', 'none');
            xlabel(ax, xText, 'Color', c.Text, 'Interpreter', 'none');
            if ~isfield(options, 'ShowLegend') || options.ShowLegend
                U.legend(ax, c, handles, 'Location', 'northeast');
            end
        end
    end

    methods (Static)
        function data = fromFeatureSequence(fs, opts)
            % FROMFEATURESEQUENCE  Data struct of a PitchContour output (2 x T FeatureSequence).
            arguments
                fs (1,1) OceanNLP.Speech.Core.FeatureSequence
                opts.VoicingThreshold double = []
                opts.Title (1,1) string = ""
            end
            if fs.numBins() ~= 2
                error('OceanViz:Strategies:Speech:PitchContourStrategy:NotPitch', ...
                    'Expected a 2 x T pitch FeatureSequence (F0; voicing), got %d rows.', fs.numBins());
            end
            data = struct('F0Hz', fs.Values(1, :), 'Voicing', fs.Values(2, :), ...
                'FrameTimes', fs.FrameTimes, 'VoicingThreshold', opts.VoicingThreshold, 'Title', opts.Title);
        end
    end
end
