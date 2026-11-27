classdef SpectrumStage < OceanViz.App.NLP.SpeechStudio.Stages.StageBase
    % SPECTRUMSTAGE  3 - Spectrum: the power spectrogram P_t[k] (dB display) and the
    %   spectrum of one selected frame. Click the spectrogram to pick the frame.

    properties (SetAccess = private)
        FrameIndex (1,1) double = 0
        LastKey = []
    end

    methods
        function t = title(~)
            t = "Spectrum";
        end

        function c = requires(~)
            c = "frontend";
        end

        function buildPlots(obj, grid)
            obj.addSlots(grid, ["Spectrogram", "FrameSpectrum"], 2, 1);
        end

        function [rows, classes] = draw(obj, ctx)
            P = ctx.artifact("Power spectrum");
            if isempty(P)
                error('The front end recorded no "Power spectrum" step for this model.');
            end
            T = size(P.Values, 2);
            if obj.FrameIndex < 1 || obj.FrameIndex > T || isempty(obj.LastKey) || ...
                    obj.LastKey{1} ~= ctx || obj.LastKey{2} ~= ctx.SegmentIndex
                [~, obj.FrameIndex] = max(sum(P.Values, 1));
            end
            obj.LastKey = {ctx, ctx.SegmentIndex};
            k = obj.FrameIndex;
            ax = obj.slotAxes("Spectrogram");
            obj.plot(ax, "Spectrogram", P);
            obj.addTimeAxes(ax);
            ax = obj.slotAxes("FrameSpectrum");
            db = 10 * log10(max(P.Values(:, k), max(P.Values(:)) * 1e-8));
            tk = NaN;
            if ~isempty(P.FrameTimes), tk = P.FrameTimes(k); end
            obj.plot(ax, "SpeechContour", struct('X', P.FrequencyHz(:)', ...
                'Series', struct('Name', sprintf("10 log10 P_t[k], frame t = %d", k), 'Y', db(:)'), ...
                'XLabel', "Frequency (Hz)", 'YLabel', "Power (dB, floor 80 dB below max)", ...
                'Title', sprintf("Spectrum of frame t = %d (centre %.3f s)", k, tk)));
            [~, peak] = max(P.Values(:, k));
            rows = [ ...
                "Power spectrum (K x T)", obj.shape(P.Values); ...
                "Bins K = NFFT/2 + 1", string(numel(P.FrequencyHz)); ...
                "Bin spacing f_s / NFFT", sprintf("%.2f Hz", P.FrequencyHz(2) - P.FrequencyHz(1)); ...
                "Frequency range", sprintf("0 - %g Hz", P.FrequencyHz(end)); ...
                "Units", string(P.Units); ...
                "Selected frame t", string(k); ...
                "Peak bin of frame t", sprintf("%.1f Hz", P.FrequencyHz(peak))];
            classes = ["OceanNLP.Speech.Features.LogMel", "OceanNLP.Speech.Inspect.SpeechProcessAdapter"];
        end

        function onClick(obj, t)
            if isempty(obj.Context) || isnan(t), return; end
            P = obj.Context.artifact("Power spectrum");
            if isempty(P) || isempty(P.FrameTimes), return; end
            [~, obj.FrameIndex] = min(abs(P.FrameTimes - t));
            obj.render(obj.Context);
        end

        function L = notation(~)
            L = ["$X_t[k] = \sum_{n=0}^{L-1} w[n]\,x_t[n]\,e^{-j 2\pi k n / N}$", ...
                "$P_t[k] = |X_t[k]|^2,\quad f_k = k\,f_s / N$", ...
                "Display: $10\log_{10} P_t[k]$ (dB)", ...
                "$N$: NFFT; $k$: frequency bin; $t$: frame", ...
                "$X_t[k]$: DFT of the windowed frame; $P_t[k]$: power"];
        end

        function s = description(~)
            s = ["The DFT of each windowed frame gives its spectrum; its squared magnitude is the power " + ...
                "spectrum P_t[k]. Stacking frames over time gives the spectrogram.", "", ...
                "Voiced speech shows harmonics (horizontal ridges at multiples of F0) shaped by the vocal-tract " + ...
                "resonances (formants); fricatives such as /s/ show broadband high-frequency energy.", "", ...
                "Values are the front end's own power spectrum (LogMel.intermediates); the decibel display " + ...
                "is a display transform only (the model consumes the linear power through the mel bank and log)."];
        end
    end
end
