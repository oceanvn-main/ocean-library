classdef FeaturesStage < OceanViz.App.NLP.SpeechStudio.Stages.StageBase
    % FEATURESSTAGE  5 - Features: the model's input features (recognizer.Features.forward),
    %   the per-utterance CMVN-normalised input the network consumes, and the pitch (YIN)
    %   and log-energy contours on the same frame grid.

    methods
        function t = title(~)
            t = "Features";
        end

        function c = requires(~)
            c = strings(1, 0);
        end

        function buildPlots(obj, grid)
            obj.addSlots(grid, ["Features", "Normalised", "Pitch", "Energy"], 2, 2);
        end

        function [rows, classes] = draw(obj, ctx)
            F = ctx.artifact("Model features");
            if isempty(F), error('No "Model features" step was recorded.'); end
            ax = obj.slotAxes("Features");
            obj.plot(ax, "SpeechFeatures", F);
            obj.addTimeAxes(ax);
            N = ctx.artifact("Normalised input");
            if isempty(N)
                obj.emptySlot("Normalised", "No normalised-input step recorded");
            else
                ax = obj.slotAxes("Normalised");
                obj.plot(ax, "SpeechFeatures", N);
                obj.addTimeAxes(ax);
            end
            p = ctx.pitch();
            ax = obj.slotAxes("Pitch");
            obj.plot(ax, "PitchContour", p);
            obj.addTimeAxes(ax);
            e = ctx.energy();
            ax = obj.slotAxes("Energy");
            obj.plot(ax, "SpeechContour", struct('X', e.FrameTimes, ...
                'Series', struct('Name', "E_t = ln sum_n x_t[n]^2", 'Y', e.Values), ...
                'XLabel', "Time (s), frame centres", 'YLabel', string(e.Units), ...
                'Title', "Energy contour (Features.EnergyContour)"));
            obj.addTimeAxes(ax);
            voiced = p.F0Hz > 0;
            rows = [ ...
                "Feature type", string(F.Config.Type); ...
                "Features (F x T)", obj.shape(F.Values); ...
                "Units", string(F.Units); ...
                "Model InputNorm", obj.inputNorm(ctx); ...
                "Pitch frames voiced", sprintf("%d of %d", nnz(voiced), numel(voiced)); ...
                "Median F0 (voiced)", obj.medianText(p.F0Hz(voiced)); ...
                "YIN threshold", sprintf("%g", p.VoicingThreshold); ...
                "Energy range", sprintf("[%.2f, %.2f]", min(e.Values), max(e.Values))];
            if ~isempty(N)
                v = N.Values;
                rows = [rows; "Normalised (F x T)", obj.shape(v); ...
                    "Normalised mean / std", sprintf("%.2g / %.3f", mean(v(:)), std(v(:)))];
            end
            classes = ["OceanNLP.Speech.Features." + string(F.Config.Type), "OceanNLP.Speech.Features.PitchContour", ...
                "OceanNLP.Speech.Features.EnergyContour", "OceanNLP.Speech.Inspect.SpeechProcessAdapter"];
        end

        function L = notation(~)
            L = ["$\tilde{x}_t = (x_t - \mu) \oslash \sqrt{\sigma^2 + 10^{-8}}$", ...
                "$\mu = \frac{1}{T}\sum_t x_t,\;\; \sigma^2 = \frac{1}{T}\sum_t (x_t - \mu)^2$", ...
                "YIN: $d'(\tau)$ cumulative-mean difference; voiced if $d'(\tau^*) < \theta$", ...
                "$x_t$: feature vector of frame $t$ ($F$ values); $T$: frames", ...
                "$\mu, \sigma^2$: per-utterance mean, variance (CMVN)", ...
                "$F_0 = f_s / \tau^*$: fundamental frequency"];
        end

        function s = description(~)
            s = ["The features are exactly what the recogniser consumes: recognizer.Features.forward(waveform) " + ...
                "(here log-mel or MFCC), recorded by SpeechProcessAdapter.", "", ...
                "Per-utterance cepstral mean and variance normalisation (CMVN) removes the speaker / channel " + ...
                "offset that the log turned additive, so each feature has zero mean and unit variance over the " + ...
                "utterance's valid frames.", "", ...
                "Pitch (YIN, Features.PitchContour) and log energy (Features.EnergyContour) are NOT model inputs " + ...
                "here; they are computed on the same frame grid to help read the features: voiced frames carry " + ...
                "F0 harmonics, silence has low energy."];
        end
    end

    methods (Static, Access = private)
        function s = inputNorm(ctx)
            s = "?";
            if isprop(ctx.Recognizer, 'InputNorm'), s = string(ctx.Recognizer.InputNorm); end
        end

        function s = medianText(f)
            if isempty(f), s = "no voiced frames"; else, s = sprintf("%.1f Hz", median(f)); end
        end
    end
end
