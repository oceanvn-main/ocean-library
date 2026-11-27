classdef MelStage < OceanViz.App.NLP.SpeechStudio.Stages.StageBase
    % MELSTAGE  4 - Mel: the triangular mel filter bank W and the log-mel spectrogram
    %   Y = ln(max(W P, floor)) of the waveform the model saw.

    methods
        function t = title(~)
            t = "Mel";
        end

        function c = requires(~)
            c = "frontend";
        end

        function buildPlots(obj, grid)
            obj.addSlots(grid, ["Bank", "LogMel"], 2, 1);
        end

        function [rows, classes] = draw(obj, ctx)
            B = ctx.artifact("Mel filter bank");
            Y = ctx.artifact("Log-mel");
            if isempty(B) || isempty(Y)
                error('The front end recorded no mel filter bank / log-mel step for this model.');
            end
            ax = obj.slotAxes("Bank");
            obj.plot(ax, "MelFilterBank", B);
            ax = obj.slotAxes("LogMel");
            obj.plot(ax, "SpeechFeatures", Y);
            obj.addTimeAxes(ax);
            rows = [ ...
                "Filter bank W (M x K)", obj.shape(B.Weights); ...
                "Mel scale", string(B.Scale); ...
                "Area normalisation", string(B.Norm); ...
                "Frequency range", sprintf("%g - %g Hz", B.FMin, B.FMax); ...
                "Log-mel Y (M x T)", obj.shape(Y.Values); ...
                "Units", string(Y.Units); ...
                "Y range", sprintf("[%.2f, %.2f]", min(Y.Values(:)), max(Y.Values(:)))];
            classes = ["OceanNLP.Speech.Features.LogMel", "OceanMath.Signal.MelFilterBank", ...
                "OceanNLP.Speech.Inspect.SpeechProcessAdapter"];
        end

        function L = notation(~)
            L = ["$\mathrm{mel}(f) = 2595\,\log_{10}(1 + f/700)$ (HTK)", ...
                "$Y_t[m] = \ln \max\big(\sum_k W_{mk} P_t[k],\; \epsilon\big)$", ...
                "$W_{mk}$: weight of filter $m$ on bin $k$ (triangle)", ...
                "$M$: number of mel filters; $\epsilon$: log floor", ...
                "$Y_t[m]$: log-mel energy of filter $m$, frame $t$"];
        end

        function L = notationFor(obj, ctx)
            % The mel-scale formula of the scale this model's filter bank actually uses.
            L = obj.notation();
            B = ctx.artifact("Mel filter bank");
            if ~isempty(B) && isfield(B, 'Scale') && string(B.Scale) == "slaney"
                L(1) = "$\mathrm{mel}(f) = \frac{3f}{200}$ for $f < 1$ kHz, logarithmic above (Slaney)";
            end
        end

        function s = description(~)
            s = ["Human pitch perception is roughly linear below 1 kHz and logarithmic above; the mel scale " + ...
                "models this. M triangular filters, equally spaced on the mel scale, pool the power spectrum " + ...
                "into M band energies (OceanMath.Signal.MelFilterBank).", "", ...
                "The logarithm compresses the dynamic range (loudness is perceived logarithmically) and turns " + ...
                "a multiplicative channel gain into an additive offset, which the model's per-utterance " + ...
                "normalisation (Features stage) then removes.", "", ...
                "The table states the mel scale (HTK or Slaney) and normalisation this model actually uses."];
        end
    end
end
