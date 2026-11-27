classdef InputStage < OceanViz.App.NLP.SpeechStudio.Stages.StageBase
    % INPUTSTAGE  1 - Input & conditioning: raw vs conditioned waveform, the speech-activity
    %   detector's frame energy against its threshold with the detected segments, and the
    %   conditioner's level / clipping / DC / rate report.

    methods
        function t = title(~)
            t = "Input & conditioning";
        end

        function c = requires(~)
            c = strings(1, 0);
        end

        function tf = needsRecognition(~)
            tf = false;
        end

        function buildPlots(obj, grid)
            obj.addSlots(grid, ["Raw", "Conditioned", "Detector"], 3, 1);
        end

        function [rows, classes] = draw(obj, ctx)
            r = ctx.Result;
            in = r.Input;
            off = obj.segmentOffset(ctx);
            ax = obj.slotAxes("Raw");
            obj.plot(ax, "SpeechWaveform", struct('Samples', in.Samples, 'SampleRate', in.SampleRate, ...
                'Title', sprintf("Raw input: %s (%g Hz, %d ch)", ctx.InputLabel, in.SampleRate, in.numChannels())));
            obj.addTimeAxes(ax, off);

            w = r.Conditioned;
            ax = obj.slotAxes("Conditioned");
            obj.plot(ax, "SpeechWaveform", struct('Samples', w.Samples, 'SampleRate', w.SampleRate, ...
                'Title', sprintf("Conditioned (AudioConditioner): %g Hz, %.3f s", w.SampleRate, w.duration())));
            obj.addTimeAxes(ax, off);

            info = ctx.DetectorInfo;
            seg = r.Segments;
            iv = zeros(0, 2);
            if ~isempty(seg), iv = [[seg.Start]', [seg.End]']; end
            if isfield(info, 'Energy') && ~isempty(info.Energy)
                ax = obj.slotAxes("Detector");
                d = struct('X', info.FrameTimes, 'Series', struct('Name', "Frame energy E_t (dB)", 'Y', info.Energy), ...
                    'XLabel', "Time (s)", 'YLabel', "Energy (dB)", 'Intervals', iv, ...
                    'IntervalName', sprintf("Speech segments (%d)", size(iv, 1)), ...
                    'RefLines', obj.thresholds(info), ...
                    'Title', sprintf("SpeechActivityDetector (Sensitivity %g, used: %s)", ...
                    ctx.Settings.Sensitivity, string(ctx.Settings.UseDetector)));
                obj.plot(ax, "SpeechContour", d);
                obj.addTimeAxes(ax, off);
            else
                msg = "Speech-activity diagnostics unavailable";
                if isfield(info, 'Error'), msg = msg + ": " + info.Error; end
                obj.emptySlot("Detector", msg);
            end

            rep = r.Report;
            rows = strings(0, 2);
            f = ["InputRate", "OutputRate", "Channels", "DurationIn", "DurationOut", "PeakDBFS", ...
                "RMSdBFS", "ClippedFraction", "DCOffset", "GainDB"];
            for k = f
                if isfield(rep, k), rows(end+1, :) = [k, obj.fmt(rep.(k))]; end %#ok<AGROW>
            end
            rows(end+1, :) = ["Input samples", obj.shape(in.Samples)];
            rows(end+1, :) = ["Conditioned samples", obj.shape(w.Samples)];
            rows(end+1, :) = ["Segments [start, end] s", obj.segText(seg)];
            if isfield(rep, 'Steps')
                for s = reshape(string(rep.Steps), 1, [])
                    rows(end+1, :) = ["Step", s]; %#ok<AGROW>
                end
            end
            for h = reshape(w.History, 1, [])
                rows(end+1, :) = ["Waveform.History", h]; %#ok<AGROW>
            end
            if isfield(rep, 'Warnings')
                for s = reshape(string(rep.Warnings), 1, [])
                    rows(end+1, :) = ["Warning", s]; %#ok<AGROW>
                end
            end
            classes = ["OceanNLP.Speech.Inference.AudioConditioner", "OceanNLP.Speech.Inference.SpeechActivityDetector", ...
                "OceanNLP.Speech.Inference.RecognitionPipeline", "OceanNLP.Speech.Core.Waveform"];
        end

        function L = notation(~)
            L = ["$\tilde{x}[n] = g\,\big(x[n] - \bar{x}\big)$, resampled to $f_s$", ...
                "$\mathrm{dBFS} = 20\log_{10}\sqrt{\frac{1}{S}\sum_n x[n]^2}$", ...
                "$E_t = 10\log_{10}\big(\frac{1}{L}\sum_{n \in t} x[n]^2 + 10^{-12}\big)$ dBFS", ...
                "$\theta_{on} = N + a R,\; \theta_{off} = \max(N + aR/2, E_{min}),\; a = 0.6 - 0.4s$", ...
                "speech run: $E_t \ge \theta_{off}$ throughout, $\ge \theta_{on}$ somewhere", ...
                "$N, P$: 10 \% / 95 \% quantiles of $E_t$; $R = P - N$; $s$: sensitivity", ...
                "$x[n]$: input sample $n$; $\bar{x}$: DC offset; $g$: gain", ...
                "$f_s$: model sample rate; $S$: number of samples", ...
                "$E_t$: energy of frame $t$ ($L$ samples); $E_{min}$: absolute floor (dBFS)"];
        end

        function s = description(~)
            s = ["Real-world audio is conditioned before feature extraction: channels are mixed down, " + ...
                "the DC offset is removed, the signal is resampled to the rate the model was trained on " + ...
                "(OceanMath.Signal.ResampleSignal), clipping is measured and the level can be normalised " + ...
                "to a target RMS (bounded by a maximum gain).", "", ...
                "The speech-activity detector computes a short-time energy per frame, compares it with an " + ...
                "adaptive threshold (Sensitivity moves it), applies hysteresis and minimum speech / silence " + ...
                "durations and pads the result. Long recordings are split at pauses into segments that each " + ...
                "go through the recogniser; the merged transcript carries the detector's segment times.", "", ...
                "Every step is appended to Waveform.History, listed in the table on the right."];
        end
    end

    methods (Static, Access = private)
        function off = segmentOffset(ctx)
            off = ctx.segmentStart();
        end

        function r = thresholds(info)
            r = struct('Name', "On threshold theta", 'Y', info.Threshold(1));
            if isfield(info, 'ThresholdOff') && ~isempty(info.ThresholdOff) && info.ThresholdOff(1) ~= info.Threshold(1)
                r(2) = struct('Name', "Off threshold (hysteresis)", 'Y', info.ThresholdOff(1));
            end
        end

        function s = fmt(v)
            if isnumeric(v) || islogical(v)
                s = strjoin(compose("%.6g", double(v(:)')), ", ");
            else
                s = strjoin(string(v(:)'), ", ");
            end
        end

        function s = segText(seg)
            if isempty(seg), s = "none"; return; end
            s = strjoin(compose("[%.2f, %.2f]", [[seg.Start]; [seg.End]]'), " ");
        end
    end
end
