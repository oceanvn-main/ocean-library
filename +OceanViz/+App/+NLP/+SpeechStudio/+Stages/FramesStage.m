classdef FramesStage < OceanViz.App.NLP.SpeechStudio.Stages.StageBase
    % FRAMESSTAGE  2 - Frames: the frame grid over the waveform the model saw, the analysis
    %   window and one selected frame (raw and windowed). Click a time axis (or use the
    %   spinner) to select the frame; the default is the frame with the largest energy.

    properties (SetAccess = private)
        FrameIndex (1,1) double = 0
        FrameSpinner = []
        LastKey = []
    end

    methods
        function t = title(~)
            t = "Frames";
        end

        function c = requires(~)
            c = "frontend";
        end

        function buildPlots(obj, grid)
            grid.RowHeight = {'fit', '1x'};
            grid.ColumnWidth = {'1x'};
            bar = uigridlayout(grid, [1 3], 'ColumnWidth', {'fit', 90, '1x'}, 'Padding', [0 0 0 0]);
            bar.Layout.Row = 1;
            uilabel(bar, 'Text', 'Selected frame t:');
            obj.FrameSpinner = uispinner(bar, 'Limits', [1 Inf], 'Value', 1, 'Step', 1, ...
                'RoundFractionalValues', 'on', 'ValueChangedFcn', @(s, ~) obj.selectFrame(s.Value));
            uilabel(bar, 'Text', 'Click a time axis to select the frame under the pointer.', ...
                'FontAngle', 'italic');
            g = obj.addSlots(grid, ["Grid", "Frame"], 2, 1);
            g.Layout.Row = 2;
        end

        function [rows, classes] = draw(obj, ctx)
            fr = ctx.artifact("Frames + window");
            if isempty(fr)
                error('The front end recorded no "Frames + window" step for this model.');
            end
            T = size(fr.Frames, 2);
            key = {ctx, ctx.SegmentIndex};
            if obj.FrameIndex < 1 || obj.FrameIndex > T || isempty(obj.LastKey) || ...
                    obj.LastKey{1} ~= ctx || obj.LastKey{2} ~= ctx.SegmentIndex
                [~, obj.FrameIndex] = max(sum(fr.Frames .^ 2, 1));
            end
            obj.LastKey = key;
            obj.FrameSpinner.Limits = [1 max(T, 1)];
            obj.FrameSpinner.Value = obj.FrameIndex;
            obj.drawFrame(ctx, fr);
            w = ctx.segmentWave();
            rows = [ ...
                "Waveform x[n]", obj.shape(w.Samples) + " @ " + w.SampleRate + " Hz"; ...
                "Frames (L x T)", obj.shape(fr.Frames); ...
                "Window length L", string(fr.WindowLength) + sprintf(" samples (%.1f ms)", 1000 * fr.WindowLength / fr.SampleRate); ...
                "Hop H", string(fr.HopLength) + sprintf(" samples (%.1f ms)", 1000 * fr.HopLength / fr.SampleRate); ...
                "NFFT", string(fr.NFFT); ...
                "Window", string(fr.WindowType); ...
                "Selected frame t", string(obj.FrameIndex); ...
                "Frame start s_t", string(fr.FrameStarts(obj.FrameIndex)) + " samples"];
            classes = ["OceanNLP.Speech.Core.FrameGrid", "OceanNLP.Speech.Features.LogMel", ...
                "OceanNLP.Speech.Inspect.SpeechProcessAdapter"];
        end

        function onClick(obj, t)
            if isempty(obj.Context) || isnan(t), return; end
            fr = obj.Context.artifact("Frames + window");
            if isempty(fr), return; end
            [~, k] = min(abs(fr.FrameTimes - t));
            obj.selectFrame(k);
        end

        function selectFrame(obj, k)
            if isempty(obj.Context), return; end
            obj.FrameIndex = round(k);
            obj.render(obj.Context);
        end

        function L = notation(~)
            L = ["$x_t[n] = x[s_t + n],\; s_t = (t-1)H,\; n = 0..L-1$", ...
                "$\mathrm{hann}:\; w[n] = \frac{1}{2}\big(1 - \cos\frac{2\pi n}{L}\big)$ (periodic)", ...
                "$x[n]$: waveform; $x_t[n]$: frame $t$", ...
                "$L$: window length; $H$: hop; $s_t$: first sample of frame $t$", ...
                "$w[n]$: analysis window; $w[n]\,x_t[n]$: windowed frame"];
        end

        function s = description(~)
            s = ["Speech is only stationary over a few tens of milliseconds, so the waveform is cut into " + ...
                "overlapping frames of L samples every H samples (OceanNLP.Speech.Core.FrameGrid).", "", ...
                "Each frame is multiplied by a tapering window before the DFT; without it the frame edges " + ...
                "act like a rectangular window whose side lobes smear energy across frequency (leakage).", "", ...
                "The frames, window and frame starts shown are the front end's own intermediates " + ...
                "(LogMel.intermediates) recorded by SpeechProcessAdapter; the selected frame defaults to the " + ...
                "one with the largest energy sum_n x_t[n]^2."];
        end
    end

    methods (Access = private)
        function drawFrame(obj, ctx, fr)
            w = ctx.segmentWave();
            x = w.Samples(:, 1);
            t = (0:numel(x) - 1)' / w.SampleRate;
            k = obj.FrameIndex;
            s0 = fr.FrameStarts(k) / w.SampleRate;
            ax = obj.slotAxes("Grid");
            starts = fr.FrameStarts / w.SampleRate;
            obj.plot(ax, "SpeechContour", struct('X', t', 'Series', struct('Name', "Waveform x[n]", 'Y', x'), ...
                'XLabel', "Time (s)", 'YLabel', "Amplitude (full scale)", ...
                'Intervals', [s0, s0 + fr.WindowLength / w.SampleRate], ...
                'IntervalName', sprintf("Frame t = %d of %d (L = %d, H = %d)", k, numel(starts), fr.WindowLength, fr.HopLength), ...
                'Title', sprintf("Frame grid: %d frames, hop %.1f ms", numel(starts), 1000 * fr.HopLength / w.SampleRate)));
            obj.addTimeAxes(ax);
            ax = obj.slotAxes("Frame");
            obj.plot(ax, "SpeechFrame", struct('Frame', fr.Frames(:, k), 'Window', fr.Window(:), ...
                'SampleRate', fr.SampleRate, 'FrameIndex', k, 'FrameStart', fr.FrameStarts(k), ...
                'WindowType', string(fr.WindowType)));
        end
    end
end
