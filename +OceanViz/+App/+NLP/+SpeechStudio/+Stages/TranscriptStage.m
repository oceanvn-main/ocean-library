classdef TranscriptStage < OceanViz.App.NLP.SpeechStudio.Stages.StageBase
    % TRANSCRIPTSTAGE  9 - Transcript & evaluation: the merged transcript, per-segment
    %   transcripts with the transcribed pieces' times, the reference alignment (C/S/D/I) and
    %   WER / CER (Evaluation.CorpusEvaluator via RecognitionPipeline), plus domain warnings.

    properties (SetAccess = private)
        TextLabel = []
        SegmentTable = []
        WarningArea = []
    end

    methods
        function t = title(~)
            t = "Transcript & evaluation";
        end

        function c = requires(~)
            c = strings(1, 0);
        end

        function tf = needsRecognition(~)
            tf = false;
        end

        function buildPlots(obj, grid)
            grid.RowHeight = {'fit', 120, '1x', 70};
            obj.TextLabel = uilabel(grid, 'Text', '', 'FontSize', 20, 'FontWeight', 'bold', 'WordWrap', 'on');
            obj.TextLabel.Layout.Row = 1;
            obj.SegmentTable = uitable(grid, 'RowName', {}, 'ColumnEditable', false, ...
                'ColumnName', {'Piece', 'Start (s)', 'End (s)', 'Transcript', 'Score (nats)'});
            obj.SegmentTable.Layout.Row = 2;
            g = obj.addSlots(grid, "Alignment", 1, 1);
            g.Layout.Row = 3;
            obj.WarningArea = uitextarea(grid, 'Editable', 'off');
            obj.WarningArea.Layout.Row = 4;
        end

        function [rows, classes] = draw(obj, ctx)
            r = ctx.Result;
            tr = r.Transcript;
            obj.TextLabel.Text = char("""" + tr.Text + """");
            T = r.SegmentTranscripts;
            P = r.PieceTimes;                % span of each TRANSCRIBED piece (not detector segments)
            if ~ctx.hasPieces()
                obj.TextLabel.Text = 'No speech detected: nothing was recognised.';
                obj.SegmentTable.Data = {};
            else
                n = numel(T);
                data = cell(n, 5);
                for k = 1:n
                    if iscell(T), t = T{k}; else, t = T(k); end
                    data(k, :) = {k, P(k, 1), P(k, 2), char(t.Text), t.Score};
                end
                obj.SegmentTable.Data = data;
            end
            warn = ctx.Notes;
            if isempty(warn), warn = "No warnings."; end
            obj.WarningArea.Value = cellstr(warn);
            rows = [ ...
                "Transcript", """" + tr.Text + """"; ...
                "Model", ctx.ModelName + " (" + tr.ModelId + ")"; ...
                "Decoder", string(tr.Decoder); ...
                "Detector segments", string(numel(r.Segments)); ...
                "Transcribed pieces", string(numel(T)) + obj.iff(r.Segmented, " (segmented)", " (whole input)")];
            for k = 1:numel(tr.Segments)
                rows(end+1, :) = [sprintf("Transcript.Segments(%d)", k), sprintf("%.2f-%.2f s ""%s""", ...
                    tr.Segments(k).StartTime, tr.Segments(k).EndTime, tr.Segments(k).Text)]; %#ok<AGROW>
            end
            classes = ["OceanNLP.Speech.Inference.RecognitionPipeline", "OceanNLP.Speech.Core.Transcript"];
            a = ctx.alignment();
            if isempty(a)
                obj.emptySlot("Alignment", "No reference transcript: type one in the sidebar (WAV / microphone) to evaluate");
            else
                ax = obj.slotAxes("Alignment");
                obj.plot(ax, "ErrorAlignment", a);
                al = a.Alignment;
                rows = [rows; "Reference", """" + ctx.Reference + """"; ...
                    "Alignment (" + a.Unit + ")", sprintf("%d hits, %d S, %d D, %d I", al.Hits, ...
                    al.Substitutions, al.Deletions, al.Insertions)];
                classes = [classes, "OceanNLP.Speech.Evaluation.EditAlignment"];
            end
            ev = r.Evaluation;
            if ~isempty(ev) && isstruct(ev) && isfield(ev, 'WER')
                rows = [rows; "WER", obj.rateText(ev.WER); "CER", obj.rateText(ev.CER)];
                classes = [classes, "OceanNLP.Speech.Evaluation.CorpusEvaluator", ...
                    "OceanNLP.Speech.Evaluation.WordErrorRate", "OceanNLP.Speech.Evaluation.CharacterErrorRate"];
            end
        end

        function L = notation(~)
            L = ["$\mathrm{WER} = \frac{S + D + I}{N}$ (words), $\mathrm{CER}$ on characters", ...
                "$S, D, I$: substitutions, deletions, insertions", ...
                "$N$: reference length; C = correct (hit)", ...
                "Alignment: minimum edit distance (Levenshtein)"];
        end

        function s = description(~)
            s = ["The merged transcript joins the per-piece transcripts; its Segments carry the times of the " + ...
                "pieces that were actually transcribed: the speech-activity detector's start / end times when " + ...
                "the input was segmented, or [0, duration] for one unsegmented piece (no word timings are " + ...
                "invented: CTC / attention decoding alone does not produce them).", "", ...
                "With a reference, the minimum-edit-distance alignment counts substitutions, deletions and " + ...
                "insertions; WER and CER are (S + D + I) / N on words and characters (Evaluation.CorpusEvaluator, " + ...
                "validated against jiwer, Results/Speech/wer_validation.md).", "", ...
                "Warnings state the honest limits: the models know only the FSDD digit words, saw 3 training " + ...
                "speakers at 8 kHz, and real recordings may be clipped, quiet or long."];
        end
    end

    methods (Static, Access = private)
        function v = iff(c, a, b)
            if c, v = a; else, v = b; end
        end

        function s = rateText(w)
            s = "?";
            if isstruct(w)
                f = intersect(["Rate", "WER", "CER", "Value"], string(fieldnames(w))', 'stable');
                if ~isempty(f), s = sprintf("%.3f", w.(f(1))); end
                if all(isfield(w, {'S', 'D', 'I', 'N'}))
                    s = s + sprintf(" = (%d S + %d D + %d I) / %d", w.S, w.D, w.I, w.N);
                end
            elseif isnumeric(w)
                s = sprintf("%.3f", w);
            end
        end
    end
end
