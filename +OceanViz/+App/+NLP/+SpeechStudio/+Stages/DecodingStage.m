classdef DecodingStage < OceanViz.App.NLP.SpeechStudio.Stages.StageBase
    % DECODINGSTAGE  8 - Decoding.
    %   CTC: best path -> merge repeats -> drop blanks (the collapse B), the decoder's N-best
    %   and, with a language model, each candidate's score split into CTC / LM / bonus parts.
    %   Encoder-decoder: N-best of the attention beam search with its attention and (when
    %   rescoring) CTC log-probabilities, plus the recorded beam steps.

    properties (SetAccess = private)
        StepsTable = []
    end

    methods
        function t = title(~)
            t = "Decoding";
        end

        function c = requires(~)
            c = strings(1, 0);
        end

        function buildPlots(obj, grid)
            grid.RowHeight = {150, '1x'};
            obj.StepsTable = uitable(grid, 'RowName', {}, 'ColumnEditable', false);
            obj.StepsTable.Layout.Row = 1;
            g = obj.addSlots(grid, "Beam", 1, 1);
            g.Layout.Row = 2;
        end

        function [rows, classes] = draw(obj, ctx)
            B = ctx.artifact("Beam candidates");
            if isempty(B), error('No "Beam candidates" step was recorded.'); end
            tr = ctx.segmentTranscript();
            if ctx.Reference ~= "" && ctx.numSegments() == 1, B.Reference = ctx.Reference; end
            ax = obj.slotAxes("Beam");
            obj.plot(ax, "BeamCandidates", B);
            rows = [ ...
                "Decoder", string(tr.Decoder); ...
                "Method / beam width", sprintf("%s / %d", string(B.Method), B.BeamWidth); ...
                "Candidates N", string(numel(B.Candidates)); ...
                "Chosen", """" + tr.Text + """"; ...
                "Score (nats)", sprintf("%.3f", tr.Score)];
            classes = "OceanNLP.Speech.Inspect.SpeechProcessAdapter";
            if ctx.has("ctc")
                [r2, c2] = obj.drawCTC(ctx, B);
            else
                [r2, c2] = obj.drawAttention(ctx, B);
            end
            rows = [rows; r2];
            classes = [classes, c2];
        end

        function L = notationFor(obj, ctx)
            if ctx.has("attention")
                L = ["$\hat{y} = \arg\max_y \frac{\log p(y \mid X)}{|y|^{\alpha}}$ (beam search)", ...
                    "rescore: $(1-\lambda)\log p_{att}(y \mid X) + \lambda \log p_{ctc}(y \mid X)$", ...
                    "$|y|$: hypothesis length; $\alpha$: length penalty", ...
                    "$\lambda$: CTC rescoring weight; $X$: encoder memory"];
            else
                L = obj.notation();
            end
        end

        function L = notation(~)
            L = ["$\mathcal{B}(\pi)$: merge repeats, then drop blanks", ...
                "$p(l \mid x) = \sum_{\pi \in \mathcal{B}^{-1}(l)} \prod_t y_{\pi_t}(t)$", ...
                "score $= \ln p_{ctc}(l \mid x) + w \ln p_{lm}(l) + b\,|l|$", ...
                "$\pi$: frame path; $l$: labelling; $|l|$: its length", ...
                "$w$: LM weight; $b$: bonus per label"];
        end

        function s = description(~)
            s = ["CTC greedy decoding takes the best class per frame (the best path) and applies the collapse " + ...
                "B: merge consecutive repeats, then remove blanks, so 'zz-ee-r-oo' becomes 'zero'.", ...
                "Prefix beam search (CTCPrefixBeamDecoder) keeps the W most probable prefixes, summing over " + ...
                "all paths per prefix; a token n-gram LM (shallow fusion, Decoding.TokenNGramLM fitted on the " + ...
                "FSDD train transcripts only) and a length bonus can be added to the score.", "", ...
                "The attention decoder (Decoding.AttentionBeamDecoder) extends hypotheses token by token until " + ...
                "<eos>; scores are length-normalised; optional CTC rescoring re-ranks the N-best with the joint " + ...
                "CTC head. Scores are log-probabilities, not calibrated confidences."];
        end
    end

    methods (Access = private)
        function [rows, classes] = drawCTC(obj, ctx, B)
            classes = ["OceanNLP.Speech.Decoding.CTCPrefixBeamDecoder", "OceanNLP.Speech.Models.CTCRecognizer"];
            P = ctx.artifact("Posteriors");
            rows = strings(0, 2);
            if ~isempty(P)
                tok = string(P.Tokens);
                path = P.BestPath;
                keep = [true, diff(path) ~= 0];
                merged = path(keep);
                labels = merged(merged ~= P.Blank);
                show = @(ids) strjoin(replace(tok(ids), "<blank>", "-"), " ");
                obj.StepsTable.ColumnName = {'Collapse step', 'Length', 'Sequence ("-" = blank)'};
                obj.StepsTable.Data = { ...
                    '1. best path argmax_k y_k(t)', numel(path), char(show(path)); ...
                    '2. merge repeats', numel(merged), char(show(merged)); ...
                    '3. drop blanks = greedy output', numel(labels), char(strjoin(tok(labels), ""))};
                rows = [rows; "Best path length T'", string(numel(path)); ...
                    "After merge / after blanks", sprintf("%d / %d", numel(merged), numel(labels))];
            end
            S = ctx.scoreSplit(B.Candidates);
            if ~isempty(S)
                classes = [classes, "OceanNLP.Speech.Decoding.TokenNGramLM"];
                lmw = ctx.Settings.LMWeight; bo = ctx.Settings.Bonus;
                rows = [rows; "LM", sprintf("order %d, weight w = %g, bonus b = %g", ctx.Settings.LMOrder, lmw, bo)];
                for i = 1:height(S)
                    rows(end+1, :) = [sprintf("#%d ""%s""", i, S.Text(i)), sprintf("%.2f = CTC %.2f + LM %.2f + bonus %.2f", ...
                        S.Score(i), S.CTC(i), S.LM(i), S.Bonus(i))]; %#ok<AGROW>
                end
            end
        end

        function [rows, classes] = drawAttention(obj, ctx, B)
            classes = ["OceanNLP.Speech.Decoding.AttentionBeamDecoder", "OceanNLP.Speech.Models.EncoderDecoderRecognizer"];
            c = B.Candidates;
            rows = strings(0, 2);
            cols = {'Rank', 'Text', 'Score'};
            data = cell(numel(c), 3);
            extra = intersect(["LogProb", "AttentionLogProb", "CTCLogProb", "Length"], string(fieldnames(c))', 'stable');
            cols = [cols, cellstr(extra)];
            for i = 1:numel(c)
                data(i, 1:3) = {i, char(c(i).Text), c(i).Score};
                for j = 1:numel(extra)
                    data{i, 3 + j} = double(c(i).(extra(j)));
                end
            end
            obj.StepsTable.ColumnName = cols;
            obj.StepsTable.Data = data;
            S = ctx.artifact("Decoder search steps");
            if ~isempty(S) && isfield(S, 'Steps')
                v = ctx.Recognizer.Vocabulary;
                rows(end+1, :) = ["Decoder search steps", sprintf("%d step-function calls (AttentionBeamDecoder)", numel(S.Steps))];
                for i = 1:numel(S.Steps)
                    P = S.Steps(i).Prefixes;
                    txt = strings(1, size(P, 1));
                    for b = 1:size(P, 1), txt(b) = "'" + v.decode(P(b, :)) + "'"; end
                    rows(end+1, :) = [sprintf("Step %d: %d live prefix(es)", i, size(P, 1)), strjoin(txt, ", ")]; %#ok<AGROW>
                end
            end
            if isfield(ctx.Settings, 'CTCRescoreWeight') && ctx.has("ctc-rescore")
                rows(end+1, :) = ["CTC rescoring weight lambda", string(ctx.Settings.CTCRescoreWeight)];
            end
        end
    end
end
