classdef ProbabilitiesStage < OceanViz.App.NLP.SpeechStudio.Stages.StageBase
    % PROBABILITIESSTAGE  7 - Output probabilities.
    %   CTC: frame posteriors y_k(t) = softmax(u_t)_k with the best path, and the blank vs
    %   best-label probability over time (CTC's "peaky" behaviour).
    %   Encoder-decoder: per-decoder-step token posteriors and the cross-attention map.

    methods
        function t = title(~)
            t = "Output probabilities";
        end

        function c = requires(~)
            c = strings(1, 0);
        end

        function buildPlots(obj, grid)
            obj.addSlots(grid, ["Main", "Second"], 2, 1);
        end

        function [rows, classes] = draw(obj, ctx)
            if ctx.has("ctc")
                [rows, classes] = obj.drawCTC(ctx);
            else
                [rows, classes] = obj.drawAttention(ctx);
            end
        end

        function L = notationFor(obj, ctx)
            if ctx.has("attention")
                L = ["$p(y_u \mid y_{<u}, X) = \mathrm{softmax}(W\,\mathrm{LN}(g_u) + b)$", ...
                    "$\alpha_{u,t} = \mathrm{softmax}_t\big(q_u^\top k_t / \sqrt{d}\big)$", ...
                    "$y_u$: token at decoder step $u$; $X$: encoder memory", ...
                    "$g_u$: decoder state; $\alpha_{u,t}$: attention of step $u$ on frame $t$", ...
                    "$q_u, k_t$: query / key; $d$: head dimension"];
            else
                L = obj.notation();
            end
        end

        function L = notation(~)
            L = ["$y_k(t) = \mathrm{softmax}(u_t)_k = \frac{e^{u_{t,k}}}{\sum_j e^{u_{t,j}}}$", ...
                "$u_t = W h_t + b$; best path $\pi_t = \arg\max_k y_k(t)$", ...
                "$y_k(t)$: probability of class $k$ at output frame $t$", ...
                "$u_t$: logits; $h_t$: last encoder state; $K$: classes", ...
                "class 1 = $\langle$blank$\rangle$ (emit nothing)"];
        end

        function s = description(~)
            s = ["CTC (Graves et al. 2006): the model emits a distribution over the K classes (blank + units) " + ...
                "at every output frame; a labelling's probability sums over all frame paths that collapse to it.", ...
                "Trained CTC models are 'peaky': most frames are blank with probability near 1 and each unit " + ...
                "fires on a few frames, as the lower plot shows.", "", ...
                "Attention encoder-decoder (Chan et al. 2016): an autoregressive decoder predicts one token per " + ...
                "step, attending over the encoder frames. The attention map shows which frames each output step " + ...
                "used; for speech it is roughly monotonic."];
        end
    end

    methods (Access = private)
        function [rows, classes] = drawCTC(obj, ctx)
            P = ctx.artifact("Posteriors");
            if isempty(P), error('No "Posteriors" step was recorded.'); end
            ax = obj.slotAxes("Main");
            obj.plot(ax, "CTCPosterior", P);
            timed = ~isempty(P.FrameTimes);
            if timed, obj.addTimeAxes(ax); end
            Y = P.Posteriors;
            nb = setdiff(1:size(Y, 1), P.Blank);
            [pmax, kmax] = max(Y(nb, :), [], 1);
            x = P.FrameTimes; xl = "Time (s), frame centres";
            if ~timed, x = 1:size(Y, 2); xl = "Output frame"; end
            ax = obj.slotAxes("Second");
            obj.plot(ax, "SpeechContour", struct('X', x, 'Series', struct('Name', ...
                {"Blank y_blank(t)", "Best non-blank max_k y_k(t)"}, 'Y', {Y(P.Blank, :), pmax}), ...
                'XLabel', xl, 'YLabel', "Probability", 'Title', "Blank vs best label per frame"));
            if timed, obj.addTimeAxes(ax); end
            best = P.BestPath;
            tok = string(P.Tokens);
            fired = tok(nb(kmax(pmax > 0.5)));
            rows = [ ...
                "Posteriors (K x T)", obj.shape(Y); ...
                "Classes K", string(numel(tok)); ...
                "Blank frames on best path", sprintf("%d of %d (%.0f %%)", nnz(best == P.Blank), numel(best), 100 * mean(best == P.Blank)); ...
                "Frames with a label > 0.5", sprintf("%d (%s)", numel(fired), strjoin(fired, "")); ...
                "Mean max posterior", sprintf("%.3f", mean(max(Y, [], 1)))];
            classes = ["OceanNLP.Speech.Models.CTCRecognizer", "OceanNLP.Speech.Inspect.SpeechProcessAdapter"];
        end

        function [rows, classes] = drawAttention(obj, ctx)
            rows = strings(0, 2);
            classes = ["OceanNLP.Speech.Models.EncoderDecoderRecognizer", "OceanNLP.Speech.Inspect.SpeechProcessAdapter"];
            D = ctx.artifact("Decoder posteriors");
            if isempty(D)
                obj.emptySlot("Main", "No decoder posteriors were recorded for this model");
            else
                ax = obj.slotAxes("Main");
                V = D.Posteriors;
                tok = string(ctx.Recognizer.Vocabulary.Tokens);
                if isfield(D, 'Tokens'), tok = string(D.Tokens); end
                obj.plot(ax, "SpeechFeatures", struct('Values', V, 'FrameTimes', [], 'Tokens', tok, ...
                    'AxisNames', ["token", "decoder step u"], 'Units', "probability p(y_u | y_<u, X)", ...
                    'Title', "Decoder token posteriors per step (chosen hypothesis)"));
                xlabel(ax, "Decoder step u", 'Interpreter', 'none', 'Color', ctx.Viz.Colors.Text);
                [pm, km] = max(V, [], 1);
                rows = [rows; "Posteriors (K x U)", obj.shape(V); ...
                    "Argmax per step", strjoin(tok(km), " "); ...
                    "Mean max probability", sprintf("%.3f", mean(pm))];
                if isfield(D, 'StepLabels'), rows = [rows; "Step labels", strjoin(string(D.StepLabels), " ")]; end
                if isfield(D, 'LogProb'), rows = [rows; "log p(y | X) (nats)", sprintf("%.3f", sum(D.LogProb(:)))]; end
            end
            A = ctx.artifact("Cross-attention");
            if isempty(A)
                obj.emptySlot("Second", "No cross-attention was recorded for this model");
            else
                ax = obj.slotAxes("Second");
                if isfield(A, 'StepLabels') && numel(A.StepLabels) == size(A.Values, 1)
                    A.Tokens = "u" + (1:size(A.Values, 1)) + ": " + string(A.StepLabels);
                end
                obj.plot(ax, "SpeechFeatures", A);
                if isfield(A, 'FrameTimes') && ~isempty(A.FrameTimes), obj.addTimeAxes(ax); end
                rows = [rows; "Cross-attention (U x T')", obj.shape(A.Values)];
                if isfield(A, 'Title'), rows = [rows; "Attention source", string(A.Title)]; end
            end
        end
    end

    methods (Static, Access = private)
        function v = firstWith(a, fields)
            v = [];
            for k = 1:numel(a)
                if isstruct(a(k).Value) && any(isfield(a(k).Value, cellstr(fields)))
                    v = a(k).Value;
                    return;
                end
            end
        end
    end
end
