classdef BeamCandidatesStrategy < OceanViz.Strategies.BaseStrategy
    % BEAMCANDIDATESSTRATEGY  The decoder's N-best hypotheses and their scores, best first.
    %
    %   viz.plot(ax, "BeamCandidates", data)
    %
    %   DATA CONTRACT (the "Beam candidates" artifact of SpeechProcessAdapter):
    %     data.Candidates  1 x N struct array with fields Text (string), TokenIds, Score
    %                      (decoder log-score, nats; as stored in Core.Transcript.Candidates)
    %     data.Chosen      (optional, default 1) index of the hypothesis the decoder output
    %     data.ScoreLabel  (optional) what Score means (x-axis label)
    %     data.Reference   (optional) reference transcript, shown in the title
    %     data.Method      (optional) decoder method ("greedy" | "beam")
    %     data.Title       (optional)
    %   Horizontal bars of Score per candidate (rank 1 at the top), each labelled with
    %   its text. All bars start at a common baseline 15 % of the score range below the
    %   worst score, so bar length orders the candidates (best = longest) for any sign.
    %   Axes: x score (nats), y "Candidate". Legend: chosen vs other candidates.

    methods
        function plot(obj, ax, data, options)
            arguments
                obj
                ax
                data struct
                options struct = struct()
            end
            U = OceanViz.Strategies.Speech.SpeechVizUtils;
            U.requireFields(data, "Candidates", "BeamCandidates");
            c = U.colors(obj);
            cand = data.Candidates;
            if ~isempty(cand) && ~all(isfield(cand, {'Text', 'Score'}))
                error('OceanViz:Strategies:Speech:BeamCandidatesStrategy:Fields', 'Candidates need Text and Score.');
            end
            N = numel(cand);
            if N == 0
                error('OceanViz:Strategies:Speech:BeamCandidatesStrategy:Empty', 'No candidates to plot.');
            end
            scores = reshape(double([cand.Score]), 1, []);
            texts = arrayfun(@(s) string(s.Text), cand);
            labels = compose("%d. ""%s""", (1:N)', texts(:));
            labels(texts == "") = compose("%d. (empty)", find(texts == "")');
            chosen = U.valueOr(data, 'Chosen', 1);
            % Bars grow from a common baseline below the WORST score, so the best hypothesis
            % is the longest bar whatever the sign of the scores (bar end = the score itself).
            % Non-finite scores (e.g. -Inf: CTC-infeasible under rescoring) get no bar and
            % are labelled with their value at the baseline.
            fin = isfinite(scores);
            if any(fin), lo = min(scores(fin)); hi = max(scores(fin)); else, lo = 0; hi = 0; end
            span = max(hi - lo, max(abs(hi), 1) * 0.05);
            base = lo - 0.15 * span;
            drawn = scores;
            drawn(~fin) = base;
            hold(ax, 'on');
            others = setdiff(1:N, chosen);
            barh(ax, chosen, drawn(chosen), 0.7, 'FaceColor', c.Primary, 'EdgeColor', 'none', ...
                'BaseValue', base, 'ShowBaseLine', 'off', 'DisplayName', "Decoder output (chosen)");
            if ~isempty(others)
                barh(ax, others, drawn(others), 0.7, 'FaceColor', c.Secondary, 'EdgeColor', 'none', ...
                    'FaceAlpha', 0.7, 'BaseValue', base, 'ShowBaseLine', 'off', 'DisplayName', "Other candidates");
            end
            for n = 1:N
                text(ax, drawn(n), n, sprintf("  %.3f", scores(n)), 'Color', c.Text, ...
                    'Interpreter', 'none', 'VerticalAlignment', 'middle', 'FontSize', 8, ...
                    'HorizontalAlignment', 'left', 'Clipping', 'off');
            end
            hold(ax, 'off');
            set(ax, 'YDir', 'reverse');
            ax.YTick = 1:N;
            ax.YTickLabel = cellstr(labels);
            ylim(ax, [0.4, N + 0.6]);
            xlim(ax, [base, hi + 0.35 * span]);       % room for the score labels right of the bars
            ttl = sprintf("Decoder candidates (%d)", N);
            if isfield(data, 'Method') && string(data.Method) ~= "", ttl = ttl + " - " + string(data.Method); end
            if isfield(data, 'Reference') && string(data.Reference) ~= ""
                ttl = ttl + " | reference: """ + string(data.Reference) + """";
            end
            xText = string(U.valueOr(data, 'ScoreLabel', "Decoder log-score (nats)"));
            U.label(ax, c, U.pickTitle(options, data, ttl), xText, "Candidate (rank. text)");
            if ~isfield(options, 'ShowLegend') || options.ShowLegend
                U.legend(ax, c, 'Location', 'southoutside', 'Orientation', 'horizontal');
            end
        end
    end
end
