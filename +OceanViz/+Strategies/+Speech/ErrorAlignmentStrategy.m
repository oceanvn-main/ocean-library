classdef ErrorAlignmentStrategy < OceanViz.Strategies.BaseStrategy
    % ERRORALIGNMENTSTRATEGY  Reference vs hypothesis edit alignment (hits, S, D, I) as two token rows.
    %
    %   viz.plot(ax, "ErrorAlignment", data)
    %
    %   DATA CONTRACT:
    %     data.Alignment  struct from OceanNLP.Speech.Evaluation.EditAlignment.align(ref, hyp):
    %                     Ops (1 x K "C"|"S"|"D"|"I"), RefIndex, HypIndex (0 = no token),
    %                     Substitutions, Deletions, Insertions, Hits, RefLength
    %     data.Reference  1 x m string tokens that were aligned (words or characters)
    %     data.Hypothesis 1 x n string tokens
    %     data.Unit       (optional) "word" | "character" (title: WER vs CER)
    %     data.Title      (optional)
    %   Column k is alignment step k: top row the reference token, bottom row the
    %   hypothesis token ("-" where the step consumes none), coloured by the operation.
    %   The title reports (S + D + I) / N with the counts carried by the alignment.
    %   Axes: x "Alignment step", y rows "Reference" / "Hypothesis". Legend: the 4 ops.

    methods
        function plot(obj, ax, data, options)
            arguments
                obj
                ax
                data struct
                options struct = struct()
            end
            U = OceanViz.Strategies.Speech.SpeechVizUtils;
            U.requireFields(data, ["Alignment", "Reference", "Hypothesis"], "ErrorAlignment");
            c = U.colors(obj);
            a = data.Alignment;
            ref = string(data.Reference); hyp = string(data.Hypothesis);
            ops = string(a.Ops);
            K = numel(ops);
            if numel(a.RefIndex) ~= K || numel(a.HypIndex) ~= K || any(a.RefIndex > numel(ref)) || any(a.HypIndex > numel(hyp))
                error('OceanViz:Strategies:Speech:ErrorAlignmentStrategy:Inconsistent', ...
                    'Alignment indices do not match the given Reference / Hypothesis tokens.');
            end
            pal = obj.Parent.getColorSequence(4);
            opNames = ["C", "S", "D", "I"];
            opLabels = ["Correct (hit)", "Substitution", "Deletion", "Insertion"];
            opColors = [c.Grid; pal(3, :); pal(2, :); pal(1, :)];
            hold(ax, 'on');
            handles = gobjects(1, 4);
            for q = 1:4
                handles(q) = patch(ax, NaN, NaN, opColors(q, :), 'EdgeColor', 'none', 'DisplayName', opLabels(q));
            end
            for k = 1:K
                q = find(opNames == ops(k), 1);
                for row = 1:2
                    patch(ax, k + [-0.45 0.45 0.45 -0.45], row + [-0.4 -0.4 0.4 0.4], opColors(q, :), ...
                        'EdgeColor', 'none', 'HandleVisibility', 'off');
                end
                rt = "-"; ht = "-";
                if a.RefIndex(k) > 0, rt = ref(a.RefIndex(k)); end
                if a.HypIndex(k) > 0, ht = hyp(a.HypIndex(k)); end
                tcol = c.Text;
                if q > 1, tcol = 1 - round(mean(opColors(q, :))) * [1 1 1]; end   % contrast on coloured cells
                text(ax, k, 2, rt, 'HorizontalAlignment', 'center', 'Color', tcol, 'Interpreter', 'none', 'FontSize', 9);
                text(ax, k, 1, ht, 'HorizontalAlignment', 'center', 'Color', tcol, 'Interpreter', 'none', 'FontSize', 9);
            end
            hold(ax, 'off');
            xlim(ax, [0.4, max(K, 1) + 0.6]);
            ylim(ax, [0.4, 2.6]);
            ax.YTick = [1 2];
            ax.YTickLabel = {'Hypothesis', 'Reference'};
            ax.XTick = 1:max(K, 1);
            unit = string(U.valueOr(data, 'Unit', "token"));
            rateName = "Error rate";
            if unit == "word", rateName = "WER"; elseif unit == "character", rateName = "CER"; end
            N = a.RefLength;
            errs = a.Substitutions + a.Deletions + a.Insertions;
            if N > 0
                rateText = sprintf("%s = (%d S + %d D + %d I) / %d = %.3f", rateName, a.Substitutions, ...
                    a.Deletions, a.Insertions, N, errs / N);
            else
                rateText = sprintf("%s undefined (empty reference), %d insertions", rateName, a.Insertions);
            end
            U.label(ax, c, U.pickTitle(options, data, rateText), "Alignment step (" + unit + ")", "");
            if ~isfield(options, 'ShowLegend') || options.ShowLegend
                U.legend(ax, c, handles, 'Location', 'southoutside', 'Orientation', 'horizontal');
            end
        end
    end
end
