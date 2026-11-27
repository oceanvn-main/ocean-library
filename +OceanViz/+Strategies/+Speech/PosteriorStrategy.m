classdef PosteriorStrategy < OceanViz.Strategies.BaseStrategy
    % POSTERIORSTRATEGY  CTC frame posteriors y_k(t) as a class x time heat map, with the best path.
    %
    %   viz.plot(ax, "CTCPosterior", data)
    %
    %   DATA CONTRACT (the "Posteriors" artifact of SpeechProcessAdapter):
    %     data.Posteriors  K x T probabilities, columns sum to 1 (softmax of the logits)
    %     data.Tokens      1 x K class names (vocabulary order; "<blank>" = class Blank)
    %     data.FrameTimes  1 x T output-frame times (s); [] -> output-frame index
    %     data.BestPath    (optional) 1 x T argmax_k y_k(t), overlaid as markers
    %     data.Blank       (optional) blank index (best-path blanks drawn hollow)
    %     data.Title       (optional)
    %   Axes: x time (s) or output frame; y "Output class (token)"; colour bar
    %   "Posterior probability y_k(t)". Legend: best-path markers.

    methods
        function plot(obj, ax, data, options)
            arguments
                obj
                ax
                data struct
                options struct = struct()
            end
            U = OceanViz.Strategies.Speech.SpeechVizUtils;
            U.requireFields(data, ["Posteriors", "Tokens"], "CTCPosterior");
            c = U.colors(obj);
            Y = double(data.Posteriors);
            [K, T] = size(Y);
            tokens = string(data.Tokens);
            if numel(tokens) ~= K
                error('OceanViz:Strategies:Speech:PosteriorStrategy:TokenCount', '%d tokens for %d classes.', numel(tokens), K);
            end
            if any(Y(:) < -1e-12) || any(Y(:) > 1 + 1e-12)
                error('OceanViz:Strategies:Speech:PosteriorStrategy:NotProbability', 'Posteriors must lie in [0, 1].');
            end
            [x, xText] = U.timeAxis(U.valueOr(data, 'FrameTimes', []), T);
            imagesc(ax, x, 1:K, Y, [0 1]);
            set(ax, 'YDir', 'normal');
            colormap(ax, parula(256));
            ax.YTick = 1:K;
            ax.YTickLabel = cellstr(tokens);
            ylim(ax, [0.5, K + 0.5]);
            if T > 1, xlim(ax, [x(1), x(end)] + [-0.5 0.5] * (x(2) - x(1))); end
            if isfield(data, 'BestPath') && numel(data.BestPath) == T
                best = reshape(double(data.BestPath), 1, []);
                blank = U.valueOr(data, 'Blank', NaN);
                hold(ax, 'on');
                isB = best == blank;
                plot(ax, x(~isB), best(~isB), 's', 'Color', c.Secondary, 'MarkerFaceColor', c.Secondary, ...
                    'MarkerSize', 6, 'LineStyle', 'none', 'DisplayName', "Best path argmax_k y_k(t): label");
                plot(ax, x(isB), best(isB), 's', 'Color', c.Secondary, 'MarkerSize', 6, ...
                    'LineStyle', 'none', 'DisplayName', "Best path argmax_k y_k(t): blank");
                hold(ax, 'off');
                if ~isfield(options, 'ShowLegend') || options.ShowLegend
                    U.legend(ax, c, 'Location', 'northoutside', 'Orientation', 'horizontal');
                end
            end
            U.label(ax, c, U.pickTitle(options, data, "CTC posteriors"), xText, "Output class (token)");
            U.colorbar(ax, c, "Posterior probability y_k(t)");
        end
    end
end
