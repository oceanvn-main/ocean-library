classdef MelFilterBankStrategy < OceanViz.Strategies.BaseStrategy
    % MELFILTERBANKSTRATEGY  The triangular mel filters W(m, k) over linear frequency.
    %
    %   viz.plot(ax, "MelFilterBank", data)
    %
    %   DATA CONTRACT (the "Mel filter bank" artifact of SpeechProcessAdapter, i.e.
    %   OceanMath.Signal.MelFilterBank.build output):
    %     data.Weights         M x K filter weights (row m = filter m over FFT bins)
    %     data.BinFrequencyHz  K x 1 bin frequencies (Hz)
    %     data.CentersHz       (optional) M x 1 filter centre frequencies (Hz), marked
    %     data.Scale           (optional) "htk" | "slaney" (mel scale, shown in title)
    %     data.Norm            (optional) "none" | "slaney" (area normalisation; y label)
    %     data.Title           (optional)
    %   Axes: x "Frequency (Hz)", y "Filter weight". Every filter is drawn; the legend
    %   names the first, middle and last filter (centre frequency) plus "other filters".

    methods
        function plot(obj, ax, data, options)
            arguments
                obj
                ax
                data struct
                options struct = struct()
            end
            U = OceanViz.Strategies.Speech.SpeechVizUtils;
            U.requireFields(data, ["Weights", "BinFrequencyHz"], "MelFilterBank");
            c = U.colors(obj);
            W = double(data.Weights);
            f = reshape(double(data.BinFrequencyHz), 1, []);
            [M, K] = size(W);
            if numel(f) ~= K
                error('OceanViz:Strategies:Speech:MelFilterBankStrategy:FrequencySize', ...
                    '%d bin frequencies for %d filter columns.', numel(f), K);
            end
            centers = reshape(double(U.valueOr(data, 'CentersHz', NaN(M, 1))), [], 1);
            named = unique([1, ceil(M / 2), M]);
            pal = parula(M + 2);
            hold(ax, 'on');
            otherShown = false;
            for m = 1:M
                if ismember(m, named)
                    name = sprintf("Filter %d (centre %.0f Hz)", m, centers(m));
                    vis = 'on'; lw = 2;
                elseif ~otherShown
                    name = "Other filters"; vis = 'on'; lw = 0.9; otherShown = true;
                else
                    name = ""; vis = 'off'; lw = 0.9;
                end
                plot(ax, f, W(m, :), 'Color', pal(m + 1, :), 'LineWidth', lw, ...
                    'DisplayName', name, 'HandleVisibility', vis);
            end
            if all(isfinite(centers))
                peak = max(W, [], 2);
                plot(ax, centers, peak, 'v', 'Color', c.Text, 'MarkerSize', 4, ...
                    'MarkerFaceColor', c.Text, 'DisplayName', "Filter centres");
            end
            hold(ax, 'off');
            xlim(ax, [f(1), f(end)]);
            scale = string(U.valueOr(data, 'Scale', ""));
            norm = string(U.valueOr(data, 'Norm', "none"));
            ttl = sprintf("Mel filter bank: %d filters", M);
            if scale ~= "", ttl = ttl + " (" + scale + " mel scale)"; end
            ylab = "Filter weight (peak 1)";
            if norm == "slaney", ylab = "Filter weight (Slaney area-normalised, 1/Hz)"; end
            U.label(ax, c, U.pickTitle(options, data, ttl), "Frequency (Hz)", ylab);
            if ~isfield(options, 'ShowLegend') || options.ShowLegend
                U.legend(ax, c, 'Location', 'northeast');
            end
        end
    end
end
