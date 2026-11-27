classdef ClusteringLayout < OceanViz.App.Replay.Layouts.IReplayStudioLayout
    % CLUSTERINGLAYOUT  A k-means recording (KMeansRecordable): the data
    %   coloured by the assignment C(i) at the current iteration with the
    %   centroids m_k on top, the assignment strip, and the two steps of
    %   Lloyd's algorithm (ESL Sec. 14.3.6, Algorithm 14.1) with the stored numbers.

    methods
        function n = name(~), n = "Clustering"; end

        function n = timeline(~, ~), n = "sse"; end

        function tf = appliesTo(~, rec)
            n = rec.names();
            tf = all(ismember(["data" "centroids" "assignment"], n)) && rec.info("data").Kind == "points" ...
                && rec.info("centroids").Kind == "points";
        end

        function tiles = tiles(~, ~)
            tiles = [OceanViz.App.Replay.Layouts.IReplayStudioLayout.tile("Clusters: observations by C(i), centroids (x)", ["data" "centroids" "assignment"], ...
                         @OceanViz.App.Replay.Layouts.ClusteringLayout.drawClusters), ...
                     OceanViz.App.Replay.Layouts.IReplayStudioLayout.tile("Assignment of each observation", "assignment")];
        end

        function lines = notes(~, rec, coords)
            m = coords.iteration;
            lines = {sprintf('K-MEANS (Lloyd) -- ESL Sec. 14.3.6, Algorithm 14.1   iteration m = %g', m), '', ...
                'Step 1: assign each observation to its nearest centroid:  C(i) = argmin_k ||x_i - m_k||^2', ...
                'Step 2: move each centroid to the mean of its observations.', ...
                'Repeat until the assignment no longer changes.', '', ...
                'm = 0: the initial centroids and their nearest-centroid assignment;', ...
                'm >= 1: the assignment of iteration m and the means it produces.', ''};
            lines = [lines, OceanViz.App.Replay.Layouts.IReplayStudioLayout.scalarLines(rec, coords, ...
                intersect(["sse" "changed"], rec.names(), 'stable'))];
        end
    end

    methods (Static)
        function caption = drawClusters(ax, studio)
            rec = studio.Recording; T = studio.Theme;
            OceanViz.App.Replay.Renderers.IChannelRenderer.prepare(ax);
            X = double(rec.get("data"));
            [L, at] = rec.get("assignment", struct('iteration', studio.Coords.iteration));
            C = double(rec.get("centroids", struct('iteration', at.iteration)));
            L = double(L); K = size(C, 1); col = lines(K);
            names = rec.info("assignment").Categories;
            hold(ax, 'on');
            for k = 1:K
                nm = "cluster " + k; if k <= numel(names), nm = names(k); end
                scatter(ax, X(L == k, 1), X(L == k, 2), 14, col(k, :), 'filled', 'MarkerFaceAlpha', 0.7, ...
                    'DisplayName', char(nm));
            end
            scatter(ax, C(:, 1), C(:, 2), 160, col, 'x', 'LineWidth', 3, 'DisplayName', 'centroids m_k');
            hold(ax, 'off'); grid(ax, 'on'); axis(ax, 'equal');
            OceanViz.App.Replay.Renderers.IChannelRenderer.finish(ax, T, true);
            OceanViz.App.Replay.Renderers.IChannelRenderer.styleLegend(legend(ax, 'Location', 'bestoutside'), T);
            caption = sprintf('iteration m = %g (stored)', at.iteration);
        end
    end
end
