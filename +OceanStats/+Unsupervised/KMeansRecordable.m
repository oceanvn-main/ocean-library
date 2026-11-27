classdef KMeansRecordable < OceanReplay.Record.IRecordable
    % KMEANSRECORDABLE  Records Lloyd's k-means iterations for replay
    %   (OceanReplay; ESL 2nd ed. Sec. 14.3.6, Algorithm 14.1).
    %
    %   Axis "iteration" (open: known only after convergence): m = 0 is the
    %   initial centroids with their nearest-centroid assignment; m >= 1 is the
    %   state after step 2 of iteration m (assignment C_m, centroids = the
    %   cluster means of C_m). The run stops when the assignment no longer changes.
    %
    %   Channels
    %     data        the N observations (no axis)                points (p = 2, 3) / matrix
    %     centroids   the K cluster centres m_k                   points (p = 2, 3) / matrix
    %     assignment  cluster of each observation C(i)            vector (int32, categories)
    %     sse         sum_i || x_i - m_{C(i)} ||^2                scalar
    %     changed     observations whose cluster changed (m >= 2) scalar
    %   sse never increases from one recorded iteration to the next (each step of
    %   Lloyd's algorithm can only lower it) -- tested.
    %
    %   Example:
    %     d = OceanStats.Unsupervised.KMeansRecordable(X, 3, 'Seed', 4);
    %     OceanReplay.Record.Recorder.run(d, spec, "kmeans.h5");

    properties (SetAccess = private)
        X double
        K (1,1) double
        Seed (1,1) double
        MaxIter (1,1) double
        InitCentroids double
        DataName (1,1) string
    end

    methods
        function obj = KMeansRecordable(X, K, opts)
            arguments
                X double
                K (1,1) double {mustBePositive, mustBeInteger}
                opts.Seed (1,1) double = 1
                opts.MaxIter (1,1) double = 50
                opts.InitCentroids double = []
                opts.DataName (1,1) string = ""
            end
            obj.X = X; obj.K = K; obj.Seed = opts.Seed; obj.MaxIter = opts.MaxIter; obj.DataName = opts.DataName;
            if isempty(opts.InitCentroids)
                % K distinct observations chosen with the recorded seed (the usual random start).
                obj.InitCentroids = X(randperm(RandStream('mt19937ar', 'Seed', opts.Seed), size(X, 1), K), :);
            else
                obj.InitCentroids = opts.InitCentroids;
            end
        end

        function axes = axesInfo(~)
            axes = OceanReplay.Core.Axis("iteration", [], 'Symbol', "m", ...
                'Description', "Lloyd iteration (0 = initial centroids); ESL Algorithm 14.1");
        end

        function [c, v] = catalog(obj)
            C = @OceanReplay.Core.ChannelInfo;
            [N, p] = size(obj.X);
            geo = "matrix"; if ismember(p, [2 3]), geo = "points"; end
            cats = "cluster " + string(1:obj.K);
            c = [C("data", geo, 'ValueShape', [N p], 'LabelsChannel', "assignment", 'Symbol', "x_i", 'SymbolLatex', '$x_i$', ...
                   'Description', "the observations being clustered"), ...
                 C("centroids", geo, 'Axes', "iteration", 'ValueShape', [obj.K p], 'Symbol', "m_k", 'SymbolLatex', '$m_k$', ...
                   'Description', "cluster centres; for m >= 1 the means of the observations assigned to each cluster"), ...
                 C("assignment", "vector", 'Axes', "iteration", 'ValueShape', N, 'Dtype', "int32", 'Categories', cats, ...
                   'Symbol', "C(i)", 'SymbolLatex', '$C(i)$', 'Description', "cluster of each observation: the nearest centroid (step 1)"), ...
                 C("sse", "scalar", 'Axes', "iteration", 'Symbol', "sum_i ||x_i - m_C(i)||^2", 'SymbolLatex', '$\sum_i \|x_i - m_{C(i)}\|^2$', ...
                   'Description', "within-cluster sum of squared distances to the centroids; never increases"), ...
                 C("changed", "scalar", 'Axes', "iteration", 'Symbol', "#changed", 'SymbolLatex', '$\#\,\mathrm{changed}$', ...
                   'Description', "observations whose cluster changed in step 1 of iteration m (m >= 2); NaN for " + ...
                                  "m = 0, 1, which share the first assignment by construction")];
            v = OceanReplay.Core.ViewInfo.empty;
        end

        function out = run(obj, rec)
            km = OceanStats.Unsupervised.KMeans();
            prev = [];
            function onIter(it)
                L = it.Labels; Cm = it.Centroids;
                changedNow = NaN;                               % m = 0 and 1 share one assignment (see catalog)
                if it.Iteration >= 2, changedNow = nnz(L ~= prev); end
                prev = L;
                sse = sum(sum((obj.X - Cm(L, :)) .^ 2));
                rec.point(struct('iteration', it.Iteration), struct( ...
                    'data', @() obj.X, 'centroids', @() Cm, 'assignment', @() int32(L), ...
                    'sse', @() sse, 'changed', @() changedNow));
            end
            km.fit(obj.X, obj.K, 'InitCentroids', obj.InitCentroids, 'MaxIter', obj.MaxIter, ...
                'OnIteration', @onIter);
            out = struct('Centroids', km.Centroids, 'Labels', km.Labels);
        end

        function d = describe(obj)
            d = struct('model', struct('name', "k-means (Lloyd)", 'K', obj.K, 'data', obj.DataName, ...
                    'N', size(obj.X, 1), 'p', size(obj.X, 2), 'maxIter', obj.MaxIter), ...
                'references', {{'Hastie, Tibshirani & Friedman, ESL 2nd ed., Sec. 14.3.6, Algorithm 14.1'}}, ...
                'notation', struct('assignment', "C(i) in ESL; stored 1-based", 'centroids', "m_k in ESL"), ...
                'layout', "clustering");
        end
    end
end
