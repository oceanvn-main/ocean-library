classdef HNSWIndex < handle
    %HNSWINDEX  Inspectable approximate nearest-neighbour graph index.
    %   A compact single-layer HNSW-style index for dense row vectors.
    %   Search is approximate and controlled by EfSearch; increase it to
    %   trade latency for recall. Distances are squared Euclidean or
    %   negative inner-product scores, depending on Metric.

    properties (SetAccess = private)
        Points double = zeros(0, 0)
        Neighbors cell = cell(1, 0)
        Metric string = "euclidean"
        MaxNeighbors (1,1) double = 16
        EfConstruction (1,1) double = 64
        IsFitted (1,1) logical = false
    end

    methods
        function obj = HNSWIndex(options)
            arguments
                options.Metric string {mustBeMember(options.Metric,["euclidean","inner-product"])} = "euclidean"
                options.MaxNeighbors (1,1) double {mustBeInteger,mustBePositive} = 16
                options.EfConstruction (1,1) double {mustBeInteger,mustBePositive} = 64
            end
            obj.Metric = options.Metric;
            obj.MaxNeighbors = options.MaxNeighbors;
            obj.EfConstruction = max(options.EfConstruction, options.MaxNeighbors);
        end

        function fit(obj, points)
            arguments
                obj
                points (:,:) double
            end
            obj.Points = points;
            n = size(points, 1);
            obj.Neighbors = cell(1, n);
            if n == 0
                obj.IsFitted = true;
                return
            end
            % Incremental insertion keeps the graph construction visible.
            for i = 1:n
                if i == 1, continue, end
                candidate = (1:i-1)';
                d = obj.distance_(points(i,:), points(candidate,:));
                [~, order] = sort(d, 'ascend');
                chosen = candidate(order(1:min(obj.MaxNeighbors, numel(order))));
                obj.Neighbors{i} = chosen(:)';
                for j = chosen(:)'
                    obj.Neighbors{j}(end+1) = i;
                    obj.Neighbors{j} = obj.prune_(obj.Neighbors{j}, points(j,:));
                end
            end
            obj.IsFitted = true;
        end

        function [ids, scores] = search(obj, queries, k, options)
            arguments
                obj
                queries (:,:) double
                k (1,1) double {mustBeInteger,mustBePositive}
                options.EfSearch (1,1) double {mustBeInteger,mustBePositive} = 64
            end
            if ~obj.IsFitted, error('OceanNLP:HNSWIndex:NotFitted','Call fit first.'); end
            if size(queries,2) ~= size(obj.Points,2)
                error('OceanNLP:HNSWIndex:DimMismatch','Query dimension does not match the index.');
            end
            nQ = size(queries,1); k = min(k,size(obj.Points,1));
            ids = zeros(nQ,k); scores = -inf(nQ,k);
            for q = 1:nQ
                [ids(q,:), scores(q,:)] = obj.searchOne_(queries(q,:), k, options.EfSearch);
            end
        end

        function meta = getMetadata(obj)
            meta = struct('Name','HNSWIndex','IsApproximate',true, ...
                'Metric',obj.Metric,'MaxNeighbors',obj.MaxNeighbors, ...
                'EfConstruction',obj.EfConstruction, ...
                'Formula','graph search over candidate neighbours; recall depends on EfSearch');
        end
    end

    methods (Access = private)
        function [ids, scores] = searchOne_(obj, q, k, ef)
            n = size(obj.Points,1); ef = min(max(ef,k),n);
            visited = false(1,n); candidates = 1; visited(1) = true;
            best = 1; bestD = obj.distance_(q,obj.Points(1,:));
            while ~isempty(candidates)
                [~, p] = min(obj.distance_(q,obj.Points(candidates,:)));
                v = candidates(p); candidates(p) = [];
                for u = obj.Neighbors{v}
                    if visited(u), continue, end
                    visited(u) = true; du = obj.distance_(q,obj.Points(u,:));
                    best(end+1) = u; bestD(end+1) = du; %#ok<AGROW>
                    candidates(end+1) = u; %#ok<AGROW>
                end
                [~, ord] = sort(bestD,'ascend');
                ord = ord(1:min(ef,numel(ord))); best = best(ord); bestD = bestD(ord);
                if numel(visited(visited)) >= n, break, end
            end
            [bestD, ord] = sort(bestD,'ascend'); best = best(ord);
            take = min(k,numel(best)); ids = best(1:take); scores = obj.toScore_(bestD(1:take));
            ids(end+1:k) = 0; scores(end+1:k) = -inf;
        end
        function d = distance_(obj, q, x)
            if obj.Metric == "euclidean"
                d = sum((x-q).^2,2);
            else
                d = -(x*q');
            end
        end
        function s = toScore_(obj,d)
            if obj.Metric == "euclidean", s = -d; else, s = -d; end
        end
        function nb = prune_(obj, nb, center)
            if numel(nb) <= obj.MaxNeighbors, return, end
            d = obj.distance_(center,obj.Points(nb,:)); [~,ord] = sort(d,'ascend'); nb = nb(ord(1:obj.MaxNeighbors));
        end
    end
end
