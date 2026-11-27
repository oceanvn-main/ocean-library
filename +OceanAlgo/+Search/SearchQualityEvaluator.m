classdef SearchQualityEvaluator
    %SEARCHQUALITYEVALUATOR  Recall@K and distance-regret evaluation for ANN.
    methods (Static)
        function report=evaluate(approximate, exact, X, queries, k)
            arguments
                approximate (1,1) OceanAlgo.Search.IVectorIndex
                exact (1,1) OceanAlgo.Search.IVectorIndex
                X (:,:) double
                queries (:,:) double
                k (1,1) double {mustBeInteger,mustBePositive}
            end
            [a,~] = approximate.search(queries,k);
            [e,~] = exact.search(queries,k);

            for i = 1:size(queries,1)

                % Recall@K
                hit(i) = mean(ismember(e(i,:), a(i,:)));

                % TRUE distances of ANN-returned vectors
                da = sum((X(a(i,:),:) - queries(i,:)).^2, 2);

                % TRUE distances of exact-returned vectors
                de = sum((X(e(i,:),:) - queries(i,:)).^2, 2);

                % Compare rankwise after using the same true metric
                da = sort(da,'ascend');
                de = sort(de,'ascend');

                regret(i) = mean(max(0, da - de));
            end
            report=struct('RecallAtK',mean(hit),'MeanDistanceRegret',mean(regret),'PerQueryRecall',hit,'ApproximateIds',a,'ExactIds',e);
        end
    end
end
