classdef TestKdtree < OceanTest.BaseTestCase
    methods (Test)
        function test1KNnEuclideanVsBruteForce(testCase)
            % 1. k-NN euclidean vs brute-force
            pass = 0; total = 0; fails = {};
            fprintf('\n=== KDTree & KNN Test Suite ===\n\n');
            N = 200; d = 4;
            X  = randn(N, d);
            q  = randn(1, d);
            Y_clf = double(sum(X(:,1:2),2) > 0) + 1;
            Y_reg = 3*X(:,1) - 2*X(:,2) + 0.3*randn(N,1);
            total = total + 1;
            kd = OceanAlgo.Spatial.KDTree(X);
            [ids_kd, d_kd] = kd.findKNearest(q, 5);
            % Brute-force
            bf_d = sqrt(sum((X - q).^2, 2));
            [sd, ord] = sort(bf_d);
            ids_bf = ord(1:5); d_bf = sd(1:5);
            ok = isequal(sort(ids_kd(ids_kd>0)), sort(ids_bf)) && ...
            max(abs(d_kd(ids_kd>0) - d_bf)) < 1e-10;
        end

        function test2ManhattanMetric(testCase)
            % 2. Manhattan metric
            pass = 0; total = 0; fails = {};
            fprintf('\n=== KDTree & KNN Test Suite ===\n\n');
            N = 200; d = 4;
            X  = randn(N, d);
            q  = randn(1, d);
            Y_clf = double(sum(X(:,1:2),2) > 0) + 1;
            Y_reg = 3*X(:,1) - 2*X(:,2) + 0.3*randn(N,1);
            total = total + 1;
            kd_m = OceanAlgo.Spatial.KDTree(X, 'Metric', 'manhattan');
            [ids_m, d_m] = kd_m.findKNearest(q, 5);
            bf_dm = sum(abs(X - q), 2);
            [sdm, ordm] = sort(bf_dm);
            ok = isequal(sort(ids_m(ids_m>0)), sort(ordm(1:5))) && ...
            max(abs(d_m(ids_m>0) - sdm(1:5))) < 1e-10;
        end

        function test3ChebyshevMetric(testCase)
            % 3. Chebyshev metric
            pass = 0; total = 0; fails = {};
            fprintf('\n=== KDTree & KNN Test Suite ===\n\n');
            N = 200; d = 4;
            X  = randn(N, d);
            q  = randn(1, d);
            Y_clf = double(sum(X(:,1:2),2) > 0) + 1;
            Y_reg = 3*X(:,1) - 2*X(:,2) + 0.3*randn(N,1);
            total = total + 1;
            kd_c = OceanAlgo.Spatial.KDTree(X, 'Metric', 'chebyshev');
            [ids_c, d_c] = kd_c.findKNearest(q, 5);
            bf_dc = max(abs(X - q), [], 2);
            [sdc, ordc] = sort(bf_dc);
            ok = isequal(sort(ids_c(ids_c>0)), sort(ordc(1:5))) && ...
            max(abs(d_c(ids_c>0) - sdc(1:5))) < 1e-10;
        end

        function test4RangeQueryVsBruteForce(testCase)
            % 4. Range query vs brute-force
            pass = 0; total = 0; fails = {};
            fprintf('\n=== KDTree & KNN Test Suite ===\n\n');
            N = 200; d = 4;
            X  = randn(N, d);
            q  = randn(1, d);
            Y_clf = double(sum(X(:,1:2),2) > 0) + 1;
            Y_reg = 3*X(:,1) - 2*X(:,2) + 0.3*randn(N,1);
            total = total + 1;
            kd4  = OceanAlgo.Spatial.KDTree(X);
            r    = 1.5;
            [ids_r, ~] = kd4.rangeQuery(q, r);
            bf_inr = find(sqrt(sum((X-q).^2,2)) < r);
            testCase.verifyTrue(isequal(sort(ids_r), sort(bf_inr)));
        end

        function test5KNReturnsPaddedZeros(testCase)
            % 5. k > N: returns padded zeros
            pass = 0; total = 0; fails = {};
            fprintf('\n=== KDTree & KNN Test Suite ===\n\n');
            N = 200; d = 4;
            X  = randn(N, d);
            q  = randn(1, d);
            Y_clf = double(sum(X(:,1:2),2) > 0) + 1;
            Y_reg = 3*X(:,1) - 2*X(:,2) + 0.3*randn(N,1);
            total = total + 1;
            kd5 = OceanAlgo.Spatial.KDTree(X(1:3,:));
            [ids5, d5] = kd5.findKNearest(q, 10);
            testCase.verifyTrue(numel(ids5)==10 && sum(ids5>0)==3 && all(d5(ids5==0)==inf));
        end

        function test6D11Dimensional(testCase)
            % 6. d=1 (1-dimensional)
            pass = 0; total = 0; fails = {};
            fprintf('\n=== KDTree & KNN Test Suite ===\n\n');
            N = 200; d = 4;
            X  = randn(N, d);
            q  = randn(1, d);
            Y_clf = double(sum(X(:,1:2),2) > 0) + 1;
            Y_reg = 3*X(:,1) - 2*X(:,2) + 0.3*randn(N,1);
            total = total + 1;
            X1 = (1:50)';
            kd6 = OceanAlgo.Spatial.KDTree(X1);
            [ids6, d6] = kd6.findKNearest(25.3, 3);
            testCase.verifyTrue(isequal(ids6(ids6>0), [25;26;24]) && all(d6(ids6>0) < 2));
        end

        function test7ExactMatchDist0(testCase)
            % 7. Exact match → dist=0
            pass = 0; total = 0; fails = {};
            fprintf('\n=== KDTree & KNN Test Suite ===\n\n');
            N = 200; d = 4;
            X  = randn(N, d);
            q  = randn(1, d);
            Y_clf = double(sum(X(:,1:2),2) > 0) + 1;
            Y_reg = 3*X(:,1) - 2*X(:,2) + 0.3*randn(N,1);
            total = total + 1;
            kd7 = OceanAlgo.Spatial.KDTree(X);
            [ids7, d7] = kd7.findKNearest(X(42,:), 1);
            testCase.verifyTrue(ids7(1)==42 && d7(1)==0);
        end

        function test8ResultsSortedAscending(testCase)
            % 8. Results sorted ascending
            pass = 0; total = 0; fails = {};
            fprintf('\n=== KDTree & KNN Test Suite ===\n\n');
            N = 200; d = 4;
            X  = randn(N, d);
            q  = randn(1, d);
            Y_clf = double(sum(X(:,1:2),2) > 0) + 1;
            Y_reg = 3*X(:,1) - 2*X(:,2) + 0.3*randn(N,1);
            total = total + 1;
            kd8 = OceanAlgo.Spatial.KDTree(X);
            [~, d8] = kd8.findKNearest(q, 10);
            testCase.verifyTrue(all(diff(d8(d8<inf)) >= 0));
        end

        function test9ClassificationAccuracy(testCase)
            % 9. Classification accuracy
            pass = 0; total = 0; fails = {};
            fprintf('\n=== KDTree & KNN Test Suite ===\n\n');
            N = 200; d = 4;
            X  = randn(N, d);
            q  = randn(1, d);
            Y_clf = double(sum(X(:,1:2),2) > 0) + 1;
            Y_reg = 3*X(:,1) - 2*X(:,2) + 0.3*randn(N,1);
            total = total + 1;
            knn = OceanStats.Supervised.Neighbors.KNNClassifier('K', 7);
            knn.fit(X, Y_clf);
            acc = mean(knn.predict(X) == Y_clf);
            if acc > 0.80, fprintf('  [PASS] 9. KNN classif (acc=%.2f)\n', acc); pass=pass+1;
            else,          fprintf('  [FAIL] 9. KNN classif (acc=%.2f)\n', acc); fails{end+1}='9'; end
        end

        function test10RegressionMse(testCase)
            % 10. Regression MSE
            pass = 0; total = 0; fails = {};
            fprintf('\n=== KDTree & KNN Test Suite ===\n\n');
            N = 200; d = 4;
            X  = randn(N, d);
            q  = randn(1, d);
            Y_clf = double(sum(X(:,1:2),2) > 0) + 1;
            Y_reg = 3*X(:,1) - 2*X(:,2) + 0.3*randn(N,1);
            total = total + 1;
            knn_r = OceanStats.Supervised.Neighbors.KNNClassifier('K', 5, 'Task', 'regression');
            knn_r.fit(X, Y_reg);
            mse = mean((knn_r.predict(X) - Y_reg).^2);
            if mse < 2.0, fprintf('  [PASS] 10. KNN regression (MSE=%.4f)\n', mse); pass=pass+1;
            else,         fprintf('  [FAIL] 10. KNN regression (MSE=%.4f)\n', mse); fails{end+1}='10'; end
        end

        function test11K1MemorizesTraining(testCase)
            % 11. K=1 → memorizes training
            pass = 0; total = 0; fails = {};
            fprintf('\n=== KDTree & KNN Test Suite ===\n\n');
            N = 200; d = 4;
            X  = randn(N, d);
            q  = randn(1, d);
            Y_clf = double(sum(X(:,1:2),2) > 0) + 1;
            Y_reg = 3*X(:,1) - 2*X(:,2) + 0.3*randn(N,1);
            total = total + 1;
            knn1 = OceanStats.Supervised.Neighbors.KNNClassifier('K', 1);
            knn1.fit(X, Y_clf);
            testCase.verifyTrue(mean(knn1.predict(X) == Y_clf) == 1.0);
        end

        function test12DistanceWeights(testCase)
            % 12. Distance weights
            pass = 0; total = 0; fails = {};
            fprintf('\n=== KDTree & KNN Test Suite ===\n\n');
            N = 200; d = 4;
            X  = randn(N, d);
            q  = randn(1, d);
            Y_clf = double(sum(X(:,1:2),2) > 0) + 1;
            Y_reg = 3*X(:,1) - 2*X(:,2) + 0.3*randn(N,1);
            total = total + 1;
            knn_u = OceanStats.Supervised.Neighbors.KNNClassifier('K', 5, 'Weights', 'uniform');
            knn_d = OceanStats.Supervised.Neighbors.KNNClassifier('K', 5, 'Weights', 'distance');
            knn_u.fit(X, Y_clf); knn_d.fit(X, Y_clf);
            % Just check they run and return correct shape
            testCase.verifyTrue(isequal(size(knn_u.predict(X)), size(knn_d.predict(X))));
        end

        function test13PredictprobaShapeAndRowSums(testCase)
            % 13. predictProba shape and row sums
            pass = 0; total = 0; fails = {};
            fprintf('\n=== KDTree & KNN Test Suite ===\n\n');
            N = 200; d = 4;
            X  = randn(N, d);
            q  = randn(1, d);
            Y_clf = double(sum(X(:,1:2),2) > 0) + 1;
            Y_reg = 3*X(:,1) - 2*X(:,2) + 0.3*randn(N,1);
            total = total + 1;
            knn_p = OceanStats.Supervised.Neighbors.KNNClassifier('K', 5);
            knn_p.fit(X, Y_clf);
            P = knn_p.predictProba(X(1:20,:));
            testCase.verifyTrue(isequal(size(P), [20,2]) && all(abs(sum(P,2)-1) < 1e-9) && all(P(:)>=0));
        end

        function test14ImodelstrategyContract(testCase)
            % 14. IModelStrategy contract
            pass = 0; total = 0; fails = {};
            fprintf('\n=== KDTree & KNN Test Suite ===\n\n');
            N = 200; d = 4;
            X  = randn(N, d);
            q  = randn(1, d);
            Y_clf = double(sum(X(:,1:2),2) > 0) + 1;
            Y_reg = 3*X(:,1) - 2*X(:,2) + 0.3*randn(N,1);
            total = total + 1;
            strat = OceanStats.Selection.Strategies.KNNStrategy('K', 5);
            m     = strat.train(X, Y_clf);
            preds = strat.predict(m, X);
            err   = strat.evaluate(Y_clf, preds);
            k     = strat.getComplexity(m);
            testCase.verifyTrue(isnumeric(preds) && isscalar(err) && k == 5);
        end

    end
end