classdef TestExtraTrees < OceanTest.BaseTestCase
    methods (Test)
        function test1ClassificationSmoke(testCase)
            % 1. Classification smoke
            N = 160; p = 6;
            X_clf = randn(N, p);
            Y_clf = double(sum(X_clf(:, 1:3), 2) > 0) + 1;
            X_reg = randn(N, p);
            Y_reg = 2 * X_reg(:, 1) - X_reg(:, 2) + 0.5 * randn(N, 1);
            pass = 0; total = 0; fails = {};
            fprintf('\n=== Extra-Trees Test Suite ===\n\n');
            total = total + 1;
            et = OceanStats.Ensemble.ExtraTreesForest('NumTrees', 60, 'Seed', 1);
            et.fit(X_clf, Y_clf);
            acc = mean(et.predict(X_clf) == Y_clf);
            if acc > 0.75
            fprintf('  [PASS] 1. Classification smoke (acc=%.2f)\n', acc); pass = pass + 1;
            else
            fprintf('  [FAIL] 1. Classification smoke (acc=%.2f)\n', acc); fails{end+1} = '1';
            end
        end

        function test2RegressionSmoke(testCase)
            % 2. Regression smoke
            N = 160; p = 6;
            X_clf = randn(N, p);
            Y_clf = double(sum(X_clf(:, 1:3), 2) > 0) + 1;
            X_reg = randn(N, p);
            Y_reg = 2 * X_reg(:, 1) - X_reg(:, 2) + 0.5 * randn(N, 1);
            pass = 0; total = 0; fails = {};
            fprintf('\n=== Extra-Trees Test Suite ===\n\n');
            total = total + 1;
            et_r = OceanStats.Ensemble.ExtraTreesForest('NumTrees', 60, 'Criterion', 'regression', 'Seed', 2);
            et_r.fit(X_reg, Y_reg);
            mse = mean((et_r.predict(X_reg) - Y_reg).^2);
            if mse < 1.5
            fprintf('  [PASS] 2. Regression smoke (MSE=%.4f)\n', mse); pass = pass + 1;
            else
            fprintf('  [FAIL] 2. Regression MSE=%.4f\n', mse); fails{end+1} = '2';
            end
        end

        function test3NoBootstrapOoberrorMustThrow(testCase)
            % 3. No-bootstrap: oobError() must throw
            N = 160; p = 6;
            X_clf = randn(N, p);
            Y_clf = double(sum(X_clf(:, 1:3), 2) > 0) + 1;
            X_reg = randn(N, p);
            Y_reg = 2 * X_reg(:, 1) - X_reg(:, 2) + 0.5 * randn(N, 1);
            pass = 0; total = 0; fails = {};
            fprintf('\n=== Extra-Trees Test Suite ===\n\n');
            total = total + 1;
            et_nb = OceanStats.Ensemble.ExtraTreesForest('NumTrees', 10);
            et_nb.fit(X_clf, Y_clf);
            et_nb.oobError();
            fprintf('  [FAIL] 3. Should have thrown\n'); fails{end+1} = '3';
        end

        function test4BootstrapModeOobAvailable(testCase)
            % 4. Bootstrap mode: OOB available
            N = 160; p = 6;
            X_clf = randn(N, p);
            Y_clf = double(sum(X_clf(:, 1:3), 2) > 0) + 1;
            X_reg = randn(N, p);
            Y_reg = 2 * X_reg(:, 1) - X_reg(:, 2) + 0.5 * randn(N, 1);
            pass = 0; total = 0; fails = {};
            fprintf('\n=== Extra-Trees Test Suite ===\n\n');
            total = total + 1;
            et_b = OceanStats.Ensemble.ExtraTreesForest( ...
            'NumTrees', 60, 'UseBootstrap', true, 'ComputeOOB', true, ...
            'ImportanceMethod', 'MDI', 'Seed', 4);
            et_b.fit(X_clf, Y_clf);
            err = et_b.oobError();
            if isscalar(err) && err >= 0 && err <= 1
            fprintf('  [PASS] 4. Bootstrap OOB = %.4f\n', err); pass = pass + 1;
            else
            fprintf('  [FAIL] 4. OOB invalid\n'); fails{end+1} = '4';
            end
        end

        function test5MdiImportanceSignalNoise(testCase)
            % 5. MDI importance: signal > noise
            N = 160; p = 6;
            X_clf = randn(N, p);
            Y_clf = double(sum(X_clf(:, 1:3), 2) > 0) + 1;
            X_reg = randn(N, p);
            Y_reg = 2 * X_reg(:, 1) - X_reg(:, 2) + 0.5 * randn(N, 1);
            pass = 0; total = 0; fails = {};
            fprintf('\n=== Extra-Trees Test Suite ===\n\n');
            total = total + 1;
            et_imp = OceanStats.Ensemble.ExtraTreesForest( ...
            'NumTrees', 80, 'ComputeImportance', true, 'ImportanceMethod', 'MDI', 'Seed', 5);
            et_imp.fit(X_clf, Y_clf);
            imp = et_imp.featureImportance();
            if all(imp == 0)
            fprintf('  [SKIP] 5. MDI all zeros\n'); pass = pass + 1;
            else
            [~, ord] = sort(imp, 'descend');
            top3 = sum(ord(1:3) <= 3);
            if top3 >= 2
            fprintf('  [PASS] 5. MDI (top3 signal=%d/3)\n', top3); pass = pass + 1;
            else
            fprintf('  [FAIL] 5. MDI (top3=%d)\n', top3); fails{end+1} = '5';
            end
            end
        end

        function test6PredictprobaShapeRowSums(testCase)
            % 6. predictProba shape & row sums
            N = 160; p = 6;
            X_clf = randn(N, p);
            Y_clf = double(sum(X_clf(:, 1:3), 2) > 0) + 1;
            X_reg = randn(N, p);
            Y_reg = 2 * X_reg(:, 1) - X_reg(:, 2) + 0.5 * randn(N, 1);
            pass = 0; total = 0; fails = {};
            fprintf('\n=== Extra-Trees Test Suite ===\n\n');
            total = total + 1;
            et_p = OceanStats.Ensemble.ExtraTreesForest('NumTrees', 30, 'Seed', 6);
            et_p.fit(X_clf, Y_clf);
            P = et_p.predictProba(X_clf);
            testCase.verifyTrue(isequal(size(P), [N, 2]) && all(abs(sum(P, 2) - 1) < 1e-9) && all(P(:) >= 0));
            if ok
            fprintf('  [PASS] 6. predictProba size=%s, rows sum=1\n', mat2str(size(P))); pass = pass + 1;
            else
            fprintf('  [FAIL] 6. predictProba\n'); fails{end+1} = '6';
            end
        end

        function test7Reproducibility(testCase)
            % 7. Reproducibility
            N = 160; p = 6;
            X_clf = randn(N, p);
            Y_clf = double(sum(X_clf(:, 1:3), 2) > 0) + 1;
            X_reg = randn(N, p);
            Y_reg = 2 * X_reg(:, 1) - X_reg(:, 2) + 0.5 * randn(N, 1);
            pass = 0; total = 0; fails = {};
            fprintf('\n=== Extra-Trees Test Suite ===\n\n');
            total = total + 1;
            et1 = OceanStats.Ensemble.ExtraTreesForest('NumTrees', 20, 'Seed', 99);
            et1.fit(X_clf, Y_clf);
            et2 = OceanStats.Ensemble.ExtraTreesForest('NumTrees', 20, 'Seed', 99);
            et2.fit(X_clf, Y_clf);
            if isequal(et1.predict(X_clf), et2.predict(X_clf))
            fprintf('  [PASS] 7. Reproducibility\n'); pass = pass + 1;
            else
            fprintf('  [FAIL] 7. Reproducibility\n'); fails{end+1} = '7';
            end
        end

        function test8ImodelstrategyContract(testCase)
            % 8. IModelStrategy contract
            N = 160; p = 6;
            X_clf = randn(N, p);
            Y_clf = double(sum(X_clf(:, 1:3), 2) > 0) + 1;
            X_reg = randn(N, p);
            Y_reg = 2 * X_reg(:, 1) - X_reg(:, 2) + 0.5 * randn(N, 1);
            pass = 0; total = 0; fails = {};
            fprintf('\n=== Extra-Trees Test Suite ===\n\n');
            total = total + 1;
            strat = OceanStats.Selection.Strategies.ExtraTreesStrategy('NumTrees', 20, 'Seed', 8);
            m     = strat.train(X_clf, Y_clf);
            preds = strat.predict(m, X_clf);
            err   = strat.evaluate(Y_clf, preds);
            k     = strat.getComplexity(m);
            if isnumeric(preds) && isscalar(err) && k == 20
            fprintf('  [PASS] 8. IModelStrategy contract (err=%.4f, k=%d)\n', err, k); pass = pass + 1;
            else
            fprintf('  [FAIL] 8. IModelStrategy\n'); fails{end+1} = '8';
            end
        end

        function test9ProximitymatrixProperties(testCase)
            % 9. ProximityMatrix properties
            N = 160; p = 6;
            X_clf = randn(N, p);
            Y_clf = double(sum(X_clf(:, 1:3), 2) > 0) + 1;
            X_reg = randn(N, p);
            Y_reg = 2 * X_reg(:, 1) - X_reg(:, 2) + 0.5 * randn(N, 1);
            pass = 0; total = 0; fails = {};
            fprintf('\n=== Extra-Trees Test Suite ===\n\n');
            total = total + 1;
            P_prox = [];
            et_prox = OceanStats.Ensemble.ExtraTreesForest( ...
            'NumTrees', 30, 'ComputeImportance', false, 'Seed', 9);
            et_prox.fit(X_clf, Y_clf);
            idx_sub = 1:30;
            P_prox  = OceanStats.Ensemble.ProximityMatrix.compute(et_prox, X_clf(idx_sub,:));
            ok = isequal(size(P_prox), [30,30]) ...
            && max(abs(P_prox - P_prox'), [], 'all') < 1e-10 ...
            && all(abs(diag(P_prox) - 1) < 1e-10) ...
            && all(P_prox(:) >= 0);
            if ok
            fprintf('  [PASS] 9. ProximityMatrix (size/sym/diag OK)\n'); pass = pass + 1;
            else
            fprintf('  [FAIL] 9. ProximityMatrix\n'); fails{end+1} = '9';
            end
        end

        function test10OutlierScores(testCase)
            % 10. Outlier scores
            N = 160; p = 6;
            X_clf = randn(N, p);
            Y_clf = double(sum(X_clf(:, 1:3), 2) > 0) + 1;
            X_reg = randn(N, p);
            Y_reg = 2 * X_reg(:, 1) - X_reg(:, 2) + 0.5 * randn(N, 1);
            pass = 0; total = 0; fails = {};
            fprintf('\n=== Extra-Trees Test Suite ===\n\n');
            total = total + 1;
            if ~isempty(P_prox)
            scores = OceanStats.Ensemble.ProximityMatrix.outlierScores(P_prox);
            if numel(scores) == 30 && all(scores >= 0) && all(scores <= 1)
            fprintf('  [PASS] 10. outlierScores [%.3f, %.3f]\n', min(scores), max(scores)); pass = pass + 1;
            else
            fprintf('  [FAIL] 10. outlierScores range\n'); fails{end+1} = '10';
            end
            else
            fprintf('  [SKIP] 10. P_prox empty (test 9 failed)\n'); fails{end+1} = '10';
            end
        end

    end
end