# Random Forest Recipes

## 1. Quick-start classification

```matlab
rf = OceanStats.Ensemble.RandomForest('NumTrees', 100, 'Seed', 42);
rf.fit(X_train, Y_train);
Y_pred = rf.predict(X_test);
fprintf('Test error: %.4f\n', mean(Y_pred ~= Y_test));
```

---

## 2. Quick-start regression

```matlab
rf = OceanStats.Ensemble.RandomForest( ...
    'NumTrees', 200, 'Criterion', 'regression', 'Seed', 42);
rf.fit(X_train, Y_train);
Y_pred = rf.predict(X_test);
fprintf('Test MSE: %.6f\n', mean((Y_pred - Y_test).^2));
```

---

## 3. OOB error as a free validation estimate

```matlab
rf = OceanStats.Ensemble.RandomForest( ...
    'NumTrees', 100, 'ComputeOOB', true, 'Seed', 42);
rf.fit(X, Y);     % No train/test split needed for OOB estimate
fprintf('OOB Error: %.4f\n', rf.oobError());
```

---

## 4. Feature importance bar chart

```matlab
rf = OceanStats.Ensemble.RandomForest( ...
    'NumTrees', 100, 'ComputeImportance', true, ...
    'ImportanceMethod', 'MDA', 'Seed', 42);
rf.fit(X, Y);

scores = rf.featureImportance();
[imp_sorted, ord] = sort(scores, 'descend');
labels = arrayfun(@(k) sprintf('F%d', k), ord, 'UniformOutput', false);

barh(fliplr(imp_sorted));
set(gca, 'YTick', 1:numel(ord), 'YTickLabel', fliplr(labels));
xlabel('MDA Importance'); title('Feature Importance');
```

---

## 5. Sweep NumTrees to find convergence point

```matlab
Bs    = [10 25 50 100 200];
errs  = zeros(1, numel(Bs));
for i = 1:numel(Bs)
    rf = OceanStats.Ensemble.RandomForest('NumTrees', Bs(i), ...
        'ComputeOOB', true, 'ComputeImportance', false, 'Seed', 42);
    rf.fit(X, Y);
    errs(i) = rf.oobError();
end
plot(Bs, errs, '-o'); xlabel('NumTrees'); ylabel('OOB Error');
```

---

## 6. Tune Mtry with 5-fold CV via IModelStrategy

```matlab
mtry_vals = [1, 2, 4, 8, floor(sqrt(size(X,2)))];
cv_errs   = zeros(1, numel(mtry_vals));

for i = 1:numel(mtry_vals)
    strat = OceanStats.Selection.Strategies.RandomForestStrategy( ...
        'NumTrees', 50, 'Mtry', mtry_vals(i), 'Criterion', 'gini');
    cv    = OceanStats.Selection.CrossValidation.CrossValidator( ...
        strat, OceanStats.Selection.CrossValidation.KFold(5));
    cv_errs(i) = cv.evaluate(X, Y);
end

[~, best] = min(cv_errs);
fprintf('Best mtry=%d (CV err=%.4f)\n', mtry_vals(best), cv_errs(best));
```

---

## 7. Soft vote probabilities for ROC curve

```matlab
rf = OceanStats.Ensemble.RandomForest('NumTrees', 100, 'Seed', 42);
rf.fit(X_train, Y_train);

P     = rf.predictProba(X_test);   % N x K
score = P(:, 2);                    % probability of positive class

% Use MATLAB's perfcurve or OceanStats.Metrics.ROC
[fpr, tpr, ~, auc] = perfcurve(Y_test, score, max(rf.Classes_));
fprintf('AUC = %.4f\n', auc);
```

---

## 8. Use with BiasVarianceAuditor

```matlab
strat = OceanStats.Selection.Strategies.RandomForestStrategy('NumTrees', 100);
audit = OceanStats.Selection.Auditors.BiasVarianceAuditor(strat);
audit.run(X, Y, 'NumBootstrap', 50);
```

---

## See Also

- `docs/guides/random_forest.md` — Full parameter reference
- `Experiments/Scripts/Supervised/exp_random_forest_esl.m` — ESL §15 experiment
- `Experiments/Scripts/Supervised/exp_extra_trees_comparison.m` — RF vs ExtraTrees

---

## 9. Quick-start Extra-Trees

```matlab
et = OceanStats.Ensemble.ExtraTreesForest('NumTrees', 100, 'Seed', 42);
et.fit(X_train, Y_train);
Y_pred = et.predict(X_test);
fprintf('Test error: %.4f\n', mean(Y_pred ~= Y_test));
```

---

## 10. Extra-Trees with bootstrap + OOB

```matlab
et = OceanStats.Ensemble.ExtraTreesForest( ...
    'NumTrees', 100, 'UseBootstrap', true, ...
    'ComputeOOB', true, 'ImportanceMethod', 'MDI', 'Seed', 42);
et.fit(X, Y);
fprintf('OOB Error: %.4f\n', et.oobError());
```

---

## 11. RF vs ExtraTrees error sweep

```matlab
Bs = [10 25 50 100];
for i = 1:numel(Bs)
    rf = OceanStats.Ensemble.RandomForest('NumTrees', Bs(i), ...
        'ComputeOOB', false, 'ComputeImportance', false);
    rf.fit(X_tr, Y_tr);
    err_rf(i) = mean(rf.predict(X_te) ~= Y_te);

    et = OceanStats.Ensemble.ExtraTreesForest('NumTrees', Bs(i), ...
        'ComputeImportance', false);
    et.fit(X_tr, Y_tr);
    err_et(i) = mean(et.predict(X_te) ~= Y_te);
end
plot(Bs, err_rf, '-o', Bs, err_et, '-s');
legend('RF', 'ExtraTrees'); xlabel('NumTrees');
```

---

## 12. Proximity matrix + outlier detection

```matlab
rf = OceanStats.Ensemble.RandomForest('NumTrees', 100, 'Seed', 42);
rf.fit(X_train, Y_train);

P   = OceanStats.Ensemble.ProximityMatrix.compute(rf, X_train);
out = OceanStats.Ensemble.ProximityMatrix.outlierScores(P);

[~, ord] = sort(out, 'descend');
fprintf('Top outlier: sample %d (score=%.3f)\n', ord(1), out(ord(1)));
```

---

## 13. Proximity MDS scatter colored by class

```matlab
P = OceanStats.Ensemble.ProximityMatrix.compute(rf, X_train);
[coords, stress] = OceanStats.Ensemble.ProximityMatrix.mdsCoords(P, 2);

classes = unique(Y_train);
for c = 1:numel(classes)
    mask = Y_train == classes(c);
    scatter(coords(mask,1), coords(mask,2), 50, 'filled', ...
        'DisplayName', sprintf('Class %d', classes(c)));
    hold on;
end
legend; xlabel('MDS 1'); ylabel('MDS 2');
title(sprintf('Proximity MDS (stress=%.3f)', stress));
```

---

## 14. ExtraTreesStrategy in CrossValidator

```matlab
strat = OceanStats.Selection.Strategies.ExtraTreesStrategy( ...
    'NumTrees', 100, 'Criterion', 'gini');
cv    = OceanStats.Selection.CrossValidation.CrossValidator( ...
    strat, OceanStats.Selection.CrossValidation.KFold(5));
err   = cv.evaluate(X, Y);
fprintf('5-fold CV error: %.4f\n', err);
```
