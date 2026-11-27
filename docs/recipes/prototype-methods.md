# Prototype Methods Recipes

## Generate 3-class mixture data
```matlab
gen = OceanData.Generators.ESLMixtureGenerator('N', 300, 'NumClasses', 3);
[X, y] = gen.generate();
```

## Generate 5-class mixture data with custom centers
```matlab
gen = OceanData.Generators.ESLMixtureGenerator('N', 500, 'NumClasses', 5, 'CentersPerClass', 8);
[X, y] = gen.generate();
```

## Train LVQ with K-means initialization
```matlab
lvq = OceanStats.Supervised.Neighbors.LVQ1('R', 5, 'LearningRate', 0.1, 'MaxEpochs', 50);
lvq.fit(X_train, y_train);
y_pred = lvq.predict(X_test);
err = mean(y_pred ~= y_test);
```

## Train LVQ from custom initial prototypes
```matlab
lvq = OceanStats.Supervised.Neighbors.LVQ1('R', 5);
lvq.fit(X_train, y_train, 'InitPrototypes', my_protos, 'InitLabels', my_labels);
```

## Compute Bayes boundary on a grid
```matlab
gen = OceanData.Generators.ESLMixtureGenerator('NumClasses', 3);
[xx, yy] = meshgrid(linspace(-4, 4, 200));
D = gen.bayesDensity([xx(:), yy(:)]);
[~, zz_bayes] = max(D, [], 2);
contour(xx, yy, reshape(zz_bayes, size(xx)), 2, '--', 'Color', [0.7 0.3 0.9]);
```

## Visualize prototype classifier via OceanViz
```matlab
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('Prototype Classifier');
ax = axes(fig);
data = struct('X', X, 'Y', y, ...
    'Prototypes', lvq.Prototypes, 'ProtoLabels', lvq.PrototypeLabels, ...
    'Grid', struct('xx', xx, 'yy', yy, 'zz', zz));
viz.plot(ax, 'PrototypeClassifier', data, 'Title', 'LVQ Classifier');
```
