# Association Rules — Recipes

## Quick Start (Aisle Level, Instacart)

```matlab
dataDir = 'Data/InstaCart';

% Ingest
reader = OceanPipeline.Ingestion.TransactionalReader(dataDir, ...
    'Mode', 'relational', ...
    'PrimaryFile', 'order_products__prior.csv', ...
    'TransactionCol', 1, 'ItemCol', 2, ...
    'EnrichFile', 'products.csv', 'EnrichFromCol', 1, 'EnrichToCol', 3, ...
    'LookupFile', 'aisles.csv', 'LookupLabelCol', 2);
result = reader.read();

% Encode + Mine + Rules
encoder = OceanStats.Unsupervised.FrequentItemsets.TransactionEncoder();
[X, state] = encoder.fitApply(result.baskets);

miner = OceanStats.Unsupervised.FrequentItemsets.AprioriMiner();
mineResult = miner.mine(X, 0.01);

gen = OceanStats.Unsupervised.Rules.RuleGenerator();
rules = gen.generate(mineResult, 0.30, ...
    'ColMap', state.colMap, 'Vocabulary', result.vocabulary);
```

## Filter Rules (Top 50 by Lift)

```matlab
filt = OceanStats.Unsupervised.Rules.RuleFilter();
filtered = filt.filter(rules, ...
    'MinLift', 1.0, ...
    'MaxRules', 50, ...
    'RemoveRedundant', true, ...
    'SortBy', 'lift');
```

## Print Rule Table

```matlab
for i = 1:filtered.stats.nFiltered
    fprintf('%s → %s  (sup=%.2f%%, conf=%.1f%%, lift=%.2f)\n', ...
        filtered.antecedentNames{i}, ...
        filtered.consequentNames{i}, ...
        filtered.support(i)*100, ...
        filtered.confidence(i)*100, ...
        filtered.lift(i));
end
```

## Visualize (OceanViz Scatter)

```matlab
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('Association Rules');
ax = axes(fig);

vizData.confidence  = filtered.confidence;
vizData.lift        = filtered.lift;
vizData.support     = filtered.support;
vizData.itemsetSize = filtered.itemsetSize;
vizData.nAnnotate   = 5;
vizData.labels      = cellfun(@(a, c) sprintf('%s → %s', a, c), ...
    filtered.antecedentNames, filtered.consequentNames, 'UniformOutput', false);

viz.plot(ax, 'AssociationRule', vizData, ...
    'Title', 'Confidence vs Lift', ...
    'XLabel', 'Confidence', 'YLabel', 'Lift');
```

## Change Granularity (Department Level)

```matlab
reader = OceanPipeline.Ingestion.TransactionalReader(dataDir, ...
    'Mode', 'relational', ...
    'PrimaryFile', 'order_products__prior.csv', ...
    'TransactionCol', 1, 'ItemCol', 2, ...
    'EnrichFile', 'products.csv', 'EnrichFromCol', 1, 'EnrichToCol', 4, ...
    'LookupFile', 'departments.csv', 'LookupLabelCol', 2);
```

## Tune Parameters

```matlab
% Stricter: fewer rules, higher quality
mineResult = miner.mine(X, 0.05);          % 5% support
rules = gen.generate(mineResult, 0.50);     % 50% confidence
filtered = filt.filter(rules, 'MinLift', 2.0);  % 2x better than random

% Looser: more rules, more exploration
mineResult = miner.mine(X, 0.005);         % 0.5% support
rules = gen.generate(mineResult, 0.20);     % 20% confidence
filtered = filt.filter(rules, 'MinLift', 1.0);  % Just above random
```
