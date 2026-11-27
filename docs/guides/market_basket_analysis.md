# Market Basket Analysis — Guide

## Overview

The Ocean Library Market Basket Analysis (MBA) pipeline discovers which items are frequently purchased together and generates directional association rules bounded by statistical metrics.

**Pipeline:** 5 phases, each encapsulated in a dedicated class.

```
TransactionalReader → TransactionEncoder → AprioriMiner → RuleGenerator → RuleFilter
     (Phase 1)            (Phase 2)          (Phase 3)       (Phase 4)      (Phase 5)
```

## Classes

### Phase 1: `OceanPipeline.Ingestion.TransactionalReader`

Reads transactional data from CSV files and produces baskets (cell array of item-ID vectors) plus a vocabulary map.

- Supports relational (multi-file JOIN) and flat (single-file) modes
- Configurable via column indices — fully domain-agnostic
- Reports profile metrics: N, p, sparsity, memory estimates

### Phase 2: `OceanStats.Unsupervised.FrequentItemsets.TransactionEncoder`

Converts baskets into a sparse Boolean matrix using the fit/apply pattern.

- `fit(baskets)` — learns unique items, builds column mapping
- `apply(baskets, state)` — transforms baskets to sparse matrix
- `fitApply(baskets)` — convenience method
- COO construction for efficient memory usage

### Phase 3: `OceanStats.Unsupervised.FrequentItemsets.AprioriMiner`

Implements the Apriori algorithm via `IMinerStrategy` interface.

- Level-wise search: k=1 (singletons), k=2 (pairs), k=3+ (general)
- Matrix multiplication trick for k=2 pair counting
- Anti-monotone pruning eliminates guaranteed losers before database scan
- Returns `itemsets`, `support`, and `stats` (k-level breakdown)

### Phase 4: `OceanStats.Unsupervised.Rules.RuleGenerator`

Converts frequent itemsets into directed association rules.

- Each k-itemset produces 2^k - 2 directed rules
- Computes support, confidence, and lift for each rule
- Optional vocabulary translation for human-readable output

### Phase 5: `OceanStats.Unsupervised.Rules.RuleFilter`

Filters, ranks, and prunes rules.

- Threshold filters: MinSupport, MinConfidence, MinLift
- Redundancy removal: drops weaker superset rules
- Top-N ranking by lift, confidence, or support

### Visualization: `OceanViz` — `"AssociationRule"` strategy

Scatter plot: confidence (x) vs lift (y), sized by support, colored by k.

## Key Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| Support | count(A and B) / N | How common is this combination? |
| Confidence | support(A,B) / support(A) | If you buy A, how likely is B? |
| Lift | confidence / support(B) | Is this better than random? (>1 = yes) |

## Usage

```matlab
% Full pipeline
reader = OceanPipeline.Ingestion.TransactionalReader(dataDir, ...
    'Mode', 'relational', 'PrimaryFile', 'orders.csv', ...
    'TransactionCol', 1, 'ItemCol', 2);
result = reader.read();

encoder = OceanStats.Unsupervised.FrequentItemsets.TransactionEncoder();
[X, state] = encoder.fitApply(result.baskets);

miner = OceanStats.Unsupervised.FrequentItemsets.AprioriMiner();
mineResult = miner.mine(X, 0.01);

gen = OceanStats.Unsupervised.Rules.RuleGenerator();
rules = gen.generate(mineResult, 0.30, ...
    'ColMap', state.colMap, 'Vocabulary', result.vocabulary);

filt = OceanStats.Unsupervised.Rules.RuleFilter();
filtered = filt.filter(rules, 'MinLift', 1.0, 'MaxRules', 50);
```

## Examples

See `Experiments/Scripts/Unsupervised/exp_market_basket_analysis.m` for the full end-to-end pipeline.

## See Also

- `IMinerStrategy` — abstract interface for swappable mining algorithms
- `OceanViz.Core` — visualization engine
- `docs/recipes/association_rules.md` — copy-paste recipes
