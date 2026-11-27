# Source Coding & Huffman Coding Guide

## Overview

The `+OceanMath/+Information/+Coding/` package provides a complete source
coding toolkit for studying information theory. It implements Shannon's
Source Coding Theorem empirically, supporting:

- **Huffman Coding** (Huffman 1952) — optimal prefix-free binary codes
- **Kraft inequality** verification
- **Expected code length, redundancy, and efficiency** as pluggable metrics
- **Tree visualization** via `OceanViz.Core` (`'HuffmanTree'` strategy)
- **Bar chart comparison** of actual vs optimal lengths (`'CodeLengthBar'`)

### Key Theorem (Shannon 1948)

For any uniquely decodable code with codeword lengths `l_k`:

```
H(X) <= L_avg < H(X) + 1    [for binary Huffman codes]
```

where `L_avg = sum_k p_k * l_k` and `H(X) = -sum_k p_k * log2(p_k)`.

---

## Package Structure

```
+OceanMath/+Information/+Coding/
  CodeTable.m           Immutable value class holding code + all metrics
  HuffmanTree.m         Handle class, backed by DigraphCus
  ISourceCoder.m        Abstract interface (Strategy Pattern)
  HuffmanCoder.m        Huffman algorithm (implements ISourceCoder)
  huffmanEncode.m       One-liner shortcut
  optimalCodeLength.m   Theoretical -log_B(p_k)
  compareCoding.m       Full comparison struct vs Shannon bounds

+OceanMath/+Information/+Strategies/
  ExpectedCodeLengthStrategy.m    L_avg metric plugin
  RedundancyStrategy.m            R = L_avg - H(X) plugin
  CodingEfficiencyStrategy.m      eta = H(X)/L_avg plugin
  KraftInequalityStrategy.m       K = sum(B^(-l_k)) plugin
  CodingGapStrategy.m             L_avg - H(X) in bits plugin
```

---

## Usage

### Quick encode (one line)

```matlab
table = OceanMath.Information.Coding.huffmanEncode( ...
    {'a','b','c','d'}, [0.4, 0.3, 0.2, 0.1]);
table.printSummary()
```

### Full workflow

```matlab
coder = OceanMath.Information.Coding.HuffmanCoder();
table = coder.encode({'a','b','c','d','e'}, [0.35, 0.25, 0.20, 0.12, 0.08]);

% Metrics
L   = table.expectedLength()   % ~2.25 bits
H   = table.entropyBits()      % ~2.15 bits
R   = table.redundancy()       % L - H ~= 0.10
eta = table.efficiency()       % H/L ~= 0.96
K   = table.kraftSum()         % always 1.0 for Huffman
```

### Encode from DiscreteDistribution

```matlab
dist = OceanMath.Information.Distributions.DiscreteDistribution.fromPMF( ...
    {'a','b','c'}, [0.5, 0.3, 0.2]);
table = dist.huffmanEncode();
```

### Run all coding metrics via InformationEngine

```matlab
table   = OceanMath.Information.Coding.huffmanEncode(symbols, probs);
engine  = OceanMath.Information.InformationEngine(table);
results = engine.runAll('ExpectedLength', 'Redundancy', 'CodingEfficiency', ...
                        'KraftSum', 'CodingGap');
% results.ExpectedLength, results.Redundancy, ...
```

### One-shot summary

```matlab
engine = OceanMath.Information.InformationEngine();
S = engine.encodeSummary({'a','b','c'}, [0.5, 0.3, 0.2]);
S.table.printSummary()
S.metrics   % struct with all 5 metrics
```

### Compare vs Shannon bounds

```matlab
S = OceanMath.Information.Coding.compareCoding( ...
    {'a','b','c','d'}, [0.4, 0.3, 0.2, 0.1], 'Verbose', true);
% S.theorem_satisfied  should always be true for Huffman
```

---

## Visualization

### Prefix Coding Tree (Reingold-Tilford Layout)

The `OceanViz.Strategies.CodeTreeRenderer` class provides high-fidelity, interactive visualizations of prefix-free coding trees (e.g., Huffman or Shannon-Fano). It automatically calculates coordinates so that Level 0 (Root) is at the top, leaves are evenly spaced at the bottom, and internal nodes sit at average parent-child centers.

```matlab
% 1. Create a prefix-free CodeTable
table = OceanMath.Information.Coding.huffmanEncode( ...
    {'a','b','c','d'}, [0.4, 0.3, 0.2, 0.1]);

% 2. Setup figure and axes in Dark theme
fig = figure('Color', [0.08, 0.08, 0.10]);
ax  = axes(fig);

% 3. Instantiate the renderer (integrates with OceanViz.Core)
renderer = OceanViz.Strategies.CodeTreeRenderer(ax);

% 4. Render the tree
renderer.renderTree(table);

% 5. Highlight the path from root to symbol 'c' in glowing neon cyan
renderer.highlightPathTo('c');
```

### Code length comparison bar chart

```matlab
table = OceanMath.Information.Coding.huffmanEncode(symbols, probs);
viz = OceanViz.Core('Theme', 'Dark');
viz.plot(ax, 'CodeLengthBar', struct('table', table));
```

---

## Extension Points

### Add a new coder (e.g., Shannon-Fano)

1. Create `+OceanMath/+Information/+Coding/ShannonFanoCoder.m`
2. Implement `ISourceCoder`: `encode()`, `getName()`, `isOptimal()`
3. Optionally add string alias in `InformationEngine.resolveStrategy_()`

### Add a new metric

1. Create `+OceanMath/+Information/+Strategies/MyMetricStrategy.m`
2. Implement `IMetricStrategy`: `calculate(context)`, `isCompatible()`, `getName()`
3. Add alias case in `InformationEngine.resolveStrategy_()`

---

## See Also

- `docs/recipes/huffman_coding_recipes.md` — copy-paste recipes
- `OceanMath.Information.InformationEngine` — orchestrator
- `OceanMath.Information.Coding.CodeTable` — full metric API
- `OceanMath.Information.Distributions.DiscreteDistribution` — source distribution
- Shannon, C.E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*.
- Huffman, D.A. (1952). "A Method for the Construction of Minimum Redundancy Codes." *IRE*.
