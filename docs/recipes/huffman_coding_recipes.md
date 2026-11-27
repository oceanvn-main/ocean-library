# Huffman Coding Recipes

Copy-paste recipes for all common source-coding tasks using the Ocean Library.

---

## 1. Quick Huffman Encode

```matlab
table = OceanMath.Information.Coding.huffmanEncode( ...
    {'a','b','c','d'}, [0.4, 0.3, 0.2, 0.1]);
table.printSummary()
```

---

## 2. Encode and Extract All Metrics

```matlab
syms  = {'x1','x2','x3','x4','x5'};
probs = [0.35, 0.25, 0.20, 0.12, 0.08];
table = OceanMath.Information.Coding.huffmanEncode(syms, probs);

L   = table.expectedLength()   % average bits per symbol
H   = table.entropyBits()      % Shannon entropy
R   = table.redundancy()       % L - H  (always >= 0)
eta = table.efficiency()       % H/L    (always in (0,1])
K   = table.kraftSum()         % always = 1.0 for Huffman
```

---

## 3. Encode from a DiscreteDistribution

```matlab
dist = OceanMath.Information.Distributions.DiscreteDistribution.fromData( ...
    ['A','A','A','B','B','C']');
table = dist.huffmanEncode();
table.printSummary()
```

---

## 4. Run All Metrics via InformationEngine (fluent API)

```matlab
table   = OceanMath.Information.Coding.huffmanEncode(syms, probs);
engine  = OceanMath.Information.InformationEngine(table);
results = engine.runAll('ExpectedLength','Redundancy','CodingEfficiency','KraftSum','CodingGap');

results.ExpectedLength    % L_avg
results.Redundancy        % R = L - H
results.CodingEfficiency  % eta = H/L
results.KraftSum          % K (should be 1.0)
results.CodingGap         % L - H in bits
```

---

## 5. One-Shot Summary (encode + all metrics)

```matlab
engine = OceanMath.Information.InformationEngine();
S = engine.encodeSummary({'a','b','c'}, [0.5, 0.3, 0.2]);
S.table.printSummary()
disp(S.metrics)
```

---

## 6. Compare vs Shannon Bounds

```matlab
S = OceanMath.Information.Coding.compareCoding( ...
    {'a','b','c','d'}, [0.4, 0.3, 0.2, 0.1], 'Verbose', true);
% Prints full comparison table + verifies H(X) <= L_avg < H(X)+1
S.theorem_satisfied    % true always for Huffman
S.shannon_lower_bound  % H(X)
S.shannon_upper_bound  % H(X) + 1
S.huffman_L            % actual L_avg
```

---

## 7. Plot the Prefix Coding Tree (Interactive Layout)

```matlab
% Create table of Huffman codewords
table = OceanMath.Information.Coding.huffmanEncode( ...
    {'a','b','c','d'}, [0.4, 0.3, 0.2, 0.1]);

fig = figure('Color', [0.08, 0.08, 0.10]);
ax  = axes(fig);

% Render prefix tree using Reingold-Tilford coordinates
renderer = OceanViz.Strategies.CodeTreeRenderer(ax);
renderer.renderTree(table);

% Highlight path for symbol 'b' (glowing cyan edge & white node)
renderer.highlightPathTo('b');
```

---

## 8. Plot Code Length Comparison

```matlab
table = OceanMath.Information.Coding.huffmanEncode(syms, probs);

fig = figure('Color', [0.11 0.11 0.13]);
ax  = axes(fig);
viz = OceanViz.Core('Theme', 'Dark');
viz.plot(ax, 'CodeLengthBar', struct('table', table));
```

---

## 9. Inspect Theoretical Optimal Lengths

```matlab
probs  = [0.5, 0.25, 0.25];
l_opt  = OceanMath.Information.Coding.optimalCodeLength(probs);
% l_opt = [1.0, 2.0, 2.0]  — exactly achievable for this distribution

table = OceanMath.Information.Coding.huffmanEncode( ...
    {'a','b','c'}, probs);
disp([table.getLengths(); l_opt])   % compare actual vs optimal
```

---

## 10. Inspect the Tree Graph

```matlab
coder = OceanMath.Information.Coding.HuffmanCoder();
tree  = coder.buildTree({'a','b','c','d'}, [0.4, 0.3, 0.2, 0.1]);

g = tree.toDigraph();   % DigraphCus
g.describe()            % prints Nodes / Edges count
tree.depth()            % max codeword length
tree.numLeaves()        % K
tree.numInternal()      % K - 1
```

---

## 11. Ternary Huffman Code (Base=3)

```matlab
% 7 symbols for a balanced ternary tree (7-1 = 6, divisible by 3-1 = 2)
syms  = {'a','b','c','d','e','f','g'};
probs = [0.22, 0.20, 0.18, 0.15, 0.12, 0.08, 0.05];
table = OceanMath.Information.Coding.huffmanEncode(syms, probs, 'Base', 3);
table.printSummary()
```

---

## 12. Shannon-Fano Coding

```matlab
% Instantiate Shannon-Fano binary coder
coder = OceanMath.Information.Coding.ShannonFanoCoder();

% Encode a symbol distribution
syms  = {'a','b','c','d'};
probs = [0.40, 0.30, 0.20, 0.10];
table = coder.encode(syms, probs);

% Print the resulting prefix-free code table and metrics
table.printSummary()

% Extract individual metrics
L   = table.expectedLength()   % average bits per symbol
H   = table.entropyBits()      % Shannon entropy
eta = table.efficiency()       % H / L
```

---

## See Also

- `docs/guides/source_coding.md` — full conceptual guide
- `OceanMath.Information.Coding.ShannonFanoCoder` — Shannon-Fano source coder
- `OceanMath.Information.Coding.CodeTable` — full metric API
