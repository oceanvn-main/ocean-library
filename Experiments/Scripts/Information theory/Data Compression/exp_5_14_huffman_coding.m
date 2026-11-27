%% exp_5_14_huffman_coding.m
% Problem 5.14 — Huffman Code (Cover & Thomas, Elements of Information Theory)
%
% Find (a) binary and (b) ternary Huffman codes for X with:
%
%   p = (1/21, 2/21, 3/21, 4/21, 5/21, 6/21)
%
% (c) Calculate L = sum(p_i * l_i) in each case.
%
% Assumption: probabilities given as exact fractions k/21. Derived purely
% from the PMF — no encoded answers.
%
% Usage:
%   >> run('Experiments/Scripts/Information theory/exp_5_14_huffman_coding.m')

clc;
fprintf('==========================================================\n');
fprintf('  Problem 5.14 — Huffman Coding\n');
fprintf('==========================================================\n\n');

% =========================================================================
%% 0. Define source
% =========================================================================
symbols = {'x1','x2','x3','x4','x5','x6'};
numer   = [1, 2, 3, 4, 5, 6];
probs   = numer / 21;          % exact fractions k/21

fprintf('Source X:\n');
fprintf('  %-6s', symbols{:}); fprintf('\n');
fracLabels = arrayfun(@(n) sprintf('%d/21',n), numer, 'UniformOutput', false);
fprintf('  %-6s', fracLabels{:}); fprintf('\n');
fprintf('  %-6.4f', probs);    fprintf('\n\n');

assert(abs(sum(probs) - 1.0) < 1e-12, 'Probabilities must sum to 1.');

% Encode shortcuts
hEnc = @OceanMath.Information.Coding.huffmanEncode;

% =========================================================================
%% (a) Binary Huffman Code
% =========================================================================
fprintf('----------------------------------------------------------\n');
fprintf('(a) Binary Huffman Code  [Base = 2]\n');
fprintf('----------------------------------------------------------\n');

table_bin = hEnc(symbols, probs, 'Base', 2);
table_bin.printSummary();

% =========================================================================
%% (b) Ternary Huffman Code
% =========================================================================
fprintf('----------------------------------------------------------\n');
fprintf('(b) Ternary Huffman Code  [Base = 3]\n');
fprintf('----------------------------------------------------------\n');

% Padding check: (K-1) mod (k-1) must be 0
K = numel(symbols); k = 3;
r = mod(K - 1, k - 1);
if r ~= 0
    nDummy = (k - 1) - r;
    fprintf('  Padding: need %d dummy symbol(s) (p=0) to balance ternary tree.\n\n', nDummy);
else
    fprintf('  No padding needed.\n\n');
end

table_ter = hEnc(symbols, probs, 'Base', 3);
table_ter.printSummary();

% =========================================================================
%% (c) Expected Code Length L = sum(p_i * l_i)
% =========================================================================
fprintf('----------------------------------------------------------\n');
fprintf('(c) Expected Code Length  L = sum(p_i * l_i)\n');
fprintf('----------------------------------------------------------\n');

H_bits  = table_bin.entropyBits();
H_trits = H_bits / log2(3);

L_bin   = table_bin.expectedLength();
L_ter   = table_ter.expectedLength();

R_bin   = table_bin.redundancy();
R_ter   = table_ter.redundancy();

eta_bin = table_bin.efficiency();
eta_ter = table_ter.efficiency();

% Per-symbol breakdown
fprintf('\n  Per-symbol breakdown:\n');
fprintf('  %-5s  %-8s  %-5s  %-8s  %-5s  %-8s\n', ...
    'Sym', 'p_i', 'l_bin', 'p*l_bin', 'l_ter', 'p*l_ter');
fprintf('  %s\n', repmat('-',1,52));

lens_bin = table_bin.getLengths();
lens_ter = table_ter.getLengths();

for i = 1:numel(symbols)
    fprintf('  %-5s  %-8s  %-5d  %-8.4f  %-5d  %-8.4f\n', ...
        symbols{i}, sprintf('%d/21',numer(i)), ...
        lens_bin(i), probs(i)*lens_bin(i), ...
        lens_ter(i), probs(i)*lens_ter(i));
end
fprintf('  %s\n', repmat('-',1,52));
fprintf('  %-5s  %-8s  %-5s  %-8.4f  %-5s  %-8.4f\n', ...
    '', '1.0000', '', L_bin, '', L_ter);

fprintf('\n  =======================================================\n');
fprintf('  %-22s  %-12s  %-12s\n', 'Metric', 'Binary', 'Ternary');
fprintf('  %s\n', repmat('-',1,50));
fprintf('  %-22s  %-12.6f  %-12.6f\n', 'H(X) [bits]',            H_bits,  H_bits);
fprintf('  %-22s  %-12.6f  %-12.6f\n', 'H(X) [base units]',      H_bits,  H_trits);
fprintf('  %-22s  %-12.6f  %-12.6f\n', 'L_avg [base units]',     L_bin,   L_ter);
fprintf('  %-22s  %-12.6f  %-12.6f\n', 'Redundancy R = L-H',     R_bin,   R_ter);
fprintf('  %-22s  %-10.2f%%  %-10.2f%%\n', 'Efficiency eta = H/L', eta_bin*100, eta_ter*100);
fprintf('  %-22s  %-12.6f  %-12.6f\n', 'Kraft sum K',            table_bin.kraftSum(), table_ter.kraftSum());
fprintf('  %s\n', repmat('-',1,50));

% Fraction form for L_bin
[L_num, L_den] = rat(L_bin, 1e-6);
fprintf('\n  L_binary  = %d/%d  ≈  %.4f bits/symbol\n', L_num, L_den, L_bin);
[L_num3, L_den3] = rat(L_ter, 1e-6);
fprintf('  L_ternary = %d/%d  ≈  %.4f trits/symbol\n\n', L_num3, L_den3, L_ter);

% =========================================================================
%% Visualization
% =========================================================================
fprintf('[Visualization] Generating figures...\n');

coder_bin = OceanMath.Information.Coding.HuffmanCoder('Base', 2);
coder_ter = OceanMath.Information.Coding.HuffmanCoder('Base', 3);
viz       = OceanViz.Core('Theme', 'Dark');

% Fig 1: Binary tree
fig1 = viz.createFigure('Problem 5.14 — Binary Huffman Tree');
ax1  = axes(fig1);
viz.plot(ax1, 'HuffmanTree', struct( ...
    'tree',           coder_bin.buildTree(symbols, probs), ...
    'showCodewords',  true, ...
    'showWeights',    true, ...
    'showEdgeLabels', true, ...
    'title',          'Binary Huffman Tree  (Base = 2)'));

% Fig 2: Ternary tree
fig2 = viz.createFigure('Problem 5.14 — Ternary Huffman Tree');
ax2  = axes(fig2);
viz.plot(ax2, 'HuffmanTree', struct( ...
    'tree',           coder_ter.buildTree(symbols, probs), ...
    'showCodewords',  true, ...
    'showWeights',    true, ...
    'showEdgeLabels', true, ...
    'title',          'Ternary Huffman Tree  (Base = 3)'));

% Fig 3: Code length comparison
fig3 = viz.createFigure('Problem 5.14 — Code Length Comparison');
ax3  = axes(fig3);
viz.plot(ax3, 'CodeLengthBar', struct( ...
    'table', table_bin, ...
    'title', 'Binary Huffman: Actual vs Shannon Optimal Lengths'));

fprintf('[Done]\n\n');
