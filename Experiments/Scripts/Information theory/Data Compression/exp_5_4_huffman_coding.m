%% exp_5_4_huffman_coding.m
% Problem 5.4 — Huffman Coding (Cover & Thomas, Elements of Information Theory)
%
% Random variable:
%   X = | x1    x2    x3    x4    x5    x6    x7  |
%       | 0.49  0.26  0.12  0.04  0.04  0.03  0.02 |
%
% Tasks:
%   (a) Find a binary Huffman code for X
%   (b) Find the expected code length for this encoding
%   (c) Find a ternary Huffman code for X
%
% Assumption: All probabilities are given exactly and sum to 1.00.
% No encoded answers — all results are derived from the PMF.
%
% Usage: Run directly in MATLAB from the workspace root.
%   >> run('Experiments/Scripts/exp_5_4_huffman_coding.m')

clc;
fprintf('==========================================================\n');
fprintf('  Problem 5.4 — Huffman Coding\n');
fprintf('==========================================================\n\n');

% =========================================================================
%% 0. Define the source
% =========================================================================
symbols = {'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'};
probs   = [0.49,  0.26,  0.12,  0.04,  0.04,  0.03,  0.02];

fprintf('Source X:\n');
fprintf('  %-6s', symbols{:}); fprintf('\n');
fprintf('  %-6.2f', probs);    fprintf('\n\n');

assert(abs(sum(probs) - 1.0) < 1e-12, 'Probabilities must sum to 1.');

% =========================================================================
%% (a) Binary Huffman Code
% =========================================================================
fprintf('----------------------------------------------------------\n');
fprintf('(a) Binary Huffman Code  [Base = 2]\n');
fprintf('----------------------------------------------------------\n');

coder_bin  = OceanMath.Information.Coding.HuffmanCoder('Base', 2);
table_bin  = coder_bin.encode(symbols, probs);
table_bin.printSummary();

% =========================================================================
%% (b) Expected Code Length
% =========================================================================
fprintf('----------------------------------------------------------\n');
fprintf('(b) Expected Code Length\n');
fprintf('----------------------------------------------------------\n');

H   = table_bin.entropyBits();
L   = table_bin.expectedLength();
R   = table_bin.redundancy();
eta = table_bin.efficiency();
K_kraft = table_bin.kraftSum();

fprintf('  H(X)               = %.6f bits\n', H);
fprintf('  L_avg (binary)     = %.6f bits/symbol\n', L);
fprintf('  Redundancy R       = L_avg - H(X)  = %.6f bits\n', R);
fprintf('  Efficiency eta     = H(X) / L_avg  = %.4f  (%.2f%%)\n', eta, eta*100);
fprintf('  Kraft sum          = %.6f  (complete: %s)\n', K_kraft, mat2str(table_bin.isComplete()));
fprintf('\n  Shannon bounds:    H(X) <= L_avg < H(X)+1\n');
fprintf('  Check:             %.6f <= %.6f < %.6f  --> %s\n', ...
    H, L, H+1, mat2str(L >= H-1e-9 && L < H+1+1e-9));

% Verify per-symbol contributions manually
fprintf('\n  Per-symbol check: L_avg = sum(p_k * l_k)\n');
lens = table_bin.getLengths();
fprintf('    %-6s  p_k     l_k   p_k*l_k\n', 'sym');
fprintf('    %s\n', repmat('-',1,35));
for i = 1:numel(symbols)
    fprintf('    %-6s  %.4f  %3d   %.4f\n', symbols{i}, probs(i), lens(i), probs(i)*lens(i));
end
fprintf('    %s\n', repmat('-',1,35));
fprintf('    %-6s  %-6s  %-3s   %.4f  <-- L_avg\n', '', '1.0000', '', L);

% =========================================================================
%% (c) Ternary Huffman Code
% =========================================================================
fprintf('\n----------------------------------------------------------\n');
fprintf('(c) Ternary Huffman Code  [Base = 3]\n');
fprintf('----------------------------------------------------------\n');

% Check padding requirement: (K-1) mod (k-1) must equal 0
K_sym = numel(symbols);
k     = 3;
r = mod(K_sym - 1, k - 1);
fprintf('  Alphabet size K = %d, Base k = %d\n', K_sym, k);
fprintf('  Padding check: (K-1) mod (k-1) = (%d-1) mod (%d-1) = %d mod %d = %d\n', ...
    K_sym, k, K_sym-1, k-1, r);
if r == 0
    fprintf('  --> No dummy symbols needed. Tree is perfectly balanced.\n\n');
else
    nDummy = k - 1 - r;
    fprintf('  --> Need %d dummy symbol(s) with p=0 to balance.\n\n', nDummy);
end

coder_ter  = OceanMath.Information.Coding.HuffmanCoder('Base', 3);
table_ter  = coder_ter.encode(symbols, probs);
table_ter.printSummary();

% Ternary expected length
H_ter  = table_ter.entropyBits();    % same H(X) in bits
H_trit = H_ter / log2(3);           % entropy in trits
L_ter  = table_ter.expectedLength();
R_ter  = table_ter.redundancy();
eta_ter = table_ter.efficiency();

fprintf('  H(X) in trits      = H_bits / log2(3) = %.6f / %.6f = %.6f trits\n', ...
    H_ter, log2(3), H_trit);
fprintf('  L_avg (ternary)    = %.6f trits/symbol\n', L_ter);
fprintf('  Redundancy R       = %.6f trits\n', R_ter);
fprintf('  Efficiency eta     = %.4f  (%.2f%%)\n', eta_ter, eta_ter*100);
fprintf('  Kraft sum          = %.6f  (complete: %s)\n', table_ter.kraftSum(), mat2str(table_ter.isComplete()));

% =========================================================================
%% Summary comparison
% =========================================================================
fprintf('\n==========================================================\n');
fprintf('  Summary\n');
fprintf('==========================================================\n');
fprintf('  %-22s  %-12s  %-12s\n', 'Metric', 'Binary (b=2)', 'Ternary (b=3)');
fprintf('  %s\n', repmat('-',1,50));
fprintf('  %-22s  %-12.6f  %-12.6f\n', 'H(X) [bits]',   H,     H_ter);
fprintf('  %-22s  %-12.6f  %-12.6f\n', 'H(X) [base units]', H/log2(2), H_trit);
fprintf('  %-22s  %-12.6f  %-12.6f\n', 'L_avg [base units]',L,     L_ter);
fprintf('  %-22s  %-12.6f  %-12.6f\n', 'Redundancy R',   R,     R_ter);
fprintf('  %-22s  %-12.2f%%  %-12.2f%%\n', 'Efficiency eta', eta*100, eta_ter*100);
fprintf('  %-22s  %-12.6f  %-12.6f\n', 'Kraft sum',      K_kraft, table_ter.kraftSum());
fprintf('  %s\n', repmat('-',1,50));

% =========================================================================
%% Visualization
% =========================================================================
fprintf('\n[Visualization] Generating figures...\n');

viz = OceanViz.Core('Theme', 'Dark');

% --- Figure 1: Binary Huffman Tree ---
fig1 = viz.createFigure('Problem 5.4 — Binary Huffman Tree');
ax1  = axes(fig1);
viz.plot(ax1, 'HuffmanTree', struct( ...
    'tree',           coder_bin.buildTree(symbols, probs), ...
    'showCodewords',  true, ...
    'showWeights',    true, ...
    'showEdgeLabels', true, ...
    'title',          'Binary Huffman Tree  (Base = 2)'));

% --- Figure 2: Ternary Huffman Tree ---
fig2 = viz.createFigure('Problem 5.4 — Ternary Huffman Tree');
ax2  = axes(fig2);
viz.plot(ax2, 'HuffmanTree', struct( ...
    'tree',           coder_ter.buildTree(symbols, probs), ...
    'showCodewords',  true, ...
    'showWeights',    true, ...
    'showEdgeLabels', true, ...
    'title',          'Ternary Huffman Tree  (Base = 3)'));

% --- Figure 3: Code length comparison (binary) ---
fig3 = viz.createFigure('Problem 5.4 — Code Length Comparison');
ax3  = axes(fig3);
viz.plot(ax3, 'CodeLengthBar', struct( ...
    'table', table_bin, ...
    'title', 'Binary Huffman: Actual vs Optimal Codeword Lengths'));

fprintf('[Done] All computations and figures complete.\n\n');
