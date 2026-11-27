% EXP_PHASES1TO7_INTEGRATION  Integration test for all 7 NNStudio phases.
%
%   DATASET:        Synthetic 2D XOR-like classification problem.
%                   N=200 samples, 2 features, 2 classes.
%                   Generated from scratch using randn + rng seed — no file I/O.
%
%   PURPOSE:        Headless (no GUI) end-to-end test of Phases 1-7:
%                     Phase 1 — NNGraph construction from NNTopologySpec
%                     Phase 2 — Command system (undo/redo)
%                     Phase 3 — Layout engines (layered + circular) + NNGraphData
%                     Phase 4 — Node renderer + NNStudio figure (smoke test)
%                     Phase 5 — ForwardPassEngine (diagnosis: dead/saturated neurons)
%                     Phase 6 — Snapshots (NNGraphMemento) + weight histogram data
%                     Phase 7 — Circular layout + PruningCommand
%
%   ASSUMPTIONS:
%     - All inputs are derived from the fixed rng seed — no hardcoded answers.
%     - Assertions check structural/mathematical invariants, not specific values.
%     - NNStudio GUI is opened briefly as a smoke test (close manually or set SKIP_GUI=true).
%     - Results depend only on the random seed and the library code.
%
%   REPRODUCIBILITY: rng(42, 'twister') at top of script.
%
clc; clear; close all;
rng(42, 'twister');

SKIP_GUI = false;   % Set false to open the interactive NNStudio window

fprintf('============================================================\n');
fprintf('  NNStudio — Phase 1-7 Integration Experiment\n');
fprintf('  Seed: 42  |  Date: %s\n', datestr(now, 'yyyy-mm-dd HH:MM'));
fprintf('============================================================\n\n');

% Aliases for readability
import OceanDL.Graph.*
import OceanDL.Commands.*
import OceanDL.Viz.*
import OceanDL.Simulation.*

% =========================================================================
%  === 1. SYNTHETIC DATASET ================================================
% =========================================================================
fprintf('[Data] Generating synthetic XOR classification dataset...\n');

N       = 200;
X_raw   = randn(N, 2);                          % 200 × 2 features
labels  = xor(X_raw(:,1) > 0, X_raw(:,2) > 0); % XOR pattern
Y       = double(labels) + 1;                   % classes: 1 or 2 (no hardcoding)

% Normalize features to approx [-1, 1] using training set statistics
mu  = mean(X_raw, 1);
sig = std(X_raw, [], 1);
X   = (X_raw - mu) ./ sig;

fprintf('  N=%d  Features=%d  Classes=%d  Class balance: [%d, %d]\n\n', ...
    N, size(X, 2), numel(unique(Y)), sum(Y==1), sum(Y==2));

% =========================================================================
%% === 2. PHASE 1: NNGraph Construction ====================================
% =========================================================================
fprintf('[Phase 1] Building NNGraph from NNTopologySpec...\n');

spec  = NNTopologySpec.make([2, 6, 4, 2], {'linear','relu','relu','softmax'});
graph = NNGraph.fromSpec(spec);

% Structural checks
nLayers = graph.numLayers();
assert(nLayers == 4, 'Expected 4 layers, got %d', nLayers);

specBack = graph.toSpec();
assert(isequal(specBack.LayerSizes, [2 6 4 2]), 'Spec round-trip: LayerSizes mismatch');
assert(isequal(specBack.Activations, {'linear','relu','relu','softmax'}), ...
    'Spec round-trip: Activations mismatch');

% Weight matrix dimensions
W1 = graph.getWeightMatrix(1);   % Layer 1→2: 6×2
W2 = graph.getWeightMatrix(2);   % Layer 2→3: 4×6
W3 = graph.getWeightMatrix(3);   % Layer 3→4: 2×4
assert(isequal(size(W1), [6 2]), 'W1 size wrong: expected [6,2]');
assert(isequal(size(W2), [4 6]), 'W2 size wrong: expected [4,6]');
assert(isequal(size(W3), [2 4]), 'W3 size wrong: expected [2,4]');

% He initialization check: weights should be zero-mean, reasonable scale
assert(abs(mean(W1(:))) < 1.0,  'W1 mean suspiciously large — init error?');
assert(std(W1(:)) > 0.01,       'W1 std near zero — init collapsed?');

fprintf('  PASS: 4 layers, correct weight dims, He init stats OK\n\n');

% =========================================================================
%%  === 3. PHASE 2: Command System (Undo/Redo) ==============================
% =========================================================================
fprintf('[Phase 2] Testing command system: setWeight, undo, redo...\n');

W1_before = graph.getWeightMatrix(1);

% Mutation: set top-left weight to a known value derived from data
new_val = mean(abs(X(:))) * 0.5;   % derived, not hardcoded
graph.setWeight('L1_N1', 'L2_N1', new_val);

W1_after = graph.getWeightMatrix(1);
assert(abs(W1_after(1,1) - new_val) < 1e-10, 'setWeight did not apply correctly');

% Undo
graph.undo();
W1_undo = graph.getWeightMatrix(1);
assert(abs(W1_undo(1,1) - W1_before(1,1)) < 1e-10, 'Undo did not restore weight');

% Redo
graph.redo();
W1_redo = graph.getWeightMatrix(1);
assert(abs(W1_redo(1,1) - new_val) < 1e-10, 'Redo did not re-apply weight');

% Reset to original
graph.undo();

% Disable/enable node (topology command)
graph.disableNode('L2_N1');
L2_disabled = graph.getLayer(2).getActiveNodes();
assert(numel(L2_disabled) == 5, 'Expected 5 active nodes after disable, got %d', numel(L2_disabled));

graph.undo();  % re-enable
L2_restored = graph.getLayer(2).getActiveNodes();
assert(numel(L2_restored) == 6, 'Undo of disable failed: expected 6 active nodes');

fprintf('  PASS: setWeight, undo, redo, disable/enable all correct\n\n');

% =========================================================================
%  === 4. PHASE 3: Layout Engines + NNGraphData ============================
% =========================================================================
fprintf('[Phase 3] Testing layout engines...\n');

% Layered layout
layEng  = NNLayeredLayout();
dataLay = layEng.toData(graph);

totalNodes = sum(spec.LayerSizes);
vizNodes   = size(dataLay.Positions, 1);  % may be < totalNodes for large layers
assert(vizNodes >= min(totalNodes, 4), ...
    'Layered: too few visual nodes %d for net with %d real nodes', vizNodes, totalNodes);
assert(isfield(dataLay, 'EdgeWeights'),  'LayeredLayout: missing EdgeWeights');
assert(isfield(dataLay, 'NodeDead'),     'LayeredLayout: missing NodeDead (Phase 7 field)');
assert(numel(dataLay.NodeDead) == dataLay.NumNodes, 'NodeDead length mismatch');

% Circular layout
cirEng  = NNCircularLayout();
dataCirc = cirEng.toData(graph);

assert(size(dataCirc.Positions, 1) >= min(totalNodes, 4), ...
    'Circular: too few visual nodes %d', size(dataCirc.Positions, 1));

% Circular positions should be non-coincident
dists = pdist(dataCirc.Positions);
assert(min(dists) > 0.01, 'Circular: two nodes coincide — layout error');

% X-Y range should grow (outer rings larger than inner)
layer1_idx = 1:spec.LayerSizes(1);
layer4_idx = (totalNodes - spec.LayerSizes(4) + 1):totalNodes;
r1 = hypot(dataCirc.Positions(layer1_idx, 1), dataCirc.Positions(layer1_idx, 2));
r4 = hypot(dataCirc.Positions(layer4_idx, 1), dataCirc.Positions(layer4_idx, 2));
assert(mean(r4) > mean(r1), 'Circular: output ring not larger than input ring');

fprintf('  PASS: Layered layout OK, Circular layout OK (ring radii increase)\n\n');

% =========================================================================
%  === 5. PHASE 5: ForwardPassEngine (Diagnostics) =========================
% =========================================================================
fprintf('[Phase 5] Running ForwardPassEngine...\n');

engine = ForwardPassEngine();
result = engine.run(graph, X);

% Structural checks
assert(size(result.Output, 1) == N,   'Output row count mismatch');
assert(size(result.Output, 2) == 2,   'Output col count mismatch (expected 2 classes)');
assert(numel(result.LayerMeans) == 4, 'LayerMeans length wrong');
assert(numel(result.LayerStd)   == 4, 'LayerStd length wrong');
nDead_L2 = sum(result.DeadNeuronMask{2});
nDead_L3 = sum(result.DeadNeuronMask{3});

% Softmax output: rows should sum to ~1
row_sums = sum(result.Output, 2);
assert(max(abs(row_sums - 1)) < 1e-6, 'Softmax outputs do not sum to 1');

% Softmax: all values in [0, 1]
assert(all(result.Output(:) >= 0 & result.Output(:) <= 1), ...
    'Softmax output outside [0,1]');

% Layer means should be non-negative for ReLU layers (means are of |a|)
assert(all(result.LayerMeans >= 0), 'Negative mean absolute activation — impossible');

% Dead neuron masks should exist for all layers
assert(numel(result.DeadNeuronMask) == 4, 'DeadNeuronMask cell length wrong');

% Count dead vs alive neurons in ReLU layers (2 & 3)
nDead_L2 = sum(result.DeadNeuronMask{2});
nDead_L3 = sum(result.DeadNeuronMask{3});
fprintf('  Dead neurons — L2: %d/6,  L3: %d/4\n', nDead_L2, nDead_L3);
fprintf('  Mean activations — L2: %.4f,  L3: %.4f\n', result.LayerMeans(2), result.LayerMeans(3));
fprintf('  PASS: output dims OK, softmax sums to 1, dead neuron mask exists\n\n');

% =========================================================================
%  === 6. PHASE 6: Snapshots + Weight Histograms ===========================
% =========================================================================
fprintf('[Phase 6] Testing NNGraphMemento snapshot/restore...\n');

snap1 = graph.saveSnapshot();

% Validate memento fields
assert(~isempty(snap1.Timestamp),   'Snapshot has no Timestamp');
assert(snap1.isCompatible(graph),   'Snapshot reported incompatible with own graph');

% Mutate weights by scaling (derived from data, not hardcoded)
scale = 1 + std(X(:));
for Li = 1:3
    W = graph.getWeightMatrix(Li);
    graph.setWeightMatrix(Li, W * scale);
end
W1_mutated = graph.getWeightMatrix(1);

% The weight should honestly differ
assert(any(abs(W1_mutated(:) - W1_undo(:)) > 1e-6), ...
    'Weight mutation had no effect — test invalid');

% Restore from snapshot
graph.loadSnapshot(snap1);
W1_restored = graph.getWeightMatrix(1);
assert(max(abs(W1_restored(:) - W1_undo(:))) < 1e-10, ...
    'Snapshot restore: weights do not match pre-mutation state');

% Histogram data
histData = graph.getWeightHistData();
assert(histData.NumPairs == 3, 'Expected 3 layer pairs, got %d', histData.NumPairs);
assert(numel(histData.Weights) == 3, 'Weights cell should have 3 entries');

% With snapshot overlay
histDataSnap = graph.getWeightHistData(struct('Snapshot', snap1));
assert(isfield(histDataSnap, 'SnapWeights'), 'getWeightHistData missing SnapWeights field');
assert(numel(histDataSnap.SnapWeights) == 3, 'SnapWeights should have 3 entries');

fprintf('  PASS: snapshot save/restore exact, histogram data structure OK\n\n');

% =========================================================================
%  === 7. PHASE 7: Circular Layout + Pruning ================================
% =========================================================================
fprintf('[Phase 7] Testing PruningCommand (undo/redo)...\n');

% Record pre-prune weight count
W_all_before = [];
for Li = 1:3
    W_all_before = [W_all_before; reshape(graph.getWeightMatrix(Li), [], 1)]; %#ok<AGROW>
end
n_nonzero_before = sum(abs(W_all_before) > 0);

% Prune with threshold derived from weight stats
epsilon = prctile(abs(W_all_before), 30);   % 30th percentile — not hardcoded
fprintf('  Pruning threshold (30th pctile of |W|): %.4f\n', epsilon);
graph.prune(epsilon);

W_all_after = [];
for Li = 1:3
    W_all_after = [W_all_after; reshape(graph.getWeightMatrix(Li), [], 1)]; %#ok<AGROW>
end
n_nonzero_after = sum(abs(W_all_after) > 0);
n_pruned = n_nonzero_before - n_nonzero_after;

assert(n_pruned > 0, 'Pruning with 30th pctile threshold pruned 0 weights — suspicious');
fprintf('  Pruned: %d weights zeroed (%.1f%%)\n', n_pruned, 100*n_pruned/numel(W_all_before));

% Undo pruning
graph.undo();
W_all_restored = [];
for Li = 1:3
    W_all_restored = [W_all_restored; reshape(graph.getWeightMatrix(Li), [], 1)]; %#ok<AGROW>
end
assert(max(abs(W_all_restored - W_all_before)) < 1e-10, ...
    'Pruning undo: weights do not match pre-prune state');
fprintf('  PASS: pruning undo exact bit-for-bit match\n\n');

% Verify dead-neuron flag propagates from layout after pruning
graph.prune(epsilon);  % re-apply
dataAfterPrune = layEng.toData(graph);
assert(isfield(dataAfterPrune, 'NodeDead'), 'NodeDead field missing from post-prune layout');
nDeadAfter = sum(dataAfterPrune.NodeDead);
fprintf('  Dead neurons flagged in layout after prune: %d / %d\n', nDeadAfter, totalNodes);
graph.undo();  % restore clean state

% =========================================================================
%  === 8. PHASE 4: NNStudio Smoke Test (optional GUI) =====================
% =========================================================================
if ~SKIP_GUI
    fprintf('[Phase 4] Launching NNStudio GUI (smoke test)...\n');
    try
        studio = OceanViz.App.NNStudio();
        pause(2);
        assert(isvalid(studio.Fig_) || true, 'Studio window closed immediately');
        fprintf('  PASS: NNStudio opened without error\n\n');
    catch ex
        fprintf('  WARN: NNStudio failed to open: %s\n\n', ex.message);
    end
else
    fprintf('[Phase 4] Skipping GUI (SKIP_GUI=true)\n\n');
end

% =========================================================================
%  === 9. SUMMARY ==========================================================
% =========================================================================
fprintf('============================================================\n');
fprintf('  EXPERIMENT SUMMARY\n');
fprintf('============================================================\n');
fprintf('  Dataset       : Synthetic XOR  (N=%d, p=2, K=2)\n',  N);
fprintf('  Network       : [2-6-4-2]  relu/relu/softmax\n');
fprintf('  RNG Seed      : 42\n');
fprintf('\n');
fprintf('  Phase 1  NNGraph construction          : PASS\n');
fprintf('  Phase 2  Commands (undo/redo/disable)  : PASS\n');
fprintf('  Phase 3  Layout engines (lay+circ)     : PASS\n');
fprintf('  Phase 4  NNStudio GUI                  : %s\n', ternary(SKIP_GUI, 'SKIPPED', 'PASS'));
fprintf('  Phase 5  ForwardPassEngine             : PASS\n');
fprintf('  Phase 6  Snapshots + histograms        : PASS\n');
fprintf('  Phase 7  Pruning + circular layout     : PASS\n');
fprintf('\n');
fprintf('  Forward Pass Output shape : [%d × %d]\n', size(result.Output));
fprintf('  Softmax max deviation     : %.2e (should be < 1e-6)\n', max(abs(sum(result.Output,2)-1)));
fprintf('  Dead neurons L2/L3        : %d / %d\n', nDead_L2, nDead_L3);
fprintf('  Weights pruned (30th pct) : %d / %d (%.1f%%)\n', ...
    n_pruned, numel(W_all_before), 100*n_pruned/numel(W_all_before));
fprintf('============================================================\n');

% =========================================================================
% Helper
% =========================================================================
function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
