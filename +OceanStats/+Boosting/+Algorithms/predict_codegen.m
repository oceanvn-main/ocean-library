function Y_pred = predict_codegen(X, model_flat)
% PREDICT_CODEGEN  MATLAB Coder-compatible GBM predict function.
%
%   Y_pred = predict_codegen(X, model_flat)
%
%   This function is a pure-numeric re-implementation of
%   GradientBoostingMachine.predict() that uses ONLY plain double arrays —
%   no handle classes, no cell arrays, no structs with dynamic fields.
%   It can be compiled to C/MEX using codegen (MATLAB Coder).
%
%   INPUTS:
%     X          — [N x p] double feature matrix
%     model_flat — The flat model struct produced by GBM.exportForCodegen().
%                  Fields:
%                    .F0             scalar initial prediction
%                    .LearningRate   scalar nu
%                    .NumTrees       scalar M
%                    .TreeOffsets    (M+1) x 1  — start index of each tree in node arrays
%                    .FeatureIdx     (T x 1)    — split feature (0 = leaf)
%                    .Threshold      (T x 1)    — split threshold
%                    .LeftChild      (T x 1)    — index of left child node (1-based)
%                    .RightChild     (T x 1)    — index of right child node (1-based)
%                    .LeafValue      (T x 1)    — prediction value (used at leaves)
%
%   OUTPUT:
%     Y_pred — [N x 1] predictions
%
%   CODEGEN ENTRY POINT:
%     This function is suitable as the entry-point for codegen. See run_codegen.m.

%#codegen

N  = size(X, 1);
F0 = model_flat.F0;
nu = model_flat.LearningRate;
M  = model_flat.NumTrees;

Y_pred = repmat(F0, N, 1);

% --- Walk every tree in the flat array ---
for m = 1:M
    treeStart = model_flat.TreeOffsets(m);
    treeEnd   = model_flat.TreeOffsets(m + 1) - 1;

    tree_feat  = model_flat.FeatureIdx(treeStart:treeEnd);
    tree_thr   = model_flat.Threshold (treeStart:treeEnd);
    tree_left  = model_flat.LeftChild (treeStart:treeEnd);
    tree_right = model_flat.RightChild(treeStart:treeEnd);
    tree_val   = model_flat.LeafValue (treeStart:treeEnd);

    for i = 1:N
        node = 1;   % root is always node 1 within this tree's sub-array
        while tree_feat(node) ~= 0   % 0 = leaf marker
            if X(i, tree_feat(node)) <= tree_thr(node)
                node = tree_left(node);
            else
                node = tree_right(node);
            end
        end
        Y_pred(i) = Y_pred(i) + nu * tree_val(node);
    end
end
end
