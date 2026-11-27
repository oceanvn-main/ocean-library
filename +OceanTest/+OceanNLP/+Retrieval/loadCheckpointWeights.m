function W = loadCheckpointWeights(matFile)
% LOADCHECKPOINTWEIGHTS  Assemble the BertBackbone/CrossEncoder/ColBERTEncoder
% Weights struct from a <tag>_weights.mat written by export_neural_checkpoints.py.
% scipy.io.savemat cannot produce a MATLAB struct array, so per-layer fields
% arrive as (N x ...) arrays with a leading layer index; this rebuilds
% Weights.Layers(i). Matrices are PyTorch (out,in) layout, used as-is.
% Head fields (Pooler / Classifier / Projection) are added when present.
arguments
    matFile (1,1) string
end
data = load(matFile);
n = double(data.N_LAYERS);
W.TokEmbedding.WordEmbeddings = double(data.WordEmbeddings);
W.TokEmbedding.PositionEmbeddings = double(data.PositionEmbeddings);
W.TokEmbedding.TokenTypeEmbeddings = double(data.TokenTypeEmbeddings);
W.TokEmbedding.LayerNormGamma = double(data.EmbLNGamma(:));
W.TokEmbedding.LayerNormBeta = double(data.EmbLNBeta(:));

mats = {'WQ','WK','WV','WO','W1','W2'};
vecs = {'bQ','bK','bV','bO','b1','b2','LN1Gamma','LN1Beta','LN2Gamma','LN2Beta'};
Layers = repmat(struct('WQ',[],'bQ',[],'WK',[],'bK',[],'WV',[],'bV',[],'WO',[],'bO',[], ...
    'NumHeads',[],'W1',[],'b1',[],'W2',[],'b2',[],'LN1Gamma',[],'LN1Beta',[],'LN2Gamma',[],'LN2Beta',[]), 1, n);
for i = 1:n
    for k = 1:numel(mats)
        Layers(i).(mats{k}) = double(squeeze(data.(['L_' mats{k}])(i, :, :)));
    end
    for k = 1:numel(vecs)
        Layers(i).(vecs{k}) = double(squeeze(data.(['L_' vecs{k}])(i, :)))';
    end
    Layers(i).NumHeads = double(data.NUM_HEADS);
end
W.Layers = Layers;
if isfield(data, 'PoolerW'),     W.Pooler.W = double(data.PoolerW);         W.Pooler.b = double(data.PoolerB(:)); end
if isfield(data, 'ClassifierW'), W.Classifier.W = double(data.ClassifierW); W.Classifier.b = double(data.ClassifierB(:)); end
if isfield(data, 'ProjectionW'), W.Projection.W = double(data.ProjectionW); end
end
