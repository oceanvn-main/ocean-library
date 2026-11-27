% COMPARE_BERT_IN_MATLAB  Load REAL prajjwal1/bert-tiny weights (exported by
% export_bert_weights.py) into our hand-rolled BertEncoder, run its forward
% pass on the same real sentences Python embedded via transformers, and
% compare -- the layer-for-layer validation of every hand-implemented piece
% (MultiHeadAttention, TokenEmbedding, TransformerEncoderBlock, WordPiece
% tokenization) against a real trained model, not synthetic weights.

function compare_bert_in_matlab()
here = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(fileparts(here)));  % .../+OceanTest/+OceanNLP/+Retrieval -> project root
addpath(projectRoot);

data = load(fullfile(here, 'bert_tiny_weights.mat'));
vocabFile = fullfile(here, 'bert_tiny_vocab.txt');

N_LAYERS = double(data.N_LAYERS);
fprintf('=== Real prajjwal1/bert-tiny: %d layers, d=%d, d_ff=%d, %d heads, vocab=%d ===\n\n', ...
    N_LAYERS, double(data.D), double(data.D_FF), double(data.NUM_HEADS), double(data.VOCAB_SIZE));

% --------------------------------------------------------------------- %
% Assemble the Weights struct array from the flat per-layer exports.
% scipy.io.savemat cannot cleanly produce a MATLAB struct array from a
% Python list-of-dicts, so export_bert_weights.py saved each layer field as
% a (N x ...) array with a leading layer-index dimension instead; this is
% the standard, robust way to reconstruct it on the MATLAB side.
% --------------------------------------------------------------------- %
Weights.TokEmbedding.WordEmbeddings = data.WordEmbeddings;
Weights.TokEmbedding.PositionEmbeddings = data.PositionEmbeddings;
Weights.TokEmbedding.TokenTypeEmbeddings = data.TokenTypeEmbeddings;
Weights.TokEmbedding.LayerNormGamma = data.EmbLNGamma(:);
Weights.TokEmbedding.LayerNormBeta = data.EmbLNBeta(:);

Layers = repmat(struct('WQ',[],'bQ',[],'WK',[],'bK',[],'WV',[],'bV',[],'WO',[],'bO',[], ...
    'NumHeads',[],'W1',[],'b1',[],'W2',[],'b2',[], ...
    'LN1Gamma',[],'LN1Beta',[],'LN2Gamma',[],'LN2Beta',[]), 1, N_LAYERS);
for i = 1:N_LAYERS
    Layers(i).WQ = squeeze(data.L_WQ(i,:,:));   Layers(i).bQ = squeeze(data.L_bQ(i,:))';
    Layers(i).WK = squeeze(data.L_WK(i,:,:));   Layers(i).bK = squeeze(data.L_bK(i,:))';
    Layers(i).WV = squeeze(data.L_WV(i,:,:));   Layers(i).bV = squeeze(data.L_bV(i,:))';
    Layers(i).WO = squeeze(data.L_WO(i,:,:));   Layers(i).bO = squeeze(data.L_bO(i,:))';
    Layers(i).NumHeads = double(data.L_NumHeads(i));
    Layers(i).W1 = squeeze(data.L_W1(i,:,:));   Layers(i).b1 = squeeze(data.L_b1(i,:))';
    Layers(i).W2 = squeeze(data.L_W2(i,:,:));   Layers(i).b2 = squeeze(data.L_b2(i,:))';
    Layers(i).LN1Gamma = squeeze(data.L_LN1Gamma(i,:))'; Layers(i).LN1Beta = squeeze(data.L_LN1Beta(i,:))';
    Layers(i).LN2Gamma = squeeze(data.L_LN2Gamma(i,:))'; Layers(i).LN2Beta = squeeze(data.L_LN2Beta(i,:))';
end
Weights.Layers = Layers;

% --------------------------------------------------------------------- %
% Construct the hand-rolled BertEncoder from REAL weights.
% --------------------------------------------------------------------- %
enc = OceanNLP.Embedding.BertEncoder('Weights', Weights, 'VocabFile', vocabFile, 'ModelName', "bert-tiny");
fprintf('BertEncoder constructed: %d layers, d=%d (embeddingDim=%d)\n\n', ...
    enc.NumLayers, enc.HiddenDim, enc.embeddingDim());

sentences = string(data.sentences(:))';
enc.fit(sentences);
E_matlab = enc.forward(sentences);   % (5 x d)
E_python = double(data.py_embeddings);

fprintf('=== Per-sentence comparison (real trained bert-tiny) ===\n');
allPass = true;
TOL = 1e-4;  % float32(PyTorch) vs float64(MATLAB) + independent op ordering -> not bit-exact, but should be tiny
for i = 1:numel(sentences)
    d = abs(E_matlab(i,:) - E_python(i,:));
    maxd = max(d);
    cosSimObj = OceanStats.Similarity.CosineSimilarity();
    cosSim = cosSimObj.forward(E_matlab(i,:), E_python(i,:));
    status = "PASS"; if maxd >= TOL, status = "FAIL"; allPass = false; end
    fprintf('  [%s] %-45s max_abs_diff=%.3e  cosine_sim=%.8f\n', status, sentences(i), maxd, cosSim);
end

fprintf('\n=== Embedding norms sanity check ===\n');
fprintf('MATLAB norms: %s\n', mat2str(round(vecnorm(E_matlab,2,2)',4)));
fprintf('Python norms: %s\n', mat2str(round(vecnorm(E_python,2,2)',4)));

if allPass
    fprintf('\nOverall: ALL CHECKS PASSED (tolerance %.0e)\n', TOL);
else
    fprintf('\nOverall: SOME CHECKS FAILED (tolerance %.0e)\n', TOL);
end
end
