function compare_subword_in_matlab()
% COMPARE_SUBWORD_IN_MATLAB  External validation of the subword stack.
%
%   (1) INFERENCE  pretrained GPT-2 (vocab.json + merges.txt, 50,257 tokens): ids, token strings and
%       decoded text vs transformers GPT2TokenizerFast on a Unicode torture set + real Cranfield text.
%   (2) TRAINING   BPE trained on all 1400 Cranfield docs (vocab 1000, byte-level) vs HF `tokenizers`
%       BpeTrainer:
%         (a) our own merge list vs HF's: agreement prefix (tie-breaks legitimately differ),
%         (b) REPLAY: feed HF's merge list through our counting engine; every HF choice must be a
%             MAXIMUM-count pair at its step (this is tie-break independent),
%         (c) SEGMENTATION with HF's merges vs HF's encodings of the first 150 docs (token strings).
%
% Needs subword_reference.py output (gpt2_vocab.json, gpt2_merges.txt, subword_reference.json,
% bpe_train_reference.json) and ../+Retrieval/cranfield_texts.json.
% Run:  OceanTest.OceanNLP.Text.compare_subword_in_matlab()

here = fileparts(mfilename('fullpath'));
addpath(fileparts(fileparts(fileparts(here))));
allPass = true;

% ---------------- (1) GPT-2 inference ----------------
t0 = tic;
tok = OceanNLP.Subword.SubwordTokenizer.fromFiles(fullfile(here, 'gpt2_vocab.json'), fullfile(here, 'gpt2_merges.txt'));
fprintf('loaded GPT-2 (%d tokens, %d merges) in %.1fs\n', tok.Vocab.VocabSize, size(tok.Algorithm.Merges, 1), toc(t0));
cases = jsondecode(fileread(fullfile(here, 'subword_reference.json')));
badIds = 0;  badTok = 0;  badDec = 0;  first = "";
t0 = tic;
for i = 1:numel(cases)
    c = cases(i);
    text = string(c.text);
    refIds = double(c.ids(:))';
    refTok = toStrings_(c.tokens);
    tk = tok.forward(text);
    ids = tok.encode(text);
    if ~isequal(tk, refTok), badTok = badTok + 1; if first == "", first = sprintf('%d: got [%s] expected [%s]', i, strjoin(tk, '|'), strjoin(refTok, '|')); end; end
    if ~isequal(ids, refIds), badIds = badIds + 1; end
    if isempty(refIds), dec = ""; else, dec = tok.decode(ids); end
    if dec ~= string(c.decoded), badDec = badDec + 1; end
end
fprintf('GPT-2 tokenization of %d texts took %.1fs\n', numel(cases), toc(t0));
allPass = report_('GPT-2 token strings vs GPT2TokenizerFast', numel(cases), badTok, first) && allPass;
allPass = report_('GPT-2 ids vs GPT2TokenizerFast', numel(cases), badIds, "") && allPass;
allPass = report_('GPT-2 decode(encode(x)) vs HF decode', numel(cases), badDec, "") && allPass;

% ---------------- (2) BPE training ----------------
texts = jsondecode(fileread(fullfile(here, '..', '+Retrieval', 'cranfield_texts.json')));
docs = string(texts.docs(:))';
ref = jsondecode(fileread(fullfile(here, 'bpe_train_reference.json')));
hfMerges = split(string(ref.merges(:)), " ");                % (M x 2)
M = size(hfMerges, 1);

ours = OceanNLP.Subword.SubwordTokenizer();
t0 = tic;
ours.fit(docs, 'VocabSize', double(ref.vocab_size));
fprintf('trained BPE (vocab %d) on %d docs in %.1fs\n', ours.Vocab.VocabSize - 1, numel(docs), toc(t0));   % -1: fallback <unk>
om = ours.Algorithm.Merges;
n = min(size(om, 1), M);
same = all(om(1:n, :) == hfMerges(1:n, :), 2);
firstDiff = find(~same, 1);
if isempty(firstDiff), firstDiff = n + 1; end
fprintf('(a) our merges vs HF: %d merges each; identical for the first %d; identical in place: %d/%d\n', ...
    size(om, 1), firstDiff - 1, nnz(same), n);
allPass = report_('(a) our trained merge list identical to HF (all merges, in order)', M, (M - nnz(same)) + abs(size(om, 1) - M), "") && allPass;

% (b) replay HF's merges through our counting engine
pre = OceanNLP.Text.PreTokenize.Gpt2PreTokenizer();
allWords = strings(1, 0);
for d = 1:numel(docs)
    w = pre.forward(docs(d));
    for k = 1:numel(w), w(k) = OceanNLP.Subword.ByteLevel.encode(w(k)); end
    allWords = [allWords, w]; %#ok<AGROW>
end
[uw, ~, ic] = unique(allWords);
counts = accumarray(ic(:), 1)';
[~, ~, stats] = OceanNLP.Subword.Algorithm.BpeAlgorithm.trainMerges(uw, counts, double(ref.vocab_size), ...
    OceanNLP.Subword.ByteLevel.alphabet(), 0, "", hfMerges);
notMax = nnz(stats.PairCount ~= stats.MaxCount);
allPass = report_('(b) HF merge choices that are NOT a maximum-count pair (replay)', M, notMax, "") && allPass;

% (c) segmentation with HF's merges vs HF's encodings (token strings, first 150 docs)
seg = OceanNLP.Subword.SubwordTokenizer.fromMerges(hfMerges);
nDoc = numel(ref.encodings);
badSeg = 0;  badOurs = 0;  firstSeg = "";
for i = 1:nDoc
    refTok = toStrings_(ref.encodings(i).tokens);
    got = seg.forward(docs(i));
    if ~isequal(got, refTok)
        badSeg = badSeg + 1;
        if firstSeg == "", firstSeg = sprintf('doc %d: got [%s] expected [%s]', i, strjoin(got(1:min(end,12)), '|'), strjoin(refTok(1:min(end,12)), '|')); end
    end
    if ~isequal(ours.forward(docs(i)), refTok), badOurs = badOurs + 1; end
end
allPass = report_('(c) segmentation with HF merges vs HF encodings', nDoc, badSeg, firstSeg) && allPass;
fprintf('    (info) OUR trained merges vs HF encodings: %d/%d docs tokenize identically (should be all)\n', nDoc - badOurs, nDoc);

% ---------------- (3) WordPiece training vs HF WordPieceTrainer ----------------
wref = jsondecode(fileread(fullfile(here, 'wordpiece_train_reference.json')));
hfPieces = toStrings_(wref.pieces);
wp = OceanNLP.Subword.SubwordTokenizer('PreTokenizer', OceanNLP.Text.PreTokenize.BertBasicPreTokenizer('NeverSplit', strings(1, 0)), ...
    'Algorithm', OceanNLP.Subword.Algorithm.WordPieceAlgorithm(), 'ByteLevel', false);
t0 = tic;
wp.fit(docs, 'VocabSize', numel(hfPieces));
fprintf('trained WordPiece (vocab %d) in %.1fs\n', wp.Vocab.VocabSize, toc(t0));
myPieces = wp.Vocab.Tokens;
setEqual = isequal(sort(myPieces), sort(hfPieces));
inPlace = nnz(myPieces(1:min(end, numel(hfPieces))) == hfPieces(1:min(end, numel(myPieces))));
fprintf('(3a) WordPiece vocab: same SET of pieces: %d ; identical id order: %d/%d\n', setEqual, inPlace, numel(hfPieces));
allPass = report_('(3a) WordPiece vocabulary set vs HF WordPieceTrainer', numel(hfPieces), double(~setEqual), "") && allPass;
badW = 0;  firstW = "";
for i = 1:numel(wref.encodings)
    refTok = toStrings_(wref.encodings(i).tokens);
    got = wp.forward(docs(i));
    if ~isequal(got, refTok)
        badW = badW + 1;
        if firstW == "", firstW = sprintf('doc %d: got [%s] expected [%s]', i, strjoin(got(1:min(end,12)), '|'), strjoin(refTok(1:min(end,12)), '|')); end
    end
end
allPass = report_('(3b) WordPiece encodings (own vocab) vs HF encodings', numel(wref.encodings), badW, firstW) && allPass;

% ---------------- (4) additional training configurations ----------------
% Other corpora / vocab sizes / scripts, including a tie-heavy synthetic corpus, to test whether the
% Cranfield-1000 identity generalizes. PASS condition (tie-break independent): every merge HF chose is a
% MAXIMUM-count pair in our counting engine (replay). Exact merge-list identity, and WordPiece piece-set /
% encoding agreement, are REPORTED: they depend on how ties between equally frequent pairs are resolved,
% which in HF follows internal symbol ids (partly implementation-defined) and is not fully reproduced.
extra = jsondecode(fileread(fullfile(here, 'extra_train_reference.json')));
fprintf('\n--- additional training configurations ---\n');
fprintf('%-20s %5s | %-28s | %-30s | %-30s\n', 'config', 'vocab', 'HF merges = max-count (PASS)', 'BPE merge list identical / docs', 'WordPiece pieces differing / docs');
for name = string(fieldnames(extra))'
    cfg = extra.(name);
    corpus = toStrings_(cfg.corpus);
    hfM = split(string(cfg.merges(:)), " ");
    % replay HF's merges through the counting engine
    w = strings(1, 0);
    for d = 1:numel(corpus)
        ww = pre.forward(corpus(d));
        for k = 1:numel(ww), ww(k) = OceanNLP.Subword.ByteLevel.encode(ww(k)); end
        w = [w, ww]; %#ok<AGROW>
    end
    [uw2, ~, ic2] = unique(w);
    [~, ~, st] = OceanNLP.Subword.Algorithm.BpeAlgorithm.trainMerges(uw2, accumarray(ic2(:), 1)', double(cfg.vocab), ...
        OceanNLP.Subword.ByteLevel.alphabet(), 0, "", hfM);
    notMax = nnz(st.PairCount ~= st.MaxCount);
    bpeT = OceanNLP.Subword.SubwordTokenizer();
    bpeT.fit(corpus, 'VocabSize', double(cfg.vocab));
    bpeSame = isequal(bpeT.Algorithm.Merges, hfM);
    nProbe = numel(cfg.bpe_tokens);
    bpeEnc = 0;  wpEnc = 0;
    for i = 1:nProbe
        if isequal(bpeT.forward(corpus(i)), toStrings_(cfg.bpe_tokens{i})), bpeEnc = bpeEnc + 1; end
    end
    wpT = OceanNLP.Subword.SubwordTokenizer('PreTokenizer', OceanNLP.Text.PreTokenize.BertBasicPreTokenizer('NeverSplit', strings(1, 0)), ...
        'Algorithm', OceanNLP.Subword.Algorithm.WordPieceAlgorithm(), 'ByteLevel', false);
    wpT.fit(corpus, 'VocabSize', numel(cfg.pieces));
    hfPieces2 = toStrings_(cfg.pieces);
    nDiff = numel(setdiff(wpT.Vocab.Tokens, hfPieces2));
    for i = 1:nProbe
        if isequal(wpT.forward(corpus(i)), toStrings_(cfg.wp_tokens{i})), wpEnc = wpEnc + 1; end
    end
    fprintf('%-20s %5d | %-28s | %-30s | %-30s\n', name, cfg.vocab, sprintf('%d / %d violations', size(hfM, 1), notMax), ...
        sprintf('%d  (%d/%d docs)', bpeSame, bpeEnc, nProbe), sprintf('%d of %d  (%d/%d docs)', nDiff, numel(hfPieces2), wpEnc, nProbe));
    allPass = report_(sprintf('(4) HF merge choices NOT max-count (replay): %s', name), size(hfM, 1), notMax, "") && allPass;
end

if allPass
    fprintf('\nOverall: ALL CHECKS PASSED\n');
else
    fprintf('\nOverall: SOME CHECKS FAILED\n');
end
end

function ok = report_(name, n, nBad, firstMsg)
ok = nBad == 0;
if ok, s = 'PASS'; else, s = 'FAIL'; end
fprintf('[%s] %-62s %d cases, %d mismatches\n', s, name, n, nBad);
if ~ok && firstMsg ~= "", fprintf('      first: %s\n', firstMsg); end
end

function s = toStrings_(x)
if isempty(x), s = strings(1, 0); elseif iscell(x), s = string(x(:))'; else, s = string(x); end
end
