%% exp_speech_encdec_fsdd  Conformer encoder + Transformer decoder (attention encoder-decoder) on FSDD.
%
%   Consumer script only: every algorithm lives in the library
%   (OceanNLP.Speech.{Data,Features,Models,Decoding,Training,Evaluation}, OceanDL.Transformer).
%
%   SAME DATA / FEATURES / UNITS AS THE CTC BASELINE (exp_speech_ctc_fsdd.m, Results/Speech/ctc_fsdd_report.md):
%   FSDD canonical split (test = held-out speaker "theo"; validation = takes 45-49 of the three
%   training speakers, speaker-dependent, model selection only), single-digit + UtteranceComposer
%   multi-word utterances with the same counts and seeds, character units (vocabulary fitted on
%   train transcripts), 40 log-mel (25 ms / 10 ms, 8 kHz), per-utterance CMVN in the model.
%   MODEL: ConformerEncoder (Conv1d subsampling x4, Conformer blocks, RoPE) -> Transformer decoder
%   (pre-LN DecoderBlocks, causal self-attn + cross-attn) -> softmax over characters; joint CTC
%   auxiliary loss on the encoder; label smoothing.
%   TRAINING: SpeechTrainer (AdamW, warmup-cosine, clip 5, per-epoch val loss/WER, best checkpoint).
%   EVALUATION on theo: greedy and beam (length-normalised) attention decoding; the CTC baseline's
%   saved checkpoint re-evaluated on the identical test sets (greedy + beam, no LM) for a
%   like-for-like comparison: S/D/I, WER by reference length, hypothesis-length behaviour,
%   looping / early-Eos counts, cross-attention monotonicity; save -> reload -> transcribe identity.
%   Also (run 2): joint CTC N-best rescoring of the attention beam (labelled separately), and the
%   key/value-cache decoding speed-up (cached vs uncached step on 100 composed test utterances,
%   identical transcripts / scores checked). Run 1 (8 epochs) is kept in encdec_fsdd_report_run1.md.
%   Writes Results/Speech/encdec_fsdd_report.md and Results/Speech/checkpoints/encdec_fsdd_best.mat.
%
%   OCEAN_SPEECH_QUICK=1: tiny debugging run (NOT a result; writes *_quick files).

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(fileparts(here)));
addpath(repo);
quick = getenv('OCEAN_SPEECH_QUICK') == "1";
tStart = tic;
% The two new types are registered at runtime until RecognizerFactory's builtin table lists them.
OceanNLP.Speech.Models.RecognizerFactory.register("encoder", "ConformerEncoder", @OceanNLP.Speech.Models.Encoders.ConformerEncoder.fromConfig);
OceanNLP.Speech.Models.RecognizerFactory.register("recognizer", "EncoderDecoderRecognizer", @OceanNLP.Speech.Models.EncoderDecoderRecognizer.fromConfig);

%% configuration
cfg = struct();
cfg.Seed = 0;
cfg.ComposeMinWords = 2;  cfg.ComposeMaxWords = 5;
cfg.ComposeCount = struct('train', 1350, 'validation', 150, 'test', 500);
cfg.ComposeSeed = struct('train', 101, 'validation', 102, 'test', 103);
cfg.Unit = "char";
cfg.Features = struct('Type', "LogMel", 'NumMels', 40, 'WindowSeconds', 0.025, 'HopSeconds', 0.010);
cfg.Encoder = struct('Dim', 96, 'NumHeads', 4, 'NumBlocks', 3, 'FFMult', 4, 'ConvKernel', 15, ...
    'SubsampleStrides', [2 2], 'PosEnc', "rope", 'Dropout', 0.1);
cfg.Decoder = struct('Layers', 2, 'Heads', 4, 'FF', 384, 'Dropout', 0.1);
cfg.LabelSmoothing = 0.1;
cfg.CTCWeight = 0.3;
cfg.Train = struct('Epochs', 20, 'BatchSize', 32, 'LR', 1.5e-3, 'WeightDecay', 0.01, 'Clip', 5, 'WarmupFraction', 0.1);
cfg.Decode = struct('BeamWidth', 8, 'LengthPenalty', 1, 'MaxLengthRatio', 1.5, 'MaxLength', 200, 'CTCRescoreWeight', 0.3);
cfg.CacheTimingUtterances = 100;
cfg.PreviousReport = "Results/Speech/encdec_fsdd_report_run1.md";
cfg.CTCBaseline = "Results/Speech/checkpoints/ctc_fsdd_best.mat";
cfg.NumExamples = 10;
if quick
    cfg.ComposeCount = struct('train', 40, 'validation', 10, 'test', 10);
    cfg.Train.Epochs = 1;
    cfg.Encoder.NumBlocks = 1; cfg.Encoder.Dim = 32; cfg.Decoder.FF = 64; cfg.Decoder.Layers = 1;
end
resDir = fullfile(repo, 'Results', 'Speech');
ckDir = fullfile(resDir, 'checkpoints');
if ~isfolder(ckDir), mkdir(ckDir); end
suffix = ""; if quick, suffix = "_quick"; end
bestFile = fullfile(ckDir, "encdec_fsdd_best" + suffix + ".mat");
logFile = fullfile(resDir, "encdec_fsdd_train" + suffix + ".log");
if isfile(logFile), delete(logFile); end
reportFile = fullfile(resDir, "encdec_fsdd_report" + suffix + ".md");

%% data (identical construction to the CTC experiment)
man = OceanNLP.Speech.Data.SpeechManifest.fromFSDD();
canon = man.canonicalFSDDSplit();
singles = struct('train', canon.bySplit("train"), 'validation', canon.bySplit("validation"), 'test', canon.bySplit("test"));
if quick
    for f = ["train", "validation", "test"]
        s = singles.(f); singles.(f) = s.subset(round(linspace(1, s.height(), min(s.height(), 40))));
    end
end
singles.train.assertDisjoint(singles.test, 'By', "speaker");
singles.train.assertDisjoint(singles.validation, 'By', "id");
composed = struct();
for f = ["train", "validation", "test"]
    composed.(f) = OceanNLP.Speech.Data.UtteranceComposer.compose(singles.(f), 'MinWords', cfg.ComposeMinWords, ...
        'MaxWords', cfg.ComposeMaxWords, 'Count', cfg.ComposeCount.(f), 'Seed', cfg.ComposeSeed.(f));
end
both = @(a, b, name) OceanNLP.Speech.Data.SpeechManifest([a.Entries; b.Entries], 'Root', a.Root, 'Name', name);
trainMan = both(singles.train, composed.train, "FSDD-train(single+composed)");
valMan = both(singles.validation, composed.validation, "FSDD-val(single+composed)");
vocab = OceanNLP.Speech.Data.SpeechVocabulary.fit(trainMan.Entries.Transcript, 'Unit', cfg.Unit);
fs = 8000;
features = OceanNLP.Speech.Features.LogMel(fs, 'NumMels', cfg.Features.NumMels, ...
    'WindowLength', round(cfg.Features.WindowSeconds * fs), 'HopLength', round(cfg.Features.HopSeconds * fs));
mkds = @(m) OceanNLP.Speech.Data.AudioDataset(m, features, 'Vocabulary', vocab, 'SampleRate', fs);
dsTrain = mkds(trainMan);
dsVal = mkds(valMan);
dsTest = struct('single', mkds(singles.test), 'composed', mkds(composed.test));
fprintf('data ready (%.0f s): train %d, val %d, test %d+%d utterances; %d classes\n', toc(tStart), ...
    dsTrain.length(), dsVal.length(), dsTest.single.length(), dsTest.composed.length(), vocab.numClasses());

%% model
E = cfg.Encoder;
enc = OceanNLP.Speech.Models.Encoders.ConformerEncoder(features.outputDim([]), 'Dim', E.Dim, 'NumHeads', E.NumHeads, ...
    'NumBlocks', E.NumBlocks, 'FFMult', E.FFMult, 'ConvKernel', E.ConvKernel, 'SubsampleStrides', E.SubsampleStrides, ...
    'PosEnc', E.PosEnc, 'Dropout', E.Dropout, 'Seed', cfg.Seed + 1);
model = OceanNLP.Speech.Models.EncoderDecoderRecognizer(features, vocab, {enc}, 'DecoderLayers', cfg.Decoder.Layers, ...
    'DecoderHeads', cfg.Decoder.Heads, 'DecoderFF', cfg.Decoder.FF, 'Dropout', cfg.Decoder.Dropout, ...
    'LabelSmoothing', cfg.LabelSmoothing, 'CTCWeight', cfg.CTCWeight, 'Seed', cfg.Seed + 20, ...
    'ModelId', "ocean-encdec-conformer-fsdd-char-v1" + suffix);
fprintf('model: %d parameters\n', model.numParameters());
% training utterances too short for CTC after subsampling (their CTC term is zeroed, CTCZeroInfinity)
trainEncLen = model.outputLength(dsTrain.lengths());
trainTg = vocab.encode(trainMan.Entries.Transcript);
needCTC = cellfun(@(y) numel(y) + nnz(diff(y) == 0), trainTg(:)');
ctcInfeasible = nnz(reshape(trainEncLen, 1, []) < needCTC);
fprintf('%d of %d training utterances are CTC-infeasible after subsampling\n', ctcInfeasible, numel(needCTC));

%% training
bTrain = OceanNLP.Speech.Data.SpeechBatcher(dsTrain, 'BatchSize', cfg.Train.BatchSize, 'Shuffle', true, ...
    'Seed', cfg.Seed, 'SortByLength', true);
bVal = OceanNLP.Speech.Data.SpeechBatcher(dsVal, 'BatchSize', 32, 'Shuffle', false, 'SortByLength', true);
total = cfg.Train.Epochs * bTrain.numBatches();
trainer = OceanNLP.Speech.Training.SpeechTrainer(model, bTrain, 'ValBatcher', bVal, 'Epochs', cfg.Train.Epochs, ...
    'LR', cfg.Train.LR, 'WeightDecay', cfg.Train.WeightDecay, 'Clip', cfg.Train.Clip, ...
    'WarmupSteps', max(1, round(cfg.Train.WarmupFraction * total)), 'Seed', cfg.Seed, ...
    'BestCheckpoint', bestFile, 'LogFile', logFile);
tTrain = tic;
trainer.fit();
trainSeconds = toc(tTrain);
trainer.restoreBest();
H = trainer.History;
fprintf('training done in %.1f min; best epoch %d (val WER %.4f)\n', trainSeconds / 60, trainer.BestEpoch, trainer.BestScore(1));

%% evaluation: encoder-decoder (greedy, beam) and the CTC baseline on the same test sets
D = cfg.Decode;
decArgs = struct('greedy', {{'Decoder', "greedy", 'MaxLengthRatio', D.MaxLengthRatio, 'MaxLength', D.MaxLength}}, ...
    'beam', {{'Decoder', "beam", 'BeamWidth', D.BeamWidth, 'LengthPenalty', D.LengthPenalty, ...
              'MaxLengthRatio', D.MaxLengthRatio, 'MaxLength', D.MaxLength}}, ...
    'beamctc', {{'Decoder', "beam", 'BeamWidth', D.BeamWidth, 'LengthPenalty', D.LengthPenalty, ...
              'MaxLengthRatio', D.MaxLengthRatio, 'MaxLength', D.MaxLength, 'CTCRescoreWeight', D.CTCRescoreWeight}});
ctcFile = fullfile(repo, cfg.CTCBaseline);
haveCTC = isfile(ctcFile) && ~quick;
systems = {"encdec", "greedy", model, decArgs.greedy; "encdec", "beam", model, decArgs.beam; ...
           "encdec", "beamctc", model, decArgs.beamctc};
if haveCTC
    ctc = OceanNLP.Speech.Training.SpeechCheckpoint.load(ctcFile).Model;
    assert(isequal(ctc.Vocabulary.Tokens, vocab.Tokens), 'CTC baseline vocabulary differs');
    assert(isequaln(ctc.Features.config(), features.config()), 'CTC baseline features differ');
    systems(end + 1, :) = {"ctc", "greedy", ctc, {'Decoder', "greedy"}};
    systems(end + 1, :) = {"ctc", "beam", ctc, {'Decoder', "beam", 'BeamWidth', D.BeamWidth}};
end
res = struct();
rows = {};
ev = OceanNLP.Speech.Evaluation.CorpusEvaluator('Normalizer', vocab.Normalizer);
for k = 1:size(systems, 1)
    key = systems{k, 1} + "_" + systems{k, 2};
    for part = ["single", "composed"]
        t0 = tic;
        r = evalSet(systems{k, 3}, dsTest.(part), systems{k, 4});
        r.Seconds = toc(t0);
        res.(key).(part) = r;
        W = r.Result.WER;
        rows(end + 1, :) = {systems{k, 1}, systems{k, 2}, part, W.Rate, r.Result.CER.Rate, W.S, W.D, W.I, W.N, r.Seconds}; %#ok<SAGROW>
        fprintf('%s %s %s: WER %.4f CER %.4f (%.0f s)\n', systems{k, 1}, systems{k, 2}, part, W.Rate, r.Result.CER.Rate, r.Seconds);
    end
    a = ev.evaluate([res.(key).single.Refs, res.(key).composed.Refs], [res.(key).single.Hyps, res.(key).composed.Hyps]);
    rows(end + 1, :) = {systems{k, 1}, systems{k, 2}, "all", a.WER.Rate, a.CER.Rate, a.WER.S, a.WER.D, a.WER.I, a.WER.N, NaN}; %#ok<SAGROW>
end
testTable = cell2table(rows, 'VariableNames', ["Model", "Decoder", "Set", "WER", "CER", "S", "D", "I", "RefWords", "Seconds"]);
disp(testTable);

%% length behaviour: WER and S/D/I by reference word count; hypothesis - reference word count
keys = string(fieldnames(res))';
lenRows = {}; diffRows = {};
for key = keys
    refs = [res.(key).single.Refs, res.(key).composed.Refs];
    hyps = [res.(key).single.Hyps, res.(key).composed.Hyps];
    nr = arrayfun(@(s) numel(split(strtrim(s))), vocab.Normalizer.forward(refs));
    nh = arrayfun(@(s) nnz(strlength(split(strtrim(s))) > 0), vocab.Normalizer.forward(hyps));
    for L = unique(nr)
        sel = nr == L;
        w = OceanNLP.Speech.Evaluation.WordErrorRate.compute(refs(sel), hyps(sel), 'Normalizer', vocab.Normalizer);
        lenRows(end + 1, :) = {key, L, nnz(sel), w.Rate, w.S, w.D, w.I}; %#ok<SAGROW>
    end
    d = nh - nr;
    diffRows(end + 1, :) = {key, nnz(d < 0), nnz(d == 0), nnz(d > 0), mean(d), min(d), max(d)}; %#ok<SAGROW>
end
lenTable = cell2table(lenRows, 'VariableNames', ["System", "RefWords", "Utterances", "WER", "S", "D", "I"]);
diffTable = cell2table(diffRows, 'VariableNames', ["System", "HypShorter", "SameLength", "HypLonger", "MeanWordDiff", "MinDiff", "MaxDiff"]);

%% attention-specific behaviour: max-length hits (looping), early Eos, cross-attention monotonicity
behav = {};
for dec = ["greedy", "beam", "beamctc"]
    trs = [res.("encdec_" + dec).single.Transcripts, res.("encdec_" + dec).composed.Transcripts];
    hit = arrayfun(@(t) t.Candidates(1).HitMaxLength, trs);
    behav(end + 1, :) = {dec, nnz(hit), numel(trs)}; %#ok<SAGROW>
end
behavTable = cell2table(behav, 'VariableNames', ["Decoder", "HitMaxLength", "Utterances"]);
[mono, monoTable] = attentionStats(model, dsTest.composed, decArgs.greedy);

%% examples (composed, worst by encdec beam word errors + seeded random)
rb = res.encdec_beam.composed; rg = res.encdec_greedy.composed;
pu = rb.Result.PerUtterance;
[~, worst] = sort(pu.WordErrors, 'descend');
pickW = worst(1:min(5, height(pu)));
rest = setdiff(1:height(pu), pickW);
pickR = rest(randperm(RandStream('mt19937ar', 'Seed', 7), numel(rest), min(cfg.NumExamples - numel(pickW), numel(rest))));
pick = [pickW(:); pickR(:)];
exCols = {pu.Id(pick), pu.Reference(pick), rg.Hyps(pick)', rb.Hyps(pick)'};
exNames = ["Id", "Reference", "EncDecGreedy", "EncDecBeam"];
if haveCTC
    exCols{end + 1} = res.ctc_beam.composed.Hyps(pick)'; exNames(end + 1) = "CTCBeam";
end
exCols{end + 1} = arrayfun(@(t) t.Candidates(1).HitMaxLength, rb.Transcripts(pick))';
exNames(end + 1) = "BeamHitMaxLen";
ex = table(exCols{:}, 'VariableNames', exNames);
ps = res.encdec_beam.single.Result.PerUtterance;
wrong = find(ps.WordErrors > 0);
exSingle = table(ps.Id(wrong(1:min(8, end))), ps.Reference(wrong(1:min(8, end))), ps.Hypothesis(wrong(1:min(8, end))), ...
    'VariableNames', ["Id", "Reference", "EncDecBeam"]);
confRows = confusions(ps);

%% save -> reload -> transcribe identity
ck = OceanNLP.Speech.Training.SpeechCheckpoint.load(bestFile);
probe = arrayfun(@(i) dsTest.composed.item(i).Waveform, 1:min(20, dsTest.composed.length()), 'UniformOutput', false);
reloadSame = true;
for dec = ["greedy", "beam"]
    a = model.transcribe(probe, decArgs.(dec){:});
    b = ck.Model.transcribe(probe, decArgs.(dec){:});
    reloadSame = reloadSame && isequal(a.texts(), b.texts()) && isequal([a.Score], [b.Score]);
end
fprintf('save -> reload -> transcribe identical: %d\n', reloadSame);

%% key/value cache: cached vs uncached incremental decoding on real composed test utterances
nCache = min(cfg.CacheTimingUtterances, dsTest.composed.length());
bc = OceanNLP.Speech.Data.SpeechBatcher(dsTest.composed, 'BatchSize', 25, 'Shuffle', false, 'SortByLength', false);
cacheBatches = arrayfun(@(k) bc.batch(k), 1:ceil(nCache / 25), 'UniformOutput', false);
cacheRows = {};
for dec = ["greedy", "beam"]
    out = struct();
    for useCache = [false, true]
        t0 = tic;
        trs = OceanNLP.Speech.Core.Transcript.empty(1, 0);
        for k = 1:numel(cacheBatches)
            trs = [trs, model.transcribe(cacheBatches{k}, decArgs.(dec){:}, 'UseCache', useCache)]; %#ok<AGROW>
        end
        out.(char("c" + string(double(useCache)))) = struct('Seconds', toc(t0), 'Tr', trs(1:nCache));
    end
    a = out.c0.Tr; b = out.c1.Tr;
    candSame = all(arrayfun(@(i) isequal({a(i).Candidates.TokenIds}, {b(i).Candidates.TokenIds}), 1:nCache));
    dScore = Inf;
    if candSame
        dScore = max(arrayfun(@(i) max(abs([a(i).Candidates.Score] - [b(i).Candidates.Score])), 1:nCache));
    end
    cacheRows(end + 1, :) = {dec, nCache, out.c0.Seconds, out.c1.Seconds, out.c0.Seconds / out.c1.Seconds, ...
        isequal(a.texts(), b.texts()), candSame, dScore}; %#ok<SAGROW>
    fprintf('cache %s: uncached %.1f s, cached %.1f s (x%.2f), same texts %d, max |dScore| %.2g\n', dec, ...
        out.c0.Seconds, out.c1.Seconds, out.c0.Seconds / out.c1.Seconds, isequal(a.texts(), b.texts()), dScore);
end
cacheTable = cell2table(cacheRows, 'VariableNames', ["Decoder", "Utterances", "UncachedSeconds", "CachedSeconds", ...
    "SpeedUp", "SameTexts", "SameNBestTokenIds", "MaxAbsScoreDiff"]);

%% previous run (run 1, 8 epochs): read from its saved report
prev = previousRun(fullfile(repo, cfg.PreviousReport));

%% report
R = struct('cacheTable', cacheTable, 'prev', prev, 'cfg', cfg, 'quick', quick, 'model', model, 'trainer', trainer, 'H', H, 'trainSeconds', trainSeconds, ...
    'testTable', testTable, 'lenTable', lenTable, 'diffTable', diffTable, 'behavTable', behavTable, 'mono', mono, ...
    'monoTable', monoTable, 'ex', ex, 'exSingle', exSingle, 'confRows', confRows, 'reloadSame', reloadSame, ...
    'bestFile', bestFile, 'repo', repo, 'haveCTC', haveCTC, 'features', features, 'vocab', vocab, ...
    'totalMinutes', toc(tStart) / 60, 'ctcInfeasible', ctcInfeasible, 'numTrain', numel(needCTC));
writeReport(reportFile, R);
fprintf('report: %s  (total %.1f min)\n', reportFile, toc(tStart) / 60);

%% ---------------------------------------------------------------- helpers (orchestration only)
function r = evalSet(model, ds, decArgs)
b = OceanNLP.Speech.Data.SpeechBatcher(ds, 'BatchSize', 32, 'Shuffle', false, 'SortByLength', false);
refs = strings(1, 0); hyps = refs; ids = refs; spk = refs; trs = OceanNLP.Speech.Core.Transcript.empty(1, 0);
for k = 1:b.numBatches()
    B = b.batch(k);
    tr = model.transcribe(B, decArgs{:});
    refs = [refs, B.Text]; hyps = [hyps, tr.texts()]; ids = [ids, B.Ids]; spk = [spk, B.Speakers]; %#ok<AGROW>
    trs = [trs, tr]; %#ok<AGROW>
end
ev = OceanNLP.Speech.Evaluation.CorpusEvaluator('Normalizer', model.Vocabulary.Normalizer);
r = struct('Refs', refs, 'Hyps', hyps, 'Ids', ids, 'Speakers', spk, 'Transcripts', trs, ...
    'Result', ev.evaluate(refs, hyps, 'Ids', ids, 'Speakers', spk));
end

function [m, t] = attentionStats(model, ds, decArgs)
% Last-layer cross-attention (head-averaged) of each greedy hypothesis's character tokens:
% argmax frame p_u and centroid c_u = sum_t t a(u,t); fraction of non-decreasing steps and the
% Pearson correlation of p_u with the token index u (Eos row excluded).
b = OceanNLP.Speech.Data.SpeechBatcher(ds, 'BatchSize', 32, 'Shuffle', false, 'SortByLength', false);
monoA = []; monoC = []; corrA = []; spanFrac = [];
for k = 1:b.numBatches()
    B = b.batch(k);
    [~, info] = model.transcribe(B, decArgs{:}, 'ReturnAttention', true);
    for n = 1:numel(info)
        A = mean(info(n).CrossAttention, 3);            % (L+1) x T'
        A = A(1:end - 1, :);
        if size(A, 1) < 3, continue; end
        [~, p] = max(A, [], 2);
        c = A * (1:size(A, 2))';
        monoA(end + 1) = mean(diff(p) >= 0); %#ok<AGROW>
        monoC(end + 1) = mean(diff(c) >= 0); %#ok<AGROW>
        cc = corrcoef((1:numel(p))', p);
        corrA(end + 1) = cc(1, 2); %#ok<AGROW>
        spanFrac(end + 1) = (max(p) - min(p) + 1) / info(n).EncoderLength; %#ok<AGROW>
    end
end
m = struct('MonoArgmax', mean(monoA), 'MonoCentroid', mean(monoC), 'Corr', mean(corrA, 'omitnan'), ...
    'Span', mean(spanFrac), 'N', numel(monoA));
t = table(["fraction of non-decreasing argmax steps"; "fraction of non-decreasing centroid steps"; ...
    "Pearson r(token index, argmax frame)"; "argmax span / encoder frames"], ...
    [m.MonoArgmax; m.MonoCentroid; m.Corr; m.Span], 'VariableNames', ["Statistic", "MeanOverUtterances"]);
end

function t = confusions(ps)
% Single-digit (one word) reference -> hypothesis confusion counts (errors only).
err = ps.WordErrors > 0;
pairs = ps.Reference(err) + " -> " + ps.Hypothesis(err);
[u, ~, j] = unique(pairs);
c = accumarray(j, 1);
[c, o] = sort(c, 'descend');
t = table(u(o), c, 'VariableNames', ["RefToHyp", "Count"]);
t = t(1:min(10, height(t)), :);
end

function p = previousRun(file)
% Encoder-decoder held-out rows and the best-epoch line of the previous run's report (if present).
p = struct('Table', table(), 'BestLine', "");
if ~isfile(file), return; end
L = string(splitlines(fileread(file)));
best = L(startsWith(L, "Best epoch:"));
if ~isempty(best), p.BestLine = best(1); end
rows = L(startsWith(L, "| encdec |"));
if isempty(rows), return; end
cells = arrayfun(@(r) strtrim(split(extractBetween(r, 2, strlength(r) - 1), "|"))', rows, 'UniformOutput', false);
C = vertcat(cells{:});
p.Table = table(C(:, 2), C(:, 3), str2double(C(:, 4)), str2double(C(:, 5)), ...
    'VariableNames', ["Decoder", "Set", "WER", "CER"]);
end

function s = mdTable(t, fmt)
if nargin < 2, fmt = '%.4g'; end
names = string(t.Properties.VariableNames);
s = "| " + join(names, " | ") + " |" + newline + "|" + join(repmat(" --- ", 1, numel(names)), "|") + "|" + newline;
for i = 1:height(t)
    cells = strings(1, numel(names));
    for j = 1:numel(names)
        v = t.(names(j))(i);
        if iscell(v), v = v{1}; end
        if isnumeric(v) || islogical(v)
            cells(j) = string(sprintf(fmt, v));
        else
            cells(j) = replace(string(v), "|", "\|");
        end
    end
    s = s + "| " + join(cells, " | ") + " |" + newline;
end
end

function writeReport(file, R)
H = R.H; Ep = struct2table(R.H.Epochs, 'AsArray', true);
L = H.StepLoss; n = numel(L); k = max(1, floor(n / 16)); idx = unique([1:k:n, n]);
curve = table(idx', arrayfun(@(i) mean(L(max(1, i - k + 1):i)), idx)', H.StepLR(idx)', H.StepEpoch(idx)' + 1, ...
    'VariableNames', ["Step", "MeanLossLastWindow", "LR", "Epoch"]);
mcfg = R.model.config();
o = "# Attention encoder-decoder (Conformer + Transformer decoder) on FSDD (Phase 4)" + newline + newline;
if R.quick
    o = o + "> **QUICK DEBUG RUN (OCEAN_SPEECH_QUICK=1) -- tiny subsets, NOT a result.**" + newline + newline;
end
o = o + "Generated by `Experiments/Scripts/Speech/exp_speech_encdec_fsdd.m` on " + string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm')) + ...
    " (MATLAB " + version('-release') + ", CPU, double precision, CPU shared with other jobs). Total wall time " + sprintf('%.1f', R.totalMinutes) + ...
    " min, of which training " + sprintf('%.1f', R.trainSeconds / 60) + " min. All numbers are computed by the script." + newline + newline;
o = o + "Data, split, composed utterances (same counts and seeds), character vocabulary (" + R.vocab.numClasses() + ...
    " classes) and 40-dim log-mel features are constructed exactly as in `exp_speech_ctc_fsdd.m`; the script asserts that the saved CTC baseline has the same vocabulary and feature config before comparing." + newline + newline;
o = o + "## Configuration" + newline + newline + "```json" + newline + string(jsonencode(R.cfg, 'PrettyPrint', true)) + newline + "```" + newline + newline;
o = o + "Model config (`model.config()`, Vocabulary/Features omitted):" + newline + newline + "```json" + newline + ...
    string(jsonencode(rmfield(mcfg, {'Vocabulary', 'Features'}), 'PrettyPrint', true)) + newline + "```" + newline + newline;
if ~isempty(R.prev.Table)
    o = o + "> **Previous run (run 1, 8 epochs, same data / split / features / vocabulary / model config; full report kept in `" + R.cfg.PreviousReport + ...
        "`):** " + R.prev.BestLine + " Held-out encoder-decoder rows of that run (read from its report):" + newline + newline;
    o = o + mdTable(R.prev.Table) + newline + ...
        "This run trains for " + R.cfg.Train.Epochs + " epochs (the warmup-cosine schedule is stretched to the longer run; everything else is identical) because run 1 stopped with validation WER still falling." + newline + newline;
end
o = o + "Model size: **" + R.model.numParameters() + " parameters** (CTC baseline: 279061)." + newline + newline;
o = o + "## Training" + newline + newline + "Objective: 0.7 x label-smoothed (0.1) attention cross entropy per target token + 0.3 x CTC (per-utterance loss / target length; " + R.ctcInfeasible + " of " + R.numTrain + " training utterances are too short for CTC after 4x subsampling and contribute 0 CTC loss, CTCZeroInfinity). Validation WER = greedy attention decoding on the speaker-dependent validation set." + newline + newline;
o = o + mdTable(Ep(:, ["Epoch", "Steps", "TrainLoss", "ValLoss", "ValWER", "ValCER", "LR", "TrainSeconds", "ValSeconds"])) + newline;
o = o + sprintf('Best epoch: %d (val WER %.4f, val loss %.4f), saved in `%s`.', R.trainer.BestEpoch, R.trainer.BestScore(1), ...
    R.trainer.BestScore(2), replace(erase(string(R.bestFile), string(R.repo) + filesep), "\", "/")) + newline + newline;
o = o + "Training loss curve (joint objective, mean over the preceding window):" + newline + newline + mdTable(curve) + newline;
o = o + "## Held-out speaker (theo): encoder-decoder vs CTC baseline" + newline + newline;
if R.haveCTC
    o = o + "CTC rows: the saved best CTC checkpoint (`ctc_fsdd_best.mat`) re-decoded here on the identical test utterances (no LM)." + newline + newline;
end
o = o + mdTable(R.testTable) + newline;
o = o + "Beam: width " + R.cfg.Decode.BeamWidth + ", score = log P / (L+1)^" + R.cfg.Decode.LengthPenalty + ...
    " (Eos included), max output length min(" + R.cfg.Decode.MaxLength + ", ceil(" + R.cfg.Decode.MaxLengthRatio + " x encoder frames)). Seconds = batched decoding of the whole set (CPU shared, indicative only)." + newline + newline;
o = o + "`beamctc` (**joint CTC N-best rescoring**, not used for checkpoint selection): the same attention beam's " + R.cfg.Decode.BeamWidth + ...
    "-best list re-ranked by ((1 - " + R.cfg.Decode.CTCRescoreWeight + ") log P_att + " + R.cfg.Decode.CTCRescoreWeight + ...
    " log P_ctc) / (L+1)^" + R.cfg.Decode.LengthPenalty + ", log P_ctc from the model's own joint-CTC head (all alignments). It can only reorder hypotheses the attention beam found (no CTC prefix scoring inside the search)." + newline + newline;
o = o + "### Key/value-cache decoding (first " + R.cacheTable.Utterances(1) + " composed held-out utterances, batches of 25)" + newline + newline + mdTable(R.cacheTable) + newline;
o = o + "Uncached = the reference step re-running the decoder on the whole prefix every step; cached = incremental step (per-layer self-attention keys/values of the prefix and cross-attention keys/values of the memory reused; beam rows reordered by parent index). Equivalence of every step's log-probabilities to 1e-12 (greedy and beam, incl. reordering / pruning) is verified in `EncoderDecoderRecognizerTest`; the evaluation above uses the cached path. Seconds on a shared CPU, indicative only." + newline + newline;
o = o + "### WER and error types by reference length (words; 1 = single-digit set)" + newline + newline + mdTable(R.lenTable) + newline;
o = o + "### Hypothesis length vs reference length (words, single + composed)" + newline + newline + mdTable(R.diffTable) + newline;
o = o + "### Looping / max-length behaviour (encoder-decoder)" + newline + newline + mdTable(R.behavTable) + newline;
o = o + "HitMaxLength = the decoder never chose Eos and was stopped by the length guard (looping)." + newline + newline;
o = o + "### Cross-attention behaviour (last decoder layer, head-averaged, greedy hypotheses, composed test set, " + R.mono.N + " utterances with >= 3 tokens)" + newline + newline + mdTable(R.monoTable) + newline;
o = o + "## Examples (held-out speaker, composed)" + newline + newline + "First 5 rows: most beam word errors; rest: seeded random draw." + newline + newline + mdTable(R.ex) + newline;
if height(R.exSingle) > 0
    o = o + "Single-digit errors (first up to 8):" + newline + newline + mdTable(R.exSingle) + newline;
    o = o + "Most frequent single-digit confusions (encoder-decoder beam):" + newline + newline + mdTable(R.confRows) + newline;
end
o = o + "## Save -> reload -> transcribe" + newline + newline;
if R.reloadSame
    o = o + "Reloading the best checkpoint with `SpeechCheckpoint.load` (rebuilt by `RecognizerFactory.fromConfig` + weights) and transcribing 20 held-out utterances gives **identical** greedy and beam texts and scores (`isequal`)." + newline + newline;
else
    o = o + "**MISMATCH**: the reloaded checkpoint did not reproduce the transcripts exactly." + newline + newline;
end
o = o + "## Caveats" + newline + newline + ...
    "* One held-out speaker (theo); composed utterances are concatenations of isolated digits (no coarticulation)." + newline + ...
    "* Validation shares speakers with training; used only for checkpoint selection." + newline + ...
    "* Training budget is small (CPU, shared): the encoder-decoder has more parameters than the CTC baseline and fewer epochs; the comparison is of these two trained systems, not of the model families in general." + newline + ...
    "* The joint CTC head is a training objective; the greedy / beam rows decode attention-only, and the `beamctc` rows use it only for N-best rescoring (no joint CTC/attention prefix scoring inside the beam). Its weight 0.3 was set a priori (equal to the training CTC weight), not tuned on the test set." + newline;
fid = fopen(file, 'w'); fprintf(fid, '%s', o); fclose(fid);
end
