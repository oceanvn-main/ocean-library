%% exp_speech_ctc_fsdd  Native trainable CTC recogniser on the real FSDD corpus (milestone S1).
%
%   Consumer script only: every algorithm lives in the library
%   (OceanNLP.Speech.{Data,Features,Models,Decoding,Training,Evaluation}, OceanDL).
%
%   DATA (contract sec. 1): FSDD, canonical split -- test = held-out speaker "theo",
%   validation = takes 45-49 of the three training speakers (speaker-DEPENDENT, used only
%   for model selection / LM-weight choice), train = the rest.
%   (a) single-digit utterances, (b) multi-word utterances from UtteranceComposer built
%   per split (train composes from train only, etc.). Training uses (a) + (b).
%   UNITS: characters ("|" = word boundary), vocabulary fitted on train transcripts only.
%   FEATURES: 40 log-mel (25 ms Hann window, 10 ms hop, 8 kHz), per-utterance CMVN in the model.
%   MODEL: ConvEncoder (Conv1d k5 s2 + Conv1d k3 s1, LayerNorm, GELU) -> 1-layer BiGRU -> linear CTC head.
%   TRAINING: SpeechTrainer (AdamW, warmup-cosine, clip 5, per-epoch val loss/WER/CER, best checkpoint).
%   EVALUATION on theo: greedy, prefix beam, prefix beam + char 3-gram LM (train transcripts only,
%   weight and length bonus chosen on the composed validation utterances). Per-speaker table, 10 real held-out examples (worst + random),
%   time per utterance, model size, save -> reload -> transcribe identity.
%   Writes Results/Speech/ctc_fsdd_report.md and Results/Speech/checkpoints/ctc_fsdd_best.mat.
%
%   Set environment variable OCEAN_SPEECH_QUICK=1 for a tiny debugging run (NOT a result;
%   writes ctc_fsdd_report_quick.md instead).

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(fileparts(here)));
addpath(repo);
quick = getenv('OCEAN_SPEECH_QUICK') == "1";
tStart = tic;

%% configuration (everything that defines the run)
cfg = struct();
cfg.Seed = 0;
cfg.ComposeMinWords = 2;  cfg.ComposeMaxWords = 5;
cfg.ComposeCount = struct('train', 1350, 'validation', 150, 'test', 500);
cfg.ComposeSeed = struct('train', 101, 'validation', 102, 'test', 103);
cfg.Unit = "char";
cfg.Features = struct('Type', "LogMel", 'NumMels', 40, 'WindowSeconds', 0.025, 'HopSeconds', 0.010);
cfg.Conv = struct('Channels', [128 128], 'KernelSizes', [5 3], 'Strides', [2 1], 'Dropout', 0.1);
cfg.RNN = struct('HiddenSize', 128, 'NumLayers', 1, 'Cell', "gru", 'Dropout', 0);   % inter-layer dropout (unused with 1 layer)
cfg.HeadDropout = 0.1;
cfg.Train = struct('Epochs', 10, 'BatchSize', 32, 'LR', 2e-3, 'WeightDecay', 0.01, 'Clip', 5, ...
    'WarmupFraction', 0.05);
cfg.BeamWidth = 8;
cfg.LM = struct('Order', 3, 'Alpha', 0.1, 'WeightGrid', [0.3 0.6 1.0], 'BonusGrid', [0 1], 'TuneOn', "validation composed");
cfg.NumExamples = 10;
cfg.NumTimed = 50;
if quick
    cfg.ComposeCount = struct('train', 40, 'validation', 10, 'test', 10);
    cfg.Train.Epochs = 2;
    cfg.LM.WeightGrid = 0.5; cfg.LM.BonusGrid = 0;
    cfg.NumTimed = 5;
end
resDir = fullfile(repo, 'Results', 'Speech');
ckDir = fullfile(resDir, 'checkpoints');
if ~isfolder(ckDir), mkdir(ckDir); end
suffix = ""; if quick, suffix = "_quick"; end
bestFile = fullfile(ckDir, "ctc_fsdd_best" + suffix + ".mat");
logFile = fullfile(resDir, "ctc_fsdd_train" + suffix + ".log");
if isfile(logFile), delete(logFile); end
reportFile = fullfile(resDir, "ctc_fsdd_report" + suffix + ".md");

%% data
man = OceanNLP.Speech.Data.SpeechManifest.fromFSDD();
split = man.canonicalFSDDSplit();
singles = struct('train', split.bySplit("train"), 'validation', split.bySplit("validation"), 'test', split.bySplit("test"));
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
dsValComposed = mkds(composed.validation);
dsTest = struct('single', mkds(singles.test), 'composed', mkds(composed.test));
audit = [both(singles.train, singles.validation, "s").audit(); singles.test.audit()];
dataSummary = table(["train"; "validation"; "test"], ...
    [singles.train.height(); singles.validation.height(); singles.test.height()], ...
    [composed.train.height(); composed.validation.height(); composed.test.height()], ...
    [sum(singles.train.Entries.Duration) + sum(composed.train.Entries.Duration); ...
     sum(singles.validation.Entries.Duration) + sum(composed.validation.Entries.Duration); ...
     sum(singles.test.Entries.Duration) + sum(composed.test.Entries.Duration)] / 60, ...
    [join(unique(singles.train.Entries.Speaker), ","); join(unique(singles.validation.Entries.Speaker), ","); ...
     join(unique(singles.test.Entries.Speaker), ",")], ...
    'VariableNames', ["Split", "SingleDigit", "Composed", "AudioMinutes", "Speakers"]);
fprintf('data ready (%.0f s): train %d, val %d, test %d+%d utterances; %d classes\n', toc(tStart), ...
    dsTrain.length(), dsVal.length(), dsTest.single.length(), dsTest.composed.length(), vocab.numClasses());

%% model
enc = {OceanNLP.Speech.Models.Encoders.ConvEncoder(features.outputDim([]), 'Channels', cfg.Conv.Channels, ...
          'KernelSizes', cfg.Conv.KernelSizes, 'Strides', cfg.Conv.Strides, 'Dropout', cfg.Conv.Dropout, 'Seed', cfg.Seed + 1), ...
       []};
enc{2} = OceanNLP.Speech.Models.Encoders.RecurrentEncoder(enc{1}.OutputDim, 'HiddenSize', cfg.RNN.HiddenSize, ...
          'NumLayers', cfg.RNN.NumLayers, 'Cell', cfg.RNN.Cell, 'Dropout', cfg.RNN.Dropout, 'Seed', cfg.Seed + 10);
model = OceanNLP.Speech.Models.CTCRecognizer(features, vocab, enc, 'Dropout', cfg.HeadDropout, ...
    'Seed', cfg.Seed + 20, 'ModelId', "ocean-ctc-fsdd-char-v1" + suffix);
fprintf('%s\n', model.describe());

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

%% LM (train transcripts only) and its weight (chosen on validation)
lm = OceanNLP.Speech.Decoding.TokenNGramLM.fit(vocab.encode(trainMan.Entries.Transcript), vocab, ...
    'Order', cfg.LM.Order, 'Alpha', cfg.LM.Alpha);
lmGrid = zeros(0, 3);
for w = cfg.LM.WeightGrid
    for bo = cfg.LM.BonusGrid
        r = evalSet(model, dsValComposed, {'Decoder', "beam", 'BeamWidth', cfg.BeamWidth, 'LanguageModel', lm, ...
            'LMWeight', w, 'Bonus', bo});
        lmGrid(end + 1, :) = [w bo r.Result.WER.Rate]; %#ok<SAGROW>
        fprintf('  LM weight %.2f bonus %.1f: val WER %.4f\n', w, bo, r.Result.WER.Rate);
    end
end
[~, iBest] = min(lmGrid(:, 3));
lmW = lmGrid(iBest, 1); lmB = lmGrid(iBest, 2);

%% evaluation on the held-out speaker
decoders = struct('Name', {"greedy", "beam", "beam+LM"}, 'Args', { ...
    {'Decoder', "greedy"}, ...
    {'Decoder', "beam", 'BeamWidth', cfg.BeamWidth}, ...
    {'Decoder', "beam", 'BeamWidth', cfg.BeamWidth, 'LanguageModel', lm, 'LMWeight', lmW, 'Bonus', lmB}});
rows = {};
res = struct();
for d = 1:numel(decoders)
    for part = ["single", "composed"]
        r = evalSet(model, dsTest.(part), decoders(d).Args);
        res.(matlab.lang.makeValidName(decoders(d).Name)).(part) = r;
        rows(end + 1, :) = {decoders(d).Name, part, r.Result.WER.Rate, r.Result.CER.Rate, ...
            r.Result.WER.S, r.Result.WER.D, r.Result.WER.I, r.Result.WER.N, numel(r.Refs)}; %#ok<SAGROW>
    end
    rAll = evaluateConcat(res.(matlab.lang.makeValidName(decoders(d).Name)), model);
    rows(end + 1, :) = {decoders(d).Name, "all", rAll.WER.Rate, rAll.CER.Rate, rAll.WER.S, rAll.WER.D, ...
        rAll.WER.I, rAll.WER.N, height(rAll.PerUtterance)}; %#ok<SAGROW>
end
testTable = cell2table(rows, 'VariableNames', ["Decoder", "Set", "WER", "CER", "S", "D", "I", "RefWords", "Utterances"]);
disp(testTable);
valGreedy = evalSet(model, dsVal, {'Decoder', "greedy"});

%% per-speaker (test speaker + validation speakers, greedy and beam)
perSpk = [addCol(res.greedy.single.Result.PerSpeaker, "test", "single", "greedy"); ...
          addCol(res.greedy.composed.Result.PerSpeaker, "test", "composed", "greedy"); ...
          addCol(res.beam.single.Result.PerSpeaker, "test", "single", "beam"); ...
          addCol(res.beam.composed.Result.PerSpeaker, "test", "composed", "beam"); ...
          addCol(valGreedy.Result.PerSpeaker, "validation (speaker-dependent)", "single+composed", "greedy")];

%% 10 real held-out examples: the worst (beam) plus a seeded random draw of the rest
rb = res.beam.composed; rg = res.greedy.composed;
pu = rb.Result.PerUtterance;
[~, worst] = sort(pu.WordErrors, 'descend');
nWorst = min(5, cfg.NumExamples);
pickW = worst(1:min(nWorst, height(pu)));
rest = setdiff(1:height(pu), pickW);
pickR = rest(randperm(RandStream('mt19937ar', 'Seed', 7), numel(rest), min(cfg.NumExamples - numel(pickW), numel(rest))));
pick = [pickW(:); pickR(:)];
ex = table(pu.Id(pick), pu.Reference(pick), rg.Hyps(pick)', rb.Hyps(pick)', [rb.Transcripts(pick).Score]', ...
    string(arrayfun(@(t) strjoin(string({t.Candidates(1:min(3, end)).Text}), " / "), rb.Transcripts(pick), 'UniformOutput', false))', ...
    pu.WordErrors(pick), 'VariableNames', ["Id", "Reference", "Greedy", "Beam", "BeamScore", "Top3Beam", "BeamWordErrors"]);
singleWrong = find(res.beam.single.Result.PerUtterance.WordErrors > 0);
exSingle = table();
if ~isempty(singleWrong)
    ps = res.beam.single.Result.PerUtterance;
    k = singleWrong(1:min(5, end));
    exSingle = table(ps.Id(k), ps.Reference(k), ps.Hypothesis(k), 'VariableNames', ["Id", "Reference", "Beam"]);
end

%% timing (end to end from the Waveform: features + network + decoding), batch of one
nT = min(cfg.NumTimed, dsTest.composed.length());
tg = zeros(1, nT); tb = zeros(1, nT); dur = zeros(1, nT);
for i = 1:nT
    w = dsTest.composed.item(i).Waveform;
    dur(i) = w.duration();
    t0 = tic; model.transcribe(w, 'Decoder', "greedy"); tg(i) = toc(t0);
    t0 = tic; model.transcribe(w, 'Decoder', "beam", 'BeamWidth', cfg.BeamWidth); tb(i) = toc(t0);
end
timing = table(["greedy"; "beam (width " + cfg.BeamWidth + ")"], 1000 * [median(tg); median(tb)], ...
    [median(tg ./ dur); median(tb ./ dur)], 'VariableNames', ["Decoder", "MedianMsPerUtterance", "MedianRealTimeFactor"]);

%% save -> reload -> transcribe identity (the best checkpoint written by the trainer)
ck = OceanNLP.Speech.Training.SpeechCheckpoint.load(bestFile);
probe = arrayfun(@(i) dsTest.composed.item(i).Waveform, 1:min(20, dsTest.composed.length()), 'UniformOutput', false);
a = model.transcribe(probe, 'Decoder', "beam", 'BeamWidth', cfg.BeamWidth);
b = ck.Model.transcribe(probe, 'Decoder', "beam", 'BeamWidth', cfg.BeamWidth);
reloadSame = isequal(a.texts(), b.texts()) && isequal([a.Score], [b.Score]);
ag = model.transcribe(probe, 'Decoder', "greedy"); bg = ck.Model.transcribe(probe, 'Decoder', "greedy");
reloadSame = reloadSame && isequal(ag.texts(), bg.texts()) && isequal([ag.Score], [bg.Score]);
fprintf('save -> reload -> transcribe identical: %d\n', reloadSame);

%% report
totalMinutes = toc(tStart) / 60;
writeReport(reportFile, struct('cfg', cfg, 'quick', quick, 'dataSummary', dataSummary, 'audit', audit, ...
    'vocab', vocab, 'features', features, 'model', model, 'trainer', trainer, 'H', H, ...
    'trainSeconds', trainSeconds, 'testTable', testTable, 'perSpk', perSpk, 'ex', ex, 'exSingle', exSingle, ...
    'timing', timing, 'reloadSame', reloadSame, 'lmGrid', lmGrid, 'lmW', lmW, 'lmB', lmB, 'lm', lm, ...
    'bestFile', bestFile, 'repo', repo, 'totalMinutes', totalMinutes, 'valGreedy', valGreedy));
fprintf('report: %s  (total %.1f min)\n', reportFile, totalMinutes);

%% ---------------------------------------------------------------- helpers (orchestration only)
function r = evalSet(model, ds, decArgs)
% Transcribe every item of DS (batched, no shuffling) and score with CorpusEvaluator.
b = OceanNLP.Speech.Data.SpeechBatcher(ds, 'BatchSize', 32, 'Shuffle', false, 'SortByLength', false);
refs = strings(1, 0); hyps = refs; ids = refs; spk = refs; trs = OceanNLP.Speech.Core.Transcript.empty(1, 0);
t0 = tic;
for k = 1:b.numBatches()
    B = b.batch(k);
    tr = model.transcribe(B, decArgs{:});
    refs = [refs, B.Text]; hyps = [hyps, tr.texts()]; ids = [ids, B.Ids]; spk = [spk, B.Speakers]; %#ok<AGROW>
    trs = [trs, tr]; %#ok<AGROW>
end
ev = OceanNLP.Speech.Evaluation.CorpusEvaluator('Normalizer', model.Vocabulary.Normalizer);
r = struct('Refs', refs, 'Hyps', hyps, 'Ids', ids, 'Speakers', spk, 'Transcripts', trs, ...
    'Result', ev.evaluate(refs, hyps, 'Ids', ids, 'Speakers', spk), 'Seconds', toc(t0));
end

function res = evaluateConcat(parts, model)
ev = OceanNLP.Speech.Evaluation.CorpusEvaluator('Normalizer', model.Vocabulary.Normalizer);
res = ev.evaluate([parts.single.Refs, parts.composed.Refs], [parts.single.Hyps, parts.composed.Hyps], ...
    'Ids', [parts.single.Ids, parts.composed.Ids], 'Speakers', [parts.single.Speakers, parts.composed.Speakers]);
end

function t = addCol(t, split, set, dec)
t = [table(repmat(string(split), height(t), 1), repmat(string(set), height(t), 1), repmat(string(dec), height(t), 1), ...
    'VariableNames', ["Split", "Set", "Decoder"]), t];
end

function s = mdTable(t, fmt)
% Markdown table from a MATLAB table (numbers with FMT).
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
H = R.H; E = struct2table(R.H.Epochs, 'AsArray', true);
L = H.StepLoss; n = numel(L);
k = max(1, floor(n / 20));
idx = unique([1:k:n, n]);
curve = table(idx', arrayfun(@(i) mean(L(max(1, i - k + 1):i)), idx)', H.StepLR(idx)', H.StepEpoch(idx)' + 1, ...
    'VariableNames', ["Step", "MeanLossLastWindow", "LR", "Epoch"]);
m = R.model;
cfgJson = string(jsonencode(R.cfg, 'PrettyPrint', true));
mcfg = m.config(); mcfg.Vocabulary = rmfield(mcfg.Vocabulary, intersect(fieldnames(mcfg.Vocabulary), {'LexiconWords', 'LexiconPhones'}));
modelJson = string(jsonencode(rmfield(mcfg, 'Features'), 'PrettyPrint', true));
featJson = string(jsonencode(R.features.config(), 'PrettyPrint', true));
o = "# CTC recogniser on FSDD (milestone S1)" + newline + newline;
if R.quick
    o = o + "> **QUICK DEBUG RUN (OCEAN_SPEECH_QUICK=1) -- tiny subsets, NOT a result.**" + newline + newline;
end
o = o + "Generated by `Experiments/Scripts/Speech/exp_speech_ctc_fsdd.m` on " + string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm')) + ...
    " (MATLAB " + version('-release') + ", CPU, double precision). Total wall time " + sprintf('%.1f', R.totalMinutes) + ...
    " min, of which training " + sprintf('%.1f', R.trainSeconds / 60) + " min." + newline + newline;
o = o + "All numbers below are computed by the script from the real FSDD recordings; none are targets." + newline + newline;
o = o + "## Data" + newline + newline + mdTable(R.dataSummary, '%.2f') + newline;
o = o + "Canonical split: test = held-out speaker `theo` (speaker-independent); validation = takes 45-49 of the training " + ...
    "speakers (speaker-DEPENDENT; used only for checkpoint selection and the LM weight). Composed utterances concatenate " + ...
    "2-5 real recordings of one speaker within one split (UtteranceComposer). Vocabulary (fitted on train transcripts): " + ...
    vocab2str(R.vocab) + "." + newline + newline;
o = o + "## Configuration" + newline + newline + "Experiment config:" + newline + newline + "```json" + newline + cfgJson + newline + "```" + newline + newline;
o = o + "Model config (`model.config()`, Features below):" + newline + newline + "```json" + newline + modelJson + newline + "```" + newline + newline;
o = o + "Feature config:" + newline + newline + "```json" + newline + featJson + newline + "```" + newline + newline;
o = o + "Model size: **" + m.numParameters() + " parameters** (" + numel(m.parameters()) + " tensors)." + newline + newline;
o = o + "## Training" + newline + newline + "Per epoch (val = speaker-dependent validation set, greedy decoding):" + newline + newline;
o = o + mdTable(E(:, ["Epoch", "Steps", "TrainLoss", "ValLoss", "ValWER", "ValCER", "LR", "TrainSeconds", "ValSeconds"])) + newline;
o = o + sprintf('Best epoch: %d (val WER %.4f, val loss %.4f); its weights are used below and saved in `%s`.', ...
    R.trainer.BestEpoch, R.trainer.BestScore(1), R.trainer.BestScore(2), relpath(R.bestFile, R.repo)) + newline + newline;
o = o + "Training loss curve (CTC loss per target token, mean over the preceding window of steps):" + newline + newline + mdTable(curve) + newline;
o = o + "## Held-out speaker (theo) results" + newline + newline + mdTable(R.testTable) + newline;
o = o + sprintf(['LM: character %d-gram, add-%.2g smoothing, fitted on the %d TRAIN transcripts only; shallow-fusion weight %.2f and ', ...
    'length bonus %.1f chosen by WER on the composed validation utterances over the grid below.'], R.lm.Order, R.lm.Alpha, R.lm.NumSequences, R.lmW, R.lmB) + newline + newline;
o = o + mdTable(array2table(R.lmGrid, 'VariableNames', ["LMWeight", "Bonus", "ValWER"])) + newline;
o = o + "## Per speaker" + newline + newline + mdTable(R.perSpk) + newline;
o = o + "## Examples (held-out speaker, composed multi-word utterances)" + newline + newline + ...
    "The first rows are the utterances with the most beam-search word errors; the rest are a seeded random draw." + newline + newline + mdTable(R.ex, '%.2f') + newline;
if height(R.exSingle) > 0
    o = o + "Single-digit test errors (first up to 5):" + newline + newline + mdTable(R.exSingle) + newline;
end
o = o + "## Speed" + newline + newline + mdTable(R.timing, '%.3f') + newline + ...
    "End to end from the Waveform (log-mel + network + decoding), one utterance at a time on CPU; real-time factor = compute time / audio duration." + newline + newline;
o = o + "## Save -> reload -> transcribe" + newline + newline;
if R.reloadSame
    o = o + "Reloading the best checkpoint with `SpeechCheckpoint.load` (model rebuilt by `RecognizerFactory.fromConfig` + saved weights) and transcribing 20 held-out utterances gives **identical** greedy and beam texts and scores (`isequal`)." + newline + newline;
else
    o = o + "**MISMATCH**: the reloaded checkpoint did not reproduce the transcripts exactly." + newline + newline;
end
o = o + "## Caveats" + newline + newline + ...
    "* One held-out speaker (theo, 500 single + 500 composed utterances): the test WER describes this speaker, not speakers in general." + newline + ...
    "* Composed utterances are concatenations of isolated-digit recordings (no coarticulation across words), so they are easier than natural connected speech." + newline + ...
    "* Validation shares speakers with training; its WER is optimistic and was used only for model / LM-weight selection." + newline + ...
    "* The char LM knows the ten digit words from training transcripts; on this closed vocabulary it is a strong constraint." + newline;
fid = fopen(file, 'w'); fprintf(fid, '%s', o); fclose(fid);
end

function s = vocab2str(v)
s = sprintf('%d classes = 5 specials + %d units {%s}', v.numClasses(), v.numClasses() - 5, strjoin(v.Tokens(6:end), ' '));
end

function p = relpath(f, repo)
p = replace(erase(string(f), string(repo) + filesep), "\", "/");
end
