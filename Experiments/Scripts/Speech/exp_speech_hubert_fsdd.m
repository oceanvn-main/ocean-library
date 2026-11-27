%% exp_speech_hubert_fsdd  HuBERT-style masked-unit pretraining (iterations 1 + 2) + multi-seed CTC fine-tuning.
%
%   Consumer script only: algorithms live in OceanNLP.Speech.Models.Pretrain
%   (UnitCodebook, HiddenLayerSelector, HuBERTModel, MaskedUnitTrainer) and the existing
%   Speech/OceanDL classes.
%
%   UNLABELED AUDIO (contract sec. 1): FSDD train-split recordings (transcripts NEVER read) +
%   the MATLAB Audio Toolbox speech samples, resampled to 8 kHz by AudioDataset ->
%   OceanMath.Signal.ResampleSignal, long files cut into 1 s chunks. Pretraining validation =
%   FSDD validation-split audio (unlabeled). The held-out test speaker "theo" is never used
%   before the final evaluation.
%   ITERATION 1 (Hsu et al. 2021 sec. III-C): UnitCodebook (k-means++/Lloyd) on
%     MFCC+delta+delta-delta frames -> masked-unit pretraining of HuBERTModel.
%   ITERATION 2: hidden features of every Transformer layer of the iteration-1 model;
%     HiddenLayerSelector picks the layer by clustering stability (NMI of two k-means seeds on
%     the pretraining VALIDATION audio -- no labels, no test speaker); the chosen layer's
%     codebook gives new targets; a NEW model (same architecture + init seed) is pretrained on
%     them with the same unlabeled data and schedule.
%   SENSITIVITY: iteration 1 redone without the toolbox "Counting" sample (a speaker counting
%     aloud = unlabeled digit words) to quantify in-domain leakage into pretraining.
%   FINE-TUNING: CTCRecognizer({HuBERTModel without head}) on a LabeledFraction subset of the
%     train single-digit utterances (+ composed utterances of that subset), for each seed s:
%     labeled-subset draw, composition, head init, batch order and dropout all derive from s
%     and are IDENTICAL across arms; arms = it1 / it2 / scratch (+ it1-noCounting for
%     cfg.NoCountingSeeds). Evaluation on theo, greedy + prefix beam (no LM).
%
%   STAGES (env OCEAN_HUBERT_STAGE, default "all" = everything in order, one process):
%     "it1"      iteration-1 pretraining (REUSES Results/Speech/checkpoints/hubert_fsdd_pretrain.mat
%                when its stored config equals cfg and the refitted codebook has the same Version)
%     "it2"      layer selection + iteration-2 pretraining      (needs it1)
%     "nocount"  iteration-1 pretraining without Counting
%     "finetune" arms OCEAN_HUBERT_ARMS (comma list, default all applicable) x seeds
%                OCEAN_HUBERT_SEEDS (comma list, default cfg.Seeds); one result file per (arm, seed)
%     "report"   writes Results/Speech/hubert_fsdd_report.md from the saved stage outputs
%   Stages are independent processes so they can run in parallel; OCEAN_THREADS sets
%   maxNumCompThreads. OCEAN_SPEECH_QUICK=1: tiny debugging run (writes *_quick files, NOT a result).

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(fileparts(here)));
addpath(repo);
quick = getenv('OCEAN_SPEECH_QUICK') == "1";
stage = string(getenv('OCEAN_HUBERT_STAGE'));
if stage == "", stage = "all"; end
if getenv('OCEAN_THREADS') ~= "", maxNumCompThreads(str2double(getenv('OCEAN_THREADS'))); end
tStart = tic;
OceanNLP.Speech.Models.RecognizerFactory.register("encoder", "HuBERTModel", ...
    @OceanNLP.Speech.Models.Pretrain.HuBERTModel.fromConfig);

%% configuration
cfg = struct();
cfg.Seed = 0;
cfg.Features = struct('Type', "LogMel", 'NumMels', 40, 'WindowSeconds', 0.025, 'HopSeconds', 0.010);
cfg.TargetFeatures = struct('Type', "MFCC+delta+deltadelta", 'NumCoeffs', 13, 'DeltaWidth', 2);
cfg.ToolboxSamples = ["Counting-16-44p1-mono-15secs.wav", "FemaleSpeech-16-8-mono-3secs.wav", ...
    "SpeechDFT-16-8-mono-5secs.wav", "Rainbow-16-8-mono-114secs.wav", "CleanSpeech-16-mono-3secs.ogg"];
cfg.ChunkFrames = 100;          % 1 s chunks of long toolbox recordings
cfg.MinChunkFrames = 20;
cfg.Codebook = struct('NumUnits', 50, 'MaxIter', 50, 'Tol', 1e-4, 'Normalize', "utterance", 'Seed', 0);
cfg.Model = struct('ConvChannels', [128 128], 'ConvKernels', [5 3], 'ConvStrides', [2 1], 'Dim', 128, ...
    'Depth', 3, 'Heads', 4, 'FFNDim', 256, 'Dropout', 0.1, 'ProjDim', 64, 'Temperature', 0.1, ...
    'MaskProb', 0.13, 'MaskLength', 5, 'MaxSpanFraction', 0.5, 'MinSpans', 1, 'Alpha', 1, 'Seed', 1);
cfg.Pretrain = struct('Epochs', 30, 'BatchSize', 32, 'LR', 1e-3, 'WeightDecay', 0.01, 'WarmupFraction', 0.08, 'Clip', 5);
% iteration 2: same K as iteration 1 (isolates the change of target SOURCE), raw hidden features
% (no CMVN, as in HuBERT), candidate layers = every Transformer layer, two k-means seeds
cfg.Iter2 = struct('NumUnits', 50, 'Layers', 1:cfg.Model.Depth, 'Seeds', [0 1], 'Normalize', "none", 'MaxIter', 50, 'Tol', 1e-4);
cfg.LeakageSample = "Counting-16-44p1-mono-15secs.wav";
cfg.LabeledFraction = 0.10;
cfg.Seeds = [1 2 3];            % seed s -> labeled subset RandStream(s), compose seed 200+s, fine-tune seed s
cfg.NoCountingSeeds = [1 2 3];        % seeds that also fine-tune the it1-noCounting encoder
cfg.ComposeMinWords = 2; cfg.ComposeMaxWords = 5;
cfg.FineTune = struct('Epochs', 25, 'BatchSize', 16, 'LR', 1e-3, 'WeightDecay', 0.01, 'Clip', 5, 'HeadDropout', 0.1);
cfg.TestCompose = struct('Count', 500, 'Seed', 103);
cfg.BeamWidth = 8;
if quick
    cfg.Pretrain.Epochs = 1; cfg.FineTune.Epochs = 1; cfg.Seeds = 1; cfg.LabeledFraction = 0.5;
    cfg.TestCompose.Count = 10; cfg.Codebook.MaxIter = 5; cfg.Iter2.MaxIter = 5;
end
if getenv('OCEAN_HUBERT_SEEDS') ~= "", seeds = str2double(split(string(getenv('OCEAN_HUBERT_SEEDS')), ",")).'; else, seeds = cfg.Seeds; end
resDir = fullfile(repo, 'Results', 'Speech');
ckDir = fullfile(resDir, 'checkpoints');
ftDir = fullfile(ckDir, 'hubert_ft');
if ~isfolder(ftDir), mkdir(ftDir); end
suffix = ""; if quick, suffix = "_quick"; end
files = struct('it1', fullfile(ckDir, "hubert_fsdd_pretrain" + suffix + ".mat"), ...
    'it2', fullfile(ckDir, "hubert_fsdd_pretrain_it2" + suffix + ".mat"), ...
    'nocount', fullfile(ckDir, "hubert_fsdd_pretrain_nocounting" + suffix + ".mat"), ...
    'report', fullfile(resDir, "hubert_fsdd_report" + suffix + ".md"), ...
    'previous', fullfile(resDir, "hubert_fsdd_report_run1.md"));
logFile = fullfile(resDir, "hubert_fsdd_" + stage + suffix + ".log");
if stage == "finetune"
    armTag = ""; if getenv('OCEAN_HUBERT_ARMS') ~= "", armTag = "_" + replace(string(getenv('OCEAN_HUBERT_ARMS')), ",", "-"); end
    logFile = fullfile(resDir, "hubert_fsdd_finetune_s" + join(string(seeds), "-") + armTag + suffix + ".log");
end
if isfile(logFile), delete(logFile); end
logf = @(msg) logLine(logFile, msg);
logf(sprintf('stage %s, seeds %s, %d threads', stage, join(string(seeds), ","), maxNumCompThreads));

%% data
fs = 8000;
man = OceanNLP.Speech.Data.SpeechManifest.fromFSDD();
split0 = man.canonicalFSDDSplit();
trainS = split0.bySplit("train"); valS = split0.bySplit("validation"); testS = split0.bySplit("test");
if quick
    trainS = trainS.subset(round(linspace(1, trainS.height(), 60)));
    valS = valS.subset(round(linspace(1, valS.height(), 20)));
    testS = testS.subset(round(linspace(1, testS.height(), 20)));
end
trainS.assertDisjoint(testS, 'By', "speaker");
features = OceanNLP.Speech.Features.LogMel(fs, 'NumMels', cfg.Features.NumMels, ...
    'WindowLength', round(cfg.Features.WindowSeconds * fs), 'HopLength', round(cfg.Features.HopSeconds * fs));
tgtFeatures = OceanNLP.Speech.Features.FeaturePipeline({OceanNLP.Speech.Features.MFCC(fs, ...
    'NumCoeffs', cfg.TargetFeatures.NumCoeffs, 'NumMels', cfg.Features.NumMels, ...
    'WindowLength', round(cfg.Features.WindowSeconds * fs), 'HopLength', round(cfg.Features.HopSeconds * fs)), ...
    OceanNLP.Speech.Features.DeltaFeatures('Width', cfg.TargetFeatures.DeltaWidth, 'Order', 2)});
mc = cfg.Model;
newHub = @() OceanNLP.Speech.Models.Pretrain.HuBERTModel(features.outputDim([]), 'ConvChannels', mc.ConvChannels, ...
    'ConvKernels', mc.ConvKernels, 'ConvStrides', mc.ConvStrides, 'Dim', mc.Dim, 'Depth', mc.Depth, 'Heads', mc.Heads, ...
    'FFNDim', mc.FFNDim, 'Dropout', mc.Dropout, 'NumUnits', cfg.Codebook.NumUnits, 'ProjDim', mc.ProjDim, ...
    'Temperature', mc.Temperature, 'MaskProb', mc.MaskProb, 'MaskLength', mc.MaskLength, ...
    'MaxSpanFraction', mc.MaxSpanFraction, 'MinSpans', mc.MinSpans, 'Alpha', mc.Alpha, 'Seed', mc.Seed);
initCfg = newHub().config();

U = [];
if ismember(stage, ["all", "it1", "it2", "nocount", "report"])
    U = unlabeledData(trainS, valS, features, tgtFeatures, fs, cfg);
    logf(sprintf('unlabeled data ready (%.0f s): %d FSDD + %d toolbox chunks (%d from %s), %d val', toc(tStart), ...
        numel(U.inFSDD), numel(U.inTool), nnz(U.isLeak), cfg.LeakageSample, numel(U.inVal)));
end

%% iteration 1 (reuse the checkpoint when it was produced by this exact configuration)
if ismember(stage, ["all", "it1"])
    pretrainIteration1(U, true(1, numel(U.preIn)), cfg, newHub, files.it1, logFile, logf);
end

%% iteration 2: data-driven layer choice, re-clustering, new pretraining
if ismember(stage, ["all", "it2"])
    pretrainIteration2(U, cfg, newHub, files.it1, files.it2, logFile, logf);
end

%% sensitivity: iteration 1 without the Counting sample
if ismember(stage, ["all", "nocount"])
    pretrainIteration1(U, ~U.isLeak, cfg, newHub, files.nocount, logFile, logf);
end

%% fine-tuning: arms x seeds, one result file each
if ismember(stage, ["all", "finetune"])
    arms = ["it1", "it2", "scratch", "it1nocount"];
    if getenv('OCEAN_HUBERT_ARMS') ~= "", arms = strtrim(split(string(getenv('OCEAN_HUBERT_ARMS')), ",")).'; end
    vocabFeat = struct('features', features, 'fs', fs);
    testC = OceanNLP.Speech.Data.UtteranceComposer.compose(testS, 'MinWords', cfg.ComposeMinWords, ...
        'MaxWords', cfg.ComposeMaxWords, 'Count', cfg.TestCompose.Count, 'Seed', cfg.TestCompose.Seed);
    for s = seeds
        for arm = arms
            if arm == "it1nocount" && ~ismember(s, cfg.NoCountingSeeds), continue; end
            out = fullfile(ftDir, "ft_" + arm + "_seed" + s + suffix + ".mat");
            R = fineTune(arm, s, trainS, valS, testS, testC, vocabFeat, initCfg, files, cfg, ftDir, suffix, logFile);
            save(out, '-struct', 'R');
            logf(sprintf('%s seed %d: %.1f min, best epoch %d, theo greedy WER single %.4f composed %.4f -> %s', arm, s, ...
                R.Seconds / 60, R.BestEpoch, R.Eval.single.greedy.WER, R.Eval.composed.greedy.WER, out));
        end
    end
end

%% report
if ismember(stage, ["all", "report"])
    writeReport(files, ftDir, suffix, cfg, U, quick, repo, testS);
    logf(sprintf('report: %s', files.report));
end
logf(sprintf('stage %s done (%.1f min)', stage, toc(tStart) / 60));

%% ---------------------------------------------------------------- helpers (orchestration only)
function logLine(file, msg)
fprintf('%s\n', msg);
fid = fopen(file, 'a'); fprintf(fid, '%s %s\n', string(datetime('now', 'Format', 'HH:mm:ss')), msg); fclose(fid);
end

function U = unlabeledData(trainS, valS, features, tgtFeatures, fs, cfg)
% Unlabeled pretraining inputs (log-mel) + target features (MFCC+deltas); transcripts blanked.
unl = @(m) OceanNLP.Speech.Data.SpeechManifest(setTranscripts(m.Entries, ""), 'Root', m.Root, 'Name', m.Name + "-unlabeled");
toolRoot = string(fullfile(matlabroot, 'toolbox', 'audio', 'samples'));
toolMan = toolboxManifest(toolRoot, cfg.ToolboxSamples);
[U.inFSDD, U.tgFSDD, U.idFSDD] = featurize(unl(trainS), features, tgtFeatures, fs);
[inTool, tgTool, idTool] = featurize(toolMan, features, tgtFeatures, fs);
[U.inTool, U.tgTool, U.idTool] = chunk(inTool, tgTool, idTool, cfg.ChunkFrames, cfg.MinChunkFrames);
[U.inVal, U.tgVal, U.idVal] = featurize(unl(valS), features, tgtFeatures, fs);
U.preIn = [U.inFSDD, U.inTool]; U.preTg = [U.tgFSDD, U.tgTool]; U.preIds = [U.idFSDD, U.idTool];
assert(~any(contains(U.preIds, "theo")) && ~any(contains(U.idVal, "theo")), 'test speaker leaked into pretraining');
leakId = "toolbox:" + erase(cfg.LeakageSample, [".wav", ".ogg"]);
U.isLeak = startsWith(U.preIds, leakId + "#");
assert(any(U.isLeak), 'leakage sample not found in the pretraining data');
U.toolInfo = toolMan.Entries(:, ["AudioPath", "Duration", "SampleRate"]);
U.toolRoot = "<matlabroot>/toolbox/audio/samples";
end

function pretrainIteration1(U, keep, cfg, newHub, file, logFile, logf)
% K-means on MFCC+deltas of the kept unlabeled frames -> masked-unit pretraining.
preIn = U.preIn(keep); preTg = U.preTg(keep); preIds = U.preIds(keep);
c = cfg.Codebook;
cb = OceanNLP.Speech.Models.Pretrain.UnitCodebook(c.NumUnits, 'Seed', c.Seed, 'MaxIter', c.MaxIter, 'Tol', c.Tol, ...
    'Normalize', c.Normalize).fit(preTg, 'Ids', preIds);
logf(sprintf('iteration-1 codebook %s on %d utterances (%d frames, %d Lloyd iterations)', cb.Version, numel(preIn), cb.NumFrames, numel(cb.Inertia)));
if isfile(file)
    ck = load(file);
    same = isfield(ck, 'Extra') && isfield(ck.Extra, 'Config') && string(ck.CodebookVersion) == cb.Version;
    if same
        old = ck.Extra.Config;
        for f = ["Features", "TargetFeatures", "ToolboxSamples", "ChunkFrames", "MinChunkFrames", "Codebook", "Model", "Pretrain", "Seed"]
            same = same && isfield(old, f) && isequal(old.(f), cfg.(f));
        end
        same = same && ck.Step == cfg.Pretrain.Epochs * ceil(numel(preIn) / cfg.Pretrain.BatchSize);
    end
    if same
        logf(sprintf('REUSING %s: same configuration, same codebook Version %s, %d steps', file, cb.Version, ck.Step));
        return;
    end
    logf(sprintf('existing %s does not match the configuration -> retraining', file));
end
pc = cfg.Pretrain;
hub = newHub();
ptr = OceanNLP.Speech.Models.Pretrain.MaskedUnitTrainer(hub, preIn, 'Ids', preIds, 'Epochs', pc.Epochs, ...
    'BatchSize', pc.BatchSize, 'LR', pc.LR, 'WeightDecay', pc.WeightDecay, 'WarmupFraction', pc.WarmupFraction, ...
    'Clip', pc.Clip, 'Seed', cfg.Seed, 'ValInputs', U.inVal, 'ValIds', U.idVal, 'Checkpoint', file, 'LogFile', logFile);
ptr.setTargets(cb.assign(preTg), 'Rate', "input", 'Codebook', cb);
ptr.setValTargets(cb.assign(U.tgVal), 'Rate', "input");
val0 = ptr.evaluate("val");
t0 = tic;
ptr.fit();
ptr.save(file, 'Extra', struct('Codebook', cb.stateDict(), 'Config', cfg, 'Val0', val0, 'Seconds', toc(t0), ...
    'Usage', cb.usage(preTg), 'ValUsage', cb.usage(U.tgVal)));
logf(sprintf('iteration-1 pretraining -> %s (%.1f min)', file, toc(t0) / 60));
end

function pretrainIteration2(U, cfg, newHub, it1File, file, logFile, logf)
ck1 = OceanNLP.Speech.Models.Pretrain.MaskedUnitTrainer.load(it1File);
cb1 = OceanNLP.Speech.Models.Pretrain.UnitCodebook.fromStateDict(ck1.Extra.Codebook);
assert(isequal(sort(string(ck1.TrainIds(:))), sort(U.preIds(:))), 'iteration-1 checkpoint was trained on different data');
pc = cfg.Pretrain; c2 = cfg.Iter2;
probe = OceanNLP.Speech.Models.Pretrain.MaskedUnitTrainer(ck1.Model, U.preIn, 'Ids', U.preIds, 'BatchSize', pc.BatchSize, ...
    'ValInputs', U.inVal, 'ValIds', U.idVal, 'Verbose', false);
t0 = tic;
trH = cell(1, numel(c2.Layers)); vaH = trH;
for j = 1:numel(c2.Layers)
    trH{j} = probe.hiddenFeatures(c2.Layers(j), "train");
    vaH{j} = probe.hiddenFeatures(c2.Layers(j), "val");
end
featSeconds = toc(t0);
% diagnostic reference: iteration-1 MFCC units of the validation audio at the output frame rate
ref = cellfun(@(tg) unitsAtOutput(cb1.assign(tg), ck1.Model, tg.numFrames()), U.tgVal, 'UniformOutput', false);
t1 = tic;
sel = OceanNLP.Speech.Models.Pretrain.HiddenLayerSelector(c2.NumUnits, 'Seeds', c2.Seeds, 'MaxIter', c2.MaxIter, ...
    'Tol', c2.Tol, 'Normalize', c2.Normalize).fit(trH, vaH, 'Layers', c2.Layers, 'Ids', U.preIds, 'Reference', ref);
selSeconds = toc(t1);
% reproducibility of the choice: refit with the same seeds -> same scores and layer
sel2 = OceanNLP.Speech.Models.Pretrain.HiddenLayerSelector(c2.NumUnits, 'Seeds', c2.Seeds, 'MaxIter', c2.MaxIter, ...
    'Tol', c2.Tol, 'Normalize', c2.Normalize).fit(trH, vaH, 'Layers', c2.Layers, 'Ids', U.preIds, 'Reference', ref);
reproducible = isequal(sel2.Scores, sel.Scores) && sel2.Layer == sel.Layer && sel2.Codebook.Version == sel.Codebook.Version;
j = find(c2.Layers == sel.Layer);
cb2 = sel.Codebook;
logf(sprintf('layer selection (%.0f s features, %.0f s k-means): stability %s -> layer %d, codebook %s, reproducible %d', ...
    featSeconds, selSeconds, mat2str(sel.Scores.Stability.', 4), sel.Layer, cb2.Version, reproducible));
hub = newHub();
ptr = OceanNLP.Speech.Models.Pretrain.MaskedUnitTrainer(hub, U.preIn, 'Ids', U.preIds, 'Epochs', pc.Epochs, ...
    'BatchSize', pc.BatchSize, 'LR', pc.LR, 'WeightDecay', pc.WeightDecay, 'WarmupFraction', pc.WarmupFraction, ...
    'Clip', pc.Clip, 'Seed', cfg.Seed, 'ValInputs', U.inVal, 'ValIds', U.idVal, 'Checkpoint', file, 'LogFile', logFile);
ptr.setTargets(cb2.assign(trH{j}), 'Rate', "output", 'Codebook', cb2);
ptr.setValTargets(cb2.assign(vaH{j}), 'Rate', "output");
val0 = ptr.evaluate("val");
t2 = tic;
ptr.fit();
extra = struct('Codebook', cb2.stateDict(), 'Config', cfg, 'Val0', val0, 'Seconds', toc(t2), ...
    'Selection', sel.Scores, 'Layer', sel.Layer, 'SelectionReproducible', reproducible, ...
    'FeatureSeconds', featSeconds, 'SelectionSeconds', selSeconds, 'It1CodebookVersion', cb1.Version, ...
    'Usage', cb2.usage(trH{j}), 'ValUsage', cb2.usage(vaH{j}), ...
    'NMIIt1It2Val', OceanNLP.Speech.Models.Pretrain.HiddenLayerSelector.nmi(cell2mat(cb2.assign(vaH{j})), cell2mat(ref)));
ptr.save(file, 'Extra', extra);
logf(sprintf('iteration-2 pretraining -> %s (%.1f min)', file, extra.Seconds / 60));
end

function u = unitsAtOutput(uIn, model, Tin)
u = uIn(model.targetFrameIndex(model.outputLength(Tin), Tin));
end

function R = fineTune(arm, s, trainS, valS, testS, testC, vf, initCfg, files, cfg, ftDir, suffix, logFile)
% One (arm, seed) fine-tuning run; everything except the encoder weights depends only on s.
features = vf.features; fs = vf.fs; ft = cfg.FineTune;
rsL = RandStream('mt19937ar', 'Seed', s);
nLab = max(10, round(cfg.LabeledFraction * trainS.height()));
labIdx = sort(randperm(rsL, trainS.height(), nLab));
labS = trainS.subset(labIdx);
labC = OceanNLP.Speech.Data.UtteranceComposer.compose(labS, 'MinWords', cfg.ComposeMinWords, ...
    'MaxWords', cfg.ComposeMaxWords, 'Count', labS.height(), 'Seed', 200 + s);
labMan = OceanNLP.Speech.Data.SpeechManifest([labS.Entries; labC.Entries], 'Root', labS.Root, 'Name', "FSDD-labeled-subset");
vocab = OceanNLP.Speech.Data.SpeechVocabulary.fit(labMan.Entries.Transcript, 'Unit', "char");
mkds = @(m) OceanNLP.Speech.Data.AudioDataset(m, features, 'Vocabulary', vocab, 'SampleRate', fs);
enc = OceanNLP.Speech.Models.Pretrain.HuBERTModel.fromConfig(initCfg);
source = struct('File', "", 'CodebookVersion', "");
if arm ~= "scratch"
    key = struct('it1', "it1", 'it2', "it2", 'it1nocount', "nocount");
    ck = OceanNLP.Speech.Models.Pretrain.MaskedUnitTrainer.load(files.(key.(arm)));
    enc.loadStateDict(ck.Model.stateDict());
    source = struct('File', string(files.(key.(arm))), 'CodebookVersion', string(ck.CodebookVersion));
end
enc.dropPretrainHead();
model = OceanNLP.Speech.Models.CTCRecognizer(features, vocab, {enc}, 'Dropout', ft.HeadDropout, ...
    'Seed', s + 20, 'ModelId', "ocean-hubert-ctc-" + arm + "-seed" + s + suffix);
bTrain = OceanNLP.Speech.Data.SpeechBatcher(mkds(labMan), 'BatchSize', ft.BatchSize, 'Shuffle', true, 'Seed', s, 'SortByLength', true);
bVal = OceanNLP.Speech.Data.SpeechBatcher(mkds(valS), 'BatchSize', 32, 'Shuffle', false, 'SortByLength', true);
tr = OceanNLP.Speech.Training.SpeechTrainer(model, bTrain, 'ValBatcher', bVal, 'Epochs', ft.Epochs, 'LR', ft.LR, ...
    'WeightDecay', ft.WeightDecay, 'Clip', ft.Clip, 'Seed', s, ...
    'BestCheckpoint', fullfile(ftDir, "ctc_" + arm + "_seed" + s + suffix + ".mat"), 'LogFile', logFile);
t0 = tic;
tr.fit();
secs = toc(t0);
tr.restoreBest();
ev = struct();
dsTest = struct('single', mkds(testS), 'composed', mkds(testC));
for part = ["single", "composed"]
    for dec = ["greedy", "beam"]
        r = evalSet(model, dsTest.(part), {'Decoder', dec, 'BeamWidth', cfg.BeamWidth}).Result;
        ev.(part).(dec) = struct('WER', r.WER.Rate, 'CER', r.CER.Rate, 'S', r.WER.S, 'D', r.WER.D, 'I', r.WER.I, 'N', r.WER.N);
    end
end
R = struct('Arm', arm, 'Seed', s, 'LabeledIds', labS.Entries.Id, 'NumLabeled', labS.height(), 'NumComposed', labC.height(), ...
    'NumClasses', vocab.numClasses(), 'Source', source, 'Eval', ev, 'BestEpoch', tr.BestEpoch, 'Steps', tr.Step, ...
    'Seconds', secs, 'Epochs', tr.History.Epochs, 'TestSingle', testS.height(), 'TestComposed', testC.height());
end

function E = setTranscripts(E, t)
E.Transcript(:) = string(t);
end

function m = toolboxManifest(root, files)
n = numel(files);
dur = zeros(n, 1); rate = zeros(n, 1);
for i = 1:n
    info = audioinfo(fullfile(root, files(i)));
    dur(i) = info.TotalSamples / info.SampleRate; rate(i) = info.SampleRate;
end
names = erase(files(:), [".wav", ".ogg"]);
E = table("toolbox:" + names, files(:), repmat("", n, 1), repmat("en", n, 1), "toolbox-" + names, ...
    repmat("", n, 1), dur, rate, repmat("train", n, 1), repmat("MathWorks Audio Toolbox example audio", n, 1), ...
    'VariableNames', ["Id", "AudioPath", "Transcript", "Language", "Speaker", "Session", "Duration", "SampleRate", "Split", "License"]);
m = OceanNLP.Speech.Data.SpeechManifest(E, 'Root', root, 'Name', "AudioToolboxSamples");
end

function [inp, tgt, ids] = featurize(man, features, tgtFeatures, fs)
% Model-input log-mel and target MFCC+deltas of every utterance (same frame grid).
ds = OceanNLP.Speech.Data.AudioDataset(man, features, 'SampleRate', fs, 'Cache', false);
n = ds.length();
inp = cell(1, n); tgt = cell(1, n); ids = strings(1, n);
for i = 1:n
    w = ds.waveform(i).toMono("mean");
    inp{i} = features.forward(w);
    tgt{i} = tgtFeatures.forward(w);
    assert(inp{i}.numFrames() == tgt{i}.numFrames());
    ids(i) = man.Entries.Id(i);
end
end

function [inp, tgt, ids] = chunk(inp0, tgt0, ids0, L, minL)
inp = {}; tgt = {}; ids = strings(1, 0);
for i = 1:numel(inp0)
    T = inp0{i}.numFrames();
    for a = 1:L:T
        b = min(a + L - 1, T);
        if b - a + 1 < minL, continue; end
        inp{end + 1} = slice(inp0{i}, a, b); %#ok<AGROW>
        tgt{end + 1} = slice(tgt0{i}, a, b); %#ok<AGROW>
        ids(end + 1) = ids0(i) + "#" + a; %#ok<AGROW>
    end
end
end

function s = slice(f, a, b)
s = OceanNLP.Speech.Core.FeatureSequence(f.Values(:, a:b), 'FrameTimes', f.FrameTimes(a:b), 'FrequencyHz', f.FrequencyHz, ...
    'AxisNames', f.AxisNames, 'Units', f.Units, 'Config', f.Config, 'Source', f.Source + "#" + a);
end

function r = evalSet(model, ds, decArgs)
b = OceanNLP.Speech.Data.SpeechBatcher(ds, 'BatchSize', 32, 'Shuffle', false, 'SortByLength', false);
refs = strings(1, 0); hyps = refs; ids = refs; spk = refs;
for k = 1:b.numBatches()
    B = b.batch(k);
    tr = model.transcribe(B, decArgs{:});
    refs = [refs, B.Text]; hyps = [hyps, tr.texts()]; ids = [ids, B.Ids]; spk = [spk, B.Speakers]; %#ok<AGROW>
end
ev = OceanNLP.Speech.Evaluation.CorpusEvaluator('Normalizer', model.Vocabulary.Normalizer);
r = struct('Refs', refs, 'Hyps', hyps, 'Result', ev.evaluate(refs, hyps, 'Ids', ids, 'Speakers', spk));
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

function s = pm(x)
% mean +- sample std (n-1) over seeds, as text
if numel(x) > 1
    s = string(sprintf('%.4f ± %.4f', mean(x), std(x)));
else
    s = string(sprintf('%.4f', x));
end
end

function o = epochTable(ck, title)
E = struct2table(ck.History.Epochs, 'AsArray', true);
k = height(E); idx = unique([1:max(1, floor(k / 10)):k, k]);
o = title + newline + newline + mdTable(E(idx, ["Epoch", "TrainLoss", "TrainMaskedAcc", "ValLoss", "ValMaskedAcc", "ValUnmaskedAcc", "Seconds"])) + newline;
end

function writeReport(files, ftDir, suffix, cfg, U, quick, repo, testS)
nl = newline;
ck1 = load(files.it1); ck2 = load(files.it2);
hasNC = isfile(files.nocount);
if hasNC, ckN = load(files.nocount); end
R = struct([]);
for f = dir(fullfile(ftDir, "ft_*_seed*" + suffix + ".mat")).'
    if suffix == "" && endsWith(f.name, "_quick.mat"), continue; end
    r = load(fullfile(f.folder, f.name));
    if ~ismember(r.Seed, cfg.Seeds), continue; end
    R = [R, r]; %#ok<AGROW>
end
arms = string({R.Arm}); seedsR = [R.Seed];
o = "# HuBERT-style pretraining (iterations 1 + 2) + CTC fine-tuning on FSDD, 3 seeds" + nl + nl;
if quick
    o = o + "> **QUICK DEBUG RUN (OCEAN_SPEECH_QUICK=1) -- tiny subsets, NOT a result.**" + nl + nl;
end
o = o + "Generated by `Experiments/Scripts/Speech/exp_speech_hubert_fsdd.m` (stages run as separate MATLAB processes, partly in parallel, " + ...
    "CPU shared with another training job) on " + string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm')) + " (MATLAB " + version('-release') + ...
    ", CPU, double). Every number below is computed by the script from real audio; none is a target." + nl + nl;
o = o + "Previous single-seed report (iteration 1 only, with full codebook/pretraining tables): `Results/Speech/hubert_fsdd_report_run1.md`." + nl + nl;
% ---- data
nLeak = nnz(U.isLeak);
o = o + "## Unlabeled pretraining data" + nl + nl;
frames = @(c) sum(cellfun(@(s) s.numFrames(), c));
o = o + mdTable(table(["FSDD train split (transcripts unused)"; "Audio Toolbox samples (1 s chunks)"; ...
    "  of which " + cfg.LeakageSample; "FSDD validation split (pretrain val)"], ...
    [numel(U.inFSDD); numel(U.inTool); nLeak; numel(U.inVal)], ...
    [frames(U.inFSDD); frames(U.inTool); frames(U.preIn(U.isLeak)); frames(U.inVal)], ...
    'VariableNames', ["Source", "Utterances", "Frames"])) + nl;
o = o + "Toolbox files from `" + U.toolRoot + "`: " + join(U.toolInfo.AudioPath, ", ") + ". The held-out test speaker `theo` is absent from all pretraining and fine-tuning data (asserted)." + nl + nl;
% ---- iteration 1
o = o + "## Iteration 1 (MFCC k-means targets)" + nl + nl;
e1 = ck1.History.Epochs(end);
o = o + sprintf(['Checkpoint `%s` (codebook `%s`, K = %d, %d steps). Final epoch: validation masked loss %.4f, masked accuracy %.4f ', ...
    '(chance %.4f). The script refits the iteration-1 codebook and reuses an existing checkpoint only if its codebook Version and stored ', ...
    'configuration equal the current ones. '], relpath(files.it1, repo), ...
    string(ck1.CodebookVersion), cfg.Codebook.NumUnits, ck1.Step, e1.ValLoss, e1.ValMaskedAcc, 1 / cfg.Codebook.NumUnits);
if isfield(ck1, 'Extra') && isfield(ck1.Extra, 'Seconds')
    o = o + sprintf(['In this run it was RETRAINED (%.1f min): the previous checkpoint''s codebook Version differed (same frames and ', ...
        'Lloyd iterations; the Version hash also covers the feature-config metadata). The it1 fine-tuning arms, started in parallel, ', ...
        'loaded the previous run''s it1 weights before the retrain overwrote the file. '], ck1.Extra.Seconds / 60);
    [nSame, nTot] = compareWithPrevious(files.previous, ck1.History.Epochs);
    o = o + sprintf(['Reproduction check against the previous report''s per-epoch table: %d of %d printed values ', ...
        '(TrainLoss, TrainMaskedAcc, ValLoss, ValMaskedAcc, ValUnmaskedAcc; 4 significant digits) are identical.'], nSame, nTot);
else
    o = o + "It was REUSED from the previous run.";
end
o = o + nl + nl;
% ---- iteration 2
x = ck2.Extra;
o = o + "## Iteration 2 (re-clustered hidden layer)" + nl + nl;
o = o + sprintf(['Hidden features of Transformer layers %s of the iteration-1 model (eval mode, no masking, %d-dim, output frame rate 20 ms) ', ...
    'were computed for all %d pretraining and %d pretraining-validation utterances (%.0f s). Layer choice (`HiddenLayerSelector`): ', ...
    'for each layer, k-means (K = %d, k-means++ + Lloyd, raw features, no CMVN) is fitted on the pretraining frames with seeds %s; ', ...
    'Stability = NMI between the two seeds'' unit assignments of the held-out pretraining-VALIDATION frames (no labels, no test speaker). ', ...
    'Hsu et al. choose by phone purity/PNMI, which needs a phone alignment we do not have. Selected = argmax Stability.'], ...
    mat2str(cfg.Iter2.Layers), cfg.Model.Dim, numel(U.preIn), numel(U.inVal), x.FeatureSeconds, cfg.Iter2.NumUnits, mat2str(cfg.Iter2.Seeds)) + nl + nl;
o = o + mdTable(x.Selection) + nl;
o = o + "ValRelInertia = held-out mean squared distance to the assigned centroid / total variance of the validation frames; " + ...
    "NMIWithReference = NMI of the seed-0 units with the iteration-1 MFCC units on the validation frames (reported, not used for the choice)." + nl + nl;
o = o + sprintf(['**Selected layer %d.** Re-running the selection with the same seeds gives identical scores, layer and codebook Version: **%d**. ', ...
    'Iteration-2 codebook `%s`: %d/%d units used on the pretraining frames, usage entropy %.3f / %.3f bits (validation %.3f), ', ...
    'NMI with the iteration-1 units on validation %.3f.'], x.Layer, x.SelectionReproducible, x.Codebook.Version, ...
    x.Usage.UsedUnits, cfg.Iter2.NumUnits, x.Usage.Entropy, x.Usage.MaxEntropy, x.ValUsage.Entropy, x.NMIIt1It2Val) + nl + nl;
o = o + sprintf(['A NEW HuBERTModel (same architecture and init seed as iteration 1) was pretrained on these targets with the same data, ', ...
    'schedule and %d steps (%.1f min). Before training: validation masked loss %.4f, masked accuracy %.4f. ', ...
    'Masked accuracies of the two iterations are on different target sets and are not comparable with each other.'], ...
    ck2.Step, x.Seconds / 60, x.Val0.Loss, x.Val0.MaskedAccuracy) + nl + nl;
o = o + epochTable(ck2, "Iteration-2 pretraining (every ~3rd epoch):");
% ---- no-Counting
if hasNC
    n = ckN.Extra; eN = ckN.History.Epochs(end);
    o = o + "## Sensitivity: iteration 1 without the `Counting` sample" + nl + nl;
    o = o + sprintf(['The %d one-second chunks of `%s` (a speaker counting aloud, i.e. unlabeled digit words) were removed, ', ...
        'the iteration-1 codebook refitted on the remaining %d utterances (`%s`) and the model pretrained from the same init for %d steps ', ...
        '(%.1f min). Final validation masked accuracy %.4f (with Counting: %.4f; different codebooks, so only indicative).'], ...
        nLeak, cfg.LeakageSample, numel(ckN.TrainIds), string(ckN.CodebookVersion), ckN.Step, n.Seconds / 60, eN.ValMaskedAcc, e1.ValMaskedAcc) + nl + nl;
end
% ---- fine-tuning
ft = cfg.FineTune;
o = o + "## CTC fine-tuning on 10% labels, 3 seeds" + nl + nl;
o = o + sprintf(['Per seed s: %d single-digit train utterances drawn by RandStream(s) (%.0f%% of %d) + the same number of composed utterances ', ...
    '(compose seed 200+s); vocabulary fitted on those transcripts; CTCRecognizer(HuBERTModel encoder without head, head dropout %.2g, head seed s+20), ', ...
    'batch order / dropout seed s, AdamW lr %.3g warmup-cosine, clip %g, %d epochs, batch %d; best epoch by greedy WER on the ', ...
    'speaker-dependent validation split. Within a seed all arms are identical except the encoder weights. Test: held-out speaker `theo`, ', ...
    '%d single + %d composed utterances; greedy and prefix beam (width %d, no LM).'], ...
    R(1).NumLabeled, 100 * cfg.LabeledFraction, round(R(1).NumLabeled / cfg.LabeledFraction), ft.HeadDropout, ...
    ft.LR, ft.Clip, ft.Epochs, ft.BatchSize, R(1).TestSingle, R(1).TestComposed, cfg.BeamWidth) + nl + nl;
armOrder = ["it1", "it2", "scratch", "it1nocount"];
armName = struct('it1', "pretrained it1", 'it2', "pretrained it2", 'scratch', "scratch", 'it1nocount', "pretrained it1, no Counting");
rows = {};
for a = armOrder
    for s = sort(unique(seedsR))
        k = find(arms == a & seedsR == s);
        if isempty(k), continue; end
        r = R(k); e = r.Eval;
        rows(end + 1, :) = {armName.(a), s, e.single.greedy.WER, e.single.beam.WER, e.single.greedy.CER, e.single.beam.CER, ...
            e.composed.greedy.WER, e.composed.beam.WER, e.composed.greedy.CER, e.composed.beam.CER, r.BestEpoch, r.Seconds / 60}; %#ok<AGROW>
    end
end
cols = ["Arm", "Seed", "SingleWERgreedy", "SingleWERbeam", "SingleCERgreedy", "SingleCERbeam", ...
    "ComposedWERgreedy", "ComposedWERbeam", "ComposedCERgreedy", "ComposedCERbeam", "BestEpoch", "TrainMinutes"];
o = o + "Per seed:" + nl + nl + mdTable(cell2table(rows, 'VariableNames', cols)) + nl;
metric = @(r, part, dec, m) r.Eval.(part).(dec).(m);
specs = ["single", "greedy"; "single", "beam"; "composed", "greedy"; "composed", "beam"];
rows = {};
for a = armOrder
    k = find(arms == a);
    if isempty(k), continue; end
    row = {armName.(a), numel(k)};
    for q = 1:size(specs, 1)
        row{end + 1} = pm(arrayfun(@(i) metric(R(i), specs(q, 1), specs(q, 2), "WER"), k)); %#ok<AGROW>
        row{end + 1} = pm(arrayfun(@(i) metric(R(i), specs(q, 1), specs(q, 2), "CER"), k)); %#ok<AGROW>
    end
    rows(end + 1, :) = row; %#ok<AGROW>
end
mcols = ["Arm", "Seeds", "SingleGreedyWER", "SingleGreedyCER", "SingleBeamWER", "SingleBeamCER", ...
    "ComposedGreedyWER", "ComposedGreedyCER", "ComposedBeamWER", "ComposedBeamCER"];
o = o + "Mean ± sample std over seeds:" + nl + nl + mdTable(cell2table(rows, 'VariableNames', mcols)) + nl;
% paired differences (same seed)
pairsD = ["it1", "scratch"; "it2", "scratch"; "it2", "it1"; "it1nocount", "it1"; "it1nocount", "scratch"];
rows = {};
for p = 1:size(pairsD, 1)
    ss = intersect(seedsR(arms == pairsD(p, 1)), seedsR(arms == pairsD(p, 2)));
    if isempty(ss), continue; end
    row = {pairsD(p, 1) + " − " + pairsD(p, 2), numel(ss)};
    for q = 1:size(specs, 1)
        d = arrayfun(@(s) metric(R(arms == pairsD(p, 1) & seedsR == s), specs(q, 1), specs(q, 2), "WER") - ...
            metric(R(arms == pairsD(p, 2) & seedsR == s), specs(q, 1), specs(q, 2), "WER"), ss);
        row{end + 1} = pm(d) + sprintf(' [%s]', strjoin(compose('%+.3f', d), ' ')); %#ok<AGROW>
    end
    rows(end + 1, :) = row; %#ok<AGROW>
end
o = o + "Paired WER differences (same seed; negative = first arm better; per-seed values in brackets):" + nl + nl + ...
    mdTable(cell2table(rows, 'VariableNames', ["Difference", "Seeds", "SingleGreedy", "SingleBeam", "ComposedGreedy", "ComposedBeam"])) + nl;
% counting exclusion
kN = find(arms == "it1nocount");
if ~isempty(kN)
    rows = {};
    for k = kN
        s = R(k).Seed;
        for a = ["it1", "it1nocount", "scratch"]
            r = R(arms == a & seedsR == s);
            rows(end + 1, :) = {armName.(a), s, r.Eval.single.greedy.WER, r.Eval.single.beam.WER, r.Eval.composed.greedy.WER, ...
                r.Eval.composed.beam.WER, r.Eval.single.greedy.CER, r.Eval.composed.greedy.CER}; %#ok<AGROW>
        end
    end
    o = o + "### Counting-exclusion sensitivity" + nl + nl + mdTable(cell2table(rows, 'VariableNames', ...
        ["Arm", "Seed", "SingleWERgreedy", "SingleWERbeam", "ComposedWERgreedy", "ComposedWERbeam", "SingleCERgreedy", "ComposedCERgreedy"])) + nl;
end
% per-epoch validation WER curves
o = o + "Validation WER (greedy, speaker-dependent split) per epoch, every 5th epoch:" + nl + nl;
rows = {}; ep = unique([5:5:ft.Epochs, ft.Epochs]);
for k = 1:numel(R)
    E = struct2table(R(k).Epochs, 'AsArray', true);
    rows(end + 1, :) = [{armName.(R(k).Arm), R(k).Seed}, num2cell(E.ValWER(ep).')]; %#ok<AGROW>
end
o = o + mdTable(cell2table(rows, 'VariableNames', ["Arm", "Seed", "Epoch" + ep])) + nl;
% ---- previous run
if isfile(files.previous)
    txt = string(fileread(files.previous));
    a = strfind(txt, "Held-out speaker `theo`"); b = strfind(txt, "Per-epoch (subset/pretrained)");
    if ~isempty(a) && ~isempty(b)
        o = o + "## Previous run (single seed, iteration 1 only; labeled-subset seed 7, fine-tune seed 0)" + nl + nl + ...
            "Copied from `Results/Speech/hubert_fsdd_report_run1.md`. Different labeled subset and fine-tuning seed from every seed above; its `subset` rows are one more sample of the it1/scratch comparison." + nl + nl + ...
            extractBetween(txt, a(1), b(1) - 1);
    end
end
% ---- caveats
o = o + "## Caveats" + nl + nl + ...
    "* Scale: ~11 min of unlabeled audio, ~0.5 M parameters (Hsu et al.: 960 h, 95 M); 3 seeds and ONE held-out speaker, so the seed spread does not include speaker variability." + nl + ...
    "* Seeds vary the labeled subset and the fine-tuning randomness only; pretraining was run once per iteration (one pretraining seed), so pretraining variance is not measured." + nl + ...
    "* Most unlabeled audio is the labeled speakers saying the same ten words; the Counting check isolates only the toolbox digit words, not the FSDD-train in-domain overlap (which is the setting's premise: the same audio, labels withheld)." + nl + ...
    "* Iteration 2 keeps K = 50 (paper: 100 -> 500) and chooses the layer by a label-free stability criterion instead of phone purity." + nl + ...
    "* Deviations from HuBERT (contract D5): conv feature encoder on log-mel, fixed sinusoidal positions, no masking during fine-tuning, encoder not frozen." + nl;
fid = fopen(files.report, 'w'); fprintf(fid, '%s', o); fclose(fid);
end

function [nSame, nTot] = compareWithPrevious(prevFile, epochs)
% Compare the per-epoch pretraining table printed in the previous report with EPOCHS (as printed).
nSame = 0; nTot = 0;
if ~isfile(prevFile), return; end
txt = string(fileread(prevFile));
blk = extractBetween(txt, "| Epoch | TrainLoss | TrainMaskedAcc", "Step loss curve");
if isempty(blk), return; end
lines = splitlines(blk(1));
fields = ["TrainLoss", "TrainMaskedAcc", "ValLoss", "ValMaskedAcc", "ValUnmaskedAcc"];
for ln = lines(startsWith(lines, "| ") & ~startsWith(lines, "| ---")).'
    c = strtrim(split(ln, "|"));
    c = c(2:end - 1);
    e = str2double(c(1));
    if isnan(e) || e > numel(epochs), continue; end
    for j = 1:numel(fields)
        nTot = nTot + 1;
        nSame = nSame + (string(sprintf('%.4g', epochs(e).(fields(j)))) == c(j + 1));
    end
end
end

function p = relpath(f, repo)
p = replace(erase(string(f), string(repo) + filesep), "\", "/");
end
