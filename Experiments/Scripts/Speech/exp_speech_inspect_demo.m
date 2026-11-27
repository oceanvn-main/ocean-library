%% EXP_SPEECH_INSPECT_DEMO  Look inside the trained CTC recogniser on held-out speech.
%
%   Loads the trained checkpoint (Results/Speech/checkpoints/ctc_fsdd_best.mat, produced by
%   exp_speech_ctc_fsdd.m), takes REAL recordings of the held-out speaker "theo" (FSDD test
%   split) and runs OceanNLP.Speech.Inspect.SpeechProcessAdapter, which replays the real
%   pipeline stage by stage and returns the transcript (identical to model.transcribe) plus a
%   reader over every captured intermediate. Every panel below is drawn by an
%   OceanViz.Strategies.Speech strategy from those captured values.
%
%   Figures (Results/Speech/figures/):
%     inspect_frontend.png    waveform, power spectrogram, mel filter bank, log-mel
%     inspect_model.png       normalised input, encoder states, CTC posteriors, pitch
%     inspect_decoding.png    beam candidates + character alignment (single digit) and
%                             beam candidates + word alignment (composed multi-word utterance)
%
%   Consumer only: no library code is changed; nothing is trained.

repo = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(repo);
outDir = fullfile(repo, 'Results', 'Speech', 'figures');
if ~isfolder(outDir), mkdir(outDir); end

ck = OceanNLP.Speech.Training.SpeechCheckpoint.load(string(fullfile(repo, 'Results', 'Speech', 'checkpoints', 'ctc_fsdd_best.mat')));
model = ck.Model;
manifest = OceanNLP.Speech.Data.SpeechManifest.fromFSDD().canonicalFSDDSplit();
test = manifest.bySplit("test");
assert(all(test.Entries.Speaker == "theo"), 'Test split must be the held-out speaker.');
normalizer = model.Vocabulary.Normalizer;
viz = OceanViz.Core('Theme', "Dark");

%% 1. A single held-out digit, beam search
id = "6_theo_17";
one = test.subset(test.Entries.Id == id);
ds = OceanNLP.Speech.Data.AudioDataset(one, model.Features, 'SampleRate', model.Features.SampleRate);
wave = ds.waveform(1);
reference = normalizer.forward(one.Entries.Transcript(1));
[tr, reader] = OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(model, wave, 'Decoder', "beam", 'BeamWidth', 8);
check = model.transcribe(wave, 'Decoder', "beam", 'BeamWidth', 8);
assert(tr.Text == check.Text && isequal(tr.Candidates, check.Candidates), 'Adapter must equal transcribe.');
fprintf('%s: reference "%s", hypothesis "%s" (score %.3f)\n', id, reference, tr.Text, tr.Score);
steps = reader.listSteps();
fprintf('  %d captured steps: %s\n', numel(steps), strjoin([steps.Name], ' -> '));
art = @(name) reader.getArtifact(steps([steps.Name] == name).OutputIds(1)).Value;

%% 2. Front end
fig = viz.createFigure("Speech front end");
fig.Position = [50 50 1500 900];
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
w = art("Waveform");
fr = art("Frames + window");
w.FrameStarts = fr.FrameStarts; w.WindowLength = fr.WindowLength;
w.Title = "Waveform " + id + " (held-out speaker)";
viz.plot(nexttile(tl), "SpeechWaveform", w);
viz.plot(nexttile(tl), "Spectrogram", art("Power spectrum"));
viz.plot(nexttile(tl), "MelFilterBank", art("Mel filter bank"));
viz.plot(nexttile(tl), "SpeechFeatures", art("Log-mel"));
viz.export(fig, fullfile(outDir, "inspect_frontend"));

%% 3. Model internals
fig = viz.createFigure("Speech model internals");
fig.Position = [50 50 1500 900];
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
viz.plot(nexttile(tl), "SpeechFeatures", art("Normalised input"));
encNames = [steps(startsWith([steps.Name], "Encoder")).Name];
viz.plot(nexttile(tl), "SpeechFeatures", art(encNames(end)));
post = art("Posteriors");
post.Title = sprintf("CTC posteriors -> ""%s"" (reference ""%s"")", tr.Text, reference);
viz.plot(nexttile(tl), "CTCPosterior", post);
pc = OceanNLP.Speech.Features.PitchContour(wave.SampleRate, 'WindowLength', fr.WindowLength, ...
    'HopLength', fr.HopLength, 'NFFT', fr.NFFT);
viz.plot(nexttile(tl), "PitchContour", OceanViz.Strategies.Speech.PitchContourStrategy.fromFeatureSequence( ...
    pc.forward(wave), 'VoicingThreshold', pc.VoicingThreshold));
viz.export(fig, fullfile(outDir, "inspect_model"));

%% 4. Decoding: single digit (characters) and a composed multi-word utterance (words)
composed = OceanNLP.Speech.Data.UtteranceComposer.compose(test, 'MinWords', 3, 'MaxWords', 5, ...
    'Count', 1, 'Seed', 103);
cds = OceanNLP.Speech.Data.AudioDataset(composed, model.Features, 'SampleRate', model.Features.SampleRate);
cwave = cds.waveform(1);
cref = normalizer.forward(composed.Entries.Transcript(1));
[ctr, creader] = OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(model, cwave, 'Decoder', "beam", 'BeamWidth', 8);
fprintf('%s: reference "%s", hypothesis "%s"\n', composed.Entries.Id(1), cref, ctr.Text);
csteps = creader.listSteps();
cbeam = creader.getArtifact(csteps([csteps.Name] == "Beam candidates").OutputIds(1)).Value;
cbeam.Reference = cref;

fig = viz.createFigure("Speech decoding");
fig.Position = [50 50 1500 900];
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
beam = art("Beam candidates");
beam.Reference = reference;
viz.plot(nexttile(tl), "BeamCandidates", beam);
refC = string(num2cell(char(reference)));
hypC = string(num2cell(char(tr.Text)));
if tr.Text == "", hypC = strings(1, 0); end
viz.plot(nexttile(tl), "ErrorAlignment", struct('Alignment', ...
    OceanNLP.Speech.Evaluation.EditAlignment.align(refC, hypC), 'Reference', refC, ...
    'Hypothesis', hypC, 'Unit', "character"));
viz.plot(nexttile(tl), "BeamCandidates", cbeam);
refW = split(cref, " ")'; hypW = split(ctr.Text, " ")';
refW = refW(refW ~= ""); hypW = hypW(hypW ~= "");
viz.plot(nexttile(tl), "ErrorAlignment", struct('Alignment', ...
    OceanNLP.Speech.Evaluation.EditAlignment.align(refW, hypW), 'Reference', refW, ...
    'Hypothesis', hypW, 'Unit', "word"));
viz.export(fig, fullfile(outDir, "inspect_decoding"));
close all
