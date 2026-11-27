%% Speech front end on real speech: time, frequency and time-frequency tests
%  Tests every component of OceanNLP.Speech.Core and OceanNLP.Speech.Features
%  on the Free Spoken Digit Dataset (FSDD, 8 kHz, 4 speakers, 2000 recordings,
%  Data/Datasets/Speech/FSDD) and draws the results with OceanViz.Core.
%
%  Components under test
%    Core     : Waveform, FrameGrid, FeatureSequence, IFeatureTransform
%    Features : LogMel, MFCC, DeltaFeatures, EnergyContour, PitchContour,
%               FeaturePipeline (+ private prepareWaveform / gridArgs through
%               the input policy and fromConfig round trips)
%
%  What is checked, and against what (every reference is computed independently
%  of the class under test, from the formulas in the class headers):
%    TIME       Waveform durations / segment offsets / concat; FrameGrid frame
%               count, frame starts and centre times; raw frames == x(s_t+1:s_t+L);
%               EnergyContour == ln sum x_t^2; DeltaFeatures == HTK regression
%               (conv2 with edge replication); one shared time axis for all features.
%    FREQUENCY  FrameGrid.powerSpectrum == |fft(w .* x_t, NFFT)|^2 (built-in fft);
%               Parseval: windowed energy == (1/NFFT) sum over the full spectrum;
%               mel centres equally spaced on 2595 log10(1 + f/700); LogMel ==
%               ln(W |X|^2); MFCC == built-in dct (orthonormal DCT-II); lifter;
%               c0 == sqrt(M) mean log-mel; known-answer 1 kHz tone calibration.
%    PITCH      YIN F0 vs Audio Toolbox pitch() (NCF), and F0 harmonics visible
%               as peaks in a narrow-band spectrum (cross-domain check).
%    USE        Nearest-centroid speaker identification from utterance-level
%               MFCC / log-mel statistics (train takes 0-29, test takes 30-49).
%
%  Outputs: a PASS/FAIL table in the Command Window and four figures saved as
%  PNG in ./figures next to this script.
%
%  Run:  run('Experiments/Scripts/NLP/Signal/exp_speech_features_core_signal.m')

clear; close all; clc;

%% 0. Paths, data, plotting engine
scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(fileparts(fileparts(fileparts(scriptDir))));
addpath(repoRoot);
outDir = fullfile(scriptDir, 'figures');
if ~isfolder(outDir), mkdir(outDir); end

fsddRoot = fullfile(repoRoot, 'Data', 'Datasets', 'Speech', 'FSDD');
corpus   = listFsdd(fsddRoot);
fprintf('FSDD: %d recordings, speakers: %s\n', height(corpus), strjoin(unique(corpus.Speaker), ', '));

viz = OceanViz.Core('Theme', 'Dark');
U   = OceanViz.Strategies.Speech.SpeechVizUtils;
C   = viz.Colors;
checks = struct('Section', {}, 'Name', {}, 'Pass', {}, 'Detail', {});
tolExact = 1e-9;     % relative tolerance for "same formula, different code" comparisons

%% 1. Core.Waveform (time domain container)
sec = "Waveform";
demoFiles = "recordings/" + ["1", "2", "3"] + "_jackson_0.wav";   % "one two three"
parts = OceanNLP.Speech.Core.Waveform.empty(1, 0);
for f = demoFiles
    parts(end + 1) = OceanNLP.Speech.Core.Waveform.read(f, 'Root', fsddRoot); %#ok<SAGROW>
end
w1 = parts(1);
fs = w1.SampleRate;
S1 = w1.numSamples();
info = audioinfo(fullfile(fsddRoot, demoFiles(1)));
checks = addCheck(checks, sec, "read: rate and length match audioinfo", ...
    fs == info.SampleRate && S1 == info.TotalSamples, sprintf('fs = %g Hz, S = %d', fs, S1));
tw = w1.times();
checks = addCheck(checks, sec, "duration = S / fs and times(end) = (S-1)/fs", ...
    abs(w1.duration() - S1 / fs) < eps && abs(tw(end) - (S1 - 1) / fs) < eps, ...
    sprintf('%.4f s', w1.duration()));

seg = w1.segment([0.10 0.30], 'Units', "seconds");
nExp = round(0.30 * fs) - round(0.10 * fs);
checks = addCheck(checks, sec, "segment [0.1,0.3) s: length and SourceOffset", ...
    seg.numSamples() == nExp && abs(seg.SourceOffset - round(0.10 * fs) / fs) < eps, ...
    sprintf('%d samples, offset %.4f s', seg.numSamples(), seg.SourceOffset));
rng1 = OceanNLP.Speech.Core.Waveform.read(demoFiles(1), 'Root', fsddRoot, 'Range', [0.10 0.30], 'Units', "seconds");
checks = addCheck(checks, sec, "read('Range') == read + segment", ...
    isequal(rng1.Samples, seg.Samples) && rng1.SourceOffset == seg.SourceOffset, "identical samples");

w = OceanNLP.Speech.Core.Waveform.concat(parts(1), parts(2), parts(3));
checks = addCheck(checks, sec, "concat: length = sum of parts, Source joined", ...
    w.numSamples() == sum(arrayfun(@(p) p.numSamples(), parts)) && count(w.Source, "+") == 2, ...
    sprintf('%.3f s demo utterance "one two three"', w.duration()));

n2 = min(parts(1).numSamples(), parts(2).numSamples());
stereo = OceanNLP.Speech.Core.Waveform([parts(1).Samples(1:n2), parts(2).Samples(1:n2)], fs);
mono = stereo.toMono("mean");
checks = addCheck(checks, sec, "toMono('mean') averages channels and records History", ...
    max(abs(mono.Samples - mean(stereo.Samples, 2))) == 0 && ~isempty(mono.History), mono.History(end));
checks = addCheck(checks, sec, "1 x S row rejected (orientation contract)", ...
    errorId(@() OceanNLP.Speech.Core.Waveform(w1.Samples', fs)) == "OceanNLP:Speech:Core:Waveform:Orientation", "");

x = w.Samples;
S = numel(x);

%% 2. Core.FrameGrid (time axis + short-time spectrum)
sec = "FrameGrid";
g = OceanNLP.Speech.Core.FrameGrid(fs);                  % 25 ms / 10 ms defaults
L = g.WindowLength; H = g.HopLength; NFFT = g.NFFT;
checks = addCheck(checks, sec, "defaults: L = 25 ms, H = 10 ms, NFFT = 2^nextpow2(L)", ...
    L == round(0.025 * fs) && H == round(0.010 * fs) && NFFT == 2 ^ nextpow2(L), ...
    sprintf('L = %d, H = %d, NFFT = %d', L, H, NFFT));

T = g.numFrames(S);
starts = g.frameStarts(S);
checks = addCheck(checks, sec, "T = 1 + floor((S - L)/H), s_t = (t-1)H", ...
    T == 1 + floor((S - L) / H) && isequal(starts, (0:T-1) * H), sprintf('T = %d frames', T));
checks = addCheck(checks, sec, "frame centre c_t = (s_t + (L-1)/2) / fs", ...
    max(abs(g.frameTimes(S) - (starts + (L - 1) / 2) / fs)) < eps, ...
    sprintf('c_1 = %.4f s, hop = %.1f ms', (L - 1) / 2 / fs, 1000 * H / fs));

frames = g.frame(x);
tProbe = [1, ceil(T / 2), T];
okFrames = all(arrayfun(@(t) isequal(frames(:, t), x(starts(t) + (1:L))), tProbe));
checks = addCheck(checks, sec, "raw frame t == x(s_t+1 : s_t+L)", okFrames, "frames 1, T/2, T");

gc = OceanNLP.Speech.Core.FrameGrid(fs, 'Center', true);
P0 = floor(L / 2);
tc = gc.frameTimes(S);
checks = addCheck(checks, sec, "Center=true: T = 1 + floor((S + 2P - L)/H), c_1 = ((L-1)/2 - P)/fs", ...
    gc.numFrames(S) == 1 + floor((S + 2 * P0 - L) / H) && ...
    abs(tc(1) - ((L - 1) / 2 - P0) / fs) < eps, ...
    sprintf('T = %d (vs %d uncentred)', gc.numFrames(S), T));

win = g.window();
P = g.powerSpectrum(x);                                   % K x T, |X_t[k]|^2
Xref = fft(frames .* hann(L, 'periodic'), NFFT);           % built-in reference
K = floor(NFFT / 2) + 1;
Pref = abs(Xref(1:K, :)) .^ 2;
checks = addCheck(checks, sec, "window == hann(L,'periodic')", max(abs(win - hann(L, 'periodic'))) < 1e-15, "");
checks = addCheck(checks, sec, "powerSpectrum == |fft(w .* x_t, NFFT)|^2 (built-in fft)", ...
    relErr(P, Pref) < tolExact, sprintf('rel. err %.1e', relErr(P, Pref)));
fk = g.binFrequencies();
checks = addCheck(checks, sec, "bin frequencies f_k = k fs / NFFT", ...
    isequal(fk, (0:K-1)' * fs / NFFT), sprintf('df = %.2f Hz, K = %d bins', fs / NFFT, K));

% Parseval: sum_n (w x_t)^2 = (1/NFFT) sum_{k=0}^{NFFT-1} |X[k]|^2, one-sided weights
wk = 2 * ones(K, 1); wk(1) = 1; if mod(NFFT, 2) == 0, wk(end) = 1; end
eTime = sum((frames .* win) .^ 2, 1);
eFreq = (wk' * P) / NFFT;
checks = addCheck(checks, sec, "Parseval: time-domain energy == spectral energy / NFFT", ...
    relErr(eTime, eFreq) < tolExact, sprintf('rel. err %.1e', relErr(eTime, eFreq)));

%% 3. Core.FeatureSequence + IFeatureTransform contract
sec = "FeatureSequence";
lm   = OceanNLP.Speech.Features.LogMel(fs);
mf   = OceanNLP.Speech.Features.MFCC(fs);
dl   = OceanNLP.Speech.Features.DeltaFeatures();
en   = OceanNLP.Speech.Features.EnergyContour(fs);
pc   = OceanNLP.Speech.Features.PitchContour(fs);
transforms = {lm, mf, dl, en, pc};
checks = addCheck(checks, sec, "every feature class is an IFeatureTransform", ...
    all(cellfun(@(t) isa(t, 'OceanNLP.Speech.Core.IFeatureTransform'), transforms)), "5 classes");

Ylm = lm.forward(w);
checks = addCheck(checks, sec, "toCBT is F x 1 x T", isequal(size(Ylm.toCBT()), [lm.NumMels, 1, T]), ...
    mat2str(size(Ylm.toCBT())));
checks = addCheck(checks, sec, "derive rejects a different frame count", ...
    errorId(@() Ylm.derive(Ylm.Values(:, 1:end-1))) == "OceanNLP:Speech:Core:FeatureSequence:FrameTimes", "");
checks = addCheck(checks, sec, "Source provenance carried from the Waveform", Ylm.Source == w.Source, Ylm.Source);

%% 4. LogMel (frequency: mel filter bank + log energies)
sec = "LogMel";
st = lm.intermediates(w);
checks = addCheck(checks, sec, "forward == intermediates.LogMel, axes [mel,time]", ...
    isequal(Ylm.Values, st.LogMel) && isequal(Ylm.AxisNames, ["mel", "time"]), ...
    sprintf('%d x %d', size(Ylm.Values)));
checks = addCheck(checks, sec, "FrameTimes == FrameGrid.frameTimes", isequal(Ylm.FrameTimes, g.frameTimes(S)), "");

mel  = @(f) 2595 * log10(1 + f / 700);
imel = @(m) 700 * (10 .^ (m / 2595) - 1);
cRef = imel(linspace(mel(0), mel(fs / 2), lm.NumMels + 2))';
cRef = cRef(2:end-1);
checks = addCheck(checks, sec, "mel centres equally spaced on 2595 log10(1 + f/700)", ...
    max(abs(lm.CentersHz - cRef)) < 1e-6, sprintf('%.1f ... %.1f Hz', cRef(1), cRef(end)));
[~, pk] = max(lm.Filters, [], 2);
checks = addCheck(checks, sec, "each triangle peaks within one bin of its centre", ...
    all(abs(fk(pk) - lm.CentersHz) <= fs / NFFT), "");

LMref = log(max(lm.Filters * Pref, lm.LogFloor));
checks = addCheck(checks, sec, "LogMel == ln(max(W |fft|^2, floor))", relErr(Ylm.Values, LMref) < tolExact, ...
    sprintf('rel. err %.1e', relErr(Ylm.Values, LMref)));

% Known-answer frequency calibration: a 1 kHz tone must land in the 1 kHz band.
fTone = 1000;
tone = OceanNLP.Speech.Core.Waveform(sin(2 * pi * fTone * (0:fs/2 - 1)' / fs), fs);
[~, mTone] = max(mean(lm.forward(tone).Values, 2));
[~, mNear] = min(abs(lm.CentersHz - fTone));
[~, kTone] = max(mean(g.powerSpectrum(tone.Samples), 2));
checks = addCheck(checks, sec, "1 kHz tone -> FFT peak bin and mel band at 1 kHz", ...
    mTone == mNear && abs(fk(kTone) - fTone) <= fs / NFFT / 2, ...
    sprintf('peak bin %.1f Hz, band centre %.1f Hz', fk(kTone), lm.CentersHz(mTone)));

checks = addCheck(checks, sec, "rate mismatch rejected (no silent resampling)", ...
    errorId(@() OceanNLP.Speech.Features.LogMel(16000).forward(w)) == ...
    "OceanNLP:Speech:Features:LogMel:SampleRateMismatch", "");
checks = addCheck(checks, sec, "stereo input mixed to mono (recorded)", ...
    relErr(lm.forward(stereo).Values, lm.forward(mono).Values) == 0, "toMono('mean')");

%% 5. MFCC (cepstrum of the log-mel spectrum)
sec = "MFCC";
Ymf = mf.forward(w);
M = lm.NumMels; Q = mf.NumCoeffs;
Dref = dct(Ylm.Values);                                     % built-in orthonormal DCT-II
checks = addCheck(checks, sec, "MFCC == dct(LogMel)(1:13,:) (built-in DCT-II)", ...
    relErr(Ymf.Values, Dref(1:Q, :)) < tolExact, sprintf('rel. err %.1e', relErr(Ymf.Values, Dref(1:Q, :))));
checks = addCheck(checks, sec, "c0 == sqrt(M) * mean log-mel", ...
    relErr(Ymf.Values(1, :), sqrt(M) * mean(Ylm.Values, 1)) < tolExact, "");
mfL = OceanNLP.Speech.Features.MFCC(fs, 'Lifter', 22);
ratio = mfL.forward(w).Values ./ Ymf.Values;
liftRef = 1 + 11 * sin(pi * (1:Q)' / 22);
checks = addCheck(checks, sec, "lifter 22 scales row k by 1 + 11 sin(pi k / 22)", ...
    max(abs(median(ratio, 2) - liftRef)) < 1e-9, sprintf('row gains %.2f ... %.2f', liftRef(1), liftRef(end)));
checks = addCheck(checks, sec, "same time axis as LogMel, no frequency coordinate", ...
    isequal(Ymf.FrameTimes, Ylm.FrameTimes) && isempty(Ymf.FrequencyHz), "");

%% 6. DeltaFeatures (time derivatives on the frame axis)
sec = "DeltaFeatures";
Yd = dl.forward(Ymf);
N = dl.Width;
dRef1 = htkDelta(Ymf.Values, N);
dRef2 = htkDelta(dRef1, N);
checks = addCheck(checks, sec, "[c; dc; ddc] == HTK regression (conv2, edge replication)", ...
    relErr(Yd.Values, [Ymf.Values; dRef1; dRef2]) < tolExact, sprintf('%d x %d', size(Yd.Values)));
checks = addCheck(checks, sec, "outputDim(13) = 39, FrameTimes unchanged", ...
    dl.outputDim(Q) == 39 && isequal(Yd.FrameTimes, Ymf.FrameTimes), "");
checks = addCheck(checks, sec, "FrequencyHz repeated per block for log-mel input", ...
    isequal(dl.forward(Ylm).FrequencyHz, repmat(lm.CentersHz, 3, 1)), "");
checks = addCheck(checks, sec, "Waveform input rejected (':InputType')", ...
    errorId(@() dl.forward(w)) == "OceanNLP:Speech:Features:DeltaFeatures:InputType", "");

%% 7. EnergyContour (time-domain energy, cross-checked in frequency)
sec = "EnergyContour";
Ye = en.forward(w);
checks = addCheck(checks, sec, "E_t == ln sum x_t[n]^2 (raw frames, HTK)", ...
    relErr(Ye.Values, log(sum(frames .^ 2, 1))) < tolExact, "");
enW = OceanNLP.Speech.Features.EnergyContour(fs, 'ApplyWindow', true);
YeW = enW.forward(w);
checks = addCheck(checks, sec, "windowed energy == ln((1/NFFT) sum_k |X_t[k]|^2) (Parseval)", ...
    relErr(YeW.Values, log(eFreq)) < tolExact, "time domain == frequency domain");
checks = addCheck(checks, sec, "same FrameTimes as LogMel", isequal(Ye.FrameTimes, Ylm.FrameTimes), "");

%% 8. PitchContour (YIN F0, compared in time and frequency)
sec = "PitchContour";
Yp = pc.forward(w);
f0 = Yp.Values(1, :); vs = Yp.Values(2, :);
voiced = f0 > 0;
checks = addCheck(checks, sec, "2 x T on the shared frame grid", ...
    isequal(size(Yp.Values), [2, T]) && isequal(Yp.FrameTimes, Ylm.FrameTimes), sprintf('%d voiced of %d frames', nnz(voiced), T));
checks = addCheck(checks, sec, "voiced F0 in [MinF0, MaxF0], strength in [0,1]", ...
    all(f0(voiced) >= pc.MinF0 & f0(voiced) <= pc.MaxF0) && all(vs >= 0 & vs <= 1), ...
    sprintf('median F0 %.1f Hz', median(f0(voiced))));

% Independent time-domain estimator: Audio Toolbox pitch() (normalised correlation),
% same L, H and search range, no median smoothing.
f0Ref = pitch(x, fs, 'WindowLength', L, 'OverlapLength', L - H, 'Range', [pc.MinF0 pc.MaxF0], ...
    'Method', "NCF", 'MedianFilterLength', 1)';
nCmp = min(numel(f0Ref), T);
both = voiced(1:nCmp);
relDev = abs(f0(both) - f0Ref(both)) ./ f0Ref(both);
gpe = mean(relDev > 0.20);
% A 25 ms window holds only ~2.5 periods of a 100 Hz voice, so the NCF reference
% makes octave errors. Classify each gross error (|dev| > 20 %) instead of
% treating the reference as truth:
%   ref octave  : f0Ref ~ 2 f0 or f0 / 2 (|log2 ratio| within 0.2 of 1)
%   ref ceiling : f0Ref stuck at the MaxF0 search limit
%   other       : neither -- an unexplained disagreement
gross = relDev > 0.20;
octave = abs(abs(log2(f0Ref(both) ./ f0(both))) - 1) < 0.2;
ceiling = f0Ref(both) >= pc.MaxF0 - 1e-6;
other = gross & ~octave & ~ceiling;
checks = addCheck(checks, sec, "vs Audio Toolbox pitch(): gross errors are octave/ceiling only (other <= 5 %)", ...
    mean(other) <= 0.05, sprintf(['GPE %.1f %% = octave %d + ref at %g Hz ceiling %d + other %d of %d frames; ', ...
    'median |dev| %.1f %%'], 100 * gpe, nnz(gross & octave), pc.MaxF0, nnz(gross & ceiling & ~octave), ...
    nnz(other), nnz(both), 100 * median(relDev)));
% YIN's own octave errors (no step 6 / no tracking across frames): voiced frames
% far below (subharmonic, period doubling) or far above (period halving) the
% utterance's median F0. Agreement with pitch() does not exclude these -- both
% estimators can jump an octave on the same frame.
f0v = f0(voiced);
sub = f0v < 0.6 * median(f0v);
sup = f0v > 1.6 * median(f0v);
checks = addCheck(checks, sec, "YIN octave jumps (F0 < 0.6x or > 1.6x median) <= 5 %", mean(sub | sup) <= 0.05, ...
    sprintf('down %d + up %d of %d voiced frames (%.1f %%)', nnz(sub), nnz(sup), numel(f0v), 100 * mean(sub | sup)));

% Frequency-domain check: in a narrow-band spectrum (64 ms), voiced frames show
% more power at h F0 than half-way between harmonics, (h + 1/2) F0.
gNB = OceanNLP.Speech.Core.FrameGrid(fs, 'WindowLength', 512, 'HopLength', H, 'NFFT', 1024);
PNB = gNB.powerSpectrum(x);
tNB = gNB.frameTimes(S);
fNB = gNB.binFrequencies();
hRatio = harmonicRatio(PNB, tNB, fNB, f0, Yp.FrameTimes, fs);
checks = addCheck(checks, sec, "narrow-band spectrum peaks at harmonics of F0 (>= 80 % of frames)", ...
    mean(hRatio > 1) >= 0.80, sprintf('%.1f %% of %d voiced frames, median ratio %.1f', ...
    100 * mean(hRatio > 1), numel(hRatio), median(hRatio)));

%% 9. FeaturePipeline + fromConfig round trips
sec = "FeaturePipeline";
pipe = OceanNLP.Speech.Features.FeaturePipeline({mf, dl});
Ypipe = pipe.forward(w);
checks = addCheck(checks, sec, "pipeline == stages applied by hand", isequal(Ypipe.Values, Yd.Values), "MFCC -> deltas");
checks = addCheck(checks, sec, "outputDim([]) = 39 and Config lists both stages", ...
    pipe.outputDim([]) == 39 && isequal([Ypipe.Config.Stages.Type], ["MFCC", "DeltaFeatures"]), "");
pipe2 = OceanNLP.Speech.Features.FeaturePipeline.fromConfig(pipe.config());
checks = addCheck(checks, sec, "fromConfig(config) rebuilds identical output", ...
    isequal(pipe2.forward(w).Values, Ypipe.Values), "");
rt = cellfun(@(t) isequaln(feval(class(t) + ".fromConfig", t.config()).config(), t.config()), transforms);
checks = addCheck(checks, sec, "fromConfig round trip for all 5 feature classes", all(rt), sprintf('%d / 5', nnz(rt)));

%% 10. Corpus-level analysis (200 recordings: 4 speakers x 10 digits x takes 0-4)
fprintf('\nCorpus pass over 200 recordings ...\n');
speakers = unique(corpus.Speaker)';
sub = corpus(corpus.Take <= 4, :);
ltas = zeros(K, numel(speakers));
f0All = cell(1, numel(speakers));
gpeAll = zeros(1, height(sub));
for i = 1:height(sub)
    wi = OceanNLP.Speech.Core.Waveform.read(sub.File(i), 'Root', fsddRoot);
    s = find(speakers == sub.Speaker(i));
    ltas(:, s) = ltas(:, s) + sum(g.powerSpectrum(wi.Samples), 2);
    pi_ = pc.forward(wi).Values(1, :);
    f0All{s} = [f0All{s}, pi_(pi_ > 0)];
    ref = pitch(wi.Samples, fs, 'WindowLength', L, 'OverlapLength', L - H, ...
        'Range', [pc.MinF0 pc.MaxF0], 'Method', "NCF", 'MedianFilterLength', 1)';
    n = min(numel(ref), numel(pi_)); v = pi_(1:n) > 0;
    gpeAll(i) = mean(abs(pi_(v) - ref(v)) ./ ref(v) > 0.20);
end
ltas = ltas ./ sum(ltas, 1);                               % normalised average spectrum
centroid = (fk' * ltas);
fprintf('  speaker  spectral centroid  median F0   (YIN vs pitch() GPE over corpus: %.1f %%)\n', ...
    100 * mean(gpeAll, 'omitnan'));
for s = 1:numel(speakers)
    fprintf('  %-9s %8.0f Hz        %6.1f Hz\n', speakers(s), centroid(s), median(f0All{s}));
end
checks = addCheck(checks, "PitchContour", "corpus mean GPE vs pitch() <= 20 %", ...
    mean(gpeAll, 'omitnan') <= 0.20, sprintf('%.1f %% over 200 recordings', 100 * mean(gpeAll, 'omitnan')));

%% 11. Do the features carry speaker information? (all 2000 recordings)
fprintf('Speaker identification from utterance statistics (2000 recordings) ...\n');
nF = height(corpus);
featMF = zeros(nF, 2 * Q); featLM = zeros(nF, 2 * M);
for i = 1:nF
    wi = OceanNLP.Speech.Core.Waveform.read(corpus.File(i), 'Root', fsddRoot);
    li = lm.intermediates(wi).LogMel;
    ci = dct(li); ci = ci(1:Q, :);                         % == mf.forward(wi), checked in sec. 5
    featLM(i, :) = [mean(li, 2); std(li, 0, 2)]';
    featMF(i, :) = [mean(ci, 2); std(ci, 0, 2)]';
end
train = corpus.Take <= 29; test = ~train;
accMF = nearestCentroid(featMF, corpus.Speaker, train, test);
accLM = nearestCentroid(featLM, corpus.Speaker, train, test);
checks = addCheck(checks, "MFCC", "speaker ID accuracy (MFCC mean+std) >> chance 25 %", accMF >= 0.60, ...
    sprintf('MFCC %.1f %%, log-mel %.1f %% (test takes 30-49, %d utts)', 100 * accMF, 100 * accLM, nnz(test)));

%% 12. Figures
% Figure 1 -- time domain
[~, tLoud] = max(Ye.Values);
fig1 = viz.createFigure('Speech features: time domain');
fig1.Position = [40 40 1500 950];
tl = tiledlayout(fig1, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
viz.plot(nexttile(tl), "SpeechWaveform", struct('Samples', x, 'SampleRate', fs, ...
    'FrameStarts', starts, 'WindowLength', L), 'Title', sprintf('Waveform "one two three" (%s, %g Hz)', "jackson", fs));
viz.plot(nexttile(tl), "SpeechFrame", struct('Frame', frames(:, tLoud), 'Window', win, 'SampleRate', fs, ...
    'FrameIndex', tLoud, 'FrameStart', starts(tLoud), 'WindowType', "hann"), ...
    'Title', sprintf('Loudest frame t = %d (%.3f s)', tLoud, Ylm.FrameTimes(tLoud)));
viz.plot(nexttile(tl), "SpeechContour", struct('X', Ye.FrameTimes, ...
    'Series', [lineSeries("ln energy, raw frame (HTK)", Ye.Values), ...
               lineSeries("ln energy, Hann-windowed", YeW.Values, "--"), ...
               lineSeries("ln((1/NFFT) sum |X_t[k]|^2) (Parseval)", log(eFreq), ":")], ...
    'XLabel', "Time (s), frame centres", 'YLabel', "Log energy (nats)", ...
    'Intervals', voicedIntervals(Yp.FrameTimes, voiced, H / fs), 'IntervalName', "YIN voiced frames"), ...
    'Title', "EnergyContour (time) vs spectral energy (frequency)");
viz.plot(nexttile(tl), "SpeechContour", struct('X', Yd.FrameTimes, ...
    'Series', [lineSeries("c1 (static)", Yd.Values(2, :)), ...
               lineSeries("delta c1", Yd.Values(Q + 2, :), "--"), ...
               lineSeries("delta-delta c1", Yd.Values(2 * Q + 2, :), ":")], ...
    'XLabel', "Time (s), frame centres", 'YLabel', "Cepstral value"), ...
    'Title', sprintf('DeltaFeatures on MFCC c1 (HTK, N = %d)', N));
viz.export(fig1, fullfile(outDir, 'speech_features_time'));

% Figure 2 -- frequency domain
fig2 = viz.createFigure('Speech features: frequency domain');
fig2.Position = [60 60 1500 950];
tl = tiledlayout(fig2, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
viz.plot(nexttile(tl), "SpeechContour", struct('X', fk', ...
    'Series', [lineSeries("FrameGrid.powerSpectrum", 10 * log10(P(:, tLoud) + eps)'), ...
               lineSeries("built-in fft reference", 10 * log10(Pref(:, tLoud) + eps)', "--")], ...
    'XLabel', "Frequency (Hz)", 'YLabel', "Power |X_t[k]|^2 (dB)"), ...
    'Title', sprintf('Spectrum of the loudest frame (df = %.1f Hz)', fs / NFFT));
ltasSeries = arrayfun(@(s) lineSeries(sprintf('%s (centroid %.0f Hz)', speakers(s), centroid(s)), ...
    10 * log10(ltas(:, s) + eps)'), 1:numel(speakers));
viz.plot(nexttile(tl), "SpeechContour", struct('X', fk', 'Series', ltasSeries, ...
    'XLabel', "Frequency (Hz)", 'YLabel', "Normalised long-term power (dB)"), ...
    'Title', "Long-term average spectrum per speaker (50 recordings each)");
viz.plot(nexttile(tl), "MelFilterBank", struct('Weights', lm.Filters, 'BinFrequencyHz', fk, ...
    'CentersHz', lm.CentersHz, 'Scale', lm.MelScale, 'Norm', lm.MelNorm), ...
    'Title', sprintf('LogMel filter bank (%d HTK mel bands, 0-%g Hz)', M, lm.FMax));
Csm = idct([Ymf.Values(:, tLoud); zeros(M - Q, 1)]);       % cepstral smoothing
viz.plot(nexttile(tl), "SpeechContour", struct('X', lm.CentersHz', ...
    'Series', [lineSeries("log-mel spectrum of the frame", Ylm.Values(:, tLoud)', "-", "o"), ...
               lineSeries(sprintf('inverse DCT of %d MFCCs (envelope)', Q), Csm', "--")], ...
    'XLabel', "Mel band centre (Hz)", 'YLabel', "ln mel energy"), ...
    'Title', "MFCC = smooth spectral envelope (cepstral smoothing)");
viz.export(fig2, fullfile(outDir, 'speech_features_frequency'));

% Figure 3 -- time-frequency
gWB = OceanNLP.Speech.Core.FrameGrid(fs, 'WindowLength', 40, 'HopLength', 16, 'NFFT', 256);
fig3 = viz.createFigure('Speech features: time-frequency');
fig3.Position = [80 80 1500 950];
tl = tiledlayout(fig3, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
viz.plot(nexttile(tl), "Spectrogram", struct('Values', gWB.powerSpectrum(x), 'FrequencyHz', gWB.binFrequencies(), ...
    'FrameTimes', gWB.frameTimes(S), 'Units', "power |X[k]|^2"), ...
    'Title', "Wide-band spectrogram (5 ms Hann): formants, onsets");
axNB = nexttile(tl);
viz.plot(axNB, "Spectrogram", struct('Values', PNB, 'FrequencyHz', fNB, 'FrameTimes', tNB, ...
    'Units', "power |X[k]|^2"), 'Title', "Narrow-band spectrogram (64 ms Hann) + YIN harmonics");
hold(axNB, 'on');
f0n = f0; f0n(~voiced) = NaN;
hH = gobjects(1, 3);
lsH = ["-", "--", ":"];
for h = 1:3
    hH(h) = plot(axNB, Yp.FrameTimes, h * f0n, lsH(h), 'Color', C.Text, 'LineWidth', 1.4, ...
        'DisplayName', sprintf('%d x F0 (PitchContour)', h));
end
hold(axNB, 'off');
U.legend(axNB, C, hH, 'Location', 'northeast');
viz.plot(nexttile(tl), "SpeechFeatures", OceanViz.Strategies.Speech.FeatureMatrixStrategy.fromFeatureSequence(Ylm), ...
    'Title', "LogMel (40 bands, 25 ms / 10 ms)");
viz.plot(nexttile(tl), "SpeechFeatures", OceanViz.Strategies.Speech.FeatureMatrixStrategy.fromFeatureSequence(Ypipe), ...
    'Title', "FeaturePipeline: MFCC 13 + delta + delta-delta (39 rows)");
viz.export(fig3, fullfile(outDir, 'speech_features_time_frequency'));

% Figure 4 -- pitch
fig4 = viz.createFigure('Speech features: pitch');
fig4.Position = [100 100 1500 520];
tl = tiledlayout(fig4, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
viz.plot(nexttile(tl), "PitchContour", OceanViz.Strategies.Speech.PitchContourStrategy.fromFeatureSequence(Yp, ...
    'VoicingThreshold', pc.VoicingThreshold), 'Title', "PitchContour (YIN) and voicing strength");
fr = f0Ref(1:nCmp); fr(~both) = NaN; fy = f0(1:nCmp); fy(~both) = NaN;
viz.plot(nexttile(tl), "SpeechContour", struct('X', Yp.FrameTimes(1:nCmp), ...
    'Series', [lineSeries("PitchContour (YIN)", fy, "-", "o"), lineSeries("Audio Toolbox pitch() (NCF)", fr, "--", "x")], ...
    'XLabel', "Time (s), frame centres", 'YLabel', "F0 (Hz), YIN-voiced frames"), ...
    'Title', sprintf('F0 vs independent estimator (GPE %.1f %%)', 100 * gpe));
edges = pc.MinF0:10:pc.MaxF0;
mids = edges(1:end-1) + 5;
histSeries = arrayfun(@(s) lineSeries(sprintf('%s (median %.0f Hz)', speakers(s), median(f0All{s})), ...
    histcounts(f0All{s}, edges, 'Normalization', 'probability')), 1:numel(speakers));
viz.plot(nexttile(tl), "SpeechContour", struct('X', mids, 'Series', histSeries, ...
    'XLabel', "F0 (Hz), 10 Hz bins", 'YLabel', "Fraction of voiced frames"), ...
    'Title', "F0 distribution per speaker (50 recordings each)");
viz.export(fig4, fullfile(outDir, 'speech_features_pitch'));

%% 13. Summary
nPass = nnz([checks.Pass]);
fprintf('\n%-16s %-4s %-74s %s\n', 'Section', '', 'Check', 'Detail');
for k = 1:numel(checks)
    tag = 'PASS'; if ~checks(k).Pass, tag = 'FAIL'; end
    fprintf('%-16s %-4s %-74s %s\n', checks(k).Section, tag, checks(k).Name, checks(k).Detail);
end
fprintf('\n%d / %d checks passed. Figures: %s\n', nPass, numel(checks), outDir);

%% Local functions
function T = listFsdd(root)
% LISTFSDD  Table of FSDD recordings: File (Root-relative), Digit, Speaker, Take.
d = dir(fullfile(root, 'recordings', '*.wav'));
tok = regexp({d.name}, '^(\d)_([a-z]+)_(\d+)\.wav$', 'tokens', 'once');
ok = ~cellfun(@isempty, tok);
tok = vertcat(tok{ok});
T = table("recordings/" + string({d(ok).name})', str2double(tok(:, 1)), string(tok(:, 2)), ...
    str2double(tok(:, 3)), 'VariableNames', ["File", "Digit", "Speaker", "Take"]);
end

function checks = addCheck(checks, section, name, pass, detail)
checks(end + 1) = struct('Section', string(section), 'Name', string(name), 'Pass', logical(pass), ...
    'Detail', string(detail));
tag = 'PASS'; if ~pass, tag = 'FAIL'; end
fprintf('[%s] %-16s %s  %s\n', tag, section, name, detail);
end

function id = errorId(f)
% ERRORID  Identifier of the error thrown by F ("" when it does not throw).
id = "";
try
    f();
catch err
    id = string(err.identifier);
end
end

function e = relErr(a, b)
e = max(abs(a(:) - b(:))) / max(max(abs(b(:))), realmin);
end

function d = htkDelta(c, N)
% HTKDELTA  d_t = sum_n n (c_{t+n} - c_{t-n}) / (2 sum n^2), edges replicated (via conv2).
T = size(c, 2);
if T == 0, d = c; return; end
cp = [repmat(c(:, 1), 1, N), c, repmat(c(:, end), 1, N)];
d = conv2(cp, (N:-1:-N) / (2 * sum((1:N) .^ 2)), 'valid');
end

function s = lineSeries(name, y, lineStyle, marker)
arguments
    name (1,1) string
    y (1,:) double
    lineStyle (1,1) string = "-"
    marker (1,1) string = "none"
end
s = struct('Name', name, 'Y', y, 'LineStyle', lineStyle, 'Marker', marker);
end

function iv = voicedIntervals(times, voiced, hop)
% VOICEDINTERVALS  [start end] (s) of runs of voiced frames, +-hop/2 around the centres.
d = diff([false, voiced, false]);
on = find(d == 1); off = find(d == -1) - 1;
iv = [times(on)' - hop / 2, times(off)' + hop / 2];
end

function r = harmonicRatio(PNB, tNB, fNB, f0, tF0, fs)
% HARMONICRATIO  Per voiced frame: power at h F0 / power at (h + 1/2) F0, h = 1..up to 8,
%   read from the narrow-band frame whose centre is nearest in time.
df = fNB(2) - fNB(1);
v = find(f0 > 0);
r = zeros(1, numel(v));
for i = 1:numel(v)
    [~, j] = min(abs(tNB - tF0(v(i))));
    hMax = min(8, floor((fs / 2 - f0(v(i))) / f0(v(i))));
    h = 1:max(hMax, 1);
    on = round(h * f0(v(i)) / df) + 1;
    off = round((h + 0.5) * f0(v(i)) / df) + 1;
    r(i) = sum(PNB(on, j)) / max(sum(PNB(off, j)), realmin);
end
end

function acc = nearestCentroid(X, y, train, test)
% NEARESTCENTROID  Accuracy of a z-scored nearest-class-mean classifier.
mu = mean(X(train, :), 1); sd = std(X(train, :), 0, 1); sd(sd == 0) = 1;
Z = (X - mu) ./ sd;
cls = unique(y(train));
cent = cell2mat(arrayfun(@(k) mean(Z(train & y == k, :), 1), cls, 'UniformOutput', false));
[~, idx] = min(pdist2(Z(test, :), cent), [], 2);
acc = mean(cls(idx) == y(test));
end
