%% validate_features_external.m -- external cross-checks of the Ocean speech feature stack
%
% Compares the native Ocean implementations (OceanMath.Signal.*, OceanNLP.Speech.Features.*)
% on REAL audio against independent references:
%   * librosa / scipy (Python, validate_features_external.py next to this file)
%   * MATLAB Audio/Signal Toolbox: designAuditoryFilterBank, melSpectrogram, mfcc,
%     audioDelta, pitch, resample (used ONLY here, never inside the Ocean algorithms)
% and writes Results/Speech/feature_validation.md.
%
% DATA: FSDD (CC BY-SA 4.0) -- take 0 of every digit for every speaker (40 files,
% deterministic selection rule, no hand-picking) -- plus the MATLAB Audio Toolbox
% speech samples SpeechDFT-16-8-mono-5secs, FemaleSpeech-16-8-mono-3secs (8 kHz) and
% Counting-16-44p1-mono-15secs (44.1 kHz).
%
% ASSUMPTIONS (stated, not hidden):
%   * grid at every rate: L = round(0.025 fs), H = round(0.010 fs), NFFT = 2^nextpow2(L),
%     periodic Hann, center = false; 40 HTK mels, no filter normalisation, natural log,
%     floor 1e-10; 13 orthonormal DCT-II MFCCs; delta width N = 2.
%   * pitch gross error = |F0 - F0_ref| / F0_ref > 20 % (the conventional GPE threshold),
%     evaluated only on frames that both estimators call voiced (MATLAB `pitch` has no
%     voicing decision, so there: frames Ocean calls voiced). No pass/fail tolerance is
%     invented for pitch -- the rates are reported.
%
% Run from anywhere:  run('Experiments/Scripts/Speech/validate_features_external.m')

%% 0. Paths (derived from this file's location; no developer-specific paths)
repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(repoRoot);
fsddRec = fullfile(repoRoot, 'Data', 'Datasets', 'Speech', 'FSDD', 'recordings');
samplesDir = fullfile(matlabroot, 'toolbox', 'audio', 'samples');
python = fullfile(repoRoot, '.venv', 'Scripts', 'python.exe');
helper = fullfile(fileparts(mfilename('fullpath')), 'validate_features_external.py');
work = fullfile(tempdir, 'ocean_speech_feature_validation');
reportFile = fullfile(repoRoot, 'Results', 'Speech', 'feature_validation.md');
assert(isfile(python), 'Python venv not found at %s', python);
assert(isfolder(fsddRec), 'FSDD recordings not found at %s', fsddRec);

%% 1. File selection (rule: take 0 of every digit x speaker; toolbox speech samples)
speakers = ["jackson", "nicolas", "theo", "yweweler"];
items = struct('key', {}, 'path', {}, 'label', {});
for s = speakers
    for d = 0:9
        f = fullfile(fsddRec, sprintf('%d_%s_0.wav', d, s));
        items(end + 1) = struct('key', sprintf('fsdd_%d_%s', d, s), 'path', f, 'label', "FSDD"); %#ok<SAGROW>
    end
end
for nm = ["SpeechDFT-16-8-mono-5secs.wav", "FemaleSpeech-16-8-mono-3secs.wav", "Counting-16-44p1-mono-15secs.wav"]
    items(end + 1) = struct('key', char(erase(extractBefore(nm, "-"), "-")), 'path', fullfile(samplesDir, nm), ...
        'label', "MATLAB sample " + nm); %#ok<SAGROW>
end
resJobs = struct('key', {'res_counting_44k_8k', 'res_fsdd_8k_16k'}, ...
    'path', {fullfile(samplesDir, 'Counting-16-44p1-mono-15secs.wav'), fullfile(fsddRec, '0_jackson_0.wav')}, ...
    'fs_out', {8000, 16000}, 'half_width', {16, 16}, 'beta', {8.6, 8.6});

%% 2. Python references (librosa / scipy)
if isfolder(work)
    rmdir(work, 's');
end
mkdir(work);
job = struct('out', work, 'files', rmfield(items, 'label'), 'resample', resJobs, 'n_mels', 40, ...
    'fmin', 50, 'fmax', 400, 'pitch', true);
jobFile = fullfile(work, 'job.json');
fid = fopen(jobFile, 'w');
fwrite(fid, jsonencode(job), 'char');
fclose(fid);
[status, out] = system(sprintf('"%s" "%s" "%s"', python, helper, jobFile));
if status ~= 0
    error('validate_features_external:Python', 'Python reference step failed:\n%s', out);
end
npy = @(name) double(OceanData.Adapters.NpyReader.read(fullfile(work, [name '.npy'])));
versions = jsondecode(fileread(fullfile(work, 'versions.json')));

%% 3. Per-file comparisons
nI = numel(items);
R = table('Size', [nI 15], 'VariableTypes', [repmat("double", 1, 14), "string"], 'VariableNames', ...
    ["fs", "T", "melbank_librosa", "melbank_matlab", "logmel_librosa", "melpow_rel_librosa", ...
     "mfcc_librosa", "d1_librosa", "d2_librosa", "logmel_matlab", "cepstral_matlab_ln10", "mfcc_matlab_ln10", ...
     "audioDelta_interior", "pitch_frames", "key"]);
pitchRows = struct('key', {}, 'nOceanVoiced', {}, 'nPyinVoiced', {}, 'nBoth', {}, 'gerPyin', {}, ...
    'gerMatlab', {}, 'voicingAgree', {}, 'medianRelErrPyin', {});
for i = 1:nI
    it = items(i);
    w = OceanNLP.Speech.Core.Waveform.read(string(it.path)).toMono();
    fs = w.SampleRate;
    x = w.Samples;
    lmT = OceanNLP.Speech.Features.LogMel(fs, 'NumMels', 40);
    g = lmT.Grid;
    L = g.WindowLength; H = g.HopLength; nfft = g.NFFT;
    st = lmT.intermediates(w);
    lm = lmT.forward(w);
    mf = OceanNLP.Speech.Features.MFCC(fs, 'NumMels', 40, 'NumCoeffs', 13).forward(w);
    dd = OceanNLP.Speech.Features.DeltaFeatures('Width', 2, 'Order', 2).forward(mf);
    F = 13;
    k = it.key;
    % --- librosa
    refMel = npy([k '_melpow']);
    big = refMel > 1e-8 * max(refMel(:));
    R.fs(i) = fs;
    R.T(i) = lm.numFrames();
    R.key(i) = string(k);
    R.melbank_librosa(i) = max(abs(lmT.Filters - npy([k '_melbank'])), [], 'all');
    R.logmel_librosa(i) = max(abs(lm.Values - npy([k '_logmel'])), [], 'all');
    R.melpow_rel_librosa(i) = max(abs(st.MelEnergies(big) - refMel(big)) ./ refMel(big));
    R.mfcc_librosa(i) = max(abs(mf.Values - npy([k '_mfcc'])), [], 'all');
    R.d1_librosa(i) = max(abs(dd.Values(F+1:2*F, :) - npy([k '_d1'])), [], 'all');
    R.d2_librosa(i) = max(abs(dd.Values(2*F+1:end, :) - npy([k '_d2'])), [], 'all');
    % --- MATLAB Audio Toolbox
    win = hann(L, 'periodic');
    fbM = designAuditoryFilterBank(fs, 'FFTLength', nfft, 'NumBands', 40, 'FrequencyRange', [0 fs/2], ...
        'Normalization', 'none', 'FilterBankDesignDomain', 'linear', 'MelStyle', 'oshaughnessy');
    R.melbank_matlab(i) = max(abs(lmT.Filters - full(fbM)), [], 'all');
    S = melSpectrogram(x, fs, 'Window', win, 'OverlapLength', L - H, 'FFTLength', nfft, 'NumBands', 40, ...
        'FrequencyRange', [0 fs/2], 'FilterBankNormalization', 'none', 'WindowNormalization', false, ...
        'SpectrumType', 'power', 'MelStyle', 'oshaughnessy', 'FilterBankDesignDomain', 'linear');
    R.logmel_matlab(i) = max(abs(lm.Values - log(max(S, 1e-10))), [], 'all');
    % DCT/log convention alone: MATLAB cepstralCoefficients (log10) on the matched mel power
    cC = cepstralCoefficients(max(S, 1e-10), 'NumCoeffs', 13)';
    R.cepstral_matlab_ln10(i) = max(abs(mf.Values - log(10) * cC), [], 'all');
    mel = OceanMath.Signal.MelFilterBank;
    edges = mel.mel2hz(linspace(mel.hz2mel(0, "htk"), mel.hz2mel(fs / 2, "htk"), 42), "htk");
    cM = mfcc(x, fs, 'Window', win, 'OverlapLength', L - H, 'FFTLength', nfft, 'BandEdges', edges, ...
        'NumCoeffs', 13, 'LogEnergy', 'ignore')';
    R.mfcc_matlab_ln10(i) = max(abs(mf.Values - log(10) * cM), [], 'all');
    aD = audioDelta(mf.Values', 5)';                          % causal, delayed by 2, scaled by 2
    Tn = mf.numFrames();
    R.audioDelta_interior(i) = max(abs(dd.Values(F+1:2*F, 3:Tn-2) - aD(:, 5:Tn) / 2), [], 'all');
    % --- pitch
    if fs <= 16000
        pc = OceanNLP.Speech.Features.PitchContour(fs).forward(w);
        f0 = pc.Values(1, :);
        pf0 = npy([k '_pyin_f0'])';
        pv = npy([k '_pyin_voiced'])' > 0;
        mp = pitch(x, fs, 'WindowLength', L, 'OverlapLength', L - H, 'Range', [50 400])';
        ov = f0 > 0;
        both = ov & pv & pf0 > 0;
        R.pitch_frames(i) = numel(f0);
        assert(numel(pf0) == numel(f0) && numel(mp) == numel(f0), 'pitch frame counts differ for %s', k);
        pitchRows(end + 1) = struct('key', string(k), 'nOceanVoiced', nnz(ov), 'nPyinVoiced', nnz(pv), ...
            'nBoth', nnz(both), 'gerPyin', nnz(abs(f0(both) - pf0(both)) ./ pf0(both) > 0.2), ...
            'gerMatlab', nnz(abs(f0(ov) - mp(ov)) ./ mp(ov) > 0.2), 'voicingAgree', nnz(ov == pv), ...
            'medianRelErrPyin', median(abs(f0(both) - pf0(both)) ./ pf0(both))); %#ok<SAGROW>
    else
        R.pitch_frames(i) = NaN;
    end
end

%% 4. Resampler
resRows = struct('key', {}, 'n', {}, 'lenOK', {}, 'scipySame', {}, 'scipyDefaultSameDesign', {}, ...
    'matlabDefaultRelRMS', {}, 'matlabSameFilter', {});
for j = 1:numel(resJobs)
    rj = resJobs(j);
    [x, fsIn] = audioread(rj.path, 'double');
    x = mean(x, 2);
    yO = OceanMath.Signal.ResampleSignal.resample(x, fsIn, rj.fs_out);
    ySc = npy([rj.key '_scipy']);
    yScDef = npy([rj.key '_scipy_default']);
    yO10 = OceanMath.Signal.ResampleSignal.resample(x, fsIn, rj.fs_out, 'HalfWidth', 10, 'KaiserBeta', 5);
    [h, p, q] = OceanMath.Signal.ResampleSignal.design(fsIn, rj.fs_out);
    yM = resample(x, p, q);
    yMh = resample(x, p, q, h);
    m = numel(yO);
    in = (ceil(0.01 * m):floor(0.99 * m))';                   % interior: away from edge transients
    resRows(end + 1) = struct('key', string(rj.key), 'n', m, 'lenOK', m == ceil(numel(x) * p / q), ...
        'scipySame', max(abs(yO(:) - ySc(:))), 'scipyDefaultSameDesign', max(abs(yO10(:) - yScDef(:))), ...
        'matlabDefaultRelRMS', norm(yO(in) - yM(in)) / norm(yM(in)), ...
        'matlabSameFilter', max(abs(yO(:) - yMh(:)))); %#ok<SAGROW>
end

%% 5. Report
P = struct2table(pitchRows);
Q = struct2table(resRows);
fsddMask = startsWith(R.key, "fsdd_");
agg = @(v) sprintf('%.3g', max(v));
lines = strings(0, 1);
lines(end + 1) = "# Speech feature external validation";
lines(end + 1) = "";
lines(end + 1) = string(sprintf("Generated by `Experiments/Scripts/Speech/validate_features_external.m` on %s with MATLAB %s, librosa %s, scipy %s, numpy %s.", ...
    string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm')), version('-release'), versions.librosa, versions.scipy, versions.numpy));
lines(end + 1) = "";
lines(end + 1) = string(sprintf("Audio: %d FSDD recordings (take 0 of every digit x speaker, 8 kHz) + MATLAB samples SpeechDFT, FemaleSpeech (8 kHz), Counting (44.1 kHz). Grid L = round(0.025 fs), H = round(0.010 fs), NFFT = 2^nextpow2(L), periodic Hann, center = false; 40 HTK mels (no norm); natural log, floor 1e-10; 13 ortho DCT-II MFCC; delta N = 2.", nnz(fsddMask)));
lines(end + 1) = "";
lines(end + 1) = "## Maximum absolute differences (max over files)";
lines(end + 1) = "";
lines(end + 1) = "| Quantity | Reference | FSDD (8 kHz) | MATLAB samples (8 / 44.1 kHz) | Note |";
lines(end + 1) = "| --- | --- | --- | --- | --- |";
rowsSpec = {
    "mel filter bank W", "librosa.filters.mel(htk=True, norm=None)", "melbank_librosa", "exact construction"
    "mel power W·P (relative)", "librosa melspectrogram", "melpow_rel_librosa", "max abs(ΔE)/E over bins with E > 1e-8·max"
    "log-mel", "log(max(librosa mel, 1e-10))", "logmel_librosa", "log domain ≈ relative error"
    "MFCC", "librosa.feature.mfcc(S=log-mel, ortho)", "mfcc_librosa", ""
    "Δ", "librosa.feature.delta(width=5, mode='nearest')", "d1_librosa", ""
    "ΔΔ", "delta of delta (nearest)", "d2_librosa", "HTK acceleration convention"
    "mel filter bank W", "designAuditoryFilterBank(Normalization='none', linear)", "melbank_matlab", ""
    "log-mel", "log melSpectrogram(WindowNormalization=false, FilterBankNormalization='none')", "logmel_matlab", ""
    "MFCC", "ln(10) · cepstralCoefficients(matched melSpectrogram)", "cepstral_matlab_ln10", "MATLAB uses log10; isolates the DCT convention"
    "MFCC", "ln(10) · mfcc(BandEdges=Ocean HTK edges, LogEnergy='ignore')", "mfcc_matlab_ln10", "NOT expected to agree: MATLAB mfcc uses abs(X) (magnitude) and bandwidth-normalised filters"
    "Δ (interior)", "audioDelta(c, 5)(t+2) / 2", "audioDelta_interior", "audioDelta is causal (delay 2) and has denominator Σn², not 2Σn²"
    };
for r = 1:size(rowsSpec, 1)
    v = R.(rowsSpec{r, 3});
    lines(end + 1) = string(sprintf("| %s | %s | %s | %s | %s |", rowsSpec{r, 1}, rowsSpec{r, 2}, agg(v(fsddMask)), agg(v(~fsddMask)), rowsSpec{r, 4}));
end
lines(end + 1) = "";
lines(end + 1) = "## Pitch (YIN, Ocean) vs librosa.pyin and MATLAB pitch (NCF)";
lines(end + 1) = "";
lines(end + 1) = "Gross error = relative deviation > 20 %. pyin: frames both call voiced. MATLAB `pitch` gives no voicing decision: frames Ocean calls voiced. Range 50–400 Hz, same frame centres (pyin frame_length 2L padded by L/2).";
lines(end + 1) = "";
lines(end + 1) = "| Set | frames | Ocean voiced | pyin voiced | voicing agreement | GER vs pyin | median rel. err vs pyin | GER vs MATLAB pitch |";
lines(end + 1) = "| --- | --- | --- | --- | --- | --- | --- | --- |";
pfr = R.pitch_frames(~isnan(R.pitch_frames));
pk = R.key(~isnan(R.pitch_frames));
for setName = ["FSDD", "MATLAB samples (8 kHz)"]
    if setName == "FSDD"
        mk = startsWith(P.key, "fsdd_");
    else
        mk = ~startsWith(P.key, "fsdd_");
    end
    frames = sum(pfr(ismember(pk, P.key(mk))));
    lines(end + 1) = string(sprintf("| %s | %d | %d | %d | %.1f %% | %d / %d = %.1f %% | %.2f %% | %d / %d = %.1f %% |", setName, frames, ...
        sum(P.nOceanVoiced(mk)), sum(P.nPyinVoiced(mk)), 100 * sum(P.voicingAgree(mk)) / frames, ...
        sum(P.gerPyin(mk)), sum(P.nBoth(mk)), 100 * sum(P.gerPyin(mk)) / max(1, sum(P.nBoth(mk))), ...
        100 * median(P.medianRelErrPyin(mk), 'omitnan'), ...
        sum(P.gerMatlab(mk)), sum(P.nOceanVoiced(mk)), 100 * sum(P.gerMatlab(mk)) / max(1, sum(P.nOceanVoiced(mk)))));
end
lines(end + 1) = "";
lines(end + 1) = "## Resampler (rational polyphase, Kaiser-windowed sinc)";
lines(end + 1) = "";
lines(end + 1) = "| Case | output samples | length = ceil(S p/q) | vs scipy resample_poly, same firwin filter (max abs) | Ocean(HalfWidth 10, β 5) vs scipy default (max abs) | vs MATLAB resample default (interior rel. RMS) | vs MATLAB resample(x,p,q,h) with Ocean filter (max abs) |";
lines(end + 1) = "| --- | --- | --- | --- | --- | --- | --- |";
for j = 1:height(Q)
    lines(end + 1) = string(sprintf("| %s | %d | %s | %.3g | %.3g | %.3g | %.3g |", Q.key(j), Q.n(j), string(Q.lenOK(j)), Q.scipySame(j), ...
        Q.scipyDefaultSameDesign(j), Q.matlabDefaultRelRMS(j), Q.matlabSameFilter(j)));
end
lines(end + 1) = "";
lines(end + 1) = "## Convention differences (documented, not hidden)";
lines(end + 1) = "";
lines(end + 1) = "* **librosa frame placement:** librosa cuts n_fft-sample frames and centres a shorter window inside them; Ocean cuts L-sample frames and zero-pads at the end (contract §3). The power spectrum is shift-invariant, so the helper pads the signal by (n_fft−L)/2 to align frames; with that alignment the results agree to rounding.";
lines(end + 1) = "* **MATLAB melSpectrogram:** defaults divide by the window sum (`WindowNormalization=true`) and use bandwidth-normalised filters; both disabled here to match Ocean. MATLAB output is NumBands × T, linear power.";
lines(end + 1) = "* **MATLAB mfcc / cepstralCoefficients:** log10 instead of natural log (× ln 10 applied above). `cepstralCoefficients` on the matched mel power isolates the DCT; `mfcc()` itself filters the *magnitude* spectrum abs(X) (not abs(X)²) with bandwidth-normalised filters (not configurable) and its default band edges are Slaney-style (Ocean's HTK edges were passed via `BandEdges`), so its residual is a convention difference, not an error.";
lines(end + 1) = "* **audioDelta:** causal FIR (output delayed by (w−1)/2 frames, zero initial state) with denominator Σn²; Ocean/HTK uses 2Σn² and edge replication, so only interior frames are comparable after a ×½ scale and 2-frame shift.";
lines(end + 1) = "* **ΔΔ:** Ocean = delta of delta (HTK). `librosa.feature.delta(order=2)` is a direct Savitzky–Golay second derivative and differs; the comparison uses delta(delta).";
lines(end + 1) = "* **Pitch:** Ocean is frame-independent YIN (no smoothing, no Viterbi). pyin adds HMM voicing/tracking; MATLAB NCF has no voicing decision. Differences are algorithmic, so rates are reported rather than a tolerance.";
lines(end + 1) = "* **MATLAB resample default:** different filter (Kaiser β = 5, 10 zero crossings per side, least-squares design), so only a relative interior RMS is meaningful there.";
lines(end + 1) = "";
lines(end + 1) = "## Per-file table";
lines(end + 1) = "";
lines(end + 1) = "| file | fs | T | logmel vs librosa | MFCC vs librosa | log-mel vs MATLAB | MFCC vs cepstralCoefficients·ln10 |";
lines(end + 1) = "| --- | --- | --- | --- | --- | --- | --- |";
for i = 1:height(R)
    lines(end + 1) = string(sprintf("| %s | %d | %d | %.2g | %.2g | %.2g | %.2g |", R.key(i), R.fs(i), R.T(i), R.logmel_librosa(i), ...
        R.mfcc_librosa(i), R.logmel_matlab(i), R.cepstral_matlab_ln10(i)));
end
if ~isfolder(fileparts(reportFile))
    mkdir(fileparts(reportFile));
end
fid = fopen(reportFile, 'w', 'n', 'UTF-8');
fprintf(fid, '%s\n', lines);
fclose(fid);
fprintf('Wrote %s\n', reportFile);
disp(R(:, ["key", "logmel_librosa", "mfcc_librosa", "logmel_matlab", "cepstral_matlab_ln10", "audioDelta_interior"]));
disp(P);
disp(Q);
