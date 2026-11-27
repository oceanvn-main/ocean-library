%% validate_wer_jiwer  Ocean WER/CER vs jiwer on real FSDD transcripts + FSDD manifest audit.
%
%   Phase 2 gate (Planning/NLP/Speech_ASR_Phase0_Contracts.md sec. 7): Ocean
%   WordErrorRate / CharacterErrorRate must agree with jiwer under identical
%   normalization, with exact S/D/I/H totals; the real corpus must load and audit.
%
%   1. Manifest: SpeechManifest.fromFSDD -> validate -> canonical split -> audit.
%   2. References: UtteranceComposer utterances of the held-out speaker (real FSDD
%      labels, 2-5 words). Hypotheses: seeded word edits (substitute / delete /
%      insert digit words) of each reference -- test inputs, not results.
%      A "styled" copy adds sentence case and punctuation to both sides.
%   3. jiwer (./.venv, via wer_jiwer_reference.py) and Ocean score the same strings:
%        clean       plain digit-word strings
%        raw-styled  styled strings, no normalization (jiwer default transforms)
%        normalized  styled strings after Ocean TranscriptNormalizer (both tools)
%      and the Ocean normalizer is compared with the equivalent jiwer pipeline.
%   4. Writes Results/Speech/wer_validation.md.
%
%   Run from anywhere:  run('Experiments/Scripts/Speech/validate_wer_jiwer.m')

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(fileparts(here)));
addpath(repo);
py = fullfile(repo, '.venv', 'Scripts', 'python.exe');
if ~isfile(py), py = fullfile(repo, '.venv', 'bin', 'python'); end
if ~isfile(py)
    error('validate_wer_jiwer:NoPython', 'Python venv with jiwer not found under %s/.venv', repo);
end

%% 1. Manifest + audit
tic;
m = OceanNLP.Speech.Data.SpeechManifest.fromFSDD();
rep = m.validate();
s = m.canonicalFSDDSplit();
s.bySplit("train").assertDisjoint(s.bySplit("test"), 'By', "speaker");
s.bySplit("train").assertDisjoint(s.bySplit("validation"), 'By', "id");
audit = s.audit();
fprintf('FSDD: %d entries, %d files validated (%.1f s)\n', m.height(), rep.NumFiles, toc);
disp(audit);

%% 2. References and hypotheses
words = OceanNLP.Speech.Data.SpeechManifest.DigitWords;
comp = OceanNLP.Speech.Data.UtteranceComposer.compose(s.bySplit("test"), 'Count', 300, 'Seed', 11);
refs = comp.Entries.Transcript';
rs = RandStream('mt19937ar', 'Seed', 2026);
hyps = strings(size(refs));
for u = 1:numel(refs)
    w = split(refs(u), " ")';
    for e = 1:randi(rs, [0 3])
        op = randi(rs, 3);
        if op == 1 && ~isempty(w)
            w(randi(rs, numel(w))) = words(randi(rs, 10));          % substitution
        elseif op == 2 && ~isempty(w)
            w(randi(rs, numel(w))) = [];                            % deletion
        else
            k = randi(rs, numel(w) + 1);                            % insertion
            w = [w(1:k-1), words(randi(rs, 10)), w(k:end)];
        end
    end
    hyps(u) = join(w, " ");
    if isempty(w), hyps(u) = ""; end
end
style = @(t) styleText(t, rs);
refsStyled = arrayfun(style, refs);
hypsStyled = arrayfun(style, hyps);
norm = OceanNLP.Speech.Data.TranscriptNormalizer();
sets = struct('name', {"clean", "raw-styled", "normalized"}, ...
    'refs', {refs, refsStyled, norm.forward(refsStyled)}, ...
    'hyps', {hyps, hypsStyled, norm.forward(hypsStyled)});

%% 3. jiwer reference (Python) vs Ocean
inFile = string(tempname) + "_in.json"; outFile = string(tempname) + "_out.json";
payload = struct('sets', {arrayfun(@(x) struct('name', x.name, 'refs', {cellstr(x.refs)}, ...
    'hyps', {cellstr(x.hyps)}), sets, 'UniformOutput', false)}, ...
    'normalize', {cellstr([refsStyled, hypsStyled])});
payload.sets = [payload.sets{:}];
fid = fopen(inFile, 'w', 'n', 'UTF-8'); fwrite(fid, jsonencode(payload), 'char'); fclose(fid);
cmd = sprintf('"%s" "%s" "%s" "%s"', py, fullfile(here, 'wer_jiwer_reference.py'), inFile, outFile);
[status, msg] = system(cmd);
if status ~= 0
    error('validate_wer_jiwer:Python', 'jiwer helper failed:\n%s', msg);
end
J = jsondecode(fileread(outFile));
delete(inFile); delete(outFile);

rows = table();
for k = 1:numel(sets)
    w = OceanNLP.Speech.Evaluation.WordErrorRate.compute(sets(k).refs, sets(k).hyps);
    c = OceanNLP.Speech.Evaluation.CharacterErrorRate.compute(sets(k).refs, sets(k).hyps);
    j = J.sets(k);
    pw = [w.PerUtterance.S, w.PerUtterance.D, w.PerUtterance.I, w.PerUtterance.H];
    pc = [c.PerUtterance.S, c.PerUtterance.D, c.PerUtterance.I, c.PerUtterance.H];
    rows = [rows; table(string(sets(k).name), numel(sets(k).refs), ...
        string(sprintf('%d/%d/%d/%d', w.S, w.D, w.I, w.H)), ...
        string(sprintf('%d/%d/%d/%d', j.word.S, j.word.D, j.word.I, j.word.H)), ...
        w.Rate, j.wer, nnz(any(pw ~= j.per_word, 2)), ...
        string(sprintf('%d/%d/%d/%d', c.S, c.D, c.I, c.H)), ...
        string(sprintf('%d/%d/%d/%d', j.char.S, j.char.D, j.char.I, j.char.H)), ...
        c.Rate, j.cer, nnz(any(pc ~= j.per_char, 2)), ...
        'VariableNames', ["Set", "Utterances", "OceanWordSDIH", "JiwerWordSDIH", "OceanWER", ...
        "JiwerWER", "WordPairsDiffer", "OceanCharSDIH", "JiwerCharSDIH", "OceanCER", "JiwerCER", ...
        "CharPairsDiffer"])]; %#ok<AGROW>
end
disp(rows);
normOcean = norm.forward([refsStyled, hypsStyled]);
normJiwer = string(J.normalized)';
normAgree = nnz(normOcean == normJiwer);
allExact = all(rows.OceanWordSDIH == rows.JiwerWordSDIH) && all(rows.OceanCharSDIH == rows.JiwerCharSDIH) ...
    && all(rows.WordPairsDiffer == 0) && all(rows.CharPairsDiffer == 0) ...
    && max(abs(rows.OceanWER - rows.JiwerWER)) == 0 && max(abs(rows.OceanCER - rows.JiwerCER)) == 0 ...
    && normAgree == numel(normOcean);
fprintf('Exact agreement with jiwer %s: %d\n', J.jiwer_version, allExact);

%% 4. Report
outDir = fullfile(repo, 'Results', 'Speech');
if ~isfolder(outDir), mkdir(outDir); end
L = strings(0, 1);
L(end+1) = "# Ocean WER/CER vs jiwer — validation on FSDD";
L(end+1) = "";
L(end+1) = sprintf("Generated by `Experiments/Scripts/Speech/validate_wer_jiwer.m` on %s (MATLAB %s, jiwer %s).", ...
    string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm')), version('-release'), J.jiwer_version);
L(end+1) = "Every number below is computed by the script; jiwer runs in `./.venv` via `wer_jiwer_reference.py`.";
L(end+1) = "";
L(end+1) = "## 1. Corpus audit (FSDD, canonical split)";
L(end+1) = "";
L(end+1) = sprintf("`SpeechManifest.fromFSDD` + `validate()`: %d entries, %d files exist and match `audioinfo`. " + ...
    "Split: test speaker `theo`, validation = takes 45–49 of the other speakers (speaker-dependent).", m.height(), rep.NumFiles);
L(end+1) = "";
L(end+1) = "| Split | Speaker | Utterances | Total s | Min s | Mean s | Median s | Max s | Rate Hz |";
L(end+1) = "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |";
for r = 1:height(audit)
    a = audit(r, :);
    L(end+1) = sprintf("| %s | %s | %d | %.2f | %.4f | %.4f | %.4f | %.4f | %s |", a.Split, a.Speaker, ...
        a.NumUtterances, a.TotalDuration, a.MinDuration, a.MeanDuration, a.MedianDuration, a.MaxDuration, a.SampleRates); %#ok<SAGROW>
end
L(end+1) = "";
L(end+1) = "## 2. WER / CER agreement";
L(end+1) = "";
L(end+1) = sprintf("References: %d `UtteranceComposer` utterances (2–5 real recordings of speaker `theo`, seed 11). " + ...
    "Hypotheses: 0–3 seeded word edits each (seed 2026). `raw-styled` adds sentence case and punctuation to both " + ...
    "sides independently; `normalized` passes those through `TranscriptNormalizer` and gives the SAME strings to both tools.", numel(refs));
L(end+1) = "Counts are S/D/I/H. *PairsDiffer* = utterances whose per-pair S/D/I/H differs between Ocean and jiwer.";
L(end+1) = "";
L(end+1) = "| Set | N utt | Ocean word | jiwer word | Ocean WER | jiwer WER | pairs differ | Ocean char | jiwer char | Ocean CER | jiwer CER | pairs differ |";
L(end+1) = "| --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |";
for r = 1:height(rows)
    x = rows(r, :);
    L(end+1) = sprintf("| %s | %d | %s | %s | %.12f | %.12f | %d | %s | %s | %.12f | %.12f | %d |", x.Set, x.Utterances, ...
        x.OceanWordSDIH, x.JiwerWordSDIH, x.OceanWER, x.JiwerWER, x.WordPairsDiffer, ...
        x.OceanCharSDIH, x.JiwerCharSDIH, x.OceanCER, x.JiwerCER, x.CharPairsDiffer); %#ok<SAGROW>
end
L(end+1) = "";
L(end+1) = sprintf("Normalizer: Ocean `TranscriptNormalizer` equals jiwer `Compose([ToLowerCase, RemovePunctuation, " + ...
    "RemoveWhiteSpace(replace_by_space=True), RemoveMultipleSpaces, Strip])` on %d / %d styled strings.", normAgree, numel(normOcean));
L(end+1) = "";
L(end+1) = sprintf("**Result: %s.**", ternary(allExact, "exact agreement — totals, per-utterance S/D/I/H splits and rates are identical", ...
    "DISAGREEMENT — see the table"));
L(end+1) = "";
L(end+1) = "## 3. Conventions and documented differences";
L(end+1) = "";
L(end+1) = "* Corpus rate = Σ(S+D+I) / ΣN (micro average), as jiwer.";
L(end+1) = "* Word tokens = jiwer `wer_default` (collapse ≥2 whitespace, strip, split on space). Character tokens = jiwer " + ...
    "`cer_default` (strip, every code point incl. internal spaces; space runs are not collapsed).";
L(end+1) = "* S/D/I split: `EditAlignment` strips the common prefix/suffix and traces back with the rapidfuzz priority " + ...
    "(deletion, then insertion via the column j-1 test, then diagonal), so splits on ties equal jiwer's, not just totals.";
L(end+1) = "* Empty total reference: Ocean raises `...:EmptyReference`; jiwer 4.0 returns WER = number of insertions " + ...
    "(the contract's note that jiwer raises does not hold for 4.0).";
L(end+1) = "* Ocean punctuation removal covers Unicode P* in U+0000–00FF and U+2000–206F only (jiwer: all of Unicode P*).";
fid = fopen(fullfile(outDir, 'wer_validation.md'), 'w', 'n', 'UTF-8');
fwrite(fid, join(L, newline) + newline, 'char');
fclose(fid);
fprintf('Wrote %s\n', fullfile(outDir, 'wer_validation.md'));
if ~allExact
    error('validate_wer_jiwer:Mismatch', 'Ocean and jiwer disagree; see the report.');
end

function t = styleText(t, rs)
% Sentence case + punctuation, drawn from RS (input styling only).
if t == "", return; end
w = split(t, " ")';
if randi(rs, 2) == 1, w(1) = upper(extractBefore(w(1), 2)) + extractAfter(w(1), 1); end
if numel(w) > 1 && randi(rs, 2) == 1
    k = randi(rs, numel(w) - 1); w(k) = w(k) + ",";
end
marks = [".", "?", "!", ""];
t = join(w, " ") + marks(randi(rs, 4));
end

function v = ternary(c, a, b)
if c, v = a; else, v = b; end
end
