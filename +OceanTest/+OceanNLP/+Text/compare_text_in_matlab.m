function compare_text_in_matlab()
% COMPARE_TEXT_IN_MATLAB  External validation of the text-processing upgrade
% against Python references on REAL Cranfield text.
%
%   (a) PorterStemmer vs NLTK PorterStemmer(mode="ORIGINAL_ALGORITHM") on every
%       distinct word of the collection (+ edge cases)   [porter_reference.json]
%   (b) Gpt2PreTokenizer vs the exact GPT-2 regex (Python `regex`, Unicode \p{L}\p{N})
%   (c) BertBasicPreTokenizer (uncased + cased) vs HF `tokenizers` BertNormalizer +
%       BertPreTokenizer                                 [pretok_reference.json]
%       on 39 Unicode torture strings + 210 real Cranfield texts.
%
% Run porter_reference.py and pretok_reference.py first. Run:  OceanTest.OceanNLP.Text.compare_text_in_matlab()

here = fileparts(mfilename('fullpath'));
addpath(fileparts(fileparts(fileparts(here))));
allPass = true;

% ---------------- (a) Porter ----------------
% jsondecode would rename keys that are not valid MATLAB identifiers ("10", "x10" collide), so parse by regex.
kv = regexp(fileread(fullfile(here, 'porter_reference.json')), '"([a-z0-9]+)":\s*"([a-z0-9]*)"', 'tokens');
words = strings(1, numel(kv));  expectedStems = strings(1, numel(kv));
for i = 1:numel(kv), words(i) = kv{i}{1}; expectedStems(i) = kv{i}{2}; end
nBad = 0;  bad = strings(0, 1);
st = OceanNLP.Text.Stem.PorterStemmer();
for i = 1:numel(words)
    got = st.stem(words(i));
    if got ~= expectedStems(i)
        nBad = nBad + 1;
        if numel(bad) < 15, bad(end+1, 1) = words(i) + " -> " + got + " (expected " + expectedStems(i) + ")"; end %#ok<AGROW>
    end
end
pass = nBad == 0;  allPass = allPass && pass;
fprintf('[%s] Porter vs NLTK ORIGINAL_ALGORITHM: %d words, %d mismatches\n', tf_(pass), numel(words), nBad);
if nBad > 0, fprintf('      e.g. %s\n', strjoin(bad, ' | ')); end

% ---------------- (b),(c) pre-tokenizers ----------------
cases = jsondecode(fileread(fullfile(here, 'pretok_reference.json')));
gpt2 = OceanNLP.Text.PreTokenize.Gpt2PreTokenizer();
bu = OceanNLP.Text.PreTokenize.BertBasicPreTokenizer('NeverSplit', strings(1, 0));
bc = OceanNLP.Text.PreTokenize.BertBasicPreTokenizer('Lowercase', false, 'NeverSplit', strings(1, 0));
nCases = numel(cases);
bad = struct('gpt2', 0, 'bert_un', 0, 'bert_cs', 0);
firstBad = struct('gpt2', "", 'bert_un', "", 'bert_cs', "");
for i = 1:nCases
    c = cases(i);
    text = string(c.text);
    got = struct('gpt2', gpt2.forward(text), 'bert_un', bu.forward(text), 'bert_cs', bc.forward(text));
    for f = ["gpt2" "bert_un" "bert_cs"]
        expected = toStrings_(c.(f));
        if ~isequal(got.(f), expected)
            bad.(f) = bad.(f) + 1;
            if firstBad.(f) == "", firstBad.(f) = sprintf('%d: got [%s] expected [%s]', i, strjoin(got.(f), '|'), strjoin(expected, '|')); end
        end
    end
end
for f = ["gpt2" "bert_un" "bert_cs"]
    pass = bad.(f) == 0;  allPass = allPass && pass;
    fprintf('[%s] %-8s pre-tokenizer vs Python reference: %d cases, %d mismatches\n', tf_(pass), f, nCases, bad.(f));
    if ~pass, fprintf('      first: %s\n', firstBad.(f)); end
end

% ---------------- (d) analyzer chain vs a Python re-implementation ----------------
chain = jsondecode(fileread(fullfile(here, 'analyzer_reference.json')));
an = OceanNLP.Text.Analyzer.preset("english");
badChain = 0;  firstChain = "";
for i = 1:numel(chain)
    got = an.forward(string(chain(i).text));
    expected = toStrings_(chain(i).tokens);
    if ~isequal(got, expected)
        badChain = badChain + 1;
        if firstChain == "", firstChain = sprintf('%d: got [%s] expected [%s]', i, strjoin(got(1:min(end,10)), '|'), strjoin(expected(1:min(end,10)), '|')); end
    end
end
pass = badChain == 0;  allPass = allPass && pass;
fprintf('[%s] english analyzer (lowercase > Lucene stop > Porter) vs Python chain: %d texts, %d mismatches\n', tf_(pass), numel(chain), badChain);
if ~pass, fprintf('      first: %s\n', firstChain); end

if allPass
    fprintf('\nOverall: ALL CHECKS PASSED\n');
else
    fprintf('\nOverall: SOME CHECKS FAILED\n');
end
end

function s = tf_(p)
if p, s = 'PASS'; else, s = 'FAIL'; end
end

function s = toStrings_(x)
% jsondecode gives [] for an empty list, a cell array of char for a string list.
if isempty(x), s = strings(1, 0); elseif iscell(x), s = string(x(:))'; else, s = string(x); end
end
