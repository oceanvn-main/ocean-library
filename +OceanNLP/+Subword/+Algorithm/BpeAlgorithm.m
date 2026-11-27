classdef BpeAlgorithm < OceanNLP.Subword.ISubwordAlgorithm
    % BPEALGORITHM  Byte-Pair Encoding (Sennrich et al. 2016; the GPT-2 flavour
    % when combined with byte-level mapping): learn an ordered list of pair
    % merges, then segment a word by repeatedly applying the lowest-RANK merge
    % that occurs in it.
    %
    % TRAINING (functional core trainMerges): start from single-code-point
    % symbols per word; repeat {count all adjacent symbol pairs weighted by word
    % frequency; merge the most frequent pair everywhere} until the budget is
    % used or no pair reaches MinFrequency. TIE-BREAK: among equally frequent
    % pairs take the smallest (id(left), id(right)) where ids follow symbol
    % CREATION order (alphabet in code-point order, then each merge result as it
    % is created). This is the rule that reproduces HuggingFace's trainer: on
    % Cranfield (vocab 1000, byte-level) all 744 merges are identical to HF's.
    % HF's own alphabet order for equal-count symbols is implementation-defined,
    % so identity is verified empirically per corpus, and compare_subword_in_matlab.m
    % additionally replays HF's list and checks every choice was a maximum-count
    % pair (tie-break independent).
    %
    % SEGMENTATION (segment): GPT-2 bpe(): find the adjacent pair with the lowest
    % merge rank, merge all its (non-overlapping, left-to-right) occurrences,
    % repeat until no adjacent pair has a rank. Results are cached per word.
    %
    % Options: EndOfWordSuffix ("" for GPT-2; "</w>" for Sennrich-style word BPE).
    %
    %   a = BpeAlgorithm();  a.train(words, counts, 'VocabSize', 500);
    %   a = BpeAlgorithm.fromMergesFile("merges.txt");     % pretrained GPT-2 style
    %
    % See also: OceanNLP.Subword.SubwordTokenizer

    properties (SetAccess = private)
        Merges (:,2) string = strings(0, 2)      % rank r = row r
        Alphabet (1,:) string = strings(1, 0)    % base symbols (training only)
        EndOfWordSuffix (1,1) string = ""
    end
    properties (Access = private)
        Ranks_ = []                              % containers.Map 'a<SOH>b' -> rank
        Cache_ = []                              % containers.Map word -> pieces (cell)
    end
    properties (Constant)
        Reference = "Sennrich, Haddow & Birch (2016) Neural MT of Rare Words with Subword Units; GPT-2 bpe() segmentation"
        InDomain = "token-sequence"
        OutDomain = "token-sequence"
    end

    methods
        function obj = BpeAlgorithm(options)
            arguments
                options.EndOfWordSuffix (1,1) string = ""
            end
            obj.EndOfWordSuffix = options.EndOfWordSuffix;
        end

        function train(obj, words, counts, options)
            arguments
                obj
                words (1,:) string
                counts (1,:) double
                options.VocabSize (1,1) double {mustBeInteger, mustBePositive} = 1000
                options.Seed (1,:) string = strings(1, 0)
                options.MinFrequency (1,1) double {mustBeNonnegative} = 0
            end
            if numel(words) ~= numel(counts)
                error('OceanNLP:Subword:BpeAlgorithm:CountMismatch', 'words and counts must have the same length.');
            end
            [merges, alphabet] = OceanNLP.Subword.Algorithm.BpeAlgorithm.trainMerges( ...
                words, counts, options.VocabSize, options.Seed, options.MinFrequency, obj.EndOfWordSuffix);
            obj.setMerges_(merges);
            obj.Alphabet = alphabet;
        end

        function pieces = segment(obj, word)
            arguments
                obj
                word (1,1) string
            end
            if isempty(obj.Ranks_), obj.setMerges_(obj.Merges); end
            key = char(word);
            if isKey(obj.Cache_, key)
                pieces = obj.Cache_(key);
                return
            end
            cps = OceanNLP.Text.Unicode.codepoints(word);
            sym = cell(1, numel(cps));
            for i = 1:numel(cps), sym{i} = char(OceanNLP.Text.Unicode.fromCodepoints(cps(i))); end
            if ~isempty(sym) && obj.EndOfWordSuffix ~= ""
                sym{end} = [sym{end} char(obj.EndOfWordSuffix)];
            end
            sep = char(1);
            while numel(sym) > 1
                best = Inf;  bestPair = [];
                for i = 1:numel(sym) - 1
                    k = [sym{i} sep sym{i+1}];
                    if isKey(obj.Ranks_, k)
                        r = obj.Ranks_(k);
                        if r < best, best = r; bestPair = {sym{i}, sym{i+1}}; end
                    end
                end
                if isempty(bestPair), break; end
                merged = cell(1, 0);
                i = 1;
                while i <= numel(sym)
                    if i < numel(sym) && strcmp(sym{i}, bestPair{1}) && strcmp(sym{i+1}, bestPair{2})
                        merged{end+1} = [bestPair{1} bestPair{2}]; %#ok<AGROW>
                        i = i + 2;
                    else
                        merged{end+1} = sym{i}; %#ok<AGROW>
                        i = i + 1;
                    end
                end
                sym = merged;
            end
            pieces = string(sym);
            if isempty(pieces), pieces = strings(1, 0); end
            obj.Cache_(key) = pieces;
        end

        function toks = tokens(obj)
            % TOKENS  alphabet symbols, then each merge result in merge order.
            merged = obj.Merges(:, 1) + obj.Merges(:, 2);
            toks = [obj.Alphabet, reshape(merged, 1, [])];
            [~, first] = unique(toks, 'stable');
            toks = toks(first);
        end

        function tf = isTrained(obj)
            tf = ~isempty(obj.Merges);
        end

        function text = detokenize(obj, pieces)
            arguments
                obj
                pieces (1,:) string
            end
            text = strjoin(pieces, "");
            if obj.EndOfWordSuffix ~= ""
                text = strtrim(strrep(text, obj.EndOfWordSuffix, " "));
            end
        end

        function s = name(~), s = "bpe"; end

        function setMerges(obj, merges)
            % SETMERGES  Install an ordered merge list (rank = row), e.g. from a pretrained merges.txt.
            arguments
                obj
                merges (:,2) string
            end
            obj.setMerges_(merges);
        end
    end

    methods (Access = protected)
        function a = assumptions(obj)
            a = {'segmentation: repeatedly merge the lowest-rank adjacent pair present (GPT-2 bpe())', ...
                 ['training tie-break: among equally frequent pairs the smallest (id(left), id(right)) by symbol-creation ' ...
                  'order (alphabet in code-point order, then merges); reproduces HF BpeTrainer on the validated corpora, ' ...
                  'implementation-defined in HF for equal-count alphabet ordering'], ...
                 'training is deterministic given (words, counts, seed alphabet, VocabSize, MinFrequency)', ...
                 char("end-of-word suffix: '" + obj.EndOfWordSuffix + "'")};
        end
    end

    methods (Access = private)
        function setMerges_(obj, merges)
            obj.Merges = merges;
            obj.Ranks_ = containers.Map('KeyType', 'char', 'ValueType', 'double');
            sep = char(1);
            for r = size(merges, 1):-1:1                 % first occurrence keeps the lowest rank
                obj.Ranks_([char(merges(r, 1)) sep char(merges(r, 2))]) = r;
            end
            obj.Cache_ = containers.Map('KeyType', 'char', 'ValueType', 'any');
        end
    end

    methods (Static)
        function obj = fromMergesFile(path, options)
            % FROMMERGESFILE  GPT-2/HF merges.txt: optional "#version" header, then "left right" per line.
            arguments
                path (1,1) string
                options.EndOfWordSuffix (1,1) string = ""
            end
            raw = fileread(path, 'Encoding', 'UTF-8');
            lines = string(regexp(raw, '\r?\n', 'split'));
            lines = lines(strlength(lines) > 0 & ~startsWith(lines, "#version"));
            parts = split(lines(:), " ");
            if size(parts, 2) ~= 2
                error('OceanNLP:Subword:BpeAlgorithm:BadMergesFile', ...
                    'Every merges line must be exactly "left right" (got %d columns).', size(parts, 2));
            end
            obj = OceanNLP.Subword.Algorithm.BpeAlgorithm('EndOfWordSuffix', options.EndOfWordSuffix);
            obj.setMerges_(parts);
        end

        function [merges, alphabet, stats] = trainMerges(words, counts, vocabSize, seed, minFreq, suffix, forced, prefix)
            % TRAINMERGES  Pure BPE trainer (see class header).
            %   [merges, alphabet, stats] = trainMerges(words, counts, vocabSize, seed, minFreq, suffix, forced)
            %   forced (optional Kx2 string): apply THESE merges in order instead of choosing;
            %   stats.MaxCount(k) is then the true maximum pair count at step k, stats.PairCount(k)
            %   the count of the forced pair - equal iff the forced choice was a maximum.
            arguments
                words (1,:) string
                counts (1,:) double
                vocabSize (1,1) double
                seed (1,:) string = strings(1, 0)
                minFreq (1,1) double = 0
                suffix (1,1) string = ""
                forced (:,2) string = strings(0, 2)
                prefix (1,1) string = ""          % continuing-subword prefix ("##" for WordPiece training)
            end
            U = OceanNLP.Text.Unicode();
            % --- symbol table & flat integer representation (0 = word separator)
            symStr = strings(1, 0);
            flat = zeros(1, 0);  wt = zeros(1, 0);
            wordSyms = cell(1, numel(words));
            allSyms = strings(1, 0);
            for w = 1:numel(words)
                cps = U.codepoints(words(w));
                s = strings(1, numel(cps));
                for i = 1:numel(cps), s(i) = U.fromCodepoints(cps(i)); end
                if prefix ~= "" && numel(s) > 1, s(2:end) = prefix + s(2:end); end
                if ~isempty(s) && suffix ~= "", s(end) = s(end) + suffix; end
                wordSyms{w} = s;
                allSyms = [allSyms, s]; %#ok<AGROW>
            end
            alphabet = unique([seed, allSyms]);
            symStr = alphabet;
            [~, idOf] = ismember(allSyms, symStr);
            pos = 0;
            chunks = cell(1, numel(words));  wchunks = cell(1, numel(words));
            for w = 1:numel(words)
                n = numel(wordSyms{w});
                chunks{w} = [idOf(pos+1:pos+n), 0];
                wchunks{w} = repmat(counts(w), 1, n + 1);
                pos = pos + n;
            end
            flat = [chunks{:}];  wt = [wchunks{:}];
            nMerge = max(0, vocabSize - numel(alphabet));
            if ~isempty(forced), nMerge = size(forced, 1); end
            merges = strings(nMerge, 2);
            stats = struct('PairCount', zeros(nMerge, 1), 'MaxCount', zeros(nMerge, 1));
            keyBase = 2^24;
            done = 0;
            for m = 1:nMerge
                a = flat(1:end-1);  b = flat(2:end);
                valid = a > 0 & b > 0;
                if ~any(valid), break; end
                key = a(valid) * keyBase + b(valid);
                w = wt(1:end-1);  w = w(valid);
                [uk, ~, ic] = unique(key);
                cnt = accumarray(ic(:), w(:))';
                mx = max(cnt);
                if isempty(forced)
                    if mx < max(1, minFreq), break; end
                    cand = uk(cnt == mx);
                    if numel(cand) > 1
                        pick = min(cand);              % numeric (idA,idB): creation-order tie-break
                    else
                        pick = cand;
                    end
                else
                    ia = find(symStr == forced(m, 1), 1);  ib = find(symStr == forced(m, 2), 1);
                    if isempty(ia) || isempty(ib)
                        stats.PairCount(m) = 0;  stats.MaxCount(m) = mx;
                        merges(m, :) = forced(m, :);
                        done = m;
                        continue           % pair impossible in the current state: count 0
                    end
                    pick = ia * keyBase + ib;
                end
                ka = floor(pick / keyBase);  kb = pick - ka * keyBase;
                merges(m, :) = [symStr(ka), symStr(kb)];
                stats.MaxCount(m) = mx;
                stats.PairCount(m) = sum(cnt(uk == pick));
                done = m;
                rightPart = symStr(kb);
                if prefix ~= "" && startsWith(rightPart, prefix), rightPart = extractAfter(rightPart, strlength(prefix)); end
                newTok = symStr(ka) + rightPart;
                existing = find(symStr == newTok, 1);
                if isempty(existing)
                    symStr(end+1) = newTok; %#ok<AGROW>
                    newId = numel(symStr);
                else
                    newId = existing;
                end
                idx = find(valid);  idx = idx(key == pick);
                keep = true(size(idx));  last = -Inf;
                for k = 1:numel(idx)
                    if idx(k) == last + 1, keep(k) = false; else, last = idx(k); end
                end
                idx = idx(keep);
                if ~isempty(idx)
                    flat(idx) = newId;
                    flat(idx + 1) = -1;
                    del = flat == -1;
                    flat(del) = [];  wt(del) = [];
                end
            end
            merges = merges(1:done, :);
            stats.PairCount = stats.PairCount(1:done);
            stats.MaxCount = stats.MaxCount(1:done);
        end
    end
end
