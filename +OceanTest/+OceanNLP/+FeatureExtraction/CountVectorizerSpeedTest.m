classdef CountVectorizerSpeedTest < OceanTest.BaseTestCase
    % COUNTVECTORIZERSPEEDTEST  The fast CountVectorizer path (one hash lookup for the whole corpus, one sparse() call,
    % single-regexp whitespace tokenizer, linear Vocabulary.fit) must give EXACTLY what the original per-document algorithm
    % gave, and must not regress to the original cost. The original algorithm re-sorted the whole vocabulary inside
    % Vocabulary.encode for every document (ismember(tokens, Tokens)): about 0.2 s per document for a 130k-word vocabulary,
    % i.e. ~35 minutes for the 11,314-document 20 Newsgroups training set (scikit-learn: 2.7 s). Fit/forward now take
    % 5.8 s / 3.8 s on that corpus.

    methods (Test)
        function testWhitespaceTokenizerMatchesTheOriginalTwoStepDefinitionOnTrickyAndRandomText(testCase)
            tok = OceanNLP.Tokenize.WhitespaceTokenizer();
            tricky = ["", "   ", "Hello, world!", "Isn't it -- great?", "..a..", "x", "a-b!c d", "!!!", "(a)(b)", "snake_case __init__ ", ...
                      "3.14 and 1,000", "tab" + char(9) + "sep" + newline + "lines", "caf" + char(233) + " na" + char(239) + "ve", "a'b'", "'quoted'", ...
                      "e-mail: john.doe@x.org.", "  leading and trailing  ", "UPPER Lower MiXeD"];
            for k = 1:numel(tricky)
                testCase.verifyEqual(tok.forward(tricky(k)), OceanTest.OceanNLP.FeatureExtraction.CountVectorizerSpeedTest.originalTokens_(tricky(k), false), "text #" + k);
            end
            rs = RandStream('mt19937ar', 'Seed', 11);
            alphabet = ['abcXYZ019_' '.,;:!?-''"()[]' '  ' char([9 10]) char(233)];
            for n = 1:300
                txt = string(alphabet(randi(rs, numel(alphabet), 1, randi(rs, [0 60]))));
                testCase.verifyEqual(tok.forward(txt), OceanTest.OceanNLP.FeatureExtraction.CountVectorizerSpeedTest.originalTokens_(txt, false), "random #" + n + ": " + txt);
            end
            tokCS = OceanNLP.Tokenize.WhitespaceTokenizer('CaseSensitive', true);
            testCase.verifyEqual(tokCS.forward("Hello, World!"), ["Hello" "World"]);
        end

        function testForwardEqualsTheOriginalPerDocumentAlgorithmExactly(testCase)
            corpus = ["the cat sat on the mat" "the dog ran" "" "   " "a a a a b" "the the the cat cat" "Unseen-word: the?"];
            train = corpus(1:6);
            v = OceanNLP.FeatureExtraction.CountVectorizer();
            v.fit(train);
            Xnew = v.forward(corpus);
            Xref = OceanTest.OceanNLP.FeatureExtraction.CountVectorizerSpeedTest.originalForward_(v, corpus);
            testCase.verifyEqual(Xnew, Xref);
            testCase.verifyTrue(issparse(Xnew));
            testCase.verifyEqual(size(Xnew), [7 v.Vocabulary.VocabSize]);
            testCase.verifyGreaterThan(full(Xnew(7, v.Vocabulary.OOVId)), 0, 'out-of-vocabulary tokens land in the OOV column');
            testCase.verifyEqual(nnz(Xnew(3, :)), 0);                                   % empty documents give empty rows
            % a corpus with no tokens at all
            testCase.verifyEqual(v.forward(["" "  "]), sparse(2, v.Vocabulary.VocabSize));
            % randomised corpus, several vocabularies
            rs = RandStream('mt19937ar', 'Seed', 5);
            words = "w" + string(1:200);
            for rep = 1:3
                docs = strings(1, 40);
                for d = 1:40, docs(d) = strjoin(words(randi(rs, 200, 1, randi(rs, [0 30]))), " "); end
                v2 = OceanNLP.FeatureExtraction.CountVectorizer();  v2.fit(docs(1:30));
                testCase.verifyEqual(v2.forward(docs), OceanTest.OceanNLP.FeatureExtraction.CountVectorizerSpeedTest.originalForward_(v2, docs));
            end
        end

        function testVocabularyEncodeCacheFollowsRefitsAndKeepsFirstOccurrence(testCase)
            v = OceanNLP.Vocab.Vocabulary();
            v.fit({["a" "b" "b" "c"], ["c" "c" "c"]});
            first = v.encode(["a" "b" "c" "zzz"]);
            testCase.verifyEqual(first, [find(v.Tokens == "a") find(v.Tokens == "b") find(v.Tokens == "c") v.OOVId]);
            v.fit({["x" "y"]});                                                         % a re-fit must invalidate the cache
            testCase.verifyEqual(v.encode(["a" "x" "y"]), [v.OOVId find(v.Tokens == "x") find(v.Tokens == "y")]);
            s = v.stateDict();
            w = OceanNLP.Vocab.Vocabulary();  w.fit({["p" "q"]});  w.encode("p");       % warm the cache, then load another state
            w.loadStateDict(s);
            testCase.verifyEqual(w.encode(["x" "y" "p"]), [find(w.Tokens == "x") find(w.Tokens == "y") w.OOVId]);
            testCase.verifyEqual(size(v.encode(strings(1, 0))), [1 0]);
            testCase.verifyEqual(v.encode(["x" missing]), [find(v.Tokens == "x") v.OOVId]);   % a missing string is out of vocabulary
            % duplicates in a vocabulary built from tokens: the first occurrence wins, as ismember did
            u = OceanNLP.Vocab.Vocabulary();
            u.loadStateDict(struct('Tokens', ["<unk>" "a" "b" "a"], 'Frequencies', [0 1 1 1], 'OOVToken', "<unk>", 'IsFitted', true, ...
                'UnkIdx', 1, 'SpecialTokens', struct('unk', "<unk>")));
            testCase.verifyEqual(u.encode("a"), 2);
        end

        function testForwardDoesNotScaleWithVocabularySizeTimesDocuments(testCase)
            % 3000 documents x 60 tokens over a 40,000-word vocabulary. The original algorithm needed ~0.04 s per document
            % here (a vocabulary re-sort per document): well over a minute in total; the fast path needs a couple of seconds.
            rs = RandStream('mt19937ar', 'Seed', 3);
            words = "t" + string(1:40000);
            docs = strings(1, 3000);
            for d = 1:3000, docs(d) = strjoin(words(randi(rs, 40000, 1, 60)), " "); end
            v = OceanNLP.FeatureExtraction.CountVectorizer();
            tic;  v.fit(docs);  tFit = toc;
            tic;  X = v.forward(docs);  tFwd = toc;
            testCase.verifyEqual(size(X), [3000 v.Vocabulary.VocabSize]);
            testCase.verifyEqual(full(sum(X(:))), 3000 * 60);
            testCase.verifyLessThan(tFwd, 15, sprintf('forward took %.1fs', tFwd));
            testCase.verifyLessThan(tFit, 15, sprintf('fit took %.1fs', tFit));
        end
    end

    methods (Static, Access = private)
        function tokens = originalTokens_(txt, caseSensitive)
            % the original WhitespaceTokenizer: split on whitespace, strip edge punctuation, drop empties
            txt = strtrim(txt);
            if txt == "", tokens = strings(1, 0); return; end
            if ~caseSensitive, txt = lower(txt); end
            parts = string(regexp(txt, '\s+', 'split'));
            stripped = regexprep(parts, '^[^\w]+|[^\w]+$', '');
            stripped = stripped(stripped ~= "");
            tokens = reshape(stripped, 1, []);
        end

        function X = originalForward_(v, corpus)
            % the original CountVectorizer.forward: per document ismember + unique + accumarray, one sparse() at the end
            N = numel(corpus);  V = v.Vocabulary.VocabSize;
            I = cell(1, N);  J = I;  C = I;
            for d = 1:N
                toks = OceanTest.OceanNLP.FeatureExtraction.CountVectorizerSpeedTest.originalTokens_(corpus(d), false);
                [tf, loc] = ismember(toks, v.Vocabulary.Tokens);
                ids = double(loc);  ids(~tf) = v.Vocabulary.OOVId;
                if isempty(ids), I{d} = zeros(1, 0);  J{d} = zeros(1, 0);  C{d} = zeros(1, 0);  continue; end
                [u, ~, ic] = unique(ids);
                I{d} = repmat(d, 1, numel(u));  J{d} = u;  C{d} = accumarray(ic(:), 1)';
            end
            X = sparse([I{:}], [J{:}], [C{:}], N, V);
        end
    end
end
