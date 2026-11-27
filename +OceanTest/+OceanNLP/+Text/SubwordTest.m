classdef SubwordTest < OceanTest.BaseTestCase
    % SUBWORDTEST  ByteLevel, BpeAlgorithm, WordPieceAlgorithm, SubwordTokenizer and the
    % pre-tokenizers. Hand-worked expectations; the real-data comparison against
    % HuggingFace is compare_subword_in_matlab.m / compare_text_in_matlab.m.

    methods (Test)
        % ---------------- pre-tokenizers ----------------
        function testGpt2PreTokenizerHandWorked(testCase)
            p = OceanNLP.Text.PreTokenize.Gpt2PreTokenizer();
            testCase.verifyEqual(p.forward("Hello world's  it's 42!!"), ...
                ["Hello" " world" "'s" " " " it" "'s" " 42" "!!"]);      % double space: 1st alone, 2nd joins " it"
            testCase.verifyEqual(p.forward("a  "), ["a" "  "]);          % trailing run kept whole
            testCase.verifyEqual(p.forward(""), strings(1, 0));
        end

        function testWhitespaceAndBertBasicPreTokenizers(testCase)
            testCase.verifyEqual(OceanNLP.Text.PreTokenize.WhitespacePreTokenizer().forward("  a  b	c "), ["a" "b" "c"]);
            b = OceanNLP.Text.PreTokenize.BertBasicPreTokenizer('NeverSplit', strings(1, 0));
            testCase.verifyEqual(b.forward("Hello, World!"), ["hello" "," "world" "!"]);
            c = OceanNLP.Text.PreTokenize.BertBasicPreTokenizer('Lowercase', false);
            testCase.verifyEqual(c.forward("Hello, [MASK]"), ["Hello" "," "[MASK]"]);           % NeverSplit default keeps [MASK]
            cafe = "caf" + string(char(233));
            testCase.verifyEqual(b.forward(cafe), "cafe");                                       % accent stripped
            testCase.verifyEqual(c.forward(cafe), string(cafe));                                 % cased: accent kept
        end

        % ---------------- byte level ----------------
        function testByteLevelAlphabetAndKnownMappings(testCase)
            a = OceanNLP.Subword.ByteLevel.alphabet();
            testCase.verifyEqual(numel(unique(a)), 256);
            testCase.verifyEqual(double(char(OceanNLP.Subword.ByteLevel.encode(" "))), 288);     % space -> U+0120
            testCase.verifyEqual(double(char(OceanNLP.Subword.ByteLevel.encode(newline))), 266);  % LF -> U+010A
            testCase.verifyEqual(OceanNLP.Subword.ByteLevel.encode("abc"), "abc");                % printable ASCII unchanged
        end

        function testByteLevelRoundTripIncludingMultibyteAndEmoji(testCase)
            emoji = string(char([55357 56832]));
            for t = ["hello world", "caf" + string(char(233)), "x " + emoji + " y", "tab" + string(char(9)) + "z", ""]
                testCase.verifyEqual(OceanNLP.Subword.ByteLevel.decode(OceanNLP.Subword.ByteLevel.encode(t)), t);
            end
            testCase.verifyEqual(strlength(OceanNLP.Subword.ByteLevel.encode(string(char(233)))), 2);   % 2 UTF-8 bytes
            testCase.verifyError(@() OceanNLP.Subword.ByteLevel.decode("a" + string(char(1000))), 'OceanNLP:Subword:ByteLevel:NotMapped');
        end

        % ---------------- BPE ----------------
        function testBpeTrainingHandWorked(testCase)
            % abab x3, abc x1:  (a,b)=7, (b,a)=3, (b,c)=1 -> merge (a,b); then (ab,ab)=3, (ab,c)=1 -> merge (ab,ab); then (ab,c)
            BPE = OceanNLP.Subword.Algorithm.BpeAlgorithm();
            [m, alpha] = BPE.trainMerges(["abab" "abc"], [3 1], 6);
            testCase.verifyEqual(alpha, ["a" "b" "c"]);
            testCase.verifyEqual(m, ["a" "b"; "ab" "ab"; "ab" "c"]);
            m2 = BPE.trainMerges(["abab" "abc"], [3 1], 5);
            testCase.verifyEqual(m2, ["a" "b"; "ab" "ab"]);
        end

        function testBpeOverlappingPairsMergeLeftToRight(testCase)
            BPE = OceanNLP.Subword.Algorithm.BpeAlgorithm();
            m = BPE.trainMerges("aaa", 1, 3);                       % alphabet {a}; (a,a) count 2 -> "aa","a"
            testCase.verifyEqual(m, ["a" "a"; "aa" "a"]);
            m4 = BPE.trainMerges("aaaa", 1, 3);
            testCase.verifyEqual(m4, ["a" "a"; "aa" "aa"]);         % non-overlapping: aa|aa
        end

        function testBpeSegmentAppliesLowestRankFirst(testCase)
            algo = OceanNLP.Subword.Algorithm.BpeAlgorithm();
            algo.setMerges(["a" "b"; "ab" "c"]);
            testCase.verifyEqual(algo.segment("abcab"), ["abc" "ab"]);
            testCase.verifyEqual(algo.segment("cab"), ["c" "ab"]);
            testCase.verifyEqual(algo.segment("xyz"), ["x" "y" "z"]);
            testCase.verifyEqual(algo.segment(""), strings(1, 0));
            % a later-rank merge must not pre-empt an earlier one: (b,c) rank1 beats (a,b) rank2 on "abc"
            algo2 = OceanNLP.Subword.Algorithm.BpeAlgorithm();
            algo2.setMerges(["b" "c"; "a" "b"]);
            testCase.verifyEqual(algo2.segment("abc"), ["a" "bc"]);
        end

        function testBpeEndOfWordSuffix(testCase)
            BPE = OceanNLP.Subword.Algorithm.BpeAlgorithm();
            m = BPE.trainMerges(["lo" "lo" "low"], [1 1 1], 20, strings(1, 0), 0, "</w>");
            testCase.verifyEqual(m(1, :), ["l" "o</w>"]);          % (l,o</w>)=2 beats (l,o)=1 and (o,w</w>)=1
            algo = OceanNLP.Subword.Algorithm.BpeAlgorithm('EndOfWordSuffix', "</w>");
            algo.setMerges(["o" "</w>"]);
            testCase.verifyEqual(algo.segment("lo"), ["l" "o</w>"]);                               % suffix glued to the last symbol
        end

        function testBpeTrainingStatsExposeMaximumCountChecks(testCase)
            BPE = OceanNLP.Subword.Algorithm.BpeAlgorithm();
            [~, ~, st] = BPE.trainMerges(["abab" "abc"], [3 1], 6);
            testCase.verifyEqual(st.PairCount, st.MaxCount);          % chosen pair IS the maximum each step
            [~, ~, st2] = BPE.trainMerges(["abab" "abc"], [3 1], 6, strings(1, 0), 0, "", ["b" "c"; "a" "b"]);
            testCase.verifyLessThan(st2.PairCount(1), st2.MaxCount(1));   % forcing a non-maximal pair is detected
        end

        function testBpeMinFrequencyStopsEarly(testCase)
            BPE = OceanNLP.Subword.Algorithm.BpeAlgorithm();
            m = BPE.trainMerges(["abab" "abc"], [3 1], 10, strings(1, 0), 4);
            testCase.verifyEqual(m, ["a" "b"]);                       % only (a,b)=7 reaches 4
        end

        % ---------------- WordPiece ----------------
        function testWordPieceGreedySegmentation(testCase)
            wp = OceanNLP.Subword.Algorithm.WordPieceAlgorithm.fromPieces( ...
                ["[UNK]" "un" "##able" "##believ" "##ably" "##un" "able" "a"]);
            testCase.verifyEqual(wp.segment("unable"), ["un" "##able"]);
            testCase.verifyEqual(wp.segment("unbelievably"), ["un" "##believ" "##ably"]);
            testCase.verifyEqual(wp.segment("able"), "able");                    % whole word wins
            testCase.verifyEqual(wp.segment("zzz"), "[UNK]");                    % no decomposition
            testCase.verifyEqual(wp.segment("unzz"), "[UNK]");                   % partial matches discarded
            testCase.verifyEqual(wp.segment(string(repmat('a', 1, 101))), "[UNK]");   % > 100 chars
            testCase.verifyEqual(wp.segment(""), strings(1, 0));
        end

        function testWordPieceTrainingLearnsPrefixedPieces(testCase)
            words = ["hug" "pug" "pun" "bun" "hugs"];  counts = [10 5 12 4 5];
            wp = OceanNLP.Subword.Algorithm.WordPieceAlgorithm();
            wp.train(words, counts, 'VocabSize', 20);
            toks = wp.tokens();
            testCase.verifyTrue(all(ismember(["h" "p" "b" "##u" "##g" "##n" "##s"], toks)));   % alphabet, prefixed continuations
            testCase.verifyLessThanOrEqual(numel(toks), 20);
            for w = words                                                             % every training word is segmentable, losslessly
                pieces = wp.segment(w);
                joined = pieces(1) + strjoin(erase(pieces(2:end), "##"), "");
                testCase.verifyEqual(joined, w);
            end
        end

        % ---------------- SubwordTokenizer ----------------
        function testSubwordTokenizerFitEncodeDecodeRoundTrip(testCase)
            corpus = ["the cat sat on the mat", "the dog sat on the log", "cats and dogs"];
            tok = OceanNLP.Subword.SubwordTokenizer();
            tok.fit(corpus, 'VocabSize', 280);
            for t = [corpus, "unseen words like caf" + string(char(233)) + " and emoji " + string(char([55357 56832]))]
                ids = tok.encode(t);
                testCase.verifyEqual(tok.decode(ids), t);                            % byte-level: lossless on ANY text
                testCase.verifyTrue(all(ids >= 0 & ids < tok.Vocab.VocabSize));
            end
            testCase.verifyEqual(tok.Vocab.token0(tok.encode("the")), tok.forward("the"));
        end

        function testSubwordTokenizerEnforcesFitAndWarnsOnRefit(testCase)
            tok = OceanNLP.Subword.SubwordTokenizer();
            testCase.verifyError(@() tok.forward("x"), 'OceanNLP:Subword:SubwordTokenizer:NotFitted');
            tok.fit("a b a b", 'VocabSize', 260);
            testCase.verifyWarning(@() tok.fit("c d c d", 'VocabSize', 260), 'OceanNLP:Subword:SubwordTokenizer:RefitOverwrites');
        end

        function testTwoTokenizersDoNotShareAlgorithmState(testCase)
            a = OceanNLP.Subword.SubwordTokenizer();  b = OceanNLP.Subword.SubwordTokenizer();
            a.fit("aa bb aa bb aa bb", 'VocabSize', 270);
            testCase.verifyFalse(b.isFitted());
            testCase.verifyEmpty(b.Algorithm.Merges);
            testCase.verifyFalse(a.Algorithm == b.Algorithm);
        end

        function testNonByteLevelWordPieceTokenizerViaStrategies(testCase)
            tok = OceanNLP.Subword.SubwordTokenizer('PreTokenizer', OceanNLP.Text.PreTokenize.BertBasicPreTokenizer(), ...
                'Algorithm', OceanNLP.Subword.Algorithm.WordPieceAlgorithm(), 'ByteLevel', false, 'Special', struct('pad', "[PAD]"));
            tok.fit(["hug pug pun bun hugs hug"], 'VocabSize', 30);
            testCase.verifyEqual(tok.Vocab.specialId("unk"), 0);           % unk comes from the ALGORITHM (WordPiece: [UNK])
            testCase.verifyEqual(tok.Vocab.specialId("pad"), 1);           % role -> token, looked up by role
            testCase.verifyEqual(tok.forward("hug"), "hug");
            testCase.verifyEqual(tok.forward("zebra"), "[UNK]");
            testCase.verifyEqual(tok.encode("zebra"), 0);
            testCase.verifyEqual(tok.Vocab.token0(tok.encode("zebra")), tok.forward("zebra"));   % forward/encode agree on unk
            testCase.verifyEqual(tok.decode(tok.encode("hug pun hugs")), "hug pun hugs");         % WordPiece detokenize round trip
        end

        function testSpecialsAreRolesNotPositions(testCase)
            tok = OceanNLP.Subword.SubwordTokenizer('Special', struct('pad', "[PAD]", 'unk', "[UNK]", 'cls', "[CLS]"));
            tok.fit("ab ab ab", 'VocabSize', 270);
            testCase.verifyEqual(tok.Vocab.specialId("pad"), 0);
            testCase.verifyEqual(tok.Vocab.specialId("unk"), 1);
            testCase.verifyEqual(tok.Vocab.specialId("cls"), 2);           % roles, not s1/s2/s3
            testCase.verifyEqual(tok.Vocab.UnkId, 2);                      % the "unk" role is the unknown token, not the first special
            testCase.verifyEqual(tok.Vocab.token0(0:2), ["[PAD]" "[UNK]" "[CLS]"]);
        end

        function testByteLevelWithoutSpecialsGetsFallbackUnkRole(testCase)
            tok = OceanNLP.Subword.SubwordTokenizer();
            tok.fit("ab ab", 'VocabSize', 260);
            testCase.verifyEqual(tok.Vocab.token0(tok.Vocab.specialId("unk")), "<unk>");
        end

        function testUntrainedWordPieceAndTokenizerErrorClearly(testCase)
            wp = OceanNLP.Subword.Algorithm.WordPieceAlgorithm();
            testCase.verifyError(@() wp.segment("x"), 'OceanNLP:Subword:WordPieceAlgorithm:NotTrained');
        end

        function testWordPieceLongWordLimitCountsCodePointsNotUtf16Units(testCase)
            wp = OceanNLP.Subword.Algorithm.WordPieceAlgorithm.fromPieces(["[UNK]" "a" "##a"], 'UnkToken', "[UNK]");
            astral = string(char([55357 56832]));                            % 1 code point, 2 UTF-16 units
            w = strjoin(repmat("a", 1, 99), "") + astral;                    % 100 code points (101 UTF-16 units)
            testCase.verifyEqual(wp.segment(w), "[UNK]");                    % not over the limit -> decomposes -> the emoji has no piece -> UNK anyway
            testCase.verifyEqual(numel(wp.segment(strjoin(repmat("a", 1, 100), ""))), 100);   % exactly 100 is allowed
        end

        function testFactoryDiscoversSubwordAlgorithmsByConvention(testCase)
            testCase.verifyEqual(OceanNLP.Text.Factory.validNames("subword"), {'bpe', 'wordpiece'});
            a = OceanNLP.Text.Factory.create("subword", "bpe");
            testCase.verifyClass(a, 'OceanNLP.Subword.Algorithm.BpeAlgorithm');
            tok = OceanNLP.Subword.SubwordTokenizer('Algorithm', OceanNLP.Text.Factory.create("subword", "wordpiece"), 'ByteLevel', false);
            testCase.verifyEqual(tok.name(), "subword:wordpiece");
        end

        function testSubwordTokenizerPlugsIntoCountVectorizer(testCase)
            tok = OceanNLP.Subword.SubwordTokenizer();
            tok.fit(["cats sit", "dogs sit"], 'VocabSize', 270);
            cv = OceanNLP.FeatureExtraction.CountVectorizer('Tokenizer', tok);
            cv.fit(["cats sit", "dogs sit"]);
            X = cv.forward("cats sit");
            testCase.verifyEqual(size(X, 1), 1);
            testCase.verifyGreaterThan(nnz(X), 0);
        end
    end
end
