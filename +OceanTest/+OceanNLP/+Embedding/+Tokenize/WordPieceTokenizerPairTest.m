classdef WordPieceTokenizerPairTest < OceanTest.BaseTestCase
    % WORDPIECETOKENIZERPAIRTEST  BasicTokenize punctuation splitting and
    % encodePair (ids, segment ids, truncation). Ids follow TinyBertFixture:
    % 2 [CLS] 3 [SEP] 7 cat 8 dog 9 , 10 . 11 a

    properties
        VocabFile
    end

    methods (TestMethodSetup)
        function makeVocab(testCase)
            testCase.VocabFile = OceanTest.OceanNLP.Embedding.TinyBertFixture.writeVocab();
            testCase.addTeardown(@() delete(testCase.VocabFile));
        end
    end

    methods (Test)
        function testDefaultDoesNotSplitPunctuation(testCase)
            % Backward compatibility: pre-existing callers see whitespace-only splitting.
            tok = OceanNLP.Embedding.Tokenize.WordPieceTokenizer('VocabFile', testCase.VocabFile);
            testCase.verifyEqual(tok.forward("cat, dog."), ["[UNK]" "[UNK]"]);
        end

        function testBasicTokenizeSplitsPunctuation(testCase)
            tok = OceanNLP.Embedding.Tokenize.WordPieceTokenizer('VocabFile', testCase.VocabFile, 'BasicTokenize', true);
            testCase.verifyEqual(tok.forward("cat, dog."), ["cat" "," "dog" "."]);
            testCase.verifyEqual(tok.forward("  Cat,DOG.  "), ["cat" "," "dog" "."]);   % lowercased, no spaces needed
        end

        function testEncodePairHandWorked(testCase)
            tok = OceanNLP.Embedding.Tokenize.WordPieceTokenizer('VocabFile', testCase.VocabFile, 'BasicTokenize', true);
            [ids, seg] = tok.encodePair("cat", "dog a");
            testCase.verifyEqual(ids, [2 7 3 8 11 3]);        % [CLS] cat [SEP] dog a [SEP]
            testCase.verifyEqual(seg, [0 0 0 1 1 1]);
        end

        function testLongestFirstTruncationRemovesFromLongerSide(testCase)
            tok = OceanNLP.Embedding.Tokenize.WordPieceTokenizer('VocabFile', testCase.VocabFile, 'BasicTokenize', true);
            % q=2 tokens, d=5 tokens, MaxLength 8 -> budget 5 -> trim doc 5->3
            [ids, seg] = tok.encodePair("cat dog", "a a a a a", 'MaxLength', 8);
            testCase.verifyEqual(ids, [2 7 8 3 11 11 11 3]);
            testCase.verifyEqual(numel(seg), 8);
            % both long: q=4, d=4, budget 5. HF splits the budget evenly and gives the odd
            % extra token to the (originally) longer side, the document on a tie -> q=2, d=3.
            [ids2, seg2] = tok.encodePair("cat cat cat cat", "dog dog dog dog", 'MaxLength', 8);
            testCase.verifyEqual(ids2, [2 7 7 3 8 8 8 3]);
            testCase.verifyEqual(seg2, [0 0 0 0 1 1 1 1]);
        end

        function testLongestFirstMatchesTransformersReferenceTable(testCase)
            % (queryLen, docLen, MaxLength) -> (queryLen, docLen) after truncation, taken from
            % transformers 5.16.1 (all-MiniLM-L6-v2 tokenizer, truncation=True). A token-by-token
            % "trim the longer side" loop gets several of these wrong.
            table = [ 7 5  10  4 3;    % budget 7, odd: extra token to the longer side (query)
                      5 7  10  3 4;    % ... to the longer side (doc)
                      6 6  10  3 4;    % tie: extra token to the document
                      8 4  10  4 3;
                      3 2   6  2 1;
                      2 2   6  1 2;
                      4 4   8  2 3;
                      9 3  10  4 3];
            tok = OceanNLP.Embedding.Tokenize.WordPieceTokenizer('VocabFile', testCase.VocabFile, 'BasicTokenize', true);
            for r = 1:size(table, 1)
                q = strjoin(repmat("cat", 1, table(r, 1)), " ");
                d = strjoin(repmat("dog", 1, table(r, 2)), " ");
                [ids, ~] = tok.encodePair(q, d, 'MaxLength', table(r, 3));
                testCase.verifyEqual([nnz(ids == 7), nnz(ids == 8)], table(r, 4:5), ...
                    sprintf('case %d: q=%d d=%d MaxLength=%d', r, table(r, 1:3)));
            end
        end

        function testOnlySecondErrorsWhenQueryAloneExceedsBudget(testCase)
            tok = OceanNLP.Embedding.Tokenize.WordPieceTokenizer('VocabFile', testCase.VocabFile, 'BasicTokenize', true);
            testCase.verifyError(@() tok.encodePair("cat cat cat cat cat cat", "dog", 'MaxLength', 8, 'Truncation', "only_second"), ...
                'OceanNLP:Embedding:Tokenize:WordPieceTokenizer:QueryTooLong');
        end

        function testWordLongerThanHundredCharsIsUnk(testCase)
            % Needs a vocab where a long word WOULD decompose ("a", "##a"), so the
            % 100-char rule (HF max_input_chars_per_word) is what forces [UNK].
            f = string(tempname) + ".txt";
            fid = fopen(f, 'w'); fprintf(fid, '%s\n', ["[UNK]" "a" "##a"]); fclose(fid);
            testCase.addTeardown(@() delete(f));
            tok = OceanNLP.Embedding.Tokenize.WordPieceTokenizer('VocabFile', f);
            testCase.verifyEqual(numel(tok.forward(string(repmat('a', 1, 100)))), 100);   % at the limit: decomposes
            testCase.verifyEqual(tok.forward(string(repmat('a', 1, 101))), "[UNK]");
        end

        function testOnlySecondTruncatesDocumentOnly(testCase)
            tok = OceanNLP.Embedding.Tokenize.WordPieceTokenizer('VocabFile', testCase.VocabFile, 'BasicTokenize', true);
            [ids, ~] = tok.encodePair("cat cat cat cat", "dog dog dog dog", 'MaxLength', 8, 'Truncation', "only_second");
            testCase.verifyEqual(ids, [2 7 7 7 7 3 8 3]);       % budget 5, query keeps 4, doc gets 1
        end
    end
end
