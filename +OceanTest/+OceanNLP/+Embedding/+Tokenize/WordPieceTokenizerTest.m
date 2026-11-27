classdef WordPieceTokenizerTest < OceanTest.BaseTestCase
    % WORDPIECETOKENIZERTEST  Unit tests for
    % OceanNLP.Embedding.Tokenize.WordPieceTokenizer.
    %
    % SYNTHETIC VOCAB.TXT (one token per line -- line NUMBER, 0-indexed, IS
    % the id; built fresh in TestClassSetup, cleaned up via addTeardown):
    %   line0 (id0) = [PAD]
    %   line1 (id1) = [UNK]
    %   line2 (id2) = [CLS]
    %   line3 (id3) = [SEP]
    %   line4 (id4) = the
    %   line5 (id5) = un
    %   line6 (id6) = ##aff
    %   line7 (id7) = ##able
    %   line8 (id8) = run
    %
    % Deliberately does NOT include "##ning"/"##nin"/"##ni"/"##n" -- this is
    % what makes the "running" case below a genuine discard-partial-match
    % failure (a first piece DOES match ("run"), but the remainder has no
    % valid continuation), not a trivial zero-match case.

    properties
        VocabFile
    end

    methods (TestClassSetup)
        function buildVocabFile(testCase)
            tokens = ["[PAD]"; "[UNK]"; "[CLS]"; "[SEP]"; "the"; "un"; "##aff"; "##able"; "run"];
            vocabPath = [tempname() '.txt'];
            fid = fopen(vocabPath, 'w');
            testCase.assertNotEqual(fid, -1, 'Failed to create temp vocab file.');
            for i = 1:numel(tokens)
                fprintf(fid, '%s\n', tokens(i));
            end
            fclose(fid);
            testCase.addTeardown(@() delete(vocabPath));
            testCase.VocabFile = vocabPath;
        end
    end

    methods (Test)
        function testWholeWordMatchWinsImmediately(testCase)
            import OceanNLP.Embedding.Tokenize.WordPieceTokenizer
            tok = WordPieceTokenizer('VocabFile', testCase.VocabFile);

            tokens = tok.forward("the");
            testCase.verifyEqual(tokens, "the");

            ids = tok.encode(tokens);
            testCase.verifyEqual(ids, 4);   % vocab.txt line 4 (0-indexed)
        end

        function testUnaffableStyleGreedyLongestMatchDecomposition(testCase)
            % Classic worked example: "un" + "##aff" + "##able" (none of
            % which is "unaffable" itself, present as a whole word).
            import OceanNLP.Embedding.Tokenize.WordPieceTokenizer
            tok = WordPieceTokenizer('VocabFile', testCase.VocabFile);

            tokens = tok.forward("unaffable");
            testCase.verifyEqual(tokens, ["un" "##aff" "##able"]);

            ids = tok.encode(tokens);
            testCase.verifyEqual(ids, [5 6 7]);   % vocab.txt lines 5,6,7 (0-indexed)
        end

        function testRunningStyleDiscardedPartialMatchFallsBackToUNK(testCase)
            % HARDER UNK CASE: "run" DOES match as a valid first piece, but
            % the remainder "ning" has no valid "##"-prefixed continuation
            % anywhere in this vocab -- the algorithm must DISCARD the
            % successful "run" partial match and emit a single UNK for the
            % WHOLE word, never ["run", "[UNK]"] or any partial-piece list.
            import OceanNLP.Embedding.Tokenize.WordPieceTokenizer
            tok = WordPieceTokenizer('VocabFile', testCase.VocabFile);

            tokens = tok.forward("running");
            testCase.verifyEqual(tokens, "[UNK]");   % NOT ["run", ...] or 2+ tokens
            testCase.verifySize(tokens, [1 1]);

            ids = tok.encode(tokens);
            testCase.verifyEqual(ids, tok.UnkId);
        end

        function testUnkIdReadFromActualLoadedVocabNotHardcoded(testCase)
            % UnkId must reflect THIS vocab's actual line position of
            % [UNK] (line 1, 0-indexed), not BERT's usual real-checkpoint
            % value of 100.
            import OceanNLP.Embedding.Tokenize.WordPieceTokenizer
            tok = WordPieceTokenizer('VocabFile', testCase.VocabFile);
            testCase.verifyEqual(tok.UnkId, 1);
        end

        function testEncodeIs0IndexedAgainstActualVocabLineNumbers(testCase)
            import OceanNLP.Embedding.Tokenize.WordPieceTokenizer
            tok = WordPieceTokenizer('VocabFile', testCase.VocabFile);
            ids = tok.encode(["[PAD]" "[UNK]" "[CLS]" "[SEP]"]);
            testCase.verifyEqual(ids, [0 1 2 3]);   % vocab.txt lines 0-3, 0-indexed
        end

        function testVocabSizeMatchesLineCount(testCase)
            import OceanNLP.Embedding.Tokenize.WordPieceTokenizer
            tok = WordPieceTokenizer('VocabFile', testCase.VocabFile);
            testCase.verifyEqual(tok.VocabSize, 9);
        end

        function testMultiWordSentenceTokenizesEachWordIndependently(testCase)
            import OceanNLP.Embedding.Tokenize.WordPieceTokenizer
            tok = WordPieceTokenizer('VocabFile', testCase.VocabFile);
            tokens = tok.forward("the unaffable running");
            testCase.verifyEqual(tokens, ["the" "un" "##aff" "##able" "[UNK]"]);
        end

        function testCaseFoldingLowercasesInput(testCase)
            import OceanNLP.Embedding.Tokenize.WordPieceTokenizer
            tok = WordPieceTokenizer('VocabFile', testCase.VocabFile);
            tokens = tok.forward("THE");
            testCase.verifyEqual(tokens, "the");
        end

        function testEmptyInputReturnsEmptyTokenArray(testCase)
            import OceanNLP.Embedding.Tokenize.WordPieceTokenizer
            tok = WordPieceTokenizer('VocabFile', testCase.VocabFile);
            tokens = tok.forward("   ");
            testCase.verifyEmpty(tokens);
        end

        function testMissingVocabFileErrors(testCase)
            import OceanNLP.Embedding.Tokenize.WordPieceTokenizer
            testCase.verifyError(@() WordPieceTokenizer('VocabFile', "no_such_vocab_file_xyz.txt"), ...
                'OceanNLP:Embedding:Tokenize:WordPieceTokenizer:VocabFileNotFound');
        end

        function testNoFitMethodConsistentWithFixedVocabularyPolicy(testCase)
            % WordPieceTokenizer's vocabulary is fixed at construction --
            % it must NOT expose a fit() method (unlike OceanNLP.Vocab.Vocabulary).
            import OceanNLP.Embedding.Tokenize.WordPieceTokenizer
            tok = WordPieceTokenizer('VocabFile', testCase.VocabFile);
            testCase.verifyFalse(ismethod(tok, 'fit'));
        end
    end
end
