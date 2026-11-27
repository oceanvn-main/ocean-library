classdef VocabularyTest < OceanTest.BaseTestCase
    % VOCABULARYTEST  Unit tests for OceanNLP.Vocab.Vocabulary.

    methods (Test)
        function testDeterministicTieBreakOrdering(testCase)
            % Fixed input: "the"=3, "sat"=2, "cat"=1, "dog"=1, "mat"=1.
            % Rule: descending frequency, ties broken alphabetically
            % ascending. Expected order: the, sat, cat, dog, mat.
            import OceanNLP.Vocab.Vocabulary
            vocab = Vocabulary();
            docs = {["the" "cat" "sat" "the" "mat"], ["the" "dog" "sat"]};
            vocab.fit(docs);

            testCase.verifyEqual(vocab.Tokens, ["<unk>" "the" "sat" "cat" "dog" "mat"]);
            testCase.verifyEqual(vocab.Frequencies, [0 3 2 1 1 1]);
            testCase.verifyEqual(vocab.VocabSize, 6);
        end

        function testFitIsReproducibleAcrossRepeatedCalls(testCase)
            import OceanNLP.Vocab.Vocabulary
            vocab = Vocabulary();
            docs = {["the" "cat" "sat" "the" "mat"], ["the" "dog" "sat"]};

            vocab.fit(docs);
            tokensFirst = vocab.Tokens;
            vocab.fit(docs);   % re-fit clobbers and rebuilds from scratch
            tokensSecond = vocab.Tokens;

            testCase.verifyEqual(tokensFirst, tokensSecond);
        end

        function testOOVTokenMapsToReservedId1(testCase)
            import OceanNLP.Vocab.Vocabulary
            vocab = Vocabulary();
            vocab.fit({["the" "cat" "sat" "the" "mat"], ["the" "dog" "sat"]});

            testCase.verifyEqual(vocab.OOVId, 1);
            testCase.verifyEqual(vocab.Tokens(1), vocab.OOVToken);

            ids = vocab.encode(["the" "zzz-never-seen" "cat"]);
            testCase.verifyEqual(ids, [2 1 4]);
        end

        function testReservedTokenCollisionErrors(testCase)
            import OceanNLP.Vocab.Vocabulary
            vocab = Vocabulary();   % default OOVToken = "<unk>"
            testCase.verifyError(@() vocab.fit({["hello" "<unk>"]}), ...
                'OceanNLP:Vocab:Vocabulary:ReservedTokenCollision');
        end

        function testCustomOOVTokenAvoidsCollision(testCase)
            import OceanNLP.Vocab.Vocabulary
            vocab = Vocabulary('OOVToken', "[UNK]");
            vocab.fit({["hello" "<unk>" "world"]});   % legitimate literal "<unk>"

            testCase.verifyTrue(any(vocab.Tokens == "<unk>"));
            testCase.verifyEqual(vocab.OOVToken, "[UNK]");
        end

        function testEncodeDecodeRoundTrip(testCase)
            import OceanNLP.Vocab.Vocabulary
            vocab = Vocabulary();
            vocab.fit({["the" "cat" "sat" "the" "mat"], ["the" "dog" "sat"]});

            realTokens = vocab.Tokens(2:end);   % skip reserved OOV slot
            ids = vocab.encode(realTokens);
            roundTripped = vocab.decode(ids);

            testCase.verifyEqual(roundTripped, realTokens);
        end

        function testStateDictLoadStateDictRoundTrip(testCase)
            import OceanNLP.Vocab.Vocabulary
            vocab1 = Vocabulary();
            vocab1.fit({["the" "cat" "sat" "the" "mat"], ["the" "dog" "sat"]});

            s = vocab1.stateDict();
            vocab2 = Vocabulary();
            vocab2.loadStateDict(s);

            testCase.verifyEqual(vocab2.Tokens, vocab1.Tokens);
            testCase.verifyEqual(vocab2.Frequencies, vocab1.Frequencies);
            testCase.verifyTrue(vocab2.isFitted());
            testCase.verifyEqual(vocab2.encode(["the" "cat"]), vocab1.encode(["the" "cat"]));
        end

        function testNotFittedErrorsOnEncodeAndDecode(testCase)
            import OceanNLP.Vocab.Vocabulary
            vocab = Vocabulary();

            testCase.verifyFalse(vocab.isFitted());
            testCase.verifyError(@() vocab.encode(["a"]), ...
                'OceanNLP:Vocab:Vocabulary:NotFitted');
            testCase.verifyError(@() vocab.decode([2]), ...
                'OceanNLP:Vocab:Vocabulary:NotFitted');
        end
    end
end
