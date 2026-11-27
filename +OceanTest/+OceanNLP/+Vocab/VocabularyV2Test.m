classdef VocabularyV2Test < OceanTest.BaseTestCase
    % VOCABULARYV2TEST  v2 additions to OceanNLP.Vocab.Vocabulary: fit options,
    % fixed vocabularies (fromTokens / vocab.txt / vocab.json), 0-indexed ids,
    % special-token registry. Backward compatibility of the original API is
    % covered by the untouched VocabularyTest.

    methods (Test)
        function testFitDefaultsUnchangedByNewOptions(testCase)
            docs = {["b" "a" "b"], ["c" "b" "a"]};
            v1 = OceanNLP.Vocab.Vocabulary();  v1.fit(docs);
            v2 = OceanNLP.Vocab.Vocabulary();  v2.fit(docs, 'MinCount', 1, 'MaxSize', Inf);
            testCase.verifyEqual(v1.Tokens, v2.Tokens);
            testCase.verifyEqual(v1.Tokens, ["<unk>" "b" "a" "c"]);
            testCase.verifyEqual(v1.Frequencies, [0 3 2 1]);
        end

        function testMinCountDropsRareTokens(testCase)
            v = OceanNLP.Vocab.Vocabulary();
            v.fit({["b" "a" "b"], ["c" "b" "a"]}, 'MinCount', 2);
            testCase.verifyEqual(v.Tokens, ["<unk>" "b" "a"]);
            testCase.verifyEqual(v.encode("c"), 1);                      % dropped -> OOV id
        end

        function testMaxSizeCapsIncludingOovAndBreaksTiesAlphabetically(testCase)
            v = OceanNLP.Vocab.Vocabulary();
            v.fit({["z" "y" "x" "w"]}, 'MaxSize', 3);                    % all count 1 -> keep alphabetical: w, x
            testCase.verifyEqual(v.Tokens, ["<unk>" "w" "x"]);
            testCase.verifyEqual(v.VocabSize, 3);
        end

        function testFromTokensZeroIndexedRoundTripAndUnk(testCase)
            toks = ["[PAD]" "[UNK]" "[CLS]" "cat" "dog"];
            v = OceanNLP.Vocab.Vocabulary.fromTokens(toks, 'Unk', "[UNK]", ...
                'Special', struct('cls', "[CLS]", 'pad', "[PAD]"));
            testCase.verifyEqual(v.id0(["cat" "dog" "zzz"]), [3 4 1]);   % unseen -> unk (0-indexed id 1)
            testCase.verifyEqual(v.token0([3 4 0]), ["cat" "dog" "[PAD]"]);
            testCase.verifyEqual(v.specialId("cls"), 2);
            testCase.verifyEqual(v.specialId("unk"), 1);
            testCase.verifyEqual(v.UnkId, 2);                            % 1-indexed internal position
            testCase.verifyTrue(v.contains("cat"));
            testCase.verifyFalse(v.contains("zzz"));
            % 1-indexed encode() keeps working on fixed vocabularies and honours the unk position
            testCase.verifyEqual(v.encode(["cat" "zzz"]), [4 2]);
        end

        function testFromTokensValidation(testCase)
            testCase.verifyError(@() OceanNLP.Vocab.Vocabulary.fromTokens(["a" "a" "<unk>"]), ...
                'OceanNLP:Vocab:Vocabulary:DuplicateToken');
            testCase.verifyError(@() OceanNLP.Vocab.Vocabulary.fromTokens(["a" "b"]), ...
                'OceanNLP:Vocab:Vocabulary:UnkNotInVocab');
            testCase.verifyError(@() OceanNLP.Vocab.Vocabulary.fromTokens(["a" "<unk>"], 'Special', struct('cls', "[CLS]")), ...
                'OceanNLP:Vocab:Vocabulary:SpecialNotInVocab');
            v = OceanNLP.Vocab.Vocabulary.fromTokens(["a" "<unk>"]);
            testCase.verifyError(@() v.specialId("mask"), 'OceanNLP:Vocab:Vocabulary:UnknownSpecial');
            testCase.verifyError(@() v.token0(5), 'OceanNLP:Vocab:Vocabulary:IdOutOfRange');
        end

        function testVocabTxtSaveLoadRoundTrip(testCase)
            f = string(tempname) + ".txt";
            testCase.addTeardown(@() delete(f));
            v = OceanNLP.Vocab.Vocabulary.fromTokens(["[UNK]" "cat" "##s" "caf" + string(char(233))], 'Unk', "[UNK]");
            v.saveVocabTxt(f);
            w = OceanNLP.Vocab.Vocabulary.fromVocabTxt(f, 'Unk', "[UNK]");
            testCase.verifyEqual(w.Tokens, v.Tokens);                    % incl. non-ASCII token
            testCase.verifyEqual(w.id0("##s"), 2);
        end

        function testVocabTxtToleratesBomAndCrlf(testCase)
            f = string(tempname) + ".txt";
            testCase.addTeardown(@() delete(f));
            fid = fopen(f, 'w', 'n', 'UTF-8');
            fwrite(fid, [239 187 191], 'uint8');                       % UTF-8 BOM
            fprintf(fid, '[UNK]\r\ncat\r\ndog\r\n');
            fclose(fid);
            v = OceanNLP.Vocab.Vocabulary.fromVocabTxt(f, 'Unk', "[UNK]");
            testCase.verifyEqual(v.Tokens, ["[UNK]" "cat" "dog"]);       % no BOM glued to the first token, no CR
        end

        function testStateDictCarriesVersionTag(testCase)
            v = OceanNLP.Vocab.Vocabulary();  v.fit({["a" "b"]});
            testCase.verifyEqual(v.stateDict().Version, 2);
        end

        function testVocabJsonParsesEscapesAndOrdersById(testCase)
            f = string(tempname) + ".json";
            testCase.addTeardown(@() delete(f));
            % ids deliberately out of order; keys use \u escapes, an escaped quote and backslash
            bs = char(92);                                              % backslash built from its code so this file has no escape ambiguity
            txt = ['{"b":1,"' bs 'u0120the":2,"a' bs '"q":3,"x' bs bs 'y":4,"<|endoftext|>":0}'];
            fid = fopen(f, 'w'); fprintf(fid, '%s', txt); fclose(fid);
            v = OceanNLP.Vocab.Vocabulary.fromVocabJson(f);
            testCase.verifyEqual(v.Tokens, ["<|endoftext|>" "b" string(char(288)) + "the" "a""q" "x" + string(bs) + "y"]);
            testCase.verifyEqual(v.id0(string(char(288)) + "the"), 2);
            testCase.verifyEqual(v.specialId("unk"), 0);
        end

        function testVocabJsonRejectsNonBijectiveIds(testCase)
            f = string(tempname) + ".json";
            testCase.addTeardown(@() delete(f));
            fid = fopen(f, 'w'); fprintf(fid, '%s', '{"a":0,"b":2}'); fclose(fid);
            testCase.verifyError(@() OceanNLP.Vocab.Vocabulary.fromVocabJson(f), 'OceanNLP:Vocab:Vocabulary:BadIds');
        end

        function testStateDictRoundTripPreservesFixedVocabulary(testCase)
            v = OceanNLP.Vocab.Vocabulary.fromTokens(["[PAD]" "[UNK]" "cat"], 'Unk', "[UNK]", 'Special', struct('pad', "[PAD]"));
            w = OceanNLP.Vocab.Vocabulary();
            w.loadStateDict(v.stateDict());
            testCase.verifyEqual(w.id0(["cat" "zzz"]), [2 1]);
            testCase.verifyEqual(w.specialId("pad"), 0);
        end

        function testLegacyStateDictWithoutNewFieldsStillLoads(testCase)
            v = OceanNLP.Vocab.Vocabulary();  v.fit({["a" "b" "a"]});
            s = v.stateDict();
            s = rmfield(s, {'SpecialTokens', 'UnkIdx'});
            w = OceanNLP.Vocab.Vocabulary();  w.loadStateDict(s);
            testCase.verifyEqual(w.encode(["a" "zzz"]), [2 1]);
        end
    end
end
