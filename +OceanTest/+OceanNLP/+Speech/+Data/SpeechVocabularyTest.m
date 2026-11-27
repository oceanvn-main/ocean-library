classdef SpeechVocabularyTest < OceanTest.BaseTestCase
    % SPEECHVOCABULARYTEST  Fixed indices, train-only fitting, round trips, state dict, phone unit.

    properties
        Train       % composed train transcripts (multi-word, real FSDD labels)
        Test        % composed test-speaker transcripts
    end

    methods (TestClassSetup)
        function build(testCase)
            s = OceanTest.OceanNLP.Speech.Data.fsddTestManifest().canonicalFSDDSplit();
            tr = OceanNLP.Speech.Data.UtteranceComposer.compose(s.bySplit("train"), 'Count', 200, 'Seed', 1);
            te = OceanNLP.Speech.Data.UtteranceComposer.compose(s.bySplit("test"), 'Count', 50, 'Seed', 2);
            testCase.Train = tr.Entries.Transcript;
            testCase.Test = te.Entries.Transcript;
        end
    end

    methods (Test)
        function testCharIndicesAndUnits(testCase)
            V = OceanNLP.Speech.Data.SpeechVocabulary;
            v = V.fit(testCase.Train, 'Unit', "char");
            testCase.verifyEqual(v.Tokens(1:5), ["<blank>" "<pad>" "<bos>" "<eos>" "<unk>"]);
            testCase.verifyEqual([v.Blank v.Pad v.Bos v.Eos v.Unk], 1:5);
            chars = unique(char(join(lower(testCase.Train), " ")));
            units = string(num2cell(chars)); units(units == " ") = "|";
            testCase.verifyEqual(v.Tokens(6:end), sort(units));
            testCase.verifyEqual(v.numClasses(), 5 + numel(units));
            ids = v.encode("one two");
            testCase.verifyEqual(v.Tokens(ids), ["o" "n" "e" "|" "t" "w" "o"]);
        end

        function testRoundTripOnHeldOutSpeaker(testCase)
            for unit = ["char", "word"]
                v = OceanNLP.Speech.Data.SpeechVocabulary.fit(testCase.Train, 'Unit', unit);
                for k = 1:numel(testCase.Test)
                    t = testCase.Test(k);
                    testCase.verifyEqual(v.decode(v.encode(t)), t, unit);
                end
                ids = v.encode(testCase.Test);
                testCase.verifyClass(ids, 'cell');
            end
        end

        function testTrainOnlyFittingMapsUnseenToUnk(testCase)
            v = OceanNLP.Speech.Data.SpeechVocabulary.fit(["one two", "three"], 'Unit', "char");
            ids = v.encode("four");                           % f, u unseen in training text
            testCase.verifyEqual(ids([1 3]), [5 5]);
            testCase.verifyEqual(v.decode(ids), "or");        % specials (incl. <unk>) dropped
            w = OceanNLP.Speech.Data.SpeechVocabulary.fit(["one two", "three"], 'Unit', "word");
            testCase.verifyEqual(w.encode("one Four!"), [w.encode("one"), 5]);
            testCase.verifyEqual(w.decode([1 3 6 2 4 7]), join(w.Tokens([6 7]), " "));
        end

        function testStateDictRoundTrip(testCase)
            v = OceanNLP.Speech.Data.SpeechVocabulary.fit(testCase.Train, 'Unit', "char");
            sd = v.stateDict();
            json = jsondecode(jsonencode(sd));                % survives JSON (checkpoint use)
            json.Normalizer.Type = string(json.Normalizer.Type);
            w = OceanNLP.Speech.Data.SpeechVocabulary.fromStateDict(json);
            testCase.verifyEqual(w.Tokens, v.Tokens);
            testCase.verifyEqual(w.encode(testCase.Test(1)), v.encode(testCase.Test(1)));
            testCase.verifyEqual(w.Normalizer.config(), v.Normalizer.config());
            u = OceanNLP.Speech.Data.SpeechVocabulary();
            u.loadStateDict(sd);
            testCase.verifyEqual(u.numClasses(), v.numClasses());
            bad = sd; bad.Tokens = bad.Tokens([2 1 3:end]);
            testCase.verifyError(@() u.loadStateDict(bad), 'OceanNLP:Speech:Data:SpeechVocabulary:StateDict');
        end

        function testCompatibilityAndInvalidIds(testCase)
            v = OceanNLP.Speech.Data.SpeechVocabulary.fit(testCase.Train);
            v.assertCompatible(v.numClasses());
            testCase.verifyError(@() v.assertCompatible(v.numClasses() + 1), ...
                'OceanNLP:Speech:Data:SpeechVocabulary:Incompatible');
            testCase.verifyError(@() v.decode(v.numClasses() + 1), 'OceanNLP:Speech:Data:SpeechVocabulary:InvalidId');
            testCase.verifyError(@() v.decode(1.5), 'OceanNLP:Speech:Data:SpeechVocabulary:InvalidId');
            testCase.verifyError(@() OceanNLP.Speech.Data.SpeechVocabulary().encode("a"), ...
                'OceanNLP:Speech:Data:SpeechVocabulary:NotFitted');
            testCase.verifyEqual(v.encode(""), zeros(1, 0));
        end

        function testPhoneUnitWithSuppliedLexicon(testCase)
            % Pronunciations from CMUdict 0.7b (input data, not expected output).
            words = ["zero" "one" "two" "three" "four" "five" "six" "seven" "eight" "nine"];
            prons = ["Z IH1 R OW0" "W AH1 N" "T UW1" "TH R IY1" "F AO1 R" "F AY1 V" ...
                "S IH1 K S" "S EH1 V AH0 N" "EY1 T" "N AY1 N"];
            lex = dictionary(words, prons);
            v = OceanNLP.Speech.Data.SpeechVocabulary.fit(testCase.Train, 'Unit', "phone", 'Lexicon', lex);
            used = unique(split(join(prons(ismember(words, split(join(testCase.Train, " ")))), " ")))';
            testCase.verifyEqual(v.Tokens(6:end), sort(used));
            ids = v.encode("one two");
            testCase.verifyEqual(v.Tokens(ids), ["W" "AH1" "N" "T" "UW1"]);
            testCase.verifyEqual(v.decode(ids), "W AH1 N T UW1");
            tbl = table(words', prons', 'VariableNames', ["Word", "Phones"]);
            v2 = OceanNLP.Speech.Data.SpeechVocabulary.fit(testCase.Train, 'Unit', "phone", 'Lexicon', tbl);
            testCase.verifyEqual(v2.Tokens, v.Tokens);
            small = dictionary("one", "W AH1 N");
            testCase.verifyError(@() OceanNLP.Speech.Data.SpeechVocabulary.fit("one two", 'Unit', "phone", 'Lexicon', small), ...
                'OceanNLP:Speech:Data:SpeechVocabulary:MissingPronunciation');
        end

        function testPhoneUnitWithoutLexiconNeedsPhonemeInventory(testCase)
            if exist('OceanNLP.Speech.Phonetics.PhonemeInventory', 'class') ~= 8
                testCase.verifyError(@() OceanNLP.Speech.Data.SpeechVocabulary.fit("one", 'Unit', "phone"), ...
                    'OceanNLP:Speech:Data:SpeechVocabulary:MissingLexicon');
                return;
            end
            % integration path: builder B1's CMUdict lexicon (table Word/Variant/Arpabet)
            inv = OceanNLP.Speech.Phonetics.PhonemeInventory.english();
            v = OceanNLP.Speech.Data.SpeechVocabulary.fit(testCase.Train, 'Unit', "phone");
            v2 = OceanNLP.Speech.Data.SpeechVocabulary.fit(testCase.Train, 'Unit', "phone", 'Lexicon', inv);
            testCase.verifyEqual(v2.Tokens, v.Tokens);
            for w = ["one", "seven"]
                testCase.verifyEqual(v.Tokens(v.encode(w)), inv.pronunciation(w));
            end
        end
    end
end
