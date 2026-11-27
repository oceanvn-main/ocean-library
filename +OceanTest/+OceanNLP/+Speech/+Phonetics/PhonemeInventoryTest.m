classdef PhonemeInventoryTest < OceanTest.BaseTestCase
    % PHONEMEINVENTORYTEST  Structural checks of the General American inventory and digit lexicon.
    %   (The table is cited data; the tests check its internal consistency and the
    %   conversions, not re-typed expected values.)

    methods (Test)
        function testCMUdictPhoneSetIsCompleteAndUnique(testCase)
            inv = OceanNLP.Speech.Phonetics.PhonemeInventory.english();
            cmu = inv.symbols('CMUdictOnly', true);
            testCase.verifyNumElements(cmu, 39);                     % CMUdict documents 39 phonemes
            testCase.verifyNumElements(unique(inv.Table.IPA), height(inv.Table));
            testCase.verifyNumElements(unique(inv.Table.Arpabet), height(inv.Table));
        end

        function testRoundTripsAreIdentity(testCase)
            inv = OceanNLP.Speech.Phonetics.PhonemeInventory.english();
            a = inv.symbols();
            testCase.verifyEqual(inv.toArpabet(inv.toIPA(a)), a);
            ipa = inv.Table.IPA';
            testCase.verifyEqual(inv.toIPA(inv.toArpabet(ipa)), ipa);
            % space-separated input and stress digits
            testCase.verifyEqual(inv.toIPA("S EH1 V AH0 N"), inv.toIPA(["S" "EH" "V" "AX" "N"]));
            testCase.verifyEqual(inv.toIPA("ER0"), inv.toIPA("AXR"));
            testCase.verifyEqual(inv.toIPA("AH1"), inv.toIPA("AH"));
            testCase.verifyEqual(inv.toArpabet("g"), "G");
            testCase.verifyEqual(inv.toArpabet("t" + char(865) + "ʃ"), "CH");
            testCase.verifyError(@() inv.toIPA("QQ"), 'OceanNLP:Speech:Phonetics:PhonemeInventory:UnknownSymbol');
            testCase.verifyError(@() inv.toArpabet("ʘ"), 'OceanNLP:Speech:Phonetics:PhonemeInventory:UnknownSymbol');
        end

        function testFeaturesAreConsistentWithCategories(testCase)
            inv = OceanNLP.Speech.Phonetics.PhonemeInventory.english();
            for i = 1:height(inv.Table)
                f = inv.Table.Features(i);
                testCase.verifyEqual(f.Category, inv.Table.Category(i));
                if f.Category == "consonant"
                    testCase.verifyTrue(f.Place ~= "" && f.Manner ~= "" && f.Voicing ~= "" && f.Height == "");
                else
                    testCase.verifyTrue(f.Height ~= "" && f.Backness ~= "" && f.Rounding ~= "" && f.Place == "");
                    % every multi-character vowel symbol is a diphthong and ends in its offglide
                    if strlength(inv.Table.IPA(i)) > 1
                        testCase.verifyTrue(f.Diphthong && endsWith(inv.Table.IPA(i), f.Offglide));
                    end
                end
            end
            testCase.verifyEqual(inv.features("DH").describe(), "voiced dental fricative");
            testCase.verifyEqual(inv.features("ð"), inv.features("DH"));
            testCase.verifyEqual(inv.features("IY1").Height, "close");
            % minimal pairs differ in exactly the expected feature
            testCase.verifyEqual(inv.features("P").Place, inv.features("B").Place);
            testCase.verifyNotEqual(inv.features("P").Voicing, inv.features("B").Voicing);
        end

        function testDigitLexiconCoversFSDDWords(testCase)
            inv = OceanNLP.Speech.Phonetics.PhonemeInventory.english();
            lex = inv.lexicon();
            files = dir(fullfile(testCase.LibraryRoot, 'Data', 'Datasets', 'Speech', 'FSDD', 'recordings', '*.wav'));
            digits = unique(cellfun(@(n) str2double(n(1)), {files.name}));
            words = ["zero" "one" "two" "three" "four" "five" "six" "seven" "eight" "nine"];
            testCase.verifyEqual(digits, 0:9);
            for d = digits
                testCase.verifyTrue(any(lex.Word == words(d + 1)), words(d + 1));
                p = inv.pronunciation(words(d + 1));
                testCase.verifyNotEmpty(p);
                testCase.verifyEqual(strjoin(inv.toIPA(p), " "), lex.IPA(find(lex.Word == words(d + 1), 1)));
            end
            testCase.verifyError(@() inv.pronunciation("eleven"), 'OceanNLP:Speech:Phonetics:PhonemeInventory:UnknownWord');
        end

        function testArticulatoryFeatureValidation(testCase)
            AF = 'OceanNLP.Speech.Phonetics.ArticulatoryFeatures';
            testCase.verifyError(@() feval([AF '.consonant'], "nowhere", "stop", "voiced"), 'MATLAB:validators:mustBeMember');
            testCase.verifyError(@() feval([AF '.vowel'], "close", "front", "unrounded", 'Diphthong', true), ...
                'OceanNLP:Speech:Phonetics:ArticulatoryFeatures:Offglide');
        end
    end
end
