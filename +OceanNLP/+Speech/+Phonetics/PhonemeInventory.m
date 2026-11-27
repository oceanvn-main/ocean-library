classdef PhonemeInventory < handle
    % PHONEMEINVENTORY  Language-scoped phoneme table: ARPAbet <-> IPA + articulatory features.
    %
    %   inv = OceanNLP.Speech.Phonetics.PhonemeInventory.english();   % General American English
    %   ipa = inv.toIPA("S EH1 V AH0 N")          % -> ["s" "ɛ" "v" "ə" "n"]
    %   arp = inv.toArpabet(["θ" "ɹ" "i"])        % -> ["TH" "R" "IY"]
    %   f   = inv.features("DH")                  % ArticulatoryFeatures: voiced dental fricative
    %   lex = inv.lexicon()                       % FSDD digit words, CMUdict pronunciations
    %   p   = inv.pronunciation("seven")          % first CMUdict variant, ARPAbet with stress
    %
    %   THIS IS DATA, NOT A RECOGNISER. It describes one language variety
    %   (General American English); nothing here claims language independence.
    %
    %   SOURCES
    %   * Phone set and ARPAbet<->IPA correspondence: Jurafsky & Martin, Speech and
    %     Language Processing (3rd ed. draft), ch. 14 "Phonetics", Fig. 14.1
    %     (ARPAbet and IPA symbols for American English consonants and vowels).
    %     The 39 phonemes of the CMU Pronouncing Dictionary are marked InCMUdict;
    %     AX (ə), AXR (ɚ), IX (ɨ) and DX (ɾ) are the reduced vowels and flap from
    %     the same figure (TIMIT-style), not used by CMUdict.
    %   * Articulatory classes: J&M ch. 14 sec. 14.2 and the IPA chart.
    %   * Digit pronunciations: The CMU Pronouncing Dictionary (cmudict, Carnegie
    %     Mellon University, http://www.speech.cs.cmu.edu/cgi-bin/cmudict ;
    %     github.com/cmusphinx/cmudict, BSD-style licence), entries ZERO, ONE ...
    %     NINE including alternates ZERO(2) and ONE(2). Stress digits 0/1/2 kept.
    %
    %   CONVENTIONS
    %   * ARPAbet input may carry CMUdict stress digits. toIPA drops stress except
    %     the conventional reductions AH0 -> ə and ER0 -> ɚ (no IPA stress marks are
    %     emitted: placing ˈ requires syllabification, which is not done here).
    %   * IPA symbols are unique per row, so toArpabet is a function; ə -> AX,
    %     ɚ -> AXR. ASCII "g" and "r", and tie-barred affricates (t͡ʃ, d͡ʒ), are
    %     accepted as aliases of ɡ, ɹ, tʃ, dʒ.
    %   * Unknown symbols raise 'OceanNLP:Speech:Phonetics:PhonemeInventory:UnknownSymbol'.

    properties (SetAccess = immutable)
        Language (1,1) string
        Table table                 % Arpabet, IPA, Category, InCMUdict, Features
        Lexicon table               % Word, Variant, Arpabet, IPA
        Sources (1,:) string
    end

    methods
        function obj = PhonemeInventory(language, phones, lexicon, sources)
            arguments
                language (1,1) string
                phones table
                lexicon table
                sources (1,:) string
            end
            if numel(unique(phones.Arpabet)) ~= height(phones) || numel(unique(phones.IPA)) ~= height(phones)
                error('OceanNLP:Speech:Phonetics:PhonemeInventory:Duplicate', 'ARPAbet and IPA symbols must be unique.');
            end
            obj.Language = language;
            obj.Table = phones;
            obj.Lexicon = lexicon;
            obj.Sources = sources;
        end

        function s = symbols(obj, opts)
            % SYMBOLS  ARPAbet symbols (all, or 'CMUdictOnly', true for the 39-phone set).
            arguments
                obj
                opts.CMUdictOnly (1,1) logical = false
            end
            s = obj.Table.Arpabet';
            if opts.CMUdictOnly
                s = s(obj.Table.InCMUdict');
            end
        end

        function ipa = toIPA(obj, arpabet)
            arguments
                obj
                arpabet (1,:) string
            end
            toks = OceanNLP.Speech.Phonetics.PhonemeInventory.tokens(arpabet);
            ipa = strings(1, numel(toks));
            for i = 1:numel(toks)
                t = upper(toks(i));
                base = regexprep(t, '[012]$', '');
                if t == "AH0"
                    base = "AX";
                elseif t == "ER0"
                    base = "AXR";
                end
                ipa(i) = obj.Table.IPA(obj.rowOf(base, "Arpabet"));
            end
        end

        function arp = toArpabet(obj, ipa)
            arguments
                obj
                ipa (1,:) string
            end
            toks = OceanNLP.Speech.Phonetics.PhonemeInventory.tokens(ipa);
            arp = strings(1, numel(toks));
            for i = 1:numel(toks)
                arp(i) = obj.Table.Arpabet(obj.rowOf(OceanNLP.Speech.Phonetics.PhonemeInventory.ipaAlias(toks(i)), "IPA"));
            end
        end

        function f = features(obj, symbol)
            % FEATURES  ArticulatoryFeatures of one ARPAbet (stress allowed) or IPA symbol.
            arguments
                obj
                symbol (1,1) string
            end
            base = regexprep(upper(symbol), '[012]$', '');
            r = find(obj.Table.Arpabet == base, 1);
            if isempty(r)
                r = obj.rowOf(OceanNLP.Speech.Phonetics.PhonemeInventory.ipaAlias(symbol), "IPA");
            end
            f = obj.Table.Features(r);
        end

        function lex = lexicon(obj)
            % LEXICON  Word, Variant, Arpabet (space-separated, stressed), IPA (space-separated).
            lex = obj.Lexicon;
        end

        function p = pronunciation(obj, word, variant)
            % PRONUNCIATION  ARPAbet tokens (1 x n string) of WORD's given CMUdict variant (default 1).
            arguments
                obj
                word (1,1) string
                variant (1,1) double {mustBeInteger, mustBePositive} = 1
            end
            r = find(obj.Lexicon.Word == lower(word) & obj.Lexicon.Variant == variant, 1);
            if isempty(r)
                error('OceanNLP:Speech:Phonetics:PhonemeInventory:UnknownWord', ...
                    'No pronunciation variant %d for "%s" in the %s lexicon.', variant, word, obj.Language);
            end
            p = split(obj.Lexicon.Arpabet(r), " ")';
        end
    end

    methods (Static)
        function inv = english()
            % ENGLISH  General American English inventory (see class doc for sources).
            C = @OceanNLP.Speech.Phonetics.ArticulatoryFeatures.consonant;
            V = @OceanNLP.Speech.Phonetics.ArticulatoryFeatures.vowel;
            rows = {
                % ARPAbet IPA   CMUdict features
                "B",   "b",   true,  C("bilabial", "stop", "voiced")
                "P",   "p",   true,  C("bilabial", "stop", "voiceless")
                "M",   "m",   true,  C("bilabial", "nasal", "voiced")
                "F",   "f",   true,  C("labiodental", "fricative", "voiceless")
                "V",   "v",   true,  C("labiodental", "fricative", "voiced")
                "TH",  "θ",   true,  C("dental", "fricative", "voiceless")
                "DH",  "ð",   true,  C("dental", "fricative", "voiced")
                "T",   "t",   true,  C("alveolar", "stop", "voiceless")
                "D",   "d",   true,  C("alveolar", "stop", "voiced")
                "N",   "n",   true,  C("alveolar", "nasal", "voiced")
                "S",   "s",   true,  C("alveolar", "fricative", "voiceless")
                "Z",   "z",   true,  C("alveolar", "fricative", "voiced")
                "L",   "l",   true,  C("alveolar", "lateral-approximant", "voiced")
                "R",   "ɹ",   true,  C("alveolar", "approximant", "voiced")
                "DX",  "ɾ",   false, C("alveolar", "flap", "voiced")
                "SH",  "ʃ",   true,  C("postalveolar", "fricative", "voiceless")
                "ZH",  "ʒ",   true,  C("postalveolar", "fricative", "voiced")
                "CH",  "tʃ",  true,  C("postalveolar", "affricate", "voiceless")
                "JH",  "dʒ",  true,  C("postalveolar", "affricate", "voiced")
                "Y",   "j",   true,  C("palatal", "approximant", "voiced")
                "K",   "k",   true,  C("velar", "stop", "voiceless")
                "G",   "ɡ",   true,  C("velar", "stop", "voiced")
                "NG",  "ŋ",   true,  C("velar", "nasal", "voiced")
                "W",   "w",   true,  C("labial-velar", "approximant", "voiced")
                "HH",  "h",   true,  C("glottal", "fricative", "voiceless")
                "IY",  "i",   true,  V("close", "front", "unrounded")
                "IH",  "ɪ",   true,  V("near-close", "near-front", "unrounded")
                "EY",  "eɪ",  true,  V("close-mid", "front", "unrounded", 'Diphthong', true, 'Offglide', "ɪ")
                "EH",  "ɛ",   true,  V("open-mid", "front", "unrounded")
                "AE",  "æ",   true,  V("near-open", "front", "unrounded")
                "AA",  "ɑ",   true,  V("open", "back", "unrounded")
                "AO",  "ɔ",   true,  V("open-mid", "back", "rounded")
                "UH",  "ʊ",   true,  V("near-close", "near-back", "rounded")
                "UW",  "u",   true,  V("close", "back", "rounded")
                "OW",  "oʊ",  true,  V("close-mid", "back", "rounded", 'Diphthong', true, 'Offglide', "ʊ")
                "AH",  "ʌ",   true,  V("open-mid", "back", "unrounded")
                "AX",  "ə",   false, V("mid", "central", "unrounded")
                "IX",  "ɨ",   false, V("close", "central", "unrounded")
                "ER",  "ɝ",   true,  V("mid", "central", "unrounded", 'Rhotic', true)
                "AXR", "ɚ",   false, V("mid", "central", "unrounded", 'Rhotic', true)
                "AY",  "aɪ",  true,  V("open", "front", "unrounded", 'Diphthong', true, 'Offglide', "ɪ")
                "AW",  "aʊ",  true,  V("open", "front", "unrounded", 'Diphthong', true, 'Offglide', "ʊ")
                "OY",  "ɔɪ",  true,  V("open-mid", "back", "rounded", 'Diphthong', true, 'Offglide', "ɪ")
                };
            feats = [rows{:, 4}]';
            cats = arrayfun(@(f) f.Category, feats);
            phones = table(string(rows(:, 1)), string(rows(:, 2)), cats, logical([rows{:, 3}]'), feats, ...
                'VariableNames', ["Arpabet", "IPA", "Category", "InCMUdict", "Features"]);
            % CMU Pronouncing Dictionary entries for the FSDD words (digit d -> word).
            lexRows = {
                "zero",  1, "Z IH1 R OW0"
                "zero",  2, "Z IY1 R OW0"
                "one",   1, "W AH1 N"
                "one",   2, "HH W AH1 N"
                "two",   1, "T UW1"
                "three", 1, "TH R IY1"
                "four",  1, "F AO1 R"
                "five",  1, "F AY1 V"
                "six",   1, "S IH1 K S"
                "seven", 1, "S EH1 V AH0 N"
                "eight", 1, "EY1 T"
                "nine",  1, "N AY1 N"
                };
            lexicon = table(string(lexRows(:, 1)), [lexRows{:, 2}]', string(lexRows(:, 3)), ...
                'VariableNames', ["Word", "Variant", "Arpabet"]);
            language = "en-US (General American English)";
            sources = ["Jurafsky & Martin, SLP 3rd ed. draft, ch. 14, Fig. 14.1", ...
                "CMU Pronouncing Dictionary (cmudict), Carnegie Mellon University"];
            % IPA column derived from the ARPAbet entries through the phone table itself
            lexicon.IPA = strings(height(lexicon), 1);
            base = OceanNLP.Speech.Phonetics.PhonemeInventory(language, phones, lexicon, sources);
            for i = 1:height(lexicon)
                lexicon.IPA(i) = strjoin(base.toIPA(lexicon.Arpabet(i)), " ");
            end
            inv = OceanNLP.Speech.Phonetics.PhonemeInventory(language, phones, lexicon, sources);
        end
    end

    methods (Access = private)
        function r = rowOf(obj, sym, column)
            r = find(obj.Table.(column) == sym, 1);
            if isempty(r)
                error('OceanNLP:Speech:Phonetics:PhonemeInventory:UnknownSymbol', ...
                    '"%s" is not a %s symbol of the %s inventory.', sym, column, obj.Language);
            end
        end
    end

    methods (Static, Access = private)
        function toks = tokens(s)
            if isscalar(s)
                toks = split(strtrim(s))';
                toks = toks(toks ~= "");
            else
                toks = strtrim(s);
            end
        end

        function s = ipaAlias(s)
            s = replace(s, char(865), "");            % U+0361 combining tie bar
            if s == "g"
                s = "ɡ";
            elseif s == "r"
                s = "ɹ";
            end
        end
    end
end
