classdef ArticulatoryFeatures < handle
    % ARTICULATORYFEATURES  Articulatory description of one phoneme (immutable data).
    %
    %   f = OceanNLP.Speech.Phonetics.ArticulatoryFeatures.consonant(place, manner, voicing)
    %   f = OceanNLP.Speech.Phonetics.ArticulatoryFeatures.vowel(height, backness, rounding, ...
    %         'Diphthong', false, 'Offglide', "", 'Rhotic', false)
    %   s = f.toStruct();   txt = f.describe();
    %
    %   Consonants are classified by PLACE and MANNER of articulation and VOICING;
    %   vowels by HEIGHT, BACKNESS and lip ROUNDING (Jurafsky & Martin, Speech and
    %   Language Processing 3rd ed. draft, ch. 14 "Phonetics", sec. 14.2; category
    %   names follow the IPA chart, International Phonetic Association 1999/2015).
    %   Diphthongs carry the features of their nucleus (onset) and the IPA symbol of
    %   the offglide target. Vowels are voiced. Fields that do not apply are "".

    properties (Constant)
        Places = ["bilabial", "labiodental", "dental", "alveolar", "postalveolar", ...
            "palatal", "velar", "labial-velar", "glottal"]
        Manners = ["stop", "nasal", "fricative", "affricate", "approximant", ...
            "lateral-approximant", "flap"]
        Heights = ["close", "near-close", "close-mid", "mid", "open-mid", "near-open", "open"]
        Backnesses = ["front", "near-front", "central", "near-back", "back"]
    end

    properties (SetAccess = immutable)
        Category (1,1) string      % "consonant" | "vowel"
        Place (1,1) string = ""
        Manner (1,1) string = ""
        Voicing (1,1) string = ""  % "voiced" | "voiceless"
        Height (1,1) string = ""
        Backness (1,1) string = ""
        Rounding (1,1) string = "" % "rounded" | "unrounded"
        Diphthong (1,1) logical = false
        Offglide (1,1) string = ""
        Rhotic (1,1) logical = false
    end

    methods (Access = private)
        function obj = ArticulatoryFeatures(s)
            obj.Category = s.Category;
            obj.Place = s.Place;
            obj.Manner = s.Manner;
            obj.Voicing = s.Voicing;
            obj.Height = s.Height;
            obj.Backness = s.Backness;
            obj.Rounding = s.Rounding;
            obj.Diphthong = s.Diphthong;
            obj.Offglide = s.Offglide;
            obj.Rhotic = s.Rhotic;
        end
    end

    methods (Static)
        function f = consonant(place, manner, voicing)
            arguments
                place (1,1) string {mustBeMember(place, ["bilabial", "labiodental", "dental", "alveolar", "postalveolar", "palatal", "velar", "labial-velar", "glottal"])}
                manner (1,1) string {mustBeMember(manner, ["stop", "nasal", "fricative", "affricate", "approximant", "lateral-approximant", "flap"])}
                voicing (1,1) string {mustBeMember(voicing, ["voiced", "voiceless"])}
            end
            s = OceanNLP.Speech.Phonetics.ArticulatoryFeatures.blank("consonant");
            s.Place = place;
            s.Manner = manner;
            s.Voicing = voicing;
            f = OceanNLP.Speech.Phonetics.ArticulatoryFeatures(s);
        end

        function f = vowel(height, backness, rounding, opts)
            arguments
                height (1,1) string {mustBeMember(height, ["close", "near-close", "close-mid", "mid", "open-mid", "near-open", "open"])}
                backness (1,1) string {mustBeMember(backness, ["front", "near-front", "central", "near-back", "back"])}
                rounding (1,1) string {mustBeMember(rounding, ["rounded", "unrounded"])}
                opts.Diphthong (1,1) logical = false
                opts.Offglide (1,1) string = ""
                opts.Rhotic (1,1) logical = false
            end
            if opts.Diphthong == (opts.Offglide == "")
                error('OceanNLP:Speech:Phonetics:ArticulatoryFeatures:Offglide', ...
                    'A diphthong needs an offglide symbol and a monophthong must not have one.');
            end
            s = OceanNLP.Speech.Phonetics.ArticulatoryFeatures.blank("vowel");
            s.Voicing = "voiced";
            s.Height = height;
            s.Backness = backness;
            s.Rounding = rounding;
            s.Diphthong = opts.Diphthong;
            s.Offglide = opts.Offglide;
            s.Rhotic = opts.Rhotic;
            f = OceanNLP.Speech.Phonetics.ArticulatoryFeatures(s);
        end
    end

    methods
        function s = toStruct(obj)
            s = struct('Category', obj.Category, 'Place', obj.Place, 'Manner', obj.Manner, ...
                'Voicing', obj.Voicing, 'Height', obj.Height, 'Backness', obj.Backness, ...
                'Rounding', obj.Rounding, 'Diphthong', obj.Diphthong, 'Offglide', obj.Offglide, ...
                'Rhotic', obj.Rhotic);
        end

        function txt = describe(obj)
            % DESCRIBE  e.g. "voiced bilabial stop", "close front unrounded vowel".
            if obj.Category == "consonant"
                txt = strjoin([obj.Voicing, obj.Place, obj.Manner], " ");
            else
                parts = [obj.Height, obj.Backness, obj.Rounding];
                if obj.Rhotic
                    parts(end + 1) = "rhotic";
                end
                txt = strjoin(parts, " ") + " vowel";
                if obj.Diphthong
                    txt = txt + " (diphthong onset, offglide " + obj.Offglide + ")";
                end
            end
        end
    end

    methods (Static, Access = private)
        function s = blank(category)
            s = struct('Category', category, 'Place', "", 'Manner', "", 'Voicing', "", 'Height', "", ...
                'Backness', "", 'Rounding', "", 'Diphthong', false, 'Offglide', "", 'Rhotic', false);
        end
    end
end
