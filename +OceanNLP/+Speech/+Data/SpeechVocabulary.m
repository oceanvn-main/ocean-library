classdef SpeechVocabulary < handle
    % SPEECHVOCABULARY  Token inventory for ASR targets with fixed special indices.
    %
    %   v   = OceanNLP.Speech.Data.SpeechVocabulary.fit(trainTranscripts, 'Unit', "char")
    %   v   = OceanNLP.Speech.Data.SpeechVocabulary.fit(t, 'Unit', "word", 'Normalizer', n)
    %   v   = OceanNLP.Speech.Data.SpeechVocabulary.fit(t, 'Unit', "phone", 'Lexicon', lex)
    %   ids = v.encode(text)          1 x L double (string array in -> cell of rows)
    %   txt = v.decode(ids)           string
    %   K   = v.numClasses()
    %   sd  = v.stateDict(); v.loadStateDict(sd); v = SpeechVocabulary.fromStateDict(sd)
    %   v.assertCompatible(K)
    %
    %   INDICES (frozen, Phase 0 contract sec. 2):
    %     1 <blank>  (CTC Blank)     2 <pad>    3 <bos>    4 <eos>    5 <unk>
    %     6..K       units in sorted order (MATLAB string sort = code-unit order)
    %
    %   UNITS (extracted from NORMALIZED training transcripts only -- fit on the
    %   train split, never on validation/test text):
    %     "char"   every character; the space becomes the word-boundary token "|"
    %     "word"   space-separated words
    %     "phone"  phones from a pronunciation lexicon (word -> phone sequence); no
    %              boundary token; decode joins phones with single spaces, so a WER
    %              computed on decoded phone strings is a phone error rate. Without
    %              'Lexicon' the integration lexicon
    %              OceanNLP.Speech.Phonetics.PhonemeInventory.english().lexicon() is used
    %              if that class exists, otherwise ':MissingLexicon' is raised. A training word
    %              missing from the lexicon raises ':MissingPronunciation'.
    %
    %   encode: normalize -> units -> ids; a unit not in the vocabulary -> 5 <unk>.
    %   decode: ids 1..5 (all specials, including <unk>) are dropped; char units are
    %   concatenated and "|" becomes a space (runs collapsed, ends stripped); word and
    %   phone units are joined with single spaces. Ids outside 1..K raise ':InvalidId'.
    %
    %   Lexicon formats accepted: dictionary or containers.Map (word -> pronunciation),
    %   table or struct array with fields Word and Phones|Pronunciation|Arpabet (rows
    %   with the lowest Variant per word when a Variant column exists), or any object
    %   with a lexicon() method returning one of these (PhonemeInventory). A
    %   pronunciation is a string of space-separated phones or a string array; a
    %   cell of alternatives uses the first (documented choice).

    properties (Constant)
        Blank = 1
        Pad = 2
        Bos = 3
        Eos = 4
        Unk = 5
        Specials = ["<blank>", "<pad>", "<bos>", "<eos>", "<unk>"]
        WordBoundary = "|"
    end

    properties (SetAccess = private)
        Unit (1,1) string = ""
        Tokens (1,:) string = OceanNLP.Speech.Data.SpeechVocabulary.Specials
        Normalizer = []
        Lexicon = []          % dictionary string -> string (space-joined phones), or []
    end

    properties (Access = private)
        Index                 % dictionary token -> id
    end

    methods
        function obj = SpeechVocabulary()
            % Unfitted vocabulary (specials only); use fit() or loadStateDict().
            obj.rebuildIndex();
        end

        function K = numClasses(obj)
            K = numel(obj.Tokens);
        end

        function ids = encode(obj, text)
            arguments
                obj
                text string
            end
            if obj.Unit == ""
                error('OceanNLP:Speech:Data:SpeechVocabulary:NotFitted', 'Vocabulary has not been fitted.');
            end
            if ~isscalar(text)
                ids = arrayfun(@(t) obj.encode(t), reshape(text, 1, []), 'UniformOutput', false);
                return;
            end
            units = obj.unitsOf(obj.normalize(text), false);
            ids = obj.Unk * ones(1, numel(units));
            if isempty(units)
                ids = zeros(1, 0);
                return;
            end
            known = isKey(obj.Index, units);
            ids(known) = obj.Index(units(known));
        end

        function text = decode(obj, ids)
            arguments
                obj
                ids {mustBeNumeric, mustBeReal}
            end
            ids = reshape(double(ids), 1, []);
            K = obj.numClasses();
            if any(ids ~= round(ids) | ids < 1 | ids > K)
                error('OceanNLP:Speech:Data:SpeechVocabulary:InvalidId', ...
                    'Token ids must be integers in 1..%d.', K);
            end
            toks = obj.Tokens(ids(ids > numel(obj.Specials)));
            switch obj.Unit
                case "char"
                    text = join(toks, "");
                    if isempty(toks), text = ""; end
                    text = strip(regexprep(replace(text, obj.WordBoundary, " "), ' +', ' '));
                otherwise
                    text = join(toks, " ");
                    if isempty(toks), text = ""; end
            end
        end

        function assertCompatible(obj, numClasses)
            arguments
                obj
                numClasses (1,1) double
            end
            if numClasses ~= obj.numClasses()
                error('OceanNLP:Speech:Data:SpeechVocabulary:Incompatible', ...
                    'Model has %d output classes but the vocabulary has %d.', numClasses, obj.numClasses());
            end
        end

        function sd = stateDict(obj)
            norm = struct();
            if ~isempty(obj.Normalizer), norm = obj.Normalizer.config(); end
            lexW = strings(0, 1); lexP = strings(0, 1);
            if ~isempty(obj.Lexicon)
                lexW = keys(obj.Lexicon); lexP = values(obj.Lexicon);
            end
            sd = struct('Type', "SpeechVocabulary", 'Version', 1, 'Unit', obj.Unit, ...
                'Tokens', obj.Tokens, 'Normalizer', norm, ...
                'LexiconWords', reshape(lexW, 1, []), 'LexiconPhones', reshape(lexP, 1, []));
        end

        function loadStateDict(obj, sd)
            arguments
                obj
                sd (1,1) struct
            end
            need = ["Type", "Unit", "Tokens", "Normalizer", "LexiconWords", "LexiconPhones"];
            if ~all(isfield(sd, need)) || string(sd.Type) ~= "SpeechVocabulary"
                error('OceanNLP:Speech:Data:SpeechVocabulary:StateDict', 'Not a SpeechVocabulary state dict.');
            end
            tokens = reshape(string(sd.Tokens), 1, []);
            nS = numel(obj.Specials);
            if numel(tokens) < nS || any(tokens(1:nS) ~= obj.Specials) || numel(unique(tokens)) ~= numel(tokens)
                error('OceanNLP:Speech:Data:SpeechVocabulary:StateDict', ...
                    'Tokens must start with %s and be unique.', join(obj.Specials, ' '));
            end
            unit = string(sd.Unit);
            if ~ismember(unit, ["char", "word", "phone"])
                error('OceanNLP:Speech:Data:SpeechVocabulary:StateDict', 'Unknown unit "%s".', unit);
            end
            obj.Unit = unit;
            obj.Tokens = tokens;
            if isempty(fieldnames(sd.Normalizer))
                obj.Normalizer = [];
            else
                obj.Normalizer = OceanNLP.Speech.Data.TranscriptNormalizer.fromConfig(sd.Normalizer);
            end
            w = reshape(string(sd.LexiconWords), [], 1); p = reshape(string(sd.LexiconPhones), [], 1);
            if isempty(w)
                obj.Lexicon = [];
            else
                obj.Lexicon = dictionary(w, p);
            end
            obj.rebuildIndex();
        end
    end

    methods (Static)
        function v = fit(transcripts, opts)
            arguments
                transcripts string
                opts.Unit (1,1) string {mustBeMember(opts.Unit, ["char", "word", "phone"])} = "char"
                opts.Normalizer = OceanNLP.Speech.Data.TranscriptNormalizer()
                opts.Lexicon = []
            end
            if isempty(transcripts) || any(ismissing(transcripts))
                error('OceanNLP:Speech:Data:SpeechVocabulary:NoTranscripts', ...
                    'fit needs a non-empty string array of training transcripts.');
            end
            if ~isempty(opts.Normalizer) && ~ismethod(opts.Normalizer, 'forward')
                error('OceanNLP:Speech:Data:SpeechVocabulary:Normalizer', 'Normalizer must provide forward().');
            end
            v = OceanNLP.Speech.Data.SpeechVocabulary();
            v.Unit = opts.Unit;
            v.Normalizer = opts.Normalizer;
            if opts.Unit == "phone"
                lex = opts.Lexicon;
                if isempty(lex)
                    if exist('OceanNLP.Speech.Phonetics.PhonemeInventory', 'class') ~= 8
                        error('OceanNLP:Speech:Data:SpeechVocabulary:MissingLexicon', ...
                            ['Unit "phone" needs a Lexicon, and OceanNLP.Speech.Phonetics.PhonemeInventory ', ...
                            'is not available to supply one.']);
                    end
                    lex = OceanNLP.Speech.Phonetics.PhonemeInventory.english();
                end
                if ~isa(lex, 'dictionary') && ~isa(lex, 'containers.Map') && ~istable(lex) ...
                        && ~isstruct(lex) && ismethod(lex, 'lexicon')
                    lex = lex.lexicon();                      % e.g. a PhonemeInventory object
                end
                v.Lexicon = OceanNLP.Speech.Data.SpeechVocabulary.toLexicon(lex);
            end
            norm = v.normalize(reshape(transcripts, 1, []));
            units = strings(1, 0);
            for k = 1:numel(norm)
                units = [units, v.unitsOf(norm(k), true)]; %#ok<AGROW>
            end
            units = unique(units);                       % sorted, unique
            clash = intersect(units, v.Specials);
            if ~isempty(clash)
                error('OceanNLP:Speech:Data:SpeechVocabulary:ReservedToken', ...
                    'Training units collide with special tokens: %s', join(clash, ' '));
            end
            if isempty(units)
                error('OceanNLP:Speech:Data:SpeechVocabulary:NoTranscripts', 'Training transcripts contain no units.');
            end
            v.Tokens = [v.Specials, reshape(units, 1, [])];
            v.rebuildIndex();
        end

        function v = fromStateDict(sd)
            v = OceanNLP.Speech.Data.SpeechVocabulary();
            v.loadStateDict(sd);
        end
    end

    methods (Access = private)
        function rebuildIndex(obj)
            obj.Index = dictionary(obj.Tokens, 1:numel(obj.Tokens));
        end

        function t = normalize(obj, t)
            if ~isempty(obj.Normalizer)
                t = obj.Normalizer.forward(t);
            end
        end

        function u = unitsOf(obj, text, strict)
            % Units of one normalized transcript (1 x L string).
            switch obj.Unit
                case "char"
                    c = char(text);
                    u = string(num2cell(c));
                    u(u == " ") = obj.WordBoundary;
                    if isempty(c), u = strings(1, 0); end
                case "word"
                    u = OceanNLP.Speech.Evaluation.WordErrorRate.tokenize(text);
                case "phone"
                    words = OceanNLP.Speech.Evaluation.WordErrorRate.tokenize(text);
                    u = strings(1, 0);
                    for w = words
                        if isKey(obj.Lexicon, w)
                            u = [u, split(obj.Lexicon(w), " ").']; %#ok<AGROW>
                        elseif strict
                            error('OceanNLP:Speech:Data:SpeechVocabulary:MissingPronunciation', ...
                                'Training word "%s" has no pronunciation in the lexicon.', w);
                        else
                            u = [u, obj.Specials(obj.Unk)]; %#ok<AGROW>   % maps to <unk>
                        end
                    end
            end
            u = reshape(u, 1, []);
        end
    end

    methods (Static, Access = private)
        function d = toLexicon(lex)
            % Convert an accepted lexicon format to dictionary(word -> "P1 P2 ...").
            % words (n x 1 string) and prons (n x 1 cell) are set by each branch below
            if isa(lex, 'dictionary')
                words = keys(lex); prons = num2cell(values(lex));
                if iscell(values(lex)), prons = values(lex); end
            elseif isa(lex, 'containers.Map')
                words = string(keys(lex))'; prons = values(lex)';
            elseif istable(lex) || isstruct(lex)
                if isstruct(lex), lex = struct2table(lex(:), 'AsArray', true); end
                vn = string(lex.Properties.VariableNames);
                pv = ["Phones", "Pronunciation", "Arpabet"];
                pv = pv(ismember(pv, vn));
                if ~ismember("Word", vn) || isempty(pv)
                    error('OceanNLP:Speech:Data:SpeechVocabulary:Lexicon', ...
                        'Lexicon table/struct needs Word and Phones (or Pronunciation, Arpabet).');
                end
                if ismember("Variant", vn)                     % keep the first variant per word
                    lex = sortrows(lex, ["Word", "Variant"]);
                    [~, first] = unique(string(lex.Word), 'stable');
                    lex = lex(first, :);
                end
                words = string(lex.Word);
                prons = lex.(pv(1));
                if ~iscell(prons), prons = num2cell(prons); end
            else
                error('OceanNLP:Speech:Data:SpeechVocabulary:Lexicon', 'Unsupported lexicon type %s.', class(lex));
            end
            words = reshape(words, [], 1);
            p = strings(numel(words), 1);
            for k = 1:numel(words)
                pk = prons{k};
                if iscell(pk), pk = pk{1}; end                    % first alternative
                pk = string(pk);
                p(k) = join(split(strip(join(reshape(pk, 1, []), " "))), " ");
                if p(k) == ""
                    error('OceanNLP:Speech:Data:SpeechVocabulary:Lexicon', 'Empty pronunciation for "%s".', words(k));
                end
            end
            d = dictionary(lower(words), p);
        end
    end
end
