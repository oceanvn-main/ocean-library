classdef Analyzer < OceanNLP.Tokenize.ITokenizer
    % ANALYZER  Multi-step text analysis pipeline (Lucene/Elasticsearch model):
    %
    %   raw text -> ICharFilter* -> ITokenizer -> ITokenFilter* -> tokens
    %
    % An Analyzer IS an ITokenizer, so it drops into CountVectorizer,
    % InvertedIndex, BM25Ranker, TfidfVectorizer, ... with no change to them.
    % Every stage is a strategy object: swap the stemmer, add a stop list, or
    % change the tokenizer by changing a constructor argument, never source.
    %
    % Construction
    %   Analyzer()                                  basic = RegexTokenizer() alone (today's default behaviour)
    %   Analyzer('TokenFilters', {LowercaseFilter(), StemFilter('Stemmer', PorterStemmer())})
    %   Analyzer.preset("english")                  named presets (data, see presets())
    %   Analyzer.fromSpec(struct('Tokenizer',"regex", 'TokenFilters',["stopword","stem:porter"]))
    %
    % Spec strings resolve through OceanNLP.Text.Factory (convention-based), so a
    % strategy added later is usable in a spec the day its file exists.
    % "stem:<name>" means StemFilter with that stemmer.
    %
    % SYMMETRY RULE: the SAME analyzer (or an equal spec) must be used at index
    % and query time. getPipelineSpec().ComposesFrom lists every stage so a
    % mismatch is visible.
    %
    % NOTE on case: the default tokenizers lowercase inside forward() (CaseSensitive=false).
    % A LowercaseFilter after a CaseSensitive=true tokenizer is the Lucene arrangement.
    %
    % See also: OceanNLP.Text.Factory, OceanNLP.Text.Filter.StemFilter

    properties (SetAccess = private)
        CharFilters (1,:) cell = {}
        Tokenizer = OceanNLP.Tokenize.RegexTokenizer()
        TokenFilters (1,:) cell = {}
    end

    methods
        function obj = Analyzer(options)
            arguments
                options.CharFilters (1,:) cell = {}
                options.Tokenizer (1,1) OceanNLP.Tokenize.ITokenizer = OceanNLP.Tokenize.RegexTokenizer()
                options.TokenFilters (1,:) cell = {}
            end
            for i = 1:numel(options.CharFilters)
                if ~isa(options.CharFilters{i}, 'OceanNLP.Text.ICharFilter')
                    error('OceanNLP:Text:Analyzer:BadStage', 'CharFilters{%d} must be an ICharFilter.', i);
                end
            end
            for i = 1:numel(options.TokenFilters)
                if ~isa(options.TokenFilters{i}, 'OceanNLP.Text.ITokenFilter')
                    error('OceanNLP:Text:Analyzer:BadStage', 'TokenFilters{%d} must be an ITokenFilter.', i);
                end
            end
            obj.CharFilters = options.CharFilters;
            obj.Tokenizer = options.Tokenizer;
            obj.TokenFilters = options.TokenFilters;
        end

        function tokens = forward(obj, rawText)
            arguments
                obj
                rawText (1,1) string
            end
            txt = rawText;
            for i = 1:numel(obj.CharFilters) 
                txt = obj.CharFilters{i}.forward(txt); %Run Raw text Filtering Pipeline. Example: "<p>Hello &amp; World!</p>" ──► " Hello & World! "
            end
            tokens = obj.Tokenizer.forward(txt); %Run Tokenizer. Example: " Hello & World! " ──► ["Hello", "&", "World!"]
            for i = 1:numel(obj.TokenFilters)
                tokens = obj.TokenFilters{i}.forward(tokens); %Run Token Filtering Pipeline (Linguistic trasnform, lowerase, stopwords removing, stemming,...)
            end
        end

        function s = name(obj)
            parts = strings(1, 0);
            for i = 1:numel(obj.CharFilters), parts(end+1) = obj.CharFilters{i}.name(); end %#ok<AGROW>
            parts(end+1) = obj.Tokenizer.name();
            for i = 1:numel(obj.TokenFilters), parts(end+1) = obj.TokenFilters{i}.name(); end %#ok<AGROW>
            s = "analyzer(" + strjoin(parts, ">") + ")";
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = {'VIRTUAL composition of char filters + a tokenizer + token filters; no text-processing logic is implemented here', ...
                                 'no closed form (R7): every stage is a discrete string procedure', ...
                                 char("chain: " + obj.name())};
            meta.IsSymbolic = false;
            meta.Reference = "Lucene/Elasticsearch analysis chain: char filters -> tokenizer -> token filters";
            refs = strings(1, 0);
            for i = 1:numel(obj.CharFilters), refs(end+1) = obj.CharFilters{i}.getMetadata().Reference; end %#ok<AGROW>
            for i = 1:numel(obj.TokenFilters), refs(end+1) = obj.TokenFilters{i}.getMetadata().Reference; end %#ok<AGROW>
            tokMeta = obj.Tokenizer.getMetadata();
            assert(isfield(tokMeta, 'Reference') && strlength(tokMeta.Reference) > 0, ...
                'OceanNLP:Text:Analyzer:TokenizerMetadataUnresolved', 'Tokenizer metadata Reference is empty.');
            meta.ComposedPartReference = struct('TokenizerReference', tokMeta.Reference, 'StageReferences', refs);
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.Inputs = struct('Name', "rawText", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x 1");
            spec.Outputs = struct('Name', "tokens", 'Domain', "token-sequence", 'Dtype', "string", 'Shape', "1 x T");
            spec.Kind = "VIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "analyzer");
            comp = {};
            for i = 1:numel(obj.CharFilters), comp{end+1} = class(obj.CharFilters{i}); end %#ok<AGROW>
            comp{end+1} = class(obj.Tokenizer);
            for i = 1:numel(obj.TokenFilters), comp{end+1} = class(obj.TokenFilters{i}); end %#ok<AGROW>
            spec.ComposesFrom = comp;
        end
    end

    methods (Static)
        function obj = fromSpec(spec)
            % FROMSPEC  Build an Analyzer from data: fields CharFilters (strings),
            % Tokenizer (string), TokenFilters (strings; "stem:<stemmer>" allowed).
            arguments
                spec (1,1) struct
            end
            F = @OceanNLP.Text.Factory.create;
            cf = {};  tf = {};
            if isfield(spec, 'CharFilters')
                for n = string(spec.CharFilters(:))', cf{end+1} = F("charfilter", n); end %#ok<AGROW>
            end
            tokName = "regex";
            if isfield(spec, 'Tokenizer'), tokName = string(spec.Tokenizer); end
            tok = F("tokenizer", tokName);
            if isfield(spec, 'TokenFilters')
                for n = string(spec.TokenFilters(:))'
                    if startsWith(n, "stem:")
                        tf{end+1} = OceanNLP.Text.Filter.StemFilter('Stemmer', F("stemmer", extractAfter(n, "stem:"))); %#ok<AGROW>
                    else
                        tf{end+1} = F("filter", n); %#ok<AGROW>
                    end
                end
            end
            obj = OceanNLP.Text.Analyzer('CharFilters', cf, 'Tokenizer', tok, 'TokenFilters', tf);
        end

        function obj = preset(name)
            % PRESET  Named analyzers (data, not code).
            %   "basic"    RegexTokenizer only  (identical to using the tokenizer directly)
            %   "standard" basic + LowercaseFilter on a case-preserving tokenizer (Lucene layout)
            %   "english"  standard + Lucene English stop words + Porter stemming
            arguments
                name (1,1) string
            end
            P = OceanNLP.Text.Analyzer.presets(); %Get struct
            key = lower(name);
            if ~isfield(P, key)
                error('OceanNLP:Text:Analyzer:UnknownPreset', 'Unknown preset "%s". Valid: %s.', ...
                    name, strjoin(string(fieldnames(P))', ', '));
            end
            obj = OceanNLP.Text.Analyzer.fromSpec(P.(key));
        end

        function P = presets()
            P.basic    = struct('Tokenizer', "regex");
            P.standard = struct('Tokenizer', "regex", 'TokenFilters', "lowercase");
            P.english  = struct('Tokenizer', "regex", 'TokenFilters', ["lowercase" "stopword" "stem:porter"]);
        end
    end
end
