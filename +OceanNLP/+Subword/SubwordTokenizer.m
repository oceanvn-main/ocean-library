classdef SubwordTokenizer < OceanNLP.Tokenize.ITokenizer
    % SUBWORDTOKENIZER  Trainable subword tokenizer = pre-tokenizer + (optional)
    % byte-level mapping + subword algorithm + vocabulary. It is the CONTEXT of
    % two strategy slots (OceanNLP.Text.IPreTokenizer, OceanNLP.Subword.ISubwordAlgorithm)
    % and never names a concrete algorithm, so BPE / WordPiece / anything added
    % later is a constructor argument.
    %
    %   tok = SubwordTokenizer();                                  % GPT-2 style: Gpt2PreTokenizer + byte-level + BPE
    %   tok = SubwordTokenizer('Algorithm', OceanNLP.Subword.Algorithm.BpeAlgorithm(), ...
    %                          'PreTokenizer', OceanNLP.Text.PreTokenize.WhitespacePreTokenizer(), ...
    %                          'ByteLevel', false);
    %   tok.fit(corpus, 'VocabSize', 800);      % learn (PyTorch-style: mutates buffers on the object)
    %   tokens = tok.forward("some text");      % subword strings (ITokenizer contract)
    %   ids    = tok.encode("some text");       % 0-indexed ids (BERT/HF convention)
    %   text   = tok.decode(ids);               % exact inverse for byte-level; joined pieces otherwise
    %
    % Pretrained GPT-2 style files:  SubwordTokenizer.fromFiles("vocab.json", "merges.txt")
    %
    % fit() OVERWRITES any previous state (with a warning), exactly like the
    % library's other fit() methods.
    %
    % Special tokens: a ROLE -> TOKEN struct, e.g. 'Special', struct('unk',"[UNK]", 'pad',"[PAD]", 'cls',"[CLS]").
    % They are placed BEFORE the learned pieces (ids 0..k-1, in field order) and looked up by role via
    % Vocab.specialId("cls"). Text is never matched against them. The "unk" role is filled from the
    % algorithm's own unknown token when it has one (WordPiece: "[UNK]"), else a fallback "<unk>" entry
    % is appended (byte-level vocabularies never produce it).
    %
    % See also: OceanNLP.Vocab.Vocabulary, OceanNLP.Subword.ByteLevel

    properties (SetAccess = private)
        PreTokenizer = []          % IPreTokenizer strategy (set in the constructor; NOT a property default,
        Algorithm = []             % ISubwordAlgorithm strategy  which would be one handle shared by all objects)
        ByteLevelMapping (1,1) logical = true
        Vocab = []                 % OceanNLP.Vocab.Vocabulary (fixed), set by fit()/fromFiles()
        Special (1,1) struct = struct()     % role -> token
    end

    methods
        function obj = SubwordTokenizer(options)
            arguments
                options.PreTokenizer (1,1) OceanNLP.Text.IPreTokenizer = OceanNLP.Text.PreTokenize.Gpt2PreTokenizer()
                options.Algorithm (1,1) OceanNLP.Subword.ISubwordAlgorithm = OceanNLP.Subword.Algorithm.BpeAlgorithm()
                options.ByteLevel (1,1) logical = true
                options.Special (1,1) struct = struct()
            end
            obj.PreTokenizer = options.PreTokenizer;
            obj.Algorithm = options.Algorithm;
            obj.ByteLevelMapping = options.ByteLevel;
            obj.Special = options.Special;
        end

        function fit(obj, corpus, options)
            arguments
                obj
                corpus (1,:) {mustBeText}
                options.VocabSize (1,1) double {mustBeInteger, mustBePositive} = 1000
                options.MinFrequency (1,1) double {mustBeNonnegative} = 0
            end
            if obj.isFitted()
                warning('OceanNLP:Subword:SubwordTokenizer:RefitOverwrites', ...
                    'fit() overwrites the learned merges/vocabulary of this tokenizer.');
            end
            corpus = string(corpus);
            all = strings(1, 0);
            for d = 1:numel(corpus)
                w = obj.preWords_(corpus(d));
                all = [all, w]; %#ok<AGROW>
            end
            [uw, ~, ic] = unique(all);
            counts = accumarray(ic(:), 1)';
            seed = strings(1, 0);
            if obj.ByteLevelMapping, seed = OceanNLP.Subword.ByteLevel.alphabet(); end
            [specials, roles] = obj.resolveSpecials_();
            budget = options.VocabSize - numel(specials);
            obj.Algorithm.train(uw, counts, 'VocabSize', budget, 'Seed', seed, 'MinFrequency', options.MinFrequency);
            obj.Vocab = OceanNLP.Subword.SubwordTokenizer.buildVocab_([specials, obj.Algorithm.tokens()], roles); %Special first, the bas, and learned tokens later
        end

        function tf = isFitted(obj)
            tf = ~isempty(obj.Vocab);
        end

        function tokens = forward(obj, rawText)
            arguments
                obj
                rawText (1,1) string
            end
            if ~obj.isFitted()
                error('OceanNLP:Subword:SubwordTokenizer:NotFitted', 'Call fit() (or fromFiles) before forward().');
            end
            words = obj.preWords_(rawText);
            tokens = strings(1, 0);
            for i = 1:numel(words)
                tokens = [tokens, obj.Algorithm.segment(words(i))]; %#ok<AGROW>
            end
        end

        function ids = encode(obj, rawText)
            % ENCODE  0-indexed ids of forward(rawText); unknown pieces -> the vocabulary's unk id.
            arguments
                obj
                rawText (1,1) string
            end
            ids = obj.Vocab.id0(obj.forward(rawText));
        end

        function text = decode(obj, ids)
            arguments
                obj
                ids (1,:) double
            end
            % How pieces are glued back together is the ALGORITHM's business (detokenize); only the
            % byte-level mapping, which belongs to this class, is undone afterwards.
            text = obj.Algorithm.detokenize(obj.Vocab.token0(ids));
            if obj.ByteLevelMapping
                text = OceanNLP.Subword.ByteLevel.decode(text);
            end
        end

        function s = name(obj), s = "subword:" + obj.Algorithm.name(); end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = {'VIRTUAL composition of a pre-tokenizer + (optional) byte-level mapping + a subword algorithm + a Vocabulary; the segmentation logic lives in the algorithm', ...
                                 'no closed form (R7): discrete string procedure', ...
                                 char("algorithm = " + obj.Algorithm.name() + ", pre-tokenizer = " + obj.PreTokenizer.name() + ", byte-level = " + string(obj.ByteLevelMapping))};
            meta.IsSymbolic = false;
            meta.Reference = "subword tokenization (Sennrich et al. 2016; Radford et al. 2019); composed strategies are inspectable";
            meta.ComposedPartReference = struct('AlgorithmReference', obj.Algorithm.getMetadata().Reference, ...
                'PreTokenizerReference', obj.PreTokenizer.getMetadata().Reference);
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.Inputs = struct('Name', "rawText", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x 1");
            spec.Outputs = struct('Name', "tokens", 'Domain', "token-sequence", 'Dtype', "string", 'Shape', "1 x T");
            spec.Kind = "VIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "subword");
            spec.ComposesFrom = {class(obj.PreTokenizer), class(obj.Algorithm), 'OceanNLP.Vocab.Vocabulary'};
            if obj.ByteLevelMapping, spec.ComposesFrom{end+1} = 'OceanNLP.Subword.ByteLevel'; end
        end
    end

    methods (Access = private)
        function [specials, roles] = resolveSpecials_(obj)
            roles = obj.Special;
            u = obj.Algorithm.unkToken();
            if ~isfield(roles, 'unk') && u ~= "", roles = struct('unk', u); roles = catstruct_(roles, obj.Special); end
            specials = strings(1, 0);
            for f = fieldnames(roles)'
                specials(end+1) = string(roles.(f{1})); %#ok<AGROW>
            end
        end

        function words = preWords_(obj, text)
            words = obj.PreTokenizer.forward(text);
            if obj.ByteLevelMapping
                for i = 1:numel(words), words(i) = OceanNLP.Subword.ByteLevel.encode(words(i)); end
            end
        end
    end

    methods (Static)
        function obj = fromFiles(vocabJson, mergesTxt, options)
            % FROMFILES  A pretrained byte-level BPE tokenizer (GPT-2 vocab.json + merges.txt).
            arguments
                vocabJson (1,1) string
                mergesTxt (1,1) string
                options.Unk (1,1) string = "<|endoftext|>"
            end
            algo = OceanNLP.Subword.Algorithm.BpeAlgorithm.fromMergesFile(mergesTxt);
            obj = OceanNLP.Subword.SubwordTokenizer('Algorithm', algo);
            obj.Vocab = OceanNLP.Vocab.Vocabulary.fromVocabJson(vocabJson, 'Unk', options.Unk);
        end
    end

    methods (Static)
        function obj = fromMerges(merges)
            % FROMMERGES  Byte-level BPE tokenizer from an ORDERED merge list (rank = row), with the
            % vocabulary = 256 byte symbols + every merge result. Used to run segmentation with
            % merges learned elsewhere (e.g. validating against another library's trained merges).
            arguments
                merges (:,2) string
            end
            algo = OceanNLP.Subword.Algorithm.BpeAlgorithm();
            algo.setMerges(merges);
            obj = OceanNLP.Subword.SubwordTokenizer('Algorithm', algo);
            toks = [OceanNLP.Subword.ByteLevel.alphabet(), reshape(merges(:, 1) + merges(:, 2), 1, [])];
            [~, first] = unique(toks, 'stable');
            obj.Vocab = OceanNLP.Subword.SubwordTokenizer.buildVocab_(toks(first), struct());
        end
    end

    methods (Static, Access = private)
        function v = buildVocab_(toks, roles)
            % A Vocabulary needs an unknown-token entry. Byte-level vocabularies never produce one, so if no
            % "unk" role exists a fallback "<unk>" entry is appended and registered.
            if ~isfield(roles, 'unk')
                toks = [toks, "<unk>"];
                roles.unk = "<unk>";
            end
            v = OceanNLP.Vocab.Vocabulary.fromTokens(toks, 'Unk', string(roles.unk), 'Special', roles);
        end
    end
end

function s = catstruct_(a, b)
% catstruct_  fields of a then b (b's fields win on name clashes; order: a first).
s = a;
for f = fieldnames(b)'
    s.(f{1}) = b.(f{1});
end
end
