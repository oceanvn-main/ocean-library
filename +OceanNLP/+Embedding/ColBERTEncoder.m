classdef ColBERTEncoder < handle
    % COLBERTENCODER  ColBERT late-interaction encoder: text -> (T x dim)
    % matrix of L2-normalized PER-TOKEN embeddings (Khattab & Zaharia 2020).
    %
    % BertBackbone -> LinearHead (dim x d, NO bias) -> row L2-normalize.
    %
    % PREPROCESSING IS PART OF THE MODEL (checkpoints are trained with it;
    % values below are read from the checkpoint's artifact.metadata /
    % onnx_config.json, not recalled).
    %
    % DISCLOSED CONFLICT in the checkpoint's own files: artifact.metadata
    % (the Stanford-ColBERT training config) has query_maxlen=32,
    % attend_to_mask_tokens=false, and the README's sentence-transformers
    % example returns (32, 96) query embeddings, i.e. [MASK]-padded queries;
    % but onnx_config.json (the Vespa/ONNX deployment config) says
    % do_query_expansion=false. This class defaults to the training/README
    % behaviour (QueryExpansion=true). The Python reference re-implements the
    % same choice, so agreement with it does NOT validate that choice; the
    % benchmark reports retrieval quality with each variant
    % (QueryExpansion / AttendToMaskTokens) so the sensitivity is visible.
    %   query    [CLS] [Q] tokens [SEP], then padded with [MASK] up to
    %            QueryMaxLength (query augmentation). All padded positions
    %            still produce embeddings that enter MaxSim.
    %            AttendToMaskTokens=false (checkpoint default): real tokens
    %            do NOT attend to the padded [MASK] keys.
    %   document [CLS] [D] tokens [SEP], no padding, cut to DocMaxLength.
    %            With SkipPunctuation, rows for ASCII-punctuation tokens
    %            are DROPPED after encoding ([CLS]/[D]/[SEP] are kept).
    %   [Q] / [D] are the vocab entries "[unused0]" / "[unused1]".
    %
    % WEIGHTS: BertBackbone schema plus Weights.Projection.W (dim x d),
    % <- checkpoint "linear.weight", PyTorch layout, no transpose.
    %
    % USAGE:
    %   enc = OceanNLP.Embedding.ColBERTEncoder('Weights', W, 'VocabFile', "vocab.txt");
    %   Q = enc.encodeQuery("gas turbine");        % (32 x dim)
    %   D = enc.encodeDocument("a document ...");  % (nd x dim)
    %
    % See also: OceanStats.Similarity.LateInteraction, OceanNLP.Retrieval.ColBERTRanker

    properties (SetAccess = private)
        Backbone
        Projection
        QueryMaxLength (1,1) double = 32
        DocMaxLength (1,1) double = 300
        QueryExpansion (1,1) logical = true
        AttendToMaskTokens (1,1) logical = false
        SkipPunctuation (1,1) logical = true
        QueryMarkerId (1,1) double
        DocMarkerId (1,1) double
        MaskId (1,1) double
        SkipIds (1,:) double
    end

    properties (Dependent, SetAccess = private)
        EmbeddingDim
    end

    methods
        function obj = ColBERTEncoder(options)
            arguments
                options.Weights (1,1) struct
                options.VocabFile (1,1) string
                options.ModelName (1,1) string = "colbert"
                options.QueryMaxLength (1,1) double {mustBeInteger, mustBeGreaterThan(options.QueryMaxLength, 3)} = 32
                options.DocMaxLength (1,1) double {mustBeInteger, mustBeGreaterThan(options.DocMaxLength, 3)} = 300
                options.QueryExpansion (1,1) logical = true
                options.AttendToMaskTokens (1,1) logical = false
                options.SkipPunctuation (1,1) logical = true
            end
            W = options.Weights;
            if ~isfield(W, 'Projection') || ~isfield(W.Projection, 'W')
                error('OceanNLP:Embedding:ColBERTEncoder:MissingProjection', ...
                    'Weights.Projection.W (dim x d) is required.');
            end
            obj.Backbone = OceanNLP.Embedding.BertBackbone('Weights', W, ...
                'VocabFile', options.VocabFile, 'ModelName', options.ModelName);
            if size(W.Projection.W, 2) ~= obj.Backbone.HiddenDim
                error('OceanNLP:Embedding:ColBERTEncoder:ProjectionDimMismatch', ...
                    'Projection.W has %d columns but the backbone hidden dim is %d.', ...
                    size(W.Projection.W, 2), obj.Backbone.HiddenDim);
            end
            obj.Projection = OceanNLP.Embedding.Modules.LinearHead(W.Projection.W);
            obj.QueryMaxLength = options.QueryMaxLength;
            obj.DocMaxLength = options.DocMaxLength;
            obj.QueryExpansion = options.QueryExpansion;
            obj.AttendToMaskTokens = options.AttendToMaskTokens;
            obj.SkipPunctuation = options.SkipPunctuation;

            tok = obj.Backbone.Tokenizer;
            required = ["[unused0]" "[unused1]" "[MASK]"];
            ids = tok.encode(required);
            if any(ids == tok.UnkId)
                error('OceanNLP:Embedding:ColBERTEncoder:MissingSpecialTokens', ...
                    'Vocab must contain [unused0], [unused1] and [MASK] as whole entries.');
            end
            obj.QueryMarkerId = ids(1);
            obj.DocMarkerId = ids(2);
            obj.MaskId = ids(3);
            punct = string(num2cell('!"#$%&''()*+,-./:;<=>?@[\]^_`{|}~'));
            skip = tok.encode(punct);
            obj.SkipIds = skip(skip ~= tok.UnkId);
        end

        function E = encodeQuery(obj, text)
            % ENCODEQUERY  -> (T x dim), T = QueryMaxLength when QueryExpansion.
            arguments
                obj
                text (1,1) string
            end
            tok = obj.Backbone.Tokenizer;
            sub = tok.forward(text);
            sub = sub(1:min(numel(sub), obj.QueryMaxLength - 3));
            core = [obj.Backbone.ClsId, obj.QueryMarkerId, tok.encode(sub), obj.Backbone.SepId];
            ids = core;
            keyMask = true(1, numel(core));
            if obj.QueryExpansion
                nPad = obj.QueryMaxLength - numel(core);
                ids = [core, repmat(obj.MaskId, 1, nPad)];
                keyMask = [true(1, numel(core)), repmat(obj.AttendToMaskTokens, 1, nPad)];
            end
            X = obj.Backbone.hiddenStates(ids, zeros(1, numel(ids)), keyMask);
            E = obj.projectNormalize_(X);
        end

        function E = encodeDocument(obj, text)
            arguments
                obj
                text (1,1) string
            end
            tok = obj.Backbone.Tokenizer;
            sub = tok.forward(text);
            sub = sub(1:min(numel(sub), obj.DocMaxLength - 3));
            ids = [obj.Backbone.ClsId, obj.DocMarkerId, tok.encode(sub), obj.Backbone.SepId];
            X = obj.Backbone.hiddenStates(ids, zeros(1, numel(ids)));
            E = obj.projectNormalize_(X);
            if obj.SkipPunctuation
                E = E(~ismember(ids, obj.SkipIds), :);
            end
        end

        function d = get.EmbeddingDim(obj)
            d = obj.Projection.OutDim;
        end

        function s = name(obj)
            s = "colbert:" + obj.Backbone.ModelName;
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = { ...
                'VIRTUAL composition of BertBackbone + LinearHead projection + row L2 normalization; no math implemented here', ...
                'IsSymbolic=false is forced: the tokenizer has no closed form (R7); MaxSim closed form is in OceanStats.Similarity.LateInteraction', ...
                sprintf('query: [Q] marker, [MASK]-padded to %d (QueryExpansion=%d, AttendToMaskTokens=%d)', obj.QueryMaxLength, obj.QueryExpansion, obj.AttendToMaskTokens), ...
                sprintf('document: [D] marker, max %d tokens, ASCII-punctuation rows dropped (SkipPunctuation=%d)', obj.DocMaxLength, obj.SkipPunctuation)};
            meta.IsSymbolic = false;
            meta.Reference = "VIRTUAL composite: inspect obj.Projection.getMetadata() and OceanStats.Similarity.LateInteraction (Khattab & Zaharia 2020)";
            projMeta = obj.Projection.getMetadata();
            assert(strlength(projMeta.SymbolicLatex) > 0, ...
                'OceanNLP:Embedding:ColBERTEncoder:ProjectionMetadataUnresolved', 'Projection metadata did not resolve.');
            meta.ComposedPartReference = struct( ...
                'BackboneReference', obj.Backbone.getMetadata().Reference, ...
                'ProjectionSymbolicLatex', projMeta.SymbolicLatex);
        end

        function spec = getPipelineSpec(~)
            spec = struct();
            spec.Inputs = struct('Name', "text", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x 1");
            spec.Outputs = struct('Name', "E", 'Domain', "token-state-matrix", 'Dtype', "double", 'Shape', "T x dim");
            spec.Kind = "VIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "colbert-encode");
            spec.ComposesFrom = {'OceanNLP.Embedding.BertBackbone', 'OceanNLP.Embedding.Modules.LinearHead'};
        end
    end

    methods (Access = private)
        function E = projectNormalize_(obj, X)
            Y = obj.Projection.forward(X)';                         % (T x dim)
            E = Y ./ max(sqrt(sum(Y.^2, 2)), 1e-12);
        end
    end
end
