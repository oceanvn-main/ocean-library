classdef BertBackbone < handle
    % BERTBACKBONE  Shared front half of every BERT-family model in the
    % library: tokenize -> TokenEmbedding -> embedding LayerNorm -> N x
    % TransformerEncoderBlock, returning the (d x T) final hidden states for
    % one sequence OR one query/document pair (segment ids 0/1).
    %
    % Extracted from BertEncoder (neural-retrieval addendum 5.1) so the
    % bi-encoder, cross-encoder and ColBERT models reuse ONE verified stack
    % instead of three drifting copies. BertEncoder now delegates here and
    % only adds pooling; behaviour for single sequences is unchanged.
    %
    % WEIGHTS SCHEMA: exactly BertEncoder's (Weights.TokEmbedding.*,
    % Weights.Layers(i).*; PyTorch (out,in) matrices used AS-IS, no
    % transpose). Head weights (Pooler/Classifier/Projection) are NOT read
    % here - they belong to the model that owns the head.
    %
    % Tokenizer is built with BasicTokenize=true: real checkpoints were
    % trained with BERT's punctuation splitting, so omitting it changes ids
    % (and therefore embeddings) on any punctuated text.
    %
    % USAGE:
    %   bb = OceanNLP.Embedding.BertBackbone('Weights', W, 'VocabFile', "vocab.txt");
    %   [X, ids] = bb.encodeText("some text");            % (d x T)
    %   [X, ids, seg] = bb.encodePair("query", "doc");     % (d x T)
    %
    % See also: OceanNLP.Embedding.BertEncoder

    properties (SetAccess = private)
        % Untyped at property level: classes without zero-arg constructors
        % fail at class load (see BertEncoder MATLAB CONSTRAINT note).
        Tokenizer          % WordPieceTokenizer (BasicTokenize=true)
        TokEmbed           % TokenEmbedding
        EmbeddingLayerNorm % LayerNormLayer, eps 1e-12
        Blocks             % (1 x N) cell of TransformerEncoderBlock
        ModelName (1,1) string = "bert (unspecified checkpoint)"
        ClsId (1,1) double
        SepId (1,1) double
    end

    properties (Dependent, SetAccess = private)
        HiddenDim
        NumLayers
    end

    methods
        function obj = BertBackbone(options)
            arguments
                options.Weights (1,1) struct
                options.VocabFile (1,1) string
                options.ModelName (1,1) string = "bert (unspecified checkpoint)"
            end
            W = options.Weights;
            OceanNLP.Embedding.BertBackbone.validateWeights_(W);

            obj.ModelName = options.ModelName;
            obj.Tokenizer = OceanNLP.Embedding.Tokenize.WordPieceTokenizer( ...
                'VocabFile', options.VocabFile, 'BasicTokenize', true);

            specialIds = obj.Tokenizer.encode(["[CLS]" "[SEP]"]);
            if specialIds(1) == obj.Tokenizer.UnkId || specialIds(2) == obj.Tokenizer.UnkId
                error('OceanNLP:Embedding:BertBackbone:MissingSpecialTokens', ...
                    ['VocabFile "%s" must contain the literal tokens "[CLS]" and "[SEP]" ' ...
                     'as whole-word vocabulary entries (neither may resolve to UnkId=%d).'], ...
                    options.VocabFile, obj.Tokenizer.UnkId);
            end
            obj.ClsId = specialIds(1);
            obj.SepId = specialIds(2);

            obj.TokEmbed = OceanNLP.Embedding.Modules.TokenEmbedding( ...
                W.TokEmbedding.WordEmbeddings, W.TokEmbedding.PositionEmbeddings, W.TokEmbedding.TokenTypeEmbeddings);
            dTok = obj.TokEmbed.HiddenDim;

            obj.EmbeddingLayerNorm = OceanDL.Norm.LayerNormLayer(dTok, 'Epsilon', 1e-12);
            obj.EmbeddingLayerNorm.Gamma = W.TokEmbedding.LayerNormGamma(:);
            obj.EmbeddingLayerNorm.Beta  = W.TokEmbedding.LayerNormBeta(:);

            dLayer1 = size(W.Layers(1).WQ, 1);
            if dTok ~= dLayer1
                error('OceanNLP:Embedding:BertBackbone:HiddenDimMismatch', ...
                    ['Weights.TokEmbedding hidden dim d=%d does not match Weights.Layers(1) ' ...
                     'hidden dim d=%d (size(WQ,1)). TokEmbedding and the transformer layers ' ...
                     'must share one common d.'], dTok, dLayer1);
            end

            N = numel(W.Layers);
            blocks = cell(1, N);
            for i = 1:N
                L = W.Layers(i);
                mha = OceanNLP.Embedding.Modules.MultiHeadAttention( ...
                    L.WQ, L.bQ, L.WK, L.bK, L.WV, L.bV, L.WO, L.bO, L.NumHeads);
                ln1 = OceanDL.Norm.LayerNormLayer(dTok, 'Epsilon', 1e-12);
                ln1.Gamma = L.LN1Gamma(:);  ln1.Beta = L.LN1Beta(:);
                ln2 = OceanDL.Norm.LayerNormLayer(dTok, 'Epsilon', 1e-12);
                ln2.Gamma = L.LN2Gamma(:);  ln2.Beta = L.LN2Beta(:);
                blocks{i} = OceanNLP.Embedding.Modules.TransformerEncoderBlock( ...
                    mha, ln1, ln2, L.W1, L.b1(:), L.W2, L.b2(:));
            end
            obj.Blocks = blocks;
        end

        function X = hiddenStates(obj, ids, segmentIds, keyMask)
            % HIDDENSTATES  ids/segmentIds (1 x T, 0-indexed) -> (d x T).
            % keyMask (optional, 1 x T logical, false = not attendable) is
            % forwarded to every block; used by ColBERT's [MASK]-padded queries.
            arguments
                obj
                ids (1,:) double
                segmentIds (1,:) double
                keyMask (1,:) logical = true(1, numel(ids))
            end
            X = obj.TokEmbed.forward(ids, segmentIds);
            X = obj.EmbeddingLayerNorm.normalize(X, 'train');
            for L = 1:obj.NumLayers
                X = obj.Blocks{L}.forward(X, keyMask);
            end
        end

        function [X, ids] = encodeText(obj, text, options)
            % ENCODETEXT  [CLS] tokens [SEP], single segment -> (d x T).
            arguments
                obj
                text (1,1) string
                options.MaxLength (1,1) double = 512
            end
            sub = obj.Tokenizer.forward(text);
            sub = sub(1:min(numel(sub), options.MaxLength - 2));
            ids = obj.Tokenizer.encode(["[CLS]", sub, "[SEP]"]);
            X = obj.hiddenStates(ids, zeros(1, numel(ids)));
        end

        function [X, ids, segmentIds] = encodePair(obj, query, doc, varargin)
            % ENCODEPAIR  [CLS] query [SEP] doc [SEP] -> (d x T); name-value
            % options MaxLength / Truncation forwarded to the tokenizer.
            [ids, segmentIds] = obj.Tokenizer.encodePair(query, doc, varargin{:});
            X = obj.hiddenStates(ids, segmentIds);
        end

        function d = get.HiddenDim(obj)
            d = obj.TokEmbed.HiddenDim;
        end

        function n = get.NumLayers(obj)
            n = numel(obj.Blocks);
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = { ...
                'VIRTUAL composition of WordPieceTokenizer + TokenEmbedding + embedding LayerNorm + N x TransformerEncoderBlock; no math implemented here', ...
                'IsSymbolic=false is forced: the tokenization half has no closed form (R7)', ...
                sprintf('N=%d layers, d=%d, read from the supplied Weights', obj.NumLayers, obj.HiddenDim), ...
                'single sequence or single pair at a time; no batching or padding; an optional key mask (hiddenStates keyMask) is supported for ColBERT queries'};
            meta.IsSymbolic = false;
            meta.Reference = sprintf("Shared BERT stack (neural-retrieval addendum 5.1); model = ""%s""", obj.ModelName);
            tokMeta = obj.Tokenizer.getMetadata();
            assert(strlength(tokMeta.Reference) > 0, 'OceanNLP:Embedding:BertBackbone:TokenizerMetadataUnresolved', ...
                'Tokenizer metadata Reference is empty.');
            embMeta = obj.TokEmbed.getMetadata();
            assert(all(strlength(embMeta.SymbolicLatex) > 0), 'OceanNLP:Embedding:BertBackbone:TokEmbedMetadataUnresolved', ...
                'TokEmbed metadata SymbolicLatex is empty.');
            meta.ComposedPartReference = struct('TokenizerReference', tokMeta.Reference, ...
                'TokEmbeddingSymbolicLatex', embMeta.SymbolicLatex);
        end

        function spec = getPipelineSpec(~)
            spec = struct();
            spec.Inputs = struct('Name', "text", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x 1 (or query,doc pair)");
            spec.Outputs = struct('Name', "X", 'Domain', "token-state-matrix", 'Dtype', "double", 'Shape', "d x T");
            spec.Kind = "VIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "bert-backbone");
            spec.ComposesFrom = {'OceanNLP.Embedding.Tokenize.WordPieceTokenizer', ...
                                  'OceanNLP.Embedding.Modules.TokenEmbedding', ...
                                  'OceanNLP.Embedding.Modules.MultiHeadAttention', ...
                                  'OceanNLP.Embedding.Modules.TransformerEncoderBlock'};
        end
    end

    methods (Static, Access = private)
        function validateWeights_(W)
            if ~isfield(W, 'TokEmbedding') || ~isstruct(W.TokEmbedding)
                error('OceanNLP:Embedding:BertBackbone:MissingTokEmbedding', 'Weights.TokEmbedding (struct) is required.');
            end
            tokFields = {'WordEmbeddings', 'PositionEmbeddings', 'TokenTypeEmbeddings', 'LayerNormGamma', 'LayerNormBeta'};
            for k = 1:numel(tokFields)
                if ~isfield(W.TokEmbedding, tokFields{k})
                    error('OceanNLP:Embedding:BertBackbone:MissingTokEmbeddingField', 'Weights.TokEmbedding.%s is required.', tokFields{k});
                end
            end
            if ~isfield(W, 'Layers') || ~isstruct(W.Layers) || isempty(W.Layers)
                error('OceanNLP:Embedding:BertBackbone:MissingLayers', ...
                    'Weights.Layers (non-empty struct array, one entry per transformer layer) is required.');
            end
            layerFields = {'WQ','bQ','WK','bK','WV','bV','WO','bO','NumHeads','W1','b1','W2','b2','LN1Gamma','LN1Beta','LN2Gamma','LN2Beta'};
            for i = 1:numel(W.Layers)
                for k = 1:numel(layerFields)
                    if ~isfield(W.Layers(i), layerFields{k})
                        error('OceanNLP:Embedding:BertBackbone:MissingLayerField', 'Weights.Layers(%d).%s is required.', i, layerFields{k});
                    end
                end
            end
        end
    end
end
