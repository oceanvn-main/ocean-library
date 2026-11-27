classdef CrossEncoder < handle
    % CROSSENCODER  Joint query/document relevance scorer (monoBERT style):
    % [CLS] query [SEP] doc [SEP] -> BertBackbone -> PoolerHead -> LinearHead
    % -> one scalar per pair.
    %
    % Nothing is precomputable, so this is meant to RE-RANK a shortlist
    % (see OceanNLP.Retrieval.CrossEncoderReranker), not scan a corpus.
    %
    % WEIGHTS: the BertBackbone schema plus
    %   Weights.Pooler.W (d x d), .b (d x 1)      <- pooler.dense.weight/.bias
    %   Weights.Classifier.W (1 x d), .b (1 x 1)  <- classifier.weight/.bias
    % PyTorch (out,in) matrices, no transpose.
    %
    % Activation: "identity" returns the raw logit (default, rank-equivalent);
    % "sigmoid" applies 1/(1+exp(-x)) to it, which is what
    % sentence_transformers.CrossEncoder.predict does by default.
    %
    % Truncation: pairs longer than MaxLength are cut per Truncation
    % ("longest_first" = HuggingFace truncation=True; "only_second").
    %
    % USAGE:
    %   ce = OceanNLP.Embedding.CrossEncoder('Weights', W, 'VocabFile', "vocab.txt");
    %   s  = ce.score("query", "document");            % scalar
    %   s  = ce.scoreDocs("query", docs);              % (1 x M)
    %
    % See also: OceanNLP.Embedding.BertBackbone, OceanNLP.Retrieval.CrossEncoderReranker

    properties (SetAccess = private)
        Backbone
        Pooler
        Classifier
        MaxLength (1,1) double = 512
        Truncation (1,1) string = "longest_first"
        Activation (1,1) string = "identity"
    end

    methods
        function obj = CrossEncoder(options)
            arguments
                options.Weights (1,1) struct
                options.VocabFile (1,1) string
                options.ModelName (1,1) string = "cross-encoder"
                options.MaxLength (1,1) double {mustBeInteger, mustBeGreaterThan(options.MaxLength, 3)} = 512
                options.Truncation (1,1) string {mustBeMember(options.Truncation, ["longest_first","only_second"])} = "longest_first"
                options.Activation (1,1) string {mustBeMember(options.Activation, ["identity","sigmoid"])} = "identity"
            end
            W = options.Weights;
            for f = ["Pooler","Classifier"]
                if ~isfield(W, f) || ~isfield(W.(f), 'W') || ~isfield(W.(f), 'b')
                    error('OceanNLP:Embedding:CrossEncoder:MissingHead', ...
                        'Weights.%s.W and Weights.%s.b are required.', f, f);
                end
            end
            if size(W.Classifier.W, 1) ~= 1
                error('OceanNLP:Embedding:CrossEncoder:NotSingleLabel', ...
                    'Classifier.W must be 1 x d (one relevance logit). Got %d x %d.', ...
                    size(W.Classifier.W, 1), size(W.Classifier.W, 2));
            end
            obj.Backbone = OceanNLP.Embedding.BertBackbone('Weights', W, ...
                'VocabFile', options.VocabFile, 'ModelName', options.ModelName);
            obj.Pooler     = OceanNLP.Embedding.Modules.PoolerHead(W.Pooler.W, W.Pooler.b(:));
            obj.Classifier = OceanNLP.Embedding.Modules.LinearHead(W.Classifier.W, W.Classifier.b(:));
            obj.MaxLength  = options.MaxLength;
            obj.Truncation = options.Truncation;
            obj.Activation = options.Activation;
        end

        function s = score(obj, query, doc)
            arguments
                obj
                query (1,1) string
                doc   (1,1) string
            end
            X = obj.Backbone.encodePair(query, doc, 'MaxLength', obj.MaxLength, 'Truncation', obj.Truncation);
            s = obj.Classifier.forward(obj.Pooler.forward(X));      % (1 x 1) logit
            if obj.Activation == "sigmoid"
                s = 1 ./ (1 + exp(-s));
            end
        end

        function s = scoreDocs(obj, query, docs)
            arguments
                obj
                query (1,1) string
                docs  (1,:) string
            end
            s = zeros(1, numel(docs));
            for j = 1:numel(docs)
                s(j) = obj.score(query, docs(j));
            end
        end

        function s = name(obj)
            s = "cross-encoder:" + obj.Backbone.ModelName;
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = { ...
                'VIRTUAL composition of BertBackbone + PoolerHead + LinearHead; no math implemented here', ...
                'IsSymbolic=false is forced: the tokenizer has no closed form (R7); closed forms live one level down', ...
                'scores a (query, document) pair jointly; not precomputable, intended for re-ranking a shortlist', ...
                char("activation = " + obj.Activation + " (identity = raw logit)")};
            meta.IsSymbolic = false;
            meta.Reference = "VIRTUAL composite: inspect obj.Pooler.getMetadata() and obj.Classifier.getMetadata() for the closed-form heads (Nogueira & Cho 2019)";
            poolerMeta = obj.Pooler.getMetadata();
            clsMeta = obj.Classifier.getMetadata();
            assert(strlength(poolerMeta.SymbolicLatex) > 0 && strlength(clsMeta.SymbolicLatex) > 0, ...
                'OceanNLP:Embedding:CrossEncoder:HeadMetadataUnresolved', 'Head metadata did not resolve.');
            meta.ComposedPartReference = struct( ...
                'BackboneReference', obj.Backbone.getMetadata().Reference, ...
                'PoolerSymbolicLatex', poolerMeta.SymbolicLatex, ...
                'ClassifierSymbolicLatex', clsMeta.SymbolicLatex);
        end

        function spec = getPipelineSpec(~)
            spec = struct();
            spec.Inputs = [ ...
                struct('Name', "query", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x 1"), ...
                struct('Name', "doc", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x 1")];
            spec.Outputs = struct('Name', "score", 'Domain', "ranked-result", 'Dtype', "double", 'Shape', "1 x 1");
            spec.Kind = "VIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "cross-encode");
            spec.ComposesFrom = {'OceanNLP.Embedding.BertBackbone', ...
                                  'OceanNLP.Embedding.Modules.PoolerHead', ...
                                  'OceanNLP.Embedding.Modules.LinearHead'};
        end
    end
end
