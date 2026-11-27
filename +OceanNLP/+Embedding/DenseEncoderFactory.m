classdef DenseEncoderFactory
    % DENSEENCODERFACTORY  Create IDenseEncoder objects by NAMED, PRE-CONFIGURED PROFILE.
    %
    %   Centralizes dense-encoder construction so callers never hard-code
    %   class names. Mirrors OceanStats.Similarity.SimilarityFactory's /
    %   OceanNLP.Tokenize.TokenizerFactory's Static create(name)/validNames()
    %   shape.
    %
    %   WHY THIS FACTORY IS *NOT* THE SAME MISTAKE AS SimilarityFactory'S
    %   ORIGINAL ONE-PRODUCT SITUATION (read before assuming "one concrete
    %   class => premature abstraction" applies here too — it doesn't, and
    %   the distinction matters enough to spell out):
    %     SimilarityFactory was flagged, earlier in this project, as
    %     premature abstraction because there was exactly ONE concrete
    %     ISimilarity implementation (CosineSimilarity) and no second one on
    %     any concrete roadmap — the factory added a name->class indirection
    %     with nothing to actually select BETWEEN.
    %     This factory ALSO has exactly one concrete IDenseEncoder
    %     implementation (BertEncoder) as of this writing — but the thing
    %     being selected between was NEVER "which class" here; it is "which
    %     CHECKPOINT". Master plan §1 states the premise explicitly:
    %     "embedding-based retrieval across multiple BERT-family models" —
    %     bert-tiny, bert-base, and other checkpoints are all real,
    %     concretely-planned callers of this SAME BertEncoder class with
    %     DIFFERENT Weights/VocabFile/ModelName arguments (see BertEncoder's
    %     WEIGHTS SCHEMA note: N layers, d, and d_ff are all read from
    %     whatever Weights struct is supplied — the class itself is already
    %     checkpoint-agnostic). This factory's `name` argument selects a
    %     PROFILE (a known, recognized checkpoint identity with a sensible
    %     default ModelName), not a class — the exact case a Strategy
    %     +Factory pair is meant to serve even at "one class" cardinality,
    %     because the axis of variation (checkpoint) is real and growing,
    %     unlike SimilarityFactory's axis of variation (similarity
    %     function), which had no second real implementation queued.
    %
    %   Supported profile names (case-insensitive) — see validNames():
    %     'bert-tiny'         — prajjwal1/bert-tiny (master plan §3.3's
    %                           resolved default checkpoint: 2 layers, 128
    %                           hidden dim, ~17MB)
    %     'bert-mini'         — a smaller bert-family profile name, reserved
    %                           for a future checkpoint of that shape
    %     'bert-base-uncased' — the standard 12-layer/768-dim BERT profile
    %                           name, reserved for a future checkpoint
    %     'custom'            — any OTHER checkpoint that isn't one of the
    %                           named profiles above (e.g. a small synthetic
    %                           Weights struct hand-built for testing, per
    %                           this class's own verification). ModelName
    %                           still defaults to the given name ("custom")
    %                           unless the caller overrides it — same rule
    %                           as every other profile.
    %   ALL profile names currently route to the SAME BertEncoder class
    %   (there is, honestly, no other IDenseEncoder implementation yet) —
    %   what a name buys today is (a) a validated, typo-caught identifier
    %   and (b) a sensible ModelName default for getMetadata().Reference;
    %   what it is FOR is giving a second concrete IDenseEncoder (should one
    %   ever be built, e.g. a non-BERT dense encoder) a place to be routed
    %   to without changing any caller's create(name, ...) call site.
    %
    %   NAME-VALUE PASSTHROUGH (mirrors TokenizerFactory's Pattern-
    %   passthrough pattern): trailing name-value pairs given to create()
    %   (i.e. 'Weights', 'VocabFile', and optionally 'ModelName') are
    %   forwarded verbatim to BertEncoder's constructor. If the caller does
    %   NOT supply 'ModelName' explicitly, it is defaulted to the profile
    %   `name` itself — so create('bert-tiny', 'Weights', W, 'VocabFile', p)
    %   produces an encoder whose getMetadata().Reference/name() mention
    %   "bert-tiny" without the caller having to repeat it.
    %
    %   Usage:
    %     enc = OceanNLP.Embedding.DenseEncoderFactory.create('bert-tiny', ...
    %               'Weights', W, 'VocabFile', "vocab.txt");
    %     enc.fit(corpus);
    %     E = enc.forward(corpus);
    %
    % See also: OceanNLP.Embedding.IDenseEncoder, OceanNLP.Embedding.BertEncoder,
    %           OceanStats.Similarity.SimilarityFactory, OceanNLP.Tokenize.TokenizerFactory

    methods (Static)
        function enc = create(name, options)
            % CREATE  Return an IDenseEncoder object for the given profile name.
            %
            %   enc = DenseEncoderFactory.create('bert-tiny', 'Weights', W, 'VocabFile', path)
            %
            %   Trailing name-value pairs (captured by the repeating
            %   `options` argument) are forwarded to BertEncoder's
            %   constructor unmodified (plus an auto-defaulted 'ModelName'
            %   when the caller didn't supply one — see class NAME-VALUE
            %   PASSTHROUGH note), so any unknown or mistyped option
            %   surfaces as BertEncoder's own `arguments`-block error, not a
            %   silently-ignored value.
            arguments
                name (1,1) string
            end
            arguments (Repeating)
                options
            end
            import OceanNLP.Embedding.*

            profile = lower(strtrim(name));
            valid = DenseEncoderFactory.validNames();
            if ~ismember(profile, valid)
                error('OceanNLP:Embedding:DenseEncoderFactory:UnknownProfile', ...
                    'Unknown dense-encoder profile "%s". Valid: %s.', ...
                    name, strjoin(valid, ', '));
            end

            hasModelName = false;
            for k = 1:2:numel(options)
                if strcmpi(string(options{k}), "ModelName")
                    hasModelName = true;
                    break
                end
            end
            if ~hasModelName
                options = [options, {'ModelName', name}];
            end

            enc = BertEncoder(options{:});
        end

        function names = validNames()
            % VALIDNAMES  Return cell of all recognised dense-encoder profile
            % name strings (see class header for what each profile means).
            names = {'bert-tiny', 'bert-mini', 'bert-base-uncased', 'custom'};
        end
    end
end
