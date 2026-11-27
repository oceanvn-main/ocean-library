classdef RetrievalMetricFactory
    % RETRIEVALMETRICFACTORY  Create IRetrievalMetric objects by name.
    %
    %   Centralizes retrieval-metric construction so callers never
    %   hard-code class names. Mirrors OceanNLP.Tokenize.TokenizerFactory's
    %   shape: Static create(name, options), Static validNames(), WITH
    %   name-value passthrough (needed here for the same reason
    %   TokenizerFactory needs it for RegexTokenizer's Pattern: three of
    %   the five metrics take a required constructor option).
    %
    %   Supported names (case-insensitive):
    %     'precision@k' — PrecisionAtK       (requires 'K')
    %     'recall@k'    — RecallAtK          (requires 'K')
    %     'ap'          — AveragePrecision   (no config)
    %     'map'         — MeanAveragePrecision (no config; extends 'ap' with
    %                     a multi-query aggregate(), see that class)
    %     'rr'          — ReciprocalRank     (no config)
    %     'ndcg@k'      — NDCGAtK            (requires 'K')
    %
    %   NAMING RATIONALE: the '@k' suffix flags exactly the metrics that
    %   take a required K cutoff, matching how this project already
    %   writes them in prose and in the master plan
    %   (Planning/System/Retrieval/Retrieval_System_Master_Plan.md §3.2:
    %   "Precision@K", "NDCG@K"); 'ap'/'rr' are the standard IR
    %   abbreviations (Average Precision, Reciprocal Rank) for the two
    %   whole-list metrics that take no config, so they carry no suffix.
    %
    %   Usage:
    %     m = OceanStats.Metrics.Retrieval.RetrievalMetricFactory.create('precision@k', 'K', 5);
    %     m = OceanStats.Metrics.Retrieval.RetrievalMetricFactory.create('ap');
    %     score = m.forward(rankedIdx, relevance);
    %
    %   See also: OceanStats.Metrics.Retrieval.IRetrievalMetric

    methods (Static)
        function metric = create(name, options)
            % CREATE  Return an IRetrievalMetric object for the given name.
            %
            %   Trailing name-value pairs (captured by the repeating
            %   `options` argument) are forwarded to the target metric's
            %   constructor unmodified, so a missing required 'K' (for the
            %   @K metrics) or any unknown/mistyped option surfaces as
            %   that constructor's own arguments-block error, not a
            %   silently-ignored value.
            arguments
                name (1,1) string
            end
            arguments (Repeating)
                options
            end
            import OceanStats.Metrics.Retrieval.*

            switch lower(strtrim(name))
                case 'precision@k'
                    metric = PrecisionAtK(options{:});
                case 'recall@k'
                    metric = RecallAtK(options{:});
                case 'ap'
                    metric = AveragePrecision(options{:});
                case 'map'
                    metric = MeanAveragePrecision(options{:});
                case 'rr'
                    metric = ReciprocalRank(options{:});
                case 'ndcg@k'
                    metric = NDCGAtK(options{:});
                otherwise
                    error('OceanStats:Metrics:Retrieval:RetrievalMetricFactory:UnknownMetric', ...
                        'Unknown retrieval metric "%s". Valid: %s.', ...
                        name, strjoin(RetrievalMetricFactory.validNames(), ', '));
            end
        end

        function names = validNames()
            % VALIDNAMES  Return cell of all recognised metric name strings.
            names = {'precision@k', 'recall@k', 'ap', 'map', 'rr', 'ndcg@k'};
        end
    end
end
