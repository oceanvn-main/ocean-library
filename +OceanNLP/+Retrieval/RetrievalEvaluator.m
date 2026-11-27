classdef RetrievalEvaluator < handle
    % RETRIEVALEVALUATOR  Score a fitted IRanker against labeled queries with
    % one or more IRetrievalMetric instances; report per-query + mean scores.
    %
    %   SCOPING (Retrieval System Master Plan §3.5 — frozen decision, do not
    %   relitigate): this is an ORCHESTRATION/HARNESS class, the same
    %   category as OceanPipeline.Orchestration.DataPipeline and the
    %   *Factory classes (TokenizerFactory, RetrievalMetricFactory,
    %   DenseEncoderFactory) — it does NOT map one signal to one signal, it
    %   runs a many-query evaluation PROCEDURE and produces a results report.
    %   It therefore does NOT get the 3-pillar contract (no getMetadata(),
    %   no getPipelineSpec()), same reasoning as every factory/orchestrator
    %   in this build.
    %
    %   ANTI-LEAKAGE DISCIPLINE (matches DataPipeline's fit-before-run
    %   pattern, applied at the constructor boundary instead): the
    %   constructor REQUIRES an ALREADY-FITTED IRanker and throws
    %   immediately if it is not (checked via ranker.isFitted()). This
    %   evaluator's job is to SCORE an existing index against labeled
    %   queries — it never decides what corpus the ranker was built from,
    %   never calls ranker.fit() itself, and never re-indexes anything.
    %   evaluate() defensively re-checks obj.Ranker.isFitted() at call time
    %   too (belt-and-suspenders: obj.Ranker is a handle reference, not a
    %   copy, so this guards against a ranker whose fitted state could in
    %   principle change out from under this evaluator after construction).
    %
    %   MAP/MRR NOTE (frozen, §3.5 — the one nuance that is easy to get
    %   wrong): Mean Average Precision (MAP) and Mean Reciprocal Rank (MRR)
    %   are NOT separate metric classes. They ARE
    %   OceanStats.Metrics.Retrieval.AveragePrecision /
    %   OceanStats.Metrics.Retrieval.ReciprocalRank, run once per query and
    %   averaged by THIS class. Passing either into the `metrics` argument
    %   and reading the resulting "Mean" row for that column IS MAP / IS
    %   MRR — there is nothing else to compute.
    %
    %   NaN HANDLING (per-metric contract, see each metric's own EDGE CASE
    %   docs): RecallAtK, AveragePrecision, and ReciprocalRank return NaN
    %   for a query whose relevance vector has zero relevant documents
    %   defined at all (an undefined query, not a retrieval failure — see
    %   ReciprocalRank's docstring for why that is a genuinely different
    %   case from "relevant docs exist but none were retrieved", which is a
    %   well-defined 0). This class's mean-over-queries row uses
    %   mean(scores, 1, 'omitnan') per metric — NaN queries are EXCLUDED
    %   from that metric's mean, never allowed to poison it to NaN
    %   wholesale. Per this project's "no silent dropping" discipline, the
    %   excluded count per metric is NOT swallowed: it is (a) returned as
    %   the second output `excluded`, (b) written into
    %   report.Properties.Description, and (c) raised as a
    %   (non-fatal) warning when any metric actually excludes at least one
    %   query, so a caller who ignores both outputs still sees it.
    %
    %   USAGE:
    %     bm25 = OceanNLP.Retrieval.BM25Ranker();
    %     bm25.fit(corpus);
    %
    %     metrics = { ...
    %         OceanStats.Metrics.Retrieval.PrecisionAtK('K', 3), ...
    %         OceanStats.Metrics.Retrieval.RecallAtK('K', 3), ...
    %         OceanStats.Metrics.Retrieval.AveragePrecision(), ...   % -> MAP via the Mean row
    %         OceanStats.Metrics.Retrieval.ReciprocalRank(), ...     % -> MRR via the Mean row
    %         OceanStats.Metrics.Retrieval.NDCGAtK('K', 3)};
    %
    %     evaluator = OceanNLP.Retrieval.RetrievalEvaluator(bm25, metrics);
    %     [report, excluded] = evaluator.evaluate(queries, relevanceSets);
    %     disp(report)       % rows: Query1..QueryQ, Mean; columns: one per metric
    %     disp(excluded)     % 1 row, one column per metric: # queries excluded (NaN) from its mean
    %
    %   See also: OceanNLP.Retrieval.IRanker, OceanStats.Metrics.Retrieval.IRetrievalMetric,
    %             OceanStats.Metrics.Retrieval.AveragePrecision, OceanStats.Metrics.Retrieval.ReciprocalRank,
    %             Planning/System/Retrieval/Retrieval_System_Master_Plan.md §3.5

    properties (SetAccess = private)
        % See DenseRanker's PROPERTY-TYPING NOTE for the identical
        % rationale: IRanker is Abstract (no zero-arg constructor), so a
        % property typed to it fails at CLASS-LOAD time (MATLAB tries to
        % default-construct it to validate the declaration). Left untyped
        % here on purpose; the real type guarantee lives in the
        % constructor's `arguments` block below. Actual type: a scalar
        % OceanNLP.Retrieval.IRanker.
        Ranker

        % (1 x M) cell of OceanStats.Metrics.Retrieval.IRetrievalMetric
        % instances, in the order they were supplied — this order becomes
        % the column order of evaluate()'s report.
        Metrics cell = {}
    end

    methods
        function obj = RetrievalEvaluator(ranker, metrics)
            % RETRIEVALEVALUATOR  Construct with an ALREADY-FITTED IRanker
            % and a nonempty cell array of IRetrievalMetric instances.
            %   evaluator = RetrievalEvaluator(ranker, metrics)
            arguments
                ranker (1,1) OceanNLP.Retrieval.IRanker
                metrics (1,:) cell {mustBeNonempty}
            end
            for i = 1:numel(metrics)
                if ~isa(metrics{i}, 'OceanStats.Metrics.Retrieval.IRetrievalMetric')
                    error('OceanNLP:Retrieval:RetrievalEvaluator:BadMetric', ...
                        'metrics{%d} must be an OceanStats.Metrics.Retrieval.IRetrievalMetric, got a %s.', ...
                        i, class(metrics{i}));
                end
            end
            if ~ranker.isFitted()
                error('OceanNLP:Retrieval:RetrievalEvaluator:NotFitted', ...
                    ['ranker must already be fitted (call ranker.fit(corpus) first) before ' ...
                     'constructing a RetrievalEvaluator — evaluation SCORES an existing index ' ...
                     'against labeled queries, it does not decide what corpus that index was built from.']);
            end

            obj.Ranker  = ranker;
            obj.Metrics = metrics;
        end

        function [report, excluded] = evaluate(obj, queries, relevanceSets)
            % EVALUATE  Rank every query with obj.Ranker, score each ranking
            % with every metric in obj.Metrics, and report per-query +
            % mean-over-queries scores.
            %
            %   queries        — (1 x Q) string/cellstr of raw query texts.
            %   relevanceSets  — (1 x Q) cell array; relevanceSets{q} is a
            %                    (1 x Ntotal) relevance vector for query q,
            %                    in the EXACT shape
            %                    OceanStats.Metrics.Retrieval.IRetrievalMetric.forward()
            %                    expects: indexed by ORIGINAL document id,
            %                    0 = not relevant, SAME Ntotal for every
            %                    query (same corpus, same indexing).
            %
            %   report   — a Q+1 row x M column table. Rows "Query1".."QueryQ"
            %              are that query's raw per-metric score (as
            %              returned by IRetrievalMetric.forward(), including
            %              any NaN — see class NaN HANDLING note); the final
            %              "Mean" row is mean(scores, 'omitnan') per metric,
            %              i.e. this IS MAP when the AveragePrecision column
            %              is present, MRR when the ReciprocalRank column is
            %              present. Columns are named by each metric's
            %              name() (with "_K<value>" appended when the metric
            %              has a K property, e.g. "precision@k_K3", so two
            %              @K metrics at different cutoffs never collide).
            %   excluded — a 1-row table, same columns as report, giving the
            %              COUNT of queries excluded (NaN) from each
            %              metric's Mean row — the disclosure required by
            %              the class NaN HANDLING note, never silently
            %              dropped.
            arguments
                obj
                queries (1,:) {mustBeText}
                relevanceSets (1,:) cell
            end
            if ~obj.Ranker.isFitted()
                error('OceanNLP:Retrieval:RetrievalEvaluator:NotFitted', ...
                    'obj.Ranker is not fitted; evaluate() requires a fitted ranker throughout its lifetime.');
            end

            queries = string(queries);
            Q = numel(queries);
            if Q == 0
                error('OceanNLP:Retrieval:RetrievalEvaluator:EmptyQueries', ...
                    'queries must contain at least one query.');
            end
            if numel(relevanceSets) ~= Q
                error('OceanNLP:Retrieval:RetrievalEvaluator:SizeMismatch', ...
                    ['queries has %d entries but relevanceSets has %d; they must be the same length ' ...
                     '(exactly one relevance vector per query).'], Q, numel(relevanceSets));
            end

            M = numel(obj.Metrics);
            scores = NaN(Q, M);
            Ntotal = NaN;

            for q = 1:Q
                % No special-casing on obj.Ranker's concrete class anywhere
                % here — BM25Ranker/SemanticRanker/InvertedIndexRanker/
                % DenseRanker all satisfy the identical IRanker.rank()
                % contract, which is the entire point of the interface.
                [rankedIdx, ~] = obj.Ranker.rank(queries(q));

                rel = relevanceSets{q};
                if ~(isnumeric(rel) || islogical(rel)) || ~isvector(rel)
                    error('OceanNLP:Retrieval:RetrievalEvaluator:BadRelevance', ...
                        'relevanceSets{%d} must be a numeric/logical vector, got a %s.', q, class(rel));
                end
                rel = double(rel(:))';

                if numel(rel) ~= numel(rankedIdx)
                    error('OceanNLP:Retrieval:RetrievalEvaluator:BadRelevanceShape', ...
                        ['relevanceSets{%d} has %d entries but obj.Ranker.rank() returned a ranking over ' ...
                         '%d documents; relevance must be (1 x Ntotal), indexed by ORIGINAL document id, ' ...
                         'matching the corpus obj.Ranker was fitted on.'], q, numel(rel), numel(rankedIdx));
                end
                if q == 1
                    Ntotal = numel(rel);
                elseif numel(rel) ~= Ntotal
                    error('OceanNLP:Retrieval:RetrievalEvaluator:InconsistentNtotal', ...
                        ['relevanceSets{%d} has %d entries but relevanceSets{1} had %d; every query''s ' ...
                         'relevance vector must share the same Ntotal (same corpus, same document indexing).'], ...
                        q, numel(rel), Ntotal);
                end

                for m = 1:M
                    scores(q, m) = obj.Metrics{m}.forward(rankedIdx, rel);
                end
            end

            meanScores     = mean(scores, 1, 'omitnan');   % NaN queries excluded, never poison the mean
            excludedCounts = sum(isnan(scores), 1);

            metricNames = obj.columnNames_();

            data     = [scores; meanScores];
            rowNames = [compose("Query%d", (1:Q)'); "Mean"];
            report   = array2table(data, 'VariableNames', cellstr(metricNames), 'RowNames', cellstr(rowNames));

            excluded = array2table(excludedCounts, 'VariableNames', cellstr(metricNames), ...
                'RowNames', {'ExcludedQueries'});

            summaryParts = strings(1, M);
            for m = 1:M
                summaryParts(m) = sprintf('%s: %d/%d excluded (NaN — zero relevant docs defined)', ...
                    metricNames(m), excludedCounts(m), Q);
            end
            report.Properties.Description = "Mean row = mean(scores,'omitnan'); exclusions per metric " + ...
                "(also returned as the 2nd output, `excluded`): " + strjoin(summaryParts, "; ");

            if any(excludedCounts > 0)
                warning('OceanNLP:Retrieval:RetrievalEvaluator:ExcludedQueries', '%s', ...
                    report.Properties.Description);
            end
        end
    end

    methods (Access = private)
        function nm = columnNames_(obj)
            % COLUMNNAMES_  One column name per metric: metric.name(), with
            % "_K<value>" appended when the metric exposes a K property
            % (PrecisionAtK/RecallAtK/NDCGAtK) — disambiguates e.g.
            % PrecisionAtK(K=3) from PrecisionAtK(K=10) in the same report,
            % which would otherwise both be named "precision@k" and
            % collide as table variable names. Uses reflection (isprop),
            % not a class-name check, so it works uniformly for any current
            % or future IRetrievalMetric implementation.
            M  = numel(obj.Metrics);
            nm = strings(1, M);
            for m = 1:M
                base = string(obj.Metrics{m}.name());
                if isprop(obj.Metrics{m}, 'K')
                    base = base + "_K" + string(obj.Metrics{m}.K);
                end
                nm(m) = base;
            end
            nm = matlab.lang.makeUniqueStrings(nm);   % last-resort safety net for exact duplicates
        end
    end
end
