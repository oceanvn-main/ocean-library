classdef LateInteraction
    % LATEINTERACTION  ColBERT MaxSim over SETS of token embeddings.
    %
    %   s(q,d) = sum_i max_j  q_i . d_j
    %
    % Q is (nq x dim), D is (nd x dim), one token embedding per row (rows
    % are expected unit-norm, as ColBERT produces). Each query token takes
    % its best-matching document token; those maxima are summed.
    %
    % Deliberately NOT an ISimilarity: that contract compares (N x D) row
    % batches, while MaxSim compares two sets of different sizes.
    %
    % Functional core / wrapper split (PyTorch style):
    %   LateInteraction.maxSim(Q, D)          pure static function
    %   obj.score(Q, docs)                    wrapper over a cell of doc matrices
    %
    % Doc matrices may be single; Q is cast to D's class so the multiply
    % stays in single precision (halves memory for the token store).
    %
    % See also: OceanNLP.Retrieval.ColBERTRanker

    methods (Static)
        function s = maxSim(Q, D)
            arguments
                Q (:,:) {mustBeNumeric}
                D (:,:) {mustBeNumeric}
            end
            if size(Q, 2) ~= size(D, 2)
                error('OceanStats:Similarity:LateInteraction:DimMismatch', ...
                    'Q and D must share the embedding dimension. Got %d and %d.', size(Q, 2), size(D, 2));
            end
            if isempty(Q) || isempty(D)
                s = 0;
                return
            end
            S = cast(Q, 'like', D) * D';        % (nq x nd) token-token similarities
            s = double(sum(max(S, [], 2)));
        end
    end

    methods
        function s = score(~, Q, docs)
            % SCORE  MaxSim of one query matrix against a cell of doc matrices -> (1 x M).
            arguments
                ~
                Q (:,:) {mustBeNumeric}
                docs (1,:) cell
            end
            s = zeros(1, numel(docs));
            for j = 1:numel(docs)
                s(j) = OceanStats.Similarity.LateInteraction.maxSim(Q, docs{j});
            end
        end

        function s = name(~)
            s = "late-interaction";
        end

        function meta = getMetadata(~)
            meta = struct();
            meta.SymbolicLatex = "s(q,d) = \sum_{i=1}^{\lvert q\rvert} \max_{1\le j\le\lvert d\rvert} \mathbf{q}_i^{\top}\mathbf{d}_j";
            meta.Symbols = containers.Map();
            meta.Symbols('q_i') = "embedding of query token i (unit norm)";
            meta.Symbols('d_j') = "embedding of document token j (unit norm)";
            meta.Symbols('s(q,d)') = "late-interaction (MaxSim) relevance score";
            meta.Assumptions = {'token embeddings are unit-norm, so q_i . d_j is a cosine similarity', ...
                                 'empty query or document scores 0', ...
                                 'exact full computation: no pruning or approximation'};
            meta.IsSymbolic = true;
            meta.Reference = "Khattab & Zaharia 2020 (ColBERT), eq. 3";
        end

        function spec = getPipelineSpec(~)
            spec = struct();
            spec.Inputs = [ ...
                struct('Name', "Q", 'Domain', "token-state-matrix", 'Dtype', "double", 'Shape', "nq x dim"), ...
                struct('Name', "D", 'Domain', "token-state-matrix", 'Dtype', "single", 'Shape', "nd x dim")];
            spec.Outputs = struct('Name', "s", 'Domain', "similarity-matrix", 'Dtype', "double", 'Shape', "1 x 1");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "maxsim");
            spec.ComposesFrom = {};
        end
    end
end
