classdef CosineSimilarity < OceanStats.Similarity.ISimilarity
    % COSINESIMILARITY  Cosine similarity between vectors / batches.
    %
    % THEORY:
    %   cos(A,B) = (A . B) / (||A|| * ||B||). Bounded in [-1, 1] for real
    %   inputs (assuming non-negative-only inputs like TF-IDF rows, bounded
    %   in [0, 1]). A denominator eps guard avoids division-by-zero for
    %   all-zero rows (matches torch.nn.functional.cosine_similarity's own
    %   eps default of 1e-8).
    %
    %   Stateless strategy backed by a pure functional core (COMPUTE),
    %   mirroring torch.nn.CosineSimilarity wrapping
    %   torch.nn.functional.cosine_similarity.
    %
    % USAGE (elementwise, like F.cosine_similarity(x1, x2, dim=1)):
    %   sim = OceanStats.Similarity.CosineSimilarity();
    %   s   = sim.forward(A, B);        % (N x 1), A(i,:) vs B(i,:)
    %
    % USAGE (pairwise, retrieval / all-vs-all):
    %   P = sim.pairwise(Query, Corpus); % (Nq x Nc)
    %
    % USAGE (functional, no object):
    %   s = OceanStats.Similarity.CosineSimilarity.compute(A, B);
    %
    % CONVENTION:
    %   A, B may be sparse or dense (:,:) double. forward() requires
    %   identical size (row-vs-corresponding-row); pairwise() allows
    %   different row counts but requires the same column count (feature
    %   dimension D). pairwise() row-normalizes both inputs, then computes
    %   a single dense (Nq x Nc) score matrix via matrix multiply — the
    %   output is intentionally dense (it is the similarity result itself,
    %   not an intermediate), but the row-normalization step must remain
    %   sparse-safe when A/B are sparse (verified empirically, see
    %   OceanStatsTest.Similarity.CosineSimilarityTest).
    %
    % See also: OceanStats.Similarity.ISimilarity,
    %           OceanStats.Similarity.SimilarityFactory
    %
    % Implements the 3-pillar contract (.agents/pipeline-architecture-spec.md
    % §1.1) via getMetadata() and getPipelineSpec(), in addition to forward/pairwise/compute.

    properties
        Eps (1,1) double {mustBePositive} = 1e-8   % matches torch's default
    end

    methods
        function obj = CosineSimilarity(options)
            % COSINESIMILARITY  Construct with an optional zero-norm guard.
            arguments
                options.Eps (1,1) double {mustBePositive} = 1e-8
            end
            obj.Eps = options.Eps;
        end

        function s = forward(obj, A, B)
            % FORWARD  Row-vs-corresponding-row similarity, (N x 1).
            arguments
                obj
                A (:,:) double
                B (:,:) double
            end
            if ~isequal(size(A), size(B))
                error('OceanStats:Similarity:CosineSimilarity:ShapeMismatch', ...
                    ['forward() compares corresponding rows: A and B must be ' ...
                     'the same size. Use pairwise() for all-vs-all.']);
            end
            s = OceanStats.Similarity.CosineSimilarity.compute(A, B, obj.Eps);
        end

        function P = pairwise(obj, A, B)
            % PAIRWISE  All-vs-all similarity, (Nrows(A) x Nrows(B)).
            arguments
                obj
                A (:,:) double
                B (:,:) double
            end
            if size(A, 2) ~= size(B, 2)
                error('OceanStats:Similarity:CosineSimilarity:DimMismatch', ...
                    ['pairwise() requires A and B to share the same number ' ...
                     'of columns (feature dimension). Got %d and %d.'], ...
                    size(A, 2), size(B, 2));
            end
            An = obj.normalizeRows_(A);
            Bn = obj.normalizeRows_(B);
            P  = An * Bn';
        end

        function s = name(~)
            % NAME  Registry key used by SimilarityFactory.
            s = "cosine";
        end

        function meta = getMetadata(~)
            % GETMETADATA  Pillar 1 (symbolic) — closed-form LaTeX, symbol
            % dictionary, assumptions, and reference backing forward/pairwise/compute.
            meta = struct();
            meta.SymbolicLatex = "s_i = \dfrac{\mathbf{a}_i \cdot \mathbf{b}_i}{\max\!\left(\lVert \mathbf{a}_i \rVert \, \lVert \mathbf{b}_i \rVert,\ \epsilon\right)}";
            meta.Symbols = containers.Map();
            meta.Symbols('a_i') = "row i of the first batch";
            meta.Symbols('b_i') = "row i of the second batch";
            meta.Symbols('s_i') = "cosine similarity of the pair";
            meta.Symbols('epsilon') = "stability constant preventing division by zero";
            meta.Assumptions = {'forward() requires A and B to share shape (row-corresponding comparison)', ...
                                 'pairwise() requires A and B to share column count (feature dimension)'};
            meta.IsSymbolic = true;
            meta.Reference = "standard vector-space-model cosine similarity";
        end

        function spec = getPipelineSpec(~)
            % GETPIPELINESPEC  Pillar 3 (diagram) — port signature and canvas
            % metadata for this NONVIRTUAL leaf operator.
            spec = struct();
            spec.Inputs = [ ...
                struct('Name', "A", 'Domain', "dense-vector-batch", 'Dtype', "double", 'Shape', "N x D"), ...
                struct('Name', "B", 'Domain', "dense-vector-batch", 'Dtype', "double", 'Shape', "M x D")];
            spec.Outputs = struct('Name', "s", 'Domain', "similarity-matrix", 'Dtype', "double", 'Shape', "N x 1");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "cosine");
            spec.ComposesFrom = {};
        end
    end

    methods (Access = private)
        function N = normalizeRows_(obj, X)
            % NORMALIZEROWS_  L2-normalize each row; sparse in stays sparse.
            nrm = sqrt(sum(X.^2, 2)); % Zero-stay zero
            nrm = max(nrm, obj.Eps); % All denom positive
            N   = X ./ nrm; % 0 / positive -> 0
        end
    end

    methods (Static)
        function s = compute(A, B, eps)
            % COMPUTE  Functional form — torch.nn.functional.cosine_similarity equivalent.
            %   s(i) = dot(A(i,:), B(i,:)) / (||A(i,:)|| * ||B(i,:)|| + eps)
            arguments
                A   (:,:) double
                B   (:,:) double
                eps (1,1) double {mustBePositive} = 1e-8
            end
            num  = sum(A .* B, 2);
            denA = sqrt(sum(A.^2, 2));
            denB = sqrt(sum(B.^2, 2));
            s    = num ./ max(denA .* denB, eps);
        end
    end
end
