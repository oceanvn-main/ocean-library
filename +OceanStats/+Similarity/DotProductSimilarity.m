classdef DotProductSimilarity < OceanStats.Similarity.ISimilarity
    % DOTPRODUCTSIMILARITY  Inner-product similarity, s_i = a_i . b_i.
    %
    % Second ISimilarity implementation (after CosineSimilarity). Bi-encoders
    % that L2-normalize their outputs (all-MiniLM-L6-v2, DPR-style models)
    % are scored with a plain dot product; on unit vectors it equals cosine.
    % No epsilon and no normalization: what you pass is what is multiplied.
    %
    %   forward(A, B)  -> (N x 1)  A(i,:) . B(i,:), same-size inputs
    %   pairwise(A, B) -> (N x M)  A * B'
    %
    % See also: OceanStats.Similarity.ISimilarity, OceanStats.Similarity.CosineSimilarity

    methods
        function s = forward(~, A, B)
            arguments
                ~
                A (:,:) double
                B (:,:) double
            end
            if ~isequal(size(A), size(B))
                error('OceanStats:Similarity:DotProductSimilarity:ShapeMismatch', ...
                    ['forward() compares corresponding rows: A and B must be ' ...
                     'the same size. Use pairwise() for all-vs-all.']);
            end
            s = sum(A .* B, 2);
        end

        function P = pairwise(~, A, B)
            arguments
                ~
                A (:,:) double
                B (:,:) double
            end
            if size(A, 2) ~= size(B, 2)
                error('OceanStats:Similarity:DotProductSimilarity:DimMismatch', ...
                    ['pairwise() requires A and B to share the same number ' ...
                     'of columns (feature dimension). Got %d and %d.'], ...
                    size(A, 2), size(B, 2));
            end
            P = A * B';
        end

        function s = name(~)
            s = "dot";
        end

        function meta = getMetadata(~)
            meta = struct();
            meta.SymbolicLatex = "s_i = \mathbf{a}_i \cdot \mathbf{b}_i";
            meta.Symbols = containers.Map();
            meta.Symbols('a_i') = "row i of the first batch";
            meta.Symbols('b_i') = "row i of the second batch";
            meta.Symbols('s_i') = "inner product of the pair";
            meta.Assumptions = {'forward() requires A and B to share shape (row-corresponding comparison)', ...
                                 'pairwise() requires A and B to share column count (feature dimension)', ...
                                 'no normalization: equals cosine similarity only when rows are unit-norm'};
            meta.IsSymbolic = true;
            meta.Reference = "inner-product (maximum inner product search) scoring";
        end

        function spec = getPipelineSpec(~)
            spec = struct();
            spec.Inputs = [ ...
                struct('Name', "A", 'Domain', "dense-vector-batch", 'Dtype', "double", 'Shape', "N x D"), ...
                struct('Name', "B", 'Domain', "dense-vector-batch", 'Dtype', "double", 'Shape', "M x D")];
            spec.Outputs = struct('Name', "s", 'Domain', "similarity-matrix", 'Dtype', "double", 'Shape', "N x 1");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "dot");
            spec.ComposesFrom = {};
        end
    end
end
