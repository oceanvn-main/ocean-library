classdef CholeskyDecomposition
    % CHOLESKYDECOMPOSITION Custom Cholesky Factorization L*L' = A for Symmetric Positive Definite Matrices
    %
    %   TEXTBOOK REFERENCE:
    %       Golub, Gene H., and Charles F. Van Loan. "Matrix Computations" (4th ed.), Section 4.2.
    %       Hamilton, James D. "Time Series Analysis" (1994), Section 5.9, Eq. (5.9.2):
    %           Omega = P * P'
    %
    %   DUAL REPRESENTATION CONTRACT:
    %       1. Symbolics  : LaTeX formulas for lower-triangular factorization A = L * L' and vector packing.
    %       2. Computation: Custom Banachiewicz factorization loop without calling built-in MATLAB chol().
    %       3. Diagram    : Port signatures & pipeline block specifications (getPipelineSpec).
    %
    %   Example:
    %       A = [4, 12, -16; 12, 37, -43; -16, -43, 98];
    %       L = OceanMath.LinearAlgebra.Decomposition.CholeskyDecomposition.factorize(A);
    %       A_rec = OceanMath.LinearAlgebra.Decomposition.CholeskyDecomposition.reconstruct(L);
    %
    %   See also OceanMath.TimeSeries.ReparameterizedLikelihood, OceanMath.LinearAlgebra.Orthogonalization.HouseholderReflection

    methods (Static)
        function L = factorize(A)
            % FACTORIZE Computes lower-triangular matrix L such that A = L * L'
            arguments
                A (:, :) double
            end

            [n, m] = size(A);
            if n ~= m
                error('CholeskyDecomposition:MatrixNotSquare', 'Input matrix A must be square.');
            end

            % Enforce symmetry for numerical safety
            A_sym = (A + A') / 2;

            L = zeros(n, n);
            for j = 1:n
                sumDiag = 0;
                for k = 1:j-1
                    sumDiag = sumDiag + L(j, k)^2;
                end
                valDiag = A_sym(j, j) - sumDiag;
                if valDiag <= 0
                    valDiag = 1e-12; % Regularize for positive semi-definiteness
                end
                L(j, j) = sqrt(valDiag);

                for i = j+1:n
                    sumOff = 0;
                    for k = 1:j-1
                        sumOff = sumOff + L(i, k) * L(j, k);
                    end
                    L(i, j) = (A_sym(i, j) - sumOff) / L(j, j);
                end
            end
        end

        function A = reconstruct(L)
            % RECONSTRUCT Computes symmetric positive-definite matrix A = L * L'
            arguments
                L (:, :) double
            end
            A = L * L';
        end

        function lambdaVector = packVector(L)
            % PACKVECTOR Extracts lower-triangular elements into an unconstrained 1D vector of length n*(n+1)/2
            arguments
                L (:, :) double
            end

            n = size(L, 1);
            m = n * (n + 1) / 2;
            lambdaVector = zeros(1, m);

            idx = 1;
            for j = 1:n
                for i = j:n
                    lambdaVector(idx) = L(i, j);
                    idx = idx + 1;
                end
            end
        end

        function L = unpackVector(lambdaVector, n)
            % UNPACKVECTOR Reconstructs n x n lower-triangular matrix L from a 1D vector of length n*(n+1)/2
            arguments
                lambdaVector (1, :) double
                n (1, 1) double {mustBePositive, mustBeInteger}
            end

            m = n * (n + 1) / 2;
            if numel(lambdaVector) ~= m
                error('CholeskyDecomposition:VectorSizeMismatch', 'Expected vector of length %d for n=%d matrix.', m, n);
            end

            L = zeros(n, n);
            idx = 1;
            for j = 1:n
                for i = j:n
                    L(i, j) = lambdaVector(idx);
                    idx = idx + 1;
                end
            end
        end

        function meta = getMetadata()
            % GETMETADATA Self-describing component metadata for system registration
            meta = struct(...
                'ClassName', 'OceanMath.LinearAlgebra.Decomposition.CholeskyDecomposition', ...
                'TextbookRef', 'Golub & Van Loan (2013) §4.2 / Hamilton (1994) §5.9.2', ...
                'SymbolicLatex', 'A = L L'', \quad \Omega = P P''', ...
                'Capabilities', {{'Cholesky Factorization', 'Matrix Reconstruction', 'Lower-Triangular Vector Packing', 'Unconstrained Covariance Parameterization'}} ...
            );
        end

        function spec = getPipelineSpec()
            % GETPIPELINESPEC Exposes block diagram input/output port specifications for studio visualization
            spec = struct(...
                'ComponentName', 'CholeskyDecomposition', ...
                'Category', 'Linear Algebra & Decomposition', ...
                'Description', 'Computes custom lower-triangular Cholesky factor L such that A = L*L'' and packs/unpacks covariance parameters', ...
                'Inputs', struct('name', {'A', 'L', 'lambdaVector'}, 'type', {'double', 'double', 'double'}, 'shape', {'n x n', 'n x n', '1 x m'}), ...
                'Outputs', struct('name', {'L', 'A', 'lambdaVector'}, 'type', {'double', 'double', 'double'}, 'shape', {'n x n', 'n x n', '1 x m'}) ...
            );
        end

        function str = getLaTeXRepresentation()
            % GETLATEXREPRESENTATION Returns LaTeX mathematical representation of the factorization
            str = sprintf(['\\begin{aligned}\n' ...
                '\\text{Factorization: } & A = L L'', \\quad L_{j,j} = \\sqrt{A_{j,j} - \\sum_{k=1}^{j-1} L_{j,k}^2} \\\\\n' ...
                '\\text{Off-Diagonal: } & L_{i,j} = \\frac{1}{L_{j,j}} \\left( A_{i,j} - \\sum_{k=1}^{j-1} L_{i,k} L_{j,k} \\right) \\\\\n' ...
                '\\text{Covariance Parameterization: } & \\Omega = P P'', \\quad P \\text{ lower-triangular}\n' ...
                '\\end{aligned}']);
        end
    end
end
