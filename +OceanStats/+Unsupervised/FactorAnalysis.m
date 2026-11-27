classdef FactorAnalysis < handle
    % FACTORANALYSIS  Classical Factor Analysis (ML estimation).
    %
    % ESL §14.7.1. Models observed data as:
    %   X = A * S + ε
    %
    % Where:
    %   X  (N x p) — observed data
    %   A  (p x q) — factor LOADING matrix
    %   S  (N x q) — latent FACTOR SCORES (q << p)
    %   ε  (N x p) — specific noise (uncorrelated, var = D_ε per variable)
    %
    % Covariance model (ESL Eq. 14.81):
    %   Σ = A * A^T + D_ε
    %
    % KEY CONTRAST WITH PCA:
    %   PCA  — finds directions of maximum VARIANCE
    %   FA   — finds latent factors explaining CORRELATION structure
    %   Consequence: PCA is dominated by high-variance uncorrelated variables,
    %   while FA recovers shared latent structure regardless of scale.
    %
    % ESTIMATION:
    %   Maximum likelihood via MATLAB's factoran (iterative EM).
    %   Rotation applied to loadings for interpretability (default: 'none').
    %
    % USAGE:
    %   fa = OceanStats.Unsupervised.FactorAnalysis();
    %   fa.fit(X, 1, 'Rotation', 'none');
    %   A = fa.getLoadings();        % (p x q) loading matrix
    %   S = fa.transform(X_new);     % (M x q) factor scores
    %   fa.printSummary();
    %
    % See Also: OceanStats.Unsupervised.NMF, OceanStats.Unsupervised.SparsePCA
    %
    % Author: Ocean Library | 2026-04-29

    properties (SetAccess = private)
        Loadings      % (p x q) factor loading matrix A
        SpecificVar   % (p x 1) specific (unique) variances D_ε diagonal
        Communality   % (p x 1) communality = 1 - SpecificVar/TotalVar per variable
        Rotation      % string — rotation applied ('none', 'varimax', etc.)
        ColumnMeans   % (1 x p) training column means (data is centered internally)
        NumFactors    % scalar q
        LogLik        % scalar — log-likelihood at convergence
        IterInfo      % struct — convergence info from factoran
    end

    methods
        function fit(obj, X, q, opts)
            % FIT  Estimate factor loadings via maximum likelihood.
            %
            %   X          (N x p)   — data matrix (any range, centered internally)
            %   q          scalar    — number of latent factors
            %   Rotation   string    — 'none' | 'varimax' | 'quartimax' (default 'none')
            %   Verbose    logical   — print convergence info (default false)

            arguments
                obj
                X               (:,:) double
                q               (1,1) double {mustBePositive, mustBeInteger}
                opts.Rotation   (1,:) char   = 'none'
                opts.Verbose    (1,1) logical = false
            end

            obj.ColumnMeans = mean(X, 1);   % (1 x p)
            obj.NumFactors  = q;
            obj.Rotation    = opts.Rotation;

            % ---------------------------------------------------------------
            % OBJECTIVE (Maximum Likelihood Factor Analysis):
            %
            %   max_{A, Psi}  log L(A, Psi | X)
            %
            %   where the model covariance is:
            %     Sigma = A * A^T + diag(Psi)        (ESL Eq. 14.81)
            %
            %   and the log-likelihood of the sample covariance S = X^T*X/(N-1) is:
            %     log L = -N/2 * [ log|Sigma| + Tr(Sigma^{-1} * S) ]
            %
            %   Constraints:
            %     Psi_j >= 0  for all j   (specific variances non-negative)
            %     A^T * Psi^{-1} * A  diagonal  (factor uniqueness via rotation fixing)
            %
            %   Solved by: iterative EM algorithm (Rubin & Thayer, 1982).
            %   Rotation T applied post-hoc to loadings for interpretability (default: none).
            %   Heywood cases (Psi_j -> 0) are handled internally by factoran.
            %
            %   USING BUILT-IN: factoran (MATLAB Statistics Toolbox)
            % ---------------------------------------------------------------
            warning('off', 'stats:factoran:ZeroDegreesOfFreedom');
            [Lambda, Psi, T, stats] = factoran(X, q, 'rotate', opts.Rotation);
            warning('on',  'stats:factoran:ZeroDegreesOfFreedom');

            obj.Loadings    = Lambda;         % (p x q)
            obj.SpecificVar = Psi;            % (p x 1) specific variances
            obj.LogLik      = stats.loglike;
            obj.IterInfo    = stats;

            % Communality: fraction of each variable's variance explained by factors
            % communality_j = sum_k A_jk^2  (for orthogonal factors)
            obj.Communality = sum(Lambda .^ 2, 2);   % (p x 1)

            if opts.Verbose
                obj.printSummary();
            end
        end

        function S = transform(obj, X_new)
            % TRANSFORM  Compute factor scores for new data (Thompson regression scores).
            %   X_new  (M x p) → S  (M x q)
            % Loading now fixed. We need to found factor score for this X_new from pattern learn in A
            % Thompson scores: S = Xc * A * (A^T*A + D_ε)^{-1}
            % These are approximate (FA scores are not unique) but standard.
            if isempty(obj.Loadings)
                error('OceanStats:FA:NotFitted', 'Call fit() first.');
            end
            Xc  = X_new - obj.ColumnMeans;           % (M x p) center
            A   = obj.Loadings;                       % (p x q)
            Psi = obj.SpecificVar;                    % (p x 1)
            % Score regression: S = Xc * (Σ^{-1} * A) where Σ = A*A^T + diag(Psi)
            % Using the Woodbury identity: Σ^{-1} = D^{-1} - D^{-1}*A*(I + A^T*D^{-1}*A)^{-1}*A^T*D^{-1}
            D_inv = 1 ./ Psi;                         % (p x 1)
            M_mat = eye(obj.NumFactors) + (A' .* D_inv') * A;  % (q x q)
            % Σ^{-1} * A = D^{-1}*A - D^{-1}*A*(M^{-1})*(A^T*D^{-1}*A)... simplified:
            % Thompson: S = Xc * D^{-1} * A * (I + A^T*D^{-1}*A)^{-1}
            S = (Xc .* D_inv') * A / M_mat;           % (M x q)
        end

        function A = getLoadings(obj)
            % GETLOADINGS  Return A (p x q) factor loading matrix.
            if isempty(obj.Loadings)
                error('OceanStats:FA:NotFitted', 'Call fit() first.');
            end
            A = obj.Loadings;
        end

        function Sigma_model = getCovModel(obj)
            % GETCOVMODEL  Return model covariance Σ = A*A^T + diag(SpecificVar).
            if isempty(obj.Loadings)
                error('OceanStats:FA:NotFitted', 'Call fit() first.');
            end
            A = obj.Loadings;
            Sigma_model = A * A' + diag(obj.SpecificVar);
        end

        function printSummary(obj)
            if isempty(obj.Loadings)
                fprintf('[FA] Not fitted.\n'); return;
            end
            [p, q] = size(obj.Loadings);
            fprintf('\n=== Factor Analysis Summary ===\n');
            fprintf('  Variables (p)       : %d\n', p);
            fprintf('  Factors   (q)       : %d\n', q);
            fprintf('  Rotation            : %s\n', obj.Rotation);
            fprintf('  Log-likelihood      : %.4f\n', obj.LogLik);
            fprintf('\n  Loading Matrix A (%d x %d):\n', p, q);
            fprintf('    Var');
            for k = 1:q
                fprintf('    Factor%d', k);
            end
            fprintf('   Communality  SpecificVar\n');
            for j = 1:p
                fprintf('    X%-2d', j);
                for k = 1:q
                    fprintf('   %+8.4f', obj.Loadings(j, k));
                end
                fprintf('   %10.4f  %10.4f\n', obj.Communality(j), obj.SpecificVar(j));
            end
            fprintf('===============================\n');
        end
    end
end
