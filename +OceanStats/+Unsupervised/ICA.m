classdef ICA < handle
    % ICA  Independent Component Analysis with pluggable solver strategies.
    %
    % ESL §14.7. Generic orchestrator — handles PCA whitening then delegates
    % the unmixing optimization to an IICASolver strategy.
    %
    % AVAILABLE STRATEGIES (in +OceanStats/+Unsupervised/+ICASolvers/):
    %   FastICAStrategy     — fixed contrast function (tanh/cubic), ESL §14.7.3
    %   ProDenICAStrategy   — adaptive KDE contrast,                 ESL §14.7.4
    %   KernelICAStrategy   — HSIC dependence minimization (stub),   Bach & Jordan 2002
    %
    % WHAT THIS CLASS DOES (common to all strategies):
    %   1. Center X: Xc = X - mean(X)
    %   2. PCA whiten: X_w = Xc * V_q * diag(1/d_q)  (unit covariance, q dims)
    %   3. Call strategy.solve(X_w, q) → W (q x q) orthogonal unmixing matrix
    %   4. Compute S = X_w * W^T  (N x q) IC scores
    %              A = Dewhitening * W^T  (p x q) IC images in pixel space
    %
    % USAGE:
    %   % Default: FastICA with tanh
    %   ica = OceanStats.Unsupervised.ICA();
    %   ica.fit(X, 5);
    %
    %   % ProDenICA (adaptive density)
    %   s = OceanStats.Unsupervised.ICASolvers.ProDenICAStrategy();
    %   ica.fit(X, 5, 'Strategy', s);
    %
    %   % FastICA with cubic contrast
    %   s = OceanStats.Unsupervised.ICASolvers.FastICAStrategy('ContrastFcn','cubic');
    %   ica.fit(X, 5, 'Strategy', s);
    %
    %   S = ica.getComponents();     % (N x q) IC scores
    %   A = ica.getMixingMatrix();   % (p x q) IC images
    %   Xw = ica.getWhitenedData();  % (N x q) PCA-standardized scores
    %
    % Author: Ocean Library | 2026-04-30

    properties (SetAccess = private)
        W              % (q x q) unmixing matrix in whitened space
        A              % (p x q) mixing matrix — each column = IC image
        S              % (N x q) IC scores (training data)
        X_w            % (N x q) PCA-whitened data (standardized PCA scores)
        WhiteningMat   % (p x q): X_w = Xc * WhiteningMat
        DewhiteningMat % (p x q): Xc ≈ X_w * DewhiteningMat'
        ColumnMeans    % (1 x p) training column means
        NumComponents  % scalar q
        StrategyName   % string — name of the strategy used
        ConvergedFlags % (q x 1) logical
        Iterations     % (q x 1) iterations per component
    end

    methods
        function fit(obj, X, q, opts)
            % FIT  Whiten data and delegate ICA solve to the chosen strategy.
            %
            %   X        (N x p)  — data matrix (any range)
            %   q        scalar   — number of ICs to extract
            %   Strategy IICASolver instance (default: FastICAStrategy)
            %   Verbose  logical  — print progress (default false)

            arguments
                obj
                X              (:,:) double
                q              (1,1) double {mustBePositive, mustBeInteger}
                opts.Strategy  = OceanStats.Unsupervised.ICASolvers.FastICAStrategy()
                opts.Verbose   (1,1) logical = false
            end

            [N, ~] = size(X);
            obj.ColumnMeans   = mean(X, 1);
            obj.NumComponents = q;
            obj.StrategyName  = opts.Strategy.Name;
            Xc = X - obj.ColumnMeans;

            % -------------------------------------------------------
            % PCA WHITENING (common to all strategies)
            % Projects to q-dim subspace and scales to unit variance.
            % After: Cov(X_w) ≈ I_q, which allows W to be orthogonal.
            % -------------------------------------------------------
            [~, D_svd, V_svd] = svd(Xc, 'econ');
            d_q = diag(D_svd(1:q, 1:q)) / sqrt(N - 1);   % (q x 1) std devs
            V_q = V_svd(:, 1:q);                           % (p x q) loadings

            obj.WhiteningMat   = V_q ./ d_q';    % (p x q) Xw = Xc * WhiteningMat
            obj.DewhiteningMat = V_q .* d_q';    % (p x q) Xc ≈ Xw * DewhiteningMat'
            obj.X_w = Xc * obj.WhiteningMat;      % (N x q) whitened

            % -------------------------------------------------------
            % STRATEGY SOLVE
            % Delegate the entire unmixing optimization to the strategy.
            % Strategy receives whitened data and returns orthogonal W.
            % -------------------------------------------------------
            if opts.Verbose
                fprintf('  [ICA] Strategy: %s | q=%d\n', opts.Strategy.Name, q);
            end

            [W_ica, conv, iters] = opts.Strategy.solve(obj.X_w, q, opts.Verbose);

            obj.W              = W_ica;
            obj.ConvergedFlags = conv;
            obj.Iterations     = iters;
            obj.S              = obj.X_w * W_ica';        % (N x q) IC scores
            obj.A              = obj.DewhiteningMat * W_ica';  % (p x q) IC images
        end

        function S_new = transform(obj, X_new)
            % TRANSFORM  Project new data onto learned IC directions.
            if isempty(obj.W)
                error('OceanStats:ICA:NotFitted', 'Call fit() first.');
            end
            Xc_new  = X_new - obj.ColumnMeans;
            X_w_new = Xc_new * obj.WhiteningMat;
            S_new   = X_w_new * obj.W';
        end

        function S = getComponents(obj)
            % GETCOMPONENTS  Return S (N x q) IC scores.
            if isempty(obj.S), error('OceanStats:ICA:NotFitted', 'Call fit() first.'); end
            S = obj.S;
        end

        function A = getMixingMatrix(obj)
            % GETMIXINGMATRIX  Return A (p x q) — each column is one IC image.
            if isempty(obj.A), error('OceanStats:ICA:NotFitted', 'Call fit() first.'); end
            A = obj.A;
        end

        function Xw = getWhitenedData(obj)
            % GETWHITENEDDATA  Return X_w (N x q) — standardized PCA scores.
            if isempty(obj.X_w), error('OceanStats:ICA:NotFitted', 'Call fit() first.'); end
            Xw = obj.X_w;
        end

        function printSummary(obj)
            if isempty(obj.W)
                fprintf('[ICA] Not fitted.\n'); return;
            end
            fprintf('\n=== ICA Summary ===\n');
            fprintf('  Strategy       : %s\n', obj.StrategyName);
            fprintf('  Components (q) : %d\n', obj.NumComponents);
            for k = 1:obj.NumComponents
                fprintf('  IC%-2d: iter=%3d  converged=%s\n', ...
                    k, obj.Iterations(k), mat2str(obj.ConvergedFlags(k)));
            end
            fprintf('===================\n');
        end
    end
end
