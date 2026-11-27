classdef IICASolver < handle
    % IICASOLVER  Abstract interface for ICA optimization strategies.
    %
    % All ICA strategies receive PCA-whitened data X_w (N x q) and must
    % return an orthogonal unmixing matrix W (q x q) such that
    %   S = X_w * W^T  are maximally independent components.
    %
    % IMPLEMENTING A NEW STRATEGY:
    %   1. Subclass this interface in +OceanStats/+Unsupervised/+ICASolvers/
    %   2. Set the 'Name' property
    %   3. Implement solve(X_w, q, verbose) → [W, converged, iters]
    %   4. Use in: ica.fit(X, q, 'Strategy', MyStrategy())
    %
    % Author: Ocean Library | 2026-04-30

    properties (Abstract, SetAccess = protected)
        Name  % (1 x :) char — display name of this strategy
    end

    methods (Abstract)
        [W, converged, iters] = solve(obj, X_w, q, verbose)
        % SOLVE  Find unmixing matrix W from whitened data X_w.
        %
        %   X_w      (N x q) double  — PCA-whitened data (unit covariance)
        %   q        scalar          — number of ICs to extract
        %   verbose  logical         — print per-component progress
        %
        %   W         (q x q) double  — orthogonal unmixing matrix
        %   converged (q x 1) logical — convergence flag per component
        %   iters     (q x 1) double  — iterations used per component
    end

    methods (Access = protected)
        function w = deflate(~, w, W, k)
            % DEFLATE  Gram-Schmidt deflation of w against rows 1..k-1 of W.
            % Ensures newly found component is orthogonal to previous ones.
            if k > 1
                w = w - W(1:k-1,:)' * (W(1:k-1,:) * w);
            end
        end
    end
end
