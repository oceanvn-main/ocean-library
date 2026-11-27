classdef ProDenICAStrategy < OceanStats.Unsupervised.ICASolvers.IICASolver
    % PRODENICA  ProDenICA (Hastie & Tibshirani, 2002).
    %
    % ESL §14.7.4. Same deflation structure as FastICA but uses a DATA-ADAPTIVE
    % contrast function g estimated from the current projected distribution.
    % More robust than FastICA when source distributions are unknown or mixed.
    %
    % OBJECTIVE (per component):
    %   max_w  J(w) = [E{G(w^T X_w)} - E{G(v)}]^2  where G = log p_hat
    %   (same as FastICA, but G is estimated from data, not fixed)
    %
    % ADAPTIVE CONTRAST FUNCTION (score function of KDE):
    %   p_hat(u) = (1/Nh) * sum_i K((u - u_i)/h)     [Gaussian KDE]
    %   g(u)     = -d log p_hat(u) / du = -p_hat'(u) / p_hat(u)
    %   g'(u)    = d g(u) / du = [-p_hat''*p_hat + (p_hat')^2] / p_hat^2
    %
    %   where K(z) = (1/sqrt(2*pi)) * exp(-z^2/2)  [Gaussian kernel]
    %
    % BANDWIDTH SELECTION:
    %   'silverman' — h = 0.9 * min(std, IQR/1.34) * N^{-1/5}  (default)
    %   scalar      — manual fixed bandwidth h
    %
    % NOTE:
    %   KDE evaluation is O(N^2) per step — slower than FastICA but more
    %   adaptive. For N<1000 this is acceptable.
    %
    % USAGE:
    %   s = OceanStats.Unsupervised.ICASolvers.ProDenICAStrategy();
    %   s = OceanStats.Unsupervised.ICASolvers.ProDenICAStrategy('BandwidthMethod', 0.3);
    %   ica.fit(X, q, 'Strategy', s);
    %
    % Author: Ocean Library | 2026-04-30

    properties (SetAccess = private)
        BandwidthMethod  % 'silverman' | scalar
        MaxIter          % int
        Tol              % scalar
    end

    properties (SetAccess = protected)
        Name = 'ProDenICA'
    end

    methods
        function obj = ProDenICAStrategy(opts)
            arguments
                opts.BandwidthMethod = 'silverman'   % char or scalar
                opts.MaxIter (1,1) double {mustBePositive, mustBeInteger} = 200
                opts.Tol     (1,1) double {mustBePositive} = 1e-6
            end
            obj.BandwidthMethod = opts.BandwidthMethod;
            obj.MaxIter = opts.MaxIter;
            obj.Tol     = opts.Tol;
        end

        function [W, converged, iters] = solve(obj, X_w, q, verbose)
            [N, ~] = size(X_w);
            W         = zeros(q, q);
            converged = false(q, 1);
            iters     = zeros(q, 1);

            for k = 1:q
                w = randn(q, 1);
                w = obj.deflate(w, W, k);
                w = w / norm(w);

                for iter = 1:obj.MaxIter
                    w_old = w;
                    u = X_w * w;   % (N x 1) projected scores

                    % Adaptive contrast: estimate g and g' from current distribution
                    h = obj.silverman(u, N);
                    [gu, gdu] = obj.kdeScore(u, h, N);

                    % Fixed-point update (same as FastICA, different g)
                    w_new = (X_w' * gu) / N  -  mean(gdu) * w;
                    w_new = obj.deflate(w_new, W, k);

                    nw = norm(w_new);
                    if nw < 1e-12
                        w_new = obj.deflate(randn(q,1), W, k);
                        nw = norm(w_new);
                    end
                    w = w_new / nw;

                    if abs(abs(w' * w_old) - 1) < obj.Tol
                        converged(k) = true;
                        iters(k)     = iter;
                        break;
                    end
                end

                if ~converged(k), iters(k) = obj.MaxIter; end
                W(k, :) = w';

                if verbose
                    fprintf('  [ProDenICA] IC%d: iter=%d  converged=%s  h=%.4f\n', ...
                        k, iters(k), mat2str(converged(k)), obj.silverman(X_w*w', N));
                end
            end
        end
    end

    methods (Access = private)
        function h = silverman(obj, u, N)
            % SILVERMAN  Compute KDE bandwidth using Silverman's rule of thumb.
            if isnumeric(obj.BandwidthMethod)
                h = obj.BandwidthMethod;
            else
                sigma = std(u);
                iq    = iqr(u);
                h = 0.9 * min(sigma, iq / 1.34) * N^(-0.2);
                h = max(h, 1e-6);
            end
        end

        function [gu, gdu] = kdeScore(~, u, h, N)
            % KDESCORE  Compute score function g(u) = -d log p_hat(u) / du
            %           and its derivative g'(u) via Gaussian KDE.
            %
            % D(j,i) = u_j - u_i  →  pairwise difference matrix (N x N)
            % K(j,i) = Gaussian kernel evaluated at (u_j - u_i)/h
            %
            % p_hat(u_j)   = (1/Nh)   * sum_i K(j,i)
            % p_hat'(u_j)  = (1/Nh^2) * sum_i (-D/h) * K(j,i)   [by chain rule]
            % p_hat''(u_j) = (1/Nh^3) * sum_i (D^2/h^2 - 1) * K(j,i)

            D = u - u';                          % (N x N) pairwise differences
            Z = D / h;                           % standardized
            K = exp(-0.5 * Z.^2) / sqrt(2*pi);  % (N x N) Gaussian kernels

            p_hat    = sum(K, 2) / (N * h);            % (N x 1) density
            p_hat_d  = -sum(Z .* K, 2) / (N * h^2);   % (N x 1) first derivative
            p_hat_dd =  sum((Z.^2 - 1) .* K, 2) / (N * h^3);  % (N x 1) second derivative

            % Score function: g = -d log p / du = -p'/p
            p_safe = max(p_hat, 1e-10);
            gu  = -p_hat_d ./ p_safe;

            % Derivative of score: g' = d/du (-p'/p) = (-p''*p + (p')^2) / p^2
            gdu = (-p_hat_dd .* p_safe  +  p_hat_d .^ 2) ./ (p_safe .^ 2);
        end
    end
end
