classdef FastICAStrategy < OceanStats.Unsupervised.ICASolvers.IICASolver
    % FASTICASTRATEGY  FastICA (Hyvärinen & Oja, 2000).
    %
    % ESL §14.7.3. Sequential deflation using a FIXED contrast function g.
    % Maximizes negentropy J(w) ≈ [E{G(w^T x)} - E{G(v)}]^2 via fixed-point iteration.
    %
    % OBJECTIVE (per component):
    %   max_w  J(w) = [E{G(w^T X_w)} - E{G(v)}]^2
    %   where G is an entropy function, g = G', g' = G'', and v ~ N(0,1)
    %
    % FIXED-POINT UPDATE:
    %   w <- (1/N) * X_w^T * g(X_w*w) - E{g'(X_w*w)} * w
    %   w <- Gram-Schmidt deflate
    %   w <- w / ||w||
    %
    % CONTRAST FUNCTIONS:
    %   'tanh'   g(u) = tanh(alpha*u),    g'(u) = alpha*(1-tanh^2(alpha*u))
    %            Best for super-Gaussian sources (heavy-tailed). Default.
    %   'cubic'  g(u) = u^3,              g'(u) = 3*u^2
    %            Computationally simple. Best for sub-Gaussian.
    %
    % USAGE:
    %   s = OceanStats.Unsupervised.ICASolvers.FastICAStrategy();
    %   s = OceanStats.Unsupervised.ICASolvers.FastICAStrategy('ContrastFcn','cubic');
    %   ica.fit(X, q, 'Strategy', s);
    %
    % Author: Ocean Library | 2026-04-30

    properties (SetAccess = private)
        ContrastFcn  % 'tanh' | 'cubic'
        Alpha        % scalar — tanh steepness parameter (default 1)
        MaxIter      % int    — max iterations per component (default 200)
        Tol          % scalar — convergence: |1 - |w'*w_old|| < Tol (default 1e-6)
    end

    properties (SetAccess = protected)
        Name = 'FastICA'
    end

    methods
        function obj = FastICAStrategy(opts)
            arguments
                opts.ContrastFcn (1,:) char   = 'tanh'
                opts.Alpha       (1,1) double  = 1.0
                opts.MaxIter     (1,1) double {mustBePositive, mustBeInteger} = 200
                opts.Tol         (1,1) double {mustBePositive} = 1e-6
            end
            obj.ContrastFcn = opts.ContrastFcn;
            obj.Alpha       = opts.Alpha;
            obj.MaxIter     = opts.MaxIter;
            obj.Tol         = opts.Tol;
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
                    u = X_w * w;                        % (N x 1)
                    [gu, gdu] = obj.contrast(u);        % g and g' evaluated at u

                    % Fixed-point update
                    w_new = (X_w' * gu) / N  -  mean(gdu) * w;
                    w_new = obj.deflate(w_new, W, k);

                    % Safe normalize
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
                    fprintf('  [FastICA | %s] IC%d: iter=%d  converged=%s\n', ...
                        obj.ContrastFcn, k, iters(k), mat2str(converged(k)));
                end
            end
        end
    end

    methods (Access = private)
        function [gu, gdu] = contrast(obj, u)
            % CONTRAST  Evaluate g(u) and g'(u) for the configured contrast function.
            switch obj.ContrastFcn
                case 'tanh'
                    gu  = tanh(obj.Alpha * u);
                    gdu = obj.Alpha * (1 - gu .^ 2);
                case 'cubic'
                    gu  = u .^ 3;
                    gdu = 3 * u .^ 2;
                otherwise
                    error('OceanStats:ICA:UnknownContrast', ...
                        'Unknown contrast function: "%s". Use ''tanh'' or ''cubic''.', obj.ContrastFcn);
            end
        end
    end
end
