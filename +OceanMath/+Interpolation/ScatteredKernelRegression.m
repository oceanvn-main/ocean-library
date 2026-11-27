classdef ScatteredKernelRegression < OceanMath.Interpolation.IScatteredInterpolant
    % SCATTEREDKERNELREGRESSION  N-D Nadaraya-Watson kernel regression estimator.
    %
    %   Implements textbook equations 4.12 - 4.14 (Szeliski §4.1.1):
    %       f(x) = sum_k [ d_k * phi(||x - x_k||) ] / sum_l [ phi(||x - x_l||) ]
    %
    %   No linear system solver required. Computes a distance-weighted average
    %   at query time.
    %
    %   Usage:
    %     kr = OceanMath.Interpolation.ScatteredKernelRegression('Kernel', 'gaussian', 'Epsilon', 0.5);
    %     kr.fit(X, y);
    %     y_hat = kr.predict(Xq);
    %
    %   Author: Antigravity Code Assistant
    %   Date:   2026-08-18

    properties
        Kernel   % IRBFKernel strategy or string alias
        Epsilon  (1,1) double {mustBePositive} = 1.0  % Scale parameter c
    end

    properties (Access = private)
        KernelObj_  % Instantiated IRBFKernel object
    end

    methods
        function obj = ScatteredKernelRegression(opts)
            arguments
                opts.Kernel  = 'gaussian'
                opts.Epsilon (1,1) double {mustBePositive} = 1.0
            end
            obj.Kernel  = opts.Kernel;
            obj.Epsilon = opts.Epsilon;
        end

        function name = getName(obj)
            if isempty(obj.KernelObj_)
                name = 'KernelRegression';
            else
                fullCls = class(obj.KernelObj_);
                parts   = strsplit(fullCls, '.');
                kName   = strrep(parts{end}, 'Kernel', '');
                kName   = strrep(kName, 'RBF', '');
                name    = sprintf('KernelRegression-%s', kName);
            end
        end
    end

    methods (Access = protected)

        function fitCore_(obj, X, y)
            % Store training data and resolve kernel strategy.
            obj.X_train_   = X;
            obj.y_train_   = y;
            obj.KernelObj_ = obj.resolveKernel_(obj.Kernel, obj.Epsilon);
        end

        function y_hat = predictCore_(obj, Xq)
            % Evaluate Nadaraya-Watson estimator (eq. 4.12)
            % Xq: (M x D), X_train_: (N x D)
            R = obj.pairwiseDist_(Xq, obj.X_train_);    % (M x N)
            W = obj.KernelObj_.evaluate(R);             % (M x N)

            denom = sum(W, 2);                          % (M x 1)
            % Handle zero denominator if kernel evaluates to 0 everywhere far away
            denom(denom == 0) = eps;

            % f(x) = sum_k (y_k * w_k) / sum_l w_l
            y_hat = (W * obj.y_train_) ./ denom;         % (M x 1)
        end

    end

    methods (Access = private)

        function R = pairwiseDist_(~, X1, X2)
            % Vectorized Euclidean distance matrix between X1 (M x D) and X2 (N x D).
            diff = reshape(X1, size(X1,1), 1, []) - reshape(X2, 1, size(X2,1), []);
            R = sqrt(sum(diff.^2, 3));
        end

        function k = resolveKernel_(~, kernelArg, epsilon)
            import OceanMath.Image.Resampling.*
            if ischar(kernelArg) || isstring(kernelArg)
                switch lower(char(kernelArg))
                    case 'gaussian'
                        k = GaussianRBFKernel('Beta', 1.0 / (epsilon^2));
                    case {'inversemultiquadric', 'imq'}
                        k = InverseMultiquadricKernel('C', epsilon);
                    case {'inversedistance', 'idw'}
                        k = InverseDistanceKernel();
                    otherwise
                        error('ScatteredKernelRegression:InvalidKernel', ...
                            'Kernel Regression requires a decaying kernel (gaussian, inversemultiquadric, inversedistance). Got: %s', ...
                            kernelArg);
                end
            elseif isa(kernelArg, 'OceanMath.Image.Resampling.IRBFKernel')
                k = kernelArg;
            else
                error('ScatteredKernelRegression:InvalidKernel', ...
                    'Kernel must be a string alias or an IRBFKernel instance.');
            end
        end

    end

    methods (Static)
        function meta = getMetadata()
            % GETMETADATA  Return pedagogical metadata for the GUI learner application.
            here = fileparts(mfilename('fullpath'));
            raw  = fileread(fullfile(here, 'meta', 'ScatteredKernelRegression.meta.json'));
            meta = jsondecode(raw);
        end
    end
end
