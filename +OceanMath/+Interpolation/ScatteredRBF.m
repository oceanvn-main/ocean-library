classdef ScatteredRBF < OceanMath.Interpolation.IScatteredInterpolant
    % SCATTEREDRBF  N-D scattered data interpolation via Radial Basis Functions.
    %
    %   Implements textbook equations 4.3 - 4.11 (Szeliski §4.1.1).
    %
    %   Solves the linear system (eq. 4.8 / 4.11):
    %       (K + lambda * I) * w = y
    %   where K_ij = phi(||x_i - x_j||) and w are the RBF weights.
    %
    %   At prediction time (eq. 4.3):
    %       f(x) = sum_k w_k * phi(||x - x_k||)
    %
    %   Reuses the existing IRBFKernel strategies from +OceanMath/+Image/+Resampling/
    %   and supports Gaussian, Multiquadric, Inverse Multiquadric, Thin Plate Spline, etc.
    %
    %   Usage:
    %     rbf = OceanMath.Interpolation.ScatteredRBF('Kernel', 'gaussian', ...
    %                                                 'Epsilon', 1.0);
    %     rbf.fit(X, y);
    %     y_hat = rbf.predict(Xq);
    %
    %   Author: Antigravity Code Assistant
    %   Date:   2026-08-18

    properties
        Kernel   % IRBFKernel strategy or string alias
        Epsilon  (1,1) double {mustBePositive} = 1.0     % Scale parameter c
        Lambda   (1,1) double {mustBeNonnegative} = 0.0  % Tikhonov regularization
    end

    properties (Access = private)
        W_          % RBF weights (N x 1)
        XTr_        % Copy of training X for kernel evaluation at predict time
        KernelObj_  % Instantiated IRBFKernel object
    end

    methods
        function obj = ScatteredRBF(opts)
            % SCATTEREDRBF  Constructor.
            %
            %   obj = ScatteredRBF('Kernel', 'thinplate')
            %   obj = ScatteredRBF('Kernel', 'gaussian', 'Epsilon', 0.5)
            %   obj = ScatteredRBF('Kernel', 'inversemultiquadric', 'Epsilon', 1.0)
            %   obj = ScatteredRBF('Kernel', myKernelObj)  % concrete IRBFKernel

            arguments
                opts.Kernel  = 'thinplate'
                opts.Epsilon (1,1) double {mustBePositive} = 1.0
                opts.Lambda  (1,1) double {mustBeNonnegative} = 0.0
            end

            obj.Epsilon = opts.Epsilon;
            obj.Lambda  = opts.Lambda;
            obj.Kernel  = opts.Kernel;
        end

        function name = getName(obj)
            if isempty(obj.KernelObj_)
                name = 'RBF';
            else
                fullCls = class(obj.KernelObj_);
                parts   = strsplit(fullCls, '.');
                kName   = strrep(parts{end}, 'Kernel', '');
                kName   = strrep(kName, 'RBF', '');
                name    = sprintf('RBF-%s', kName);
            end
        end
    end

    % ── Protected hooks ───────────────────────────────────────────────────
    methods (Access = protected)

        function fitCore_(obj, X, y)
            % Build system (K + lambda * I) * w = y and solve for w.
            N = size(X, 1);
            obj.KernelObj_ = obj.resolveKernel_(obj.Kernel, obj.Epsilon);

            % 1. Pairwise distance matrix (N x N). Euclidean norm distances 
            r = obj.pairwiseDist_(X, X);

            % 2. Kernel matrix K (N x N)
            K = obj.KernelObj_.evaluate(r);

            % 3. Tikhonov regularization (eq. 4.11, quadratic penalty p=2)
            % Add small diagonal floor (1e-10) for numerical conditioning
            diagNudge = max(1e-10, obj.Lambda);
            K = K + diagNudge * eye(N);

            % 4. Solve for weights w (eq. 4.8 / 4.11)
            obj.W_   = K \ y;
            obj.XTr_ = X;
        end

        function y_hat = predictCore_(obj, Xq)
            % Evaluate RBF model at query points Xq (M x D) per eq. 4.3.
            r = obj.pairwiseDist_(Xq, obj.XTr_);       % (M x N)
            K_q = obj.KernelObj_.evaluate(r);           % (M x N)

            y_hat = K_q * obj.W_;                        % (M x 1)
        end

    end

    % ── Private helpers ───────────────────────────────────────────────────
    methods (Access = private)

        function r = pairwiseDist_(~, A, B)
            % PAIRWISEDIST_  Euclidean distance matrix between rows of A (M x D) and B (N x D).
            diff = permute(A, [1 3 2]) - permute(B, [3 1 2]);  % (M x N x D). Permute rearrange matrix M x D x 1 by [1 3 2] => M x 1 x D
            r = sqrt(sum(diff.^2, 3));                           % (M x N).
        end

        function k = resolveKernel_(~, kernelArg, epsilon)
            % Resolve string aliases or pass through concrete IRBFKernel objects.
            import OceanMath.Image.Resampling.*
            if ischar(kernelArg) || isstring(kernelArg)
                switch lower(char(kernelArg))
                    case {'thinplate', 'tps'}
                        k = ThinPlateKernel();
                    case 'gaussian'
                        k = GaussianRBFKernel('Beta', 1.0 / (epsilon^2));
                    case {'multiquadric', 'mq'}
                        k = MultiquadricKernel('C', epsilon);
                    case {'inversemultiquadric', 'imq'}
                        k = InverseMultiquadricKernel('C', epsilon);
                    case {'inversedistance', 'idw'}
                        k = InverseDistanceKernel();
                    otherwise
                        error('ScatteredRBF:UnknownKernel', ...
                            'Unknown kernel: "%s". Use thinplate/gaussian/multiquadric/inversemultiquadric/inversedistance.', ...
                            kernelArg);
                end
            elseif isa(kernelArg, 'OceanMath.Image.Resampling.IRBFKernel')
                k = kernelArg;
            else
                error('ScatteredRBF:InvalidKernel', ...
                    'Kernel must be a string alias or an IRBFKernel instance.');
            end
        end

    end

    methods (Static)
        function meta = getMetadata()
            % GETMETADATA  Return pedagogical metadata for the GUI learner application.
            here = fileparts(mfilename('fullpath'));
            raw  = fileread(fullfile(here, 'meta', 'ScatteredRBF.meta.json'));
            meta = jsondecode(raw);
        end
    end
end
