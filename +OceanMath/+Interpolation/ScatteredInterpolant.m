classdef ScatteredInterpolant < OceanMath.Interpolation.IScatteredInterpolant
    % SCATTEREDINTERPOLANT  Unified class for N-D scattered data interpolation.
    %
    %   Unifies Radial Basis Functions (RBF), Nadaraya-Watson Kernel Regression,
    %   K-Nearest Neighbours (KNN), Delaunay Triangulation (Natural Neighbour),
    %   and 1D Smoothing Splines under a single master interface.
    %
    %   Supported Methods:
    %     'rbf'               - Radial Basis Function interpolation (eq. 4.3–4.11)
    %     'kernel-regression' - Nadaraya-Watson distance-weighted regression (eq. 4.12–4.14)
    %     'knn'               - K-Nearest Neighbours regression
    %     'natural'           - Delaunay triangulation (natural, linear, nearest)
    %     'spline'            - 1D smoothing spline
    %
    %   Usage:
    %     % 1. RBF with Gaussian kernel
    %     interp = OceanMath.Interpolation.ScatteredInterpolant( ...
    %                 'Method', 'rbf', 'Kernel', 'gaussian', 'Epsilon', 0.5, 'Lambda', 1e-3);
    %
    %     % 2. Nadaraya-Watson Kernel Regression
    %     interp = OceanMath.Interpolation.ScatteredInterpolant( ...
    %                 'Method', 'kernel-regression', 'Kernel', 'gaussian', 'Epsilon', 0.5);
    %
    %     % 3. K-Nearest Neighbours
    %     interp = OceanMath.Interpolation.ScatteredInterpolant( ...
    %                 'Method', 'knn', 'K', 5, 'Weights', 'distance');
    %
    %     % 4. Delaunay Natural Neighbour
    %     interp = OceanMath.Interpolation.ScatteredInterpolant( ...
    %                 'Method', 'natural', 'SubMethod', 'natural');
    %
    %     % 5. 1D Smoothing Spline
    %     interp = OceanMath.Interpolation.ScatteredInterpolant( ...
    %                 'Method', 'spline', 'Lambda', 1e-4);
    %
    %     interp.fit(X, y);
    %     y_hat = interp.predict(Xq);
    %
    %   Author: Antigravity Code Assistant
    %   Date:   2026-08-19

    properties
        Method     (1,1) string {mustBeMember(Method, ...
                       ["rbf","kernel-regression","knn","natural","spline"])} = "rbf"
        Kernel     = "thinplate"   % RBF/KernelRegression kernel alias or IRBFKernel object
        Epsilon    (1,1) double {mustBePositive} = 1.0     % Scale parameter c
        Lambda     (1,1) double {mustBeNonnegative} = 0.0  % Regularization parameter
        K          (1,1) double {mustBePositive, mustBeInteger} = 5  % KNN neighbors
        Weights    (1,1) string {mustBeMember(Weights, ["uniform","distance"])} = "distance"
        Metric     (1,1) string {mustBeMember(Metric, ["euclidean","manhattan","chebyshev"])} = "euclidean"
        SubMethod  (1,1) string {mustBeMember(SubMethod, ["natural","linear","nearest"])} = "natural"
        ExtrapolationMethod (1,1) string {mustBeMember(ExtrapolationMethod, ["none","nearest","linear"])} = "none"
        Df         (1,1) double = NaN  % Spline target degrees of freedom
    end

    properties (Access = private)
        Delegate_  % Internal IScatteredInterpolant strategy instance
    end

    methods
        function obj = ScatteredInterpolant(opts)
            arguments
                opts.Method     (1,1) string = "rbf"
                opts.Kernel                  = "thinplate"
                opts.Epsilon    (1,1) double = 1.0
                opts.Lambda     (1,1) double = 0.0
                opts.K          (1,1) double = 5
                opts.Weights    (1,1) string = "distance"
                opts.Metric     (1,1) string = "euclidean"
                opts.SubMethod  (1,1) string = "natural"
                opts.ExtrapolationMethod (1,1) string = "none"
                opts.Df         (1,1) double = NaN
            end

            obj.Method              = opts.Method;
            obj.Kernel              = opts.Kernel;
            obj.Epsilon             = opts.Epsilon;
            obj.Lambda              = opts.Lambda;
            obj.K                   = opts.K;
            obj.Weights             = opts.Weights;
            obj.Metric              = opts.Metric;
            obj.SubMethod           = opts.SubMethod;
            obj.ExtrapolationMethod = opts.ExtrapolationMethod;
            obj.Df                  = opts.Df;

            obj.initDelegate_();
        end

        function name = getName(obj)
            if ~isempty(obj.Delegate_)
                name = obj.Delegate_.getName();
            else
                name = char(obj.Method);
            end
        end
    end

    methods (Access = protected)

        function fitCore_(obj, X, y)
            obj.initDelegate_();
            obj.Delegate_.fit(X, y);
        end

        function y_hat = predictCore_(obj, Xq)
            y_hat = obj.Delegate_.predict(Xq);
        end

    end

    methods (Access = private)

        function initDelegate_(obj)
            import OceanMath.Interpolation.*
            switch lower(obj.Method)
                case 'rbf'
                    obj.Delegate_ = ScatteredRBF('Kernel', obj.Kernel, ...
                        'Epsilon', obj.Epsilon, 'Lambda', obj.Lambda);
                case 'kernel-regression'
                    obj.Delegate_ = ScatteredKernelRegression('Kernel', obj.Kernel, ...
                        'Epsilon', obj.Epsilon);
                case 'knn'
                    obj.Delegate_ = ScatteredKNN('K', obj.K, ...
                        'Weights', obj.Weights, 'Metric', obj.Metric);
                case 'natural'
                    obj.Delegate_ = ScatteredNatural('Method', obj.SubMethod, ...
                        'ExtrapolationMethod', obj.ExtrapolationMethod);
                case 'spline'
                    obj.Delegate_ = ScatteredSpline('Lambda', obj.Lambda, 'Df', obj.Df);
            end
        end

    end

    methods (Static)
        function meta = getMetadata(methodName)
            % GETMETADATA  Return pedagogical metadata for the given method or default.
            arguments
                methodName (1,1) string = "rbf"
            end
            import OceanMath.Interpolation.*
            switch lower(methodName)
                case 'rbf'
                    meta = ScatteredRBF.getMetadata();
                case 'kernel-regression'
                    meta = ScatteredKernelRegression.getMetadata();
                case 'knn'
                    meta = ScatteredKNN.getMetadata();
                case 'natural'
                    meta = ScatteredNatural.getMetadata();
                case 'spline'
                    meta = ScatteredSpline.getMetadata();
                otherwise
                    meta = ScatteredRBF.getMetadata();
            end
        end
    end
end
