classdef KernelSmoother < OceanStats.Base.IUnivariateSmoother
    % KERNELSMOOTHER Nadaraya-Watson local constant kernel regression.
    %
    % PURPOSE:
    %   Fits a smooth non-parametric curve using kernel-weighted local averaging.
    %   Provides Gaussian, Epanechnikov, and Tricube kernels.
    %
    % USAGE:
    %   smoother = OceanStats.NonParametric.KernelSmoother('Bandwidth', 0.5);
    %   model = smoother.fit(X, Y);
    %   y_pred = smoother.predict(model, X_new);

    properties
        Bandwidth  (1,1) double {mustBePositive} = 1.0
        KernelType string {mustBeMember(KernelType, {'gaussian', 'epanechnikov', 'tricube'})} = "gaussian"
    end

    methods
        function obj = KernelSmoother(opts)
            arguments
                opts.Bandwidth  (1,1) double {mustBePositive} = 1.0
                opts.KernelType string {mustBeMember(opts.KernelType, {'gaussian', 'epanechnikov', 'tricube'})} = "gaussian"
            end
            obj.Bandwidth  = opts.Bandwidth;
            obj.KernelType = opts.KernelType;
        end

        function model = fit(obj, X, Y, Weights)
            arguments
                obj
                X (:,1) double
                Y (:,1) double
                Weights (:,1) double = ones(size(Y))
            end
            if isempty(Weights)
                Weights = ones(size(Y));
            end
            
            % Model is essentially the stored data and parameters
            model.X = X;
            model.Y = Y;
            model.Weights = Weights;
            model.Bandwidth = obj.Bandwidth;
            model.KernelType = obj.KernelType;
        end

        function y_pred = predict(~, model, X_new)
            M = length(X_new);
            y_pred = zeros(M, 1);
            
            X_train = model.X;
            Y_train = model.Y;
            W_train = model.Weights;
            bw = model.Bandwidth;
            ktype = lower(model.KernelType);
            
            for i = 1:M
                u = (X_new(i) - X_train) / bw;
                
                switch ktype
                    case 'gaussian'
                        K = exp(-0.5 * u.^2);
                    case 'epanechnikov'
                        K = max(0, 0.75 * (1 - u.^2));
                    case 'tricube'
                        K = max(0, (1 - abs(u).^3).^3);
                end
                
                W = K .* W_train;
                sumW = sum(W);
                
                if sumW < 1e-12
                    % Fallback if no points are within bandwidth (global mean)
                    y_pred(i) = mean(Y_train);
                else
                    y_pred(i) = sum(W .* Y_train) / sumW;
                end
            end
        end
    end
end
