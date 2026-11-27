classdef Dense < handle
    % DENSE  Fully-connected layer, (Din x N) -> (Dout x N), for a CNN
    %   classifier head after Flatten.
    %
    %   CONVENTION: matches VectorizedEngine/BackpropEngine's dense (d x N)
    %   column-major layout exactly: Y = W*X + b, W is (Dout x Din), b is
    %   (Dout x 1).
    %
    %   GRADIENT NORMALIZATION: like Conv2D, backward() returns the RAW SUM
    %   over the batch for dWeight/dBias (not pre-divided by N) -- see
    %   Conv2D's class doc for why a primitive should not bake in
    %   normalization.
    %
    %   USAGE:
    %     dense = OceanDL.Conv.Dense(128, 10);            % random He-normal init
    %     dense = OceanDL.Conv.Dense(128, 10, 'Weight', W, 'Bias', b);  % pretrained
    %     Y  = dense.forward(X);     % X: (128 x N) -> Y: (10 x N)
    %     dX = dense.backward(dY);   % dY: (10 x N) -> dX: (128 x N)
    %     [dW, db] = dense.getWeightGrad();

    properties (SetAccess = private)
        InFeatures  (1,1) double {mustBeNonnegative, mustBeInteger} = 1
        % NOTE: 0 is a valid sentinel meaning "not yet known -- inferred
        % from the first forward() call" (mirrors Conv2D's deferred
        % InChannels, since the flatten-adjacent Dense layer's fan-in
        % depends on the input image's spatial size, which CNNTopologySpec
        % deliberately does not declare).
        OutFeatures (1,1) double {mustBePositive, mustBeInteger} = 1
        Weight              % (Dout x Din)
        Bias                % (Dout x 1), only when HasBias
        HasBias (1,1) logical = true
    end

    properties (Access = private)
        X_       = []
        dWeight_ = []
        dBias_   = []
        Seed_    = []
    end

    methods
        function obj = Dense(inFeatures, outFeatures, opts)
            arguments
                inFeatures (1,1) double {mustBeNonnegative, mustBeInteger}
                outFeatures (1,1) double {mustBePositive, mustBeInteger}
                opts.HasBias (1,1) logical = true
                opts.Weight = []
                opts.Bias = []
                opts.Seed = []
            end
            obj.OutFeatures = outFeatures;
            obj.HasBias = opts.HasBias;
            obj.Seed_ = opts.Seed;

            if ~isempty(opts.Weight)
                W = opts.Weight;
                if inFeatures > 0 && ~isequal(size(W), [outFeatures, inFeatures])
                    error('OceanDL:Conv:Dense:BadWeightShape', ...
                        'Weight must be (%d x %d), got (%d x %d).', ...
                        outFeatures, inFeatures, size(W,1), size(W,2));
                end
                obj.InFeatures = size(W, 2);
                obj.Weight = W;
                if opts.HasBias
                    if isempty(opts.Bias)
                        obj.Bias = zeros(outFeatures, 1);
                    else
                        obj.Bias = reshape(opts.Bias, outFeatures, 1);
                    end
                end
            elseif inFeatures > 0
                obj.InFeatures = inFeatures;
                obj.initParams();
            else
                obj.InFeatures = 0; % deferred until first forward()
            end
        end

        function Y = forward(obj, X)
            arguments
                obj
                X (:,:) double
            end
            if obj.InFeatures == 0
                obj.InFeatures = size(X, 1);
                obj.initParams();
            elseif size(X, 1) ~= obj.InFeatures
                error('OceanDL:Conv:Dense:InputSizeMismatch', ...
                    'Input has %d features but this layer expects %d.', size(X,1), obj.InFeatures);
            end
            obj.X_ = X;
            Y = obj.Weight * X;
            if obj.HasBias
                Y = Y + obj.Bias;
            end
        end

        function dX = backward(obj, dY)
            arguments
                obj
                dY (:,:) double
            end
            if isempty(obj.X_)
                error('OceanDL:Conv:Dense:NoForwardCache', ...
                    'backward() called before forward().');
            end
            obj.dWeight_ = dY * obj.X_';          % (Dout x Din), raw sum over batch
            if obj.HasBias
                obj.dBias_ = sum(dY, 2);          % (Dout x 1)
            else
                obj.dBias_ = [];
            end
            dX = obj.Weight' * dY;                % (Din x N)
        end

        function [dW, db] = getWeightGrad(obj)
            if isempty(obj.dWeight_)
                error('OceanDL:Conv:Dense:NoBackwardCache', ...
                    'getWeightGrad() called before backward().');
            end
            dW = obj.dWeight_;
            db = obj.dBias_;
        end

        function applyUpdate(obj, dW, db, lr)
            arguments
                obj
                dW (:,:) double
                db = []
                lr (1,1) double = 1
            end
            obj.Weight = obj.Weight - lr * dW;
            if obj.HasBias && ~isempty(db)
                obj.Bias = obj.Bias - lr * reshape(db, obj.OutFeatures, 1);
            end
        end
    end

    methods (Access = private)
        function initParams(obj)
            if ~isempty(obj.Seed_)
                rng(obj.Seed_);
            end
            obj.Weight = randn(obj.OutFeatures, obj.InFeatures) * sqrt(2 / obj.InFeatures);
            if obj.HasBias
                obj.Bias = zeros(obj.OutFeatures, 1);
            end
        end
    end
end
