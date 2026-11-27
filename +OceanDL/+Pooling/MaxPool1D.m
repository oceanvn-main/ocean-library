classdef MaxPool1D < OceanDL.Pooling.IPoolingLayer1D
    % MAXPOOL1D  1-D max pooling layer (sequence/time-series/audio CNNs).
    %
    % THEORY:
    %   FORMULA (per output position i, channel c, batch sample n):
    %     y_{i,c,n} = max_{p in W_i} x_{p,c,n}
    %   where W_i is the PoolSize window of the (padded) input starting at
    %   (i-1)*Stride+1. Matches PyTorch nn.MaxPool1d / TensorFlow
    %   MaxPooling1D / MATLAB maxPooling1dLayer.
    %
    %   Padding uses -Inf (never wins the max) -- see MaxPool2D for the
    %   full rationale, identical here.
    %
    %   SWITCHES: forward() caches, per output position, the linear position
    %   (1..L) of the winning input element -- needed by MaxUnpool1D.
    %
    % USAGE:
    %   pool = OceanDL.Pooling.MaxPool1D(2);
    %   Y    = pool.forward(X);      % X: (L x C x N) -> Y: (Lo x C x N)
    %   dX   = pool.backward(dY);
    %   sw   = pool.getSwitches();   % (Lo x C x N) linear index into (L)
    %
    % See also: OceanDL.Pooling.AvgPool1D
    %           OceanDL.Pooling.MaxUnpool1D
    %           OceanDL.Pooling.MaxPool2D (the 2-D analog; see
    %           its header comment for the full rationale behind every
    %           design choice replicated here)

    properties (Access = private)
        Switches_  = []   % (Lo x C x N) linear index into (L), set by forward()
        InputSize_ = []   % [L C N] of the last forward() call, set by forward()
    end

    methods
        function obj = MaxPool1D(poolSize, opts)
            arguments
                poolSize (1,1) double {mustBePositive, mustBeInteger}
                opts.Stride  (1,1) double {mustBePositive, mustBeInteger} = poolSize
                opts.Padding (1,1) double {mustBeNonnegative, mustBeInteger} = 0
            end
            obj.PoolSize = poolSize;
            obj.Stride   = opts.Stride;
            obj.Padding  = opts.Padding;
            if obj.Padding >= (obj.PoolSize + 1) / 2
                error('OceanDL:Pooling:MaxPool1D:PaddingTooLarge', ...
                    'Padding %d must be < half of PoolSize %d (else a window could be entirely padding).', ...
                    obj.Padding, obj.PoolSize);
            end
        end

        function Y = forward(obj, X)
            arguments
                obj
                X (:,:,:) double
            end
            [L, C, N] = size(X);
            obj.InputSize_ = [L, C, N];
            Lo = obj.outputSize(L);

            p = obj.Padding;
            Xp = -inf(L + 2*p, C, N);
            Xp(p+1:p+L, :, :) = X;

            Y = zeros(Lo, C, N);
            switches = zeros(Lo, C, N);

            for i = 1:Lo
                idxs = obj.windowIndices(i);
                window = Xp(idxs, :, :);                 % (poolSize x C x N)
                [maxVal, localIdx] = max(window, [], 1);  % (1 x C x N)
                Y(i, :, :) = maxVal;

                unpadPos = idxs(1) - 1 + squeeze(localIdx) - p;   % (C x N), position in [1, L]
                switches(i, :, :) = reshape(unpadPos, 1, C, N);
            end
            obj.Switches_ = switches;
        end

        function dX = backward(obj, dY)
            arguments
                obj
                dY (:,:,:) double
            end
            if isempty(obj.Switches_)
                error('OceanDL:Pooling:MaxPool1D:NoForwardCache', ...
                    'backward() called before forward() -- no cached switches.');
            end
            L = obj.InputSize_(1); C = obj.InputSize_(2); N = obj.InputSize_(3);

            dX = zeros(L, C, N);
            for n = 1:N
                for c = 1:C
                    idx  = obj.Switches_(:, c, n);
                    grad = dY(:, c, n);
                    dX(:, c, n) = accumarray(idx(:), grad(:), [L, 1]);
                end
            end
        end

        function sw = getSwitches(obj)
            if isempty(obj.Switches_)
                error('OceanDL:Pooling:MaxPool1D:NoForwardCache', ...
                    'getSwitches() called before forward() -- nothing cached yet.');
            end
            sw = obj.Switches_;
        end

        function sz = getInputSize(obj)
            if isempty(obj.InputSize_)
                error('OceanDL:Pooling:MaxPool1D:NoForwardCache', ...
                    'getInputSize() called before forward() -- nothing cached yet.');
            end
            sz = obj.InputSize_;
        end

        function s = name(~)
            s = 'maxpool1d';
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = "y_{i,c,n} = \max_{p\,\in\,\mathcal{W}_{i}} x_{p,c,n}";
            meta.Symbols = containers.Map();
            meta.Symbols('y_{i,c,n}') = "pooled output at sequence position i, channel c, batch sample n";
            meta.Symbols('\mathcal{W}_{i}') = sprintf('length-%d sliding window at stride %d, padding %d', ...
                obj.PoolSize, obj.Stride, obj.Padding);
            meta.Symbols('x_{p,c,n}') = "input element at sequence position p, channel c, batch sample n";
            meta.Assumptions = {'padding is -Inf (never wins the max), not zero', ...
                                 'pooling is per-channel, per-sample -- no cross-channel mixing'};
            meta.IsSymbolic = true;
            meta.Reference = "LeCun et al. (1998); Zeiler & Fergus (2013), Visualizing and Understanding CNNs";
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.Inputs = struct('Name', "X", 'Domain', "sequence-tensor-batch", 'Dtype', "double", 'Shape', "L x C x N");
            spec.Outputs = struct('Name', "Y", 'Domain', "sequence-tensor-batch", 'Dtype', "double", ...
                'Shape', sprintf('floor((L+2*%d-%d)/%d)+1 x C x N', obj.Padding, obj.PoolSize, obj.Stride));
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "maxpool1d");
            spec.ComposesFrom = {};
        end
    end
end
