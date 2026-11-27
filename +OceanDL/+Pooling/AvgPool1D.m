classdef AvgPool1D < OceanDL.Pooling.IPoolingLayer1D
    % AVGPOOL1D  1-D average pooling layer (sequence/time-series/audio CNNs).
    %
    % THEORY:
    %   FORMULA (per output position i, channel c, batch sample n):
    %     y_{i,c,n} = (1/|W_i|) * sum_{p in W_i} x_{p,c,n}
    %   Zero-padded; CountIncludePad behaves exactly as in AvgPool2D (see
    %   its header comment for the full rationale).
    %
    % USAGE:
    %   pool = OceanDL.Pooling.AvgPool1D(2);
    %   Y    = pool.forward(X);      % X: (L x C x N) -> Y: (Lo x C x N)
    %   dX   = pool.backward(dY);
    %
    % See also: OceanDL.Pooling.MaxPool1D
    %           OceanDL.Pooling.MaxUnpool1D
    %           OceanDL.Pooling.AvgPool2D

    properties (SetAccess = private)
        CountIncludePad (1,1) logical = true
    end

    properties (Access = private)
        Divisors_  = []   % (Lo x 1) window divisor, set by forward() (shared across C,N)
        InputSize_ = []   % [L C N] of the last forward() call, set by forward()
    end

    methods
        function obj = AvgPool1D(poolSize, opts)
            arguments
                poolSize (1,1) double {mustBePositive, mustBeInteger}
                opts.Stride  (1,1) double {mustBePositive, mustBeInteger} = poolSize
                opts.Padding (1,1) double {mustBeNonnegative, mustBeInteger} = 0
                opts.CountIncludePad (1,1) logical = true
            end
            obj.PoolSize = poolSize;
            obj.Stride   = opts.Stride;
            obj.Padding  = opts.Padding;
            obj.CountIncludePad = opts.CountIncludePad;
            if obj.Padding >= (obj.PoolSize + 1) / 2
                error('OceanDL:Pooling:AvgPool1D:PaddingTooLarge', ...
                    'Padding %d must be < half of PoolSize %d (else a window could be entirely padding, and CountIncludePad=false would divide by zero).', ...
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
            Xp = zeros(L + 2*p, C, N);
            Xp(p+1:p+L, :, :) = X;

            Y = zeros(Lo, C, N);
            divisors = zeros(Lo, 1);
            for i = 1:Lo
                idxs = obj.windowIndices(i);
                window = Xp(idxs, :, :);
                Y(i, :, :) = sum(window, 1);

                if obj.CountIncludePad
                    divisors(i) = obj.PoolSize;
                else
                    validIdxs = idxs > p & idxs <= p + L;
                    divisors(i) = nnz(validIdxs);
                end
                Y(i, :, :) = Y(i, :, :) / divisors(i);
            end
            obj.Divisors_ = divisors;
        end

        function dX = backward(obj, dY)
            arguments
                obj
                dY (:,:,:) double
            end
            if isempty(obj.Divisors_)
                error('OceanDL:Pooling:AvgPool1D:NoForwardCache', ...
                    'backward() called before forward() -- no cached divisors.');
            end
            L = obj.InputSize_(1); C = obj.InputSize_(2); N = obj.InputSize_(3);
            p = obj.Padding;
            Lo = numel(obj.Divisors_);

            dXp = zeros(L + 2*p, C, N);
            for i = 1:Lo
                idxs = obj.windowIndices(i);
                contrib = dY(i, :, :) / obj.Divisors_(i);   % (1xCxN), broadcasts over the window
                dXp(idxs, :, :) = dXp(idxs, :, :) + contrib;
            end
            dX = dXp(p+1:p+L, :, :);
        end

        function s = name(~)
            s = 'avgpool1d';
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = "y_{i,c,n} = \frac{1}{|\mathcal{W}_{i}|} \sum_{p\,\in\,\mathcal{W}_{i}} x_{p,c,n}";
            meta.Symbols = containers.Map();
            meta.Symbols('y_{i,c,n}') = "pooled output at sequence position i, channel c, batch sample n";
            meta.Symbols('\mathcal{W}_{i}') = sprintf('length-%d sliding window at stride %d, padding %d', ...
                obj.PoolSize, obj.Stride, obj.Padding);
            if obj.CountIncludePad
                meta.Symbols('|\mathcal{W}_{i}|') = "PoolSize, always (zero-padding counted in the divisor)";
            else
                meta.Symbols('|\mathcal{W}_{i}|') = "number of REAL (non-padding) elements in the window";
            end
            meta.Assumptions = {'padding is zero-valued', ...
                                 'pooling is per-channel, per-sample -- no cross-channel mixing'};
            meta.IsSymbolic = true;
            meta.Reference = "LeCun, Bottou, Bengio, & Haffner (1998), Gradient-Based Learning Applied to Document Recognition";
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.Inputs = struct('Name', "X", 'Domain', "sequence-tensor-batch", 'Dtype', "double", 'Shape', "L x C x N");
            spec.Outputs = struct('Name', "Y", 'Domain', "sequence-tensor-batch", 'Dtype', "double", ...
                'Shape', sprintf('floor((L+2*%d-%d)/%d)+1 x C x N', obj.Padding, obj.PoolSize, obj.Stride));
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "avgpool1d");
            spec.ComposesFrom = {};
        end
    end
end
