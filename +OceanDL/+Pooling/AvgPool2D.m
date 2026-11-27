classdef AvgPool2D < OceanDL.Pooling.IPoolingLayer
    % AVGPOOL2D  2-D average pooling layer (LeCun et al., 1998).
    %
    % THEORY:
    %   FORMULA (per output position (i,j), channel c, batch sample n):
    %     y_{i,j,c,n} = (1 / |W_{i,j}|) * sum_{(p,q) in W_{i,j}} x_{p,q,c,n}
    %   where W_{i,j} is the PoolSize(1) x PoolSize(2) window of the (padded)
    %   input starting at row (i-1)*Stride(1)+1, col (j-1)*Stride(2)+1.
    %   Slides a window over each (H x W) spatial plane independently per
    %   channel and batch sample.
    %
    % PADDING & DIVISOR (CountIncludePad):
    %   Padding pads with ZERO (average pooling has no natural "identity"
    %   value the way max pooling does with -Inf; zero is the standard
    %   convention -- PyTorch nn.AvgPool2d, TensorFlow AveragePooling2D).
    %     CountIncludePad = true  (default, matches PyTorch's default):
    %       divisor = PoolSize(1)*PoolSize(2) for every window, ALWAYS --
    %       zero-padded cells are included in the count, diluting edge
    %       windows toward zero.
    %     CountIncludePad = false:
    %       divisor = the number of REAL (non-padding) cells covered by
    %       that window -- edge windows average only real input pixels.
    %
    % BACKWARD:
    %   Each output's gradient is distributed EQUALLY (dY/divisor) to every
    %   position in its window. Overlapping windows (Stride < PoolSize)
    %   accumulate (summed), matching the forward sum's own linearity.
    %
    % USAGE:
    %   pool = OceanDL.Pooling.AvgPool2D([2 2]);
    %   pool = OceanDL.Pooling.AvgPool2D([3 3], 'Stride', [2 2], ...
    %              'Padding', [1 1], 'CountIncludePad', false);
    %   Y    = pool.forward(X);      % X: (H x W x C x N) -> Y: (Ho x Wo x C x N)
    %   dX   = pool.backward(dY);    % dY: (Ho x Wo x C x N) -> dX: (H x W x C x N)
    %
    % See also: OceanDL.Pooling.MaxPool2D
    %           OceanDL.Pooling.MaxUnpool2D
    %
    % Implements the 3-pillar contract (.agents/pipeline-architecture-spec.md
    % §1.1) via getMetadata() and getPipelineSpec(), in addition to forward()/backward().

    properties (SetAccess = private)
        CountIncludePad (1,1) logical = true
    end

    properties (Access = private)
        Divisors_  = []   % (Ho x Wo) window divisor, set by forward() (shared across C,N)
        InputSize_ = []   % [H W C N] of the last forward() call, set by forward()
    end

    methods
        function obj = AvgPool2D(poolSize, opts)
            % AVGPOOL2D  Construct an average pooling layer.
            %   pool = AvgPool2D(poolSize)
            %   pool = AvgPool2D(poolSize, 'Stride', s, 'Padding', p, 'CountIncludePad', tf)
            arguments
                poolSize (1,:) double {mustBePositive, mustBeInteger}
                opts.Stride  (1,:) double {mustBePositive, mustBeInteger} = poolSize
                opts.Padding (1,:) double {mustBeNonnegative, mustBeInteger} = [0 0]
                opts.CountIncludePad (1,1) logical = true
            end
            obj.PoolSize = obj.normalizePair(poolSize, "PoolSize");
            obj.Stride   = obj.normalizePair(opts.Stride, "Stride");
            obj.Padding  = obj.normalizePair(opts.Padding, "Padding");
            obj.CountIncludePad = opts.CountIncludePad;
            if any(obj.Padding >= (obj.PoolSize + 1) / 2)
                error('OceanDL:Pooling:AvgPool2D:PaddingTooLarge', ...
                    'Padding [%d %d] must be < half of PoolSize [%d %d] (else a window could be entirely padding, and CountIncludePad=false would divide by zero).', ...
                    obj.Padding(1), obj.Padding(2), obj.PoolSize(1), obj.PoolSize(2));
            end
        end

        function Y = forward(obj, X)
            % FORWARD  Average-pool X: (H x W x C x N) -> Y: (Ho x Wo x C x N).
            arguments
                obj
                X (:,:,:,:) double
            end
            [H, W, C, N] = size(X);
            obj.InputSize_ = [H, W, C, N];
            [Ho, Wo] = obj.outputSize(H, W);

            ph = obj.Padding(1); pw = obj.Padding(2);
            Xp = zeros(H + 2*ph, W + 2*pw, C, N);
            Xp(ph+1:ph+H, pw+1:pw+W, :, :) = X;
            poolArea = obj.PoolSize(1) * obj.PoolSize(2);

            Y = zeros(Ho, Wo, C, N);
            divisors = zeros(Ho, Wo);
            for i = 1:Ho
                for j = 1:Wo
                    [rows, cols] = obj.windowIndices(i, j);
                    window = Xp(rows, cols, :, :);
                    Y(i, j, :, :) = sum(window, [1 2]);

                    if obj.CountIncludePad
                        divisors(i, j) = poolArea;
                    else
                        validRows = rows > ph & rows <= ph + H;
                        validCols = cols > pw & cols <= pw + W;
                        divisors(i, j) = nnz(validRows) * nnz(validCols);
                    end
                    Y(i, j, :, :) = Y(i, j, :, :) / divisors(i, j);
                end
            end
            obj.Divisors_ = divisors;
        end

        function dX = backward(obj, dY)
            % BACKWARD  Distribute dY: (Ho x Wo x C x N) equally over each
            %   window into dX: (H x W x C x N) (accumulating overlaps),
            %   using the divisors cached by the most recent forward() call.
            arguments
                obj
                dY (:,:,:,:) double
            end
            if isempty(obj.Divisors_)
                error('OceanDL:Pooling:AvgPool2D:NoForwardCache', ...
                    'backward() called before forward() -- no cached divisors.');
            end
            H = obj.InputSize_(1); W = obj.InputSize_(2);
            C = obj.InputSize_(3); N = obj.InputSize_(4);
            ph = obj.Padding(1); pw = obj.Padding(2);
            [Ho, Wo] = size(obj.Divisors_);

            dXp = zeros(H + 2*ph, W + 2*pw, C, N);
            for i = 1:Ho
                for j = 1:Wo
                    [rows, cols] = obj.windowIndices(i, j);
                    contrib = dY(i, j, :, :) / obj.Divisors_(i, j);   % (1x1xCxN), broadcasts over the window
                    dXp(rows, cols, :, :) = dXp(rows, cols, :, :) + contrib;
                end
            end
            dX = dXp(ph+1:ph+H, pw+1:pw+W, :, :);
        end

        function s = name(~)
            s = 'avgpool2d';
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = "y_{i,j,c,n} = \frac{1}{|\mathcal{W}_{i,j}|} \sum_{(p,q)\,\in\,\mathcal{W}_{i,j}} x_{p,q,c,n}";
            meta.Symbols = containers.Map();
            meta.Symbols('y_{i,j,c,n}') = "pooled output at spatial position (i,j), channel c, batch sample n";
            meta.Symbols('\mathcal{W}_{i,j}') = sprintf('%dx%d sliding window at stride [%d %d], padding [%d %d]', ...
                obj.PoolSize(1), obj.PoolSize(2), obj.Stride(1), obj.Stride(2), obj.Padding(1), obj.Padding(2));
            if obj.CountIncludePad
                meta.Symbols('|\mathcal{W}_{i,j}|') = "PoolSize(1)*PoolSize(2), always (zero-padding counted in the divisor)";
            else
                meta.Symbols('|\mathcal{W}_{i,j}|') = "number of REAL (non-padding) cells in the window";
            end
            meta.Assumptions = {'padding is zero-valued', ...
                                 'pooling is per-channel, per-sample -- no cross-channel mixing'};
            meta.IsSymbolic = true;
            meta.Reference = "LeCun, Bottou, Bengio, & Haffner (1998), Gradient-Based Learning Applied to Document Recognition";
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.Inputs = struct('Name', "X", 'Domain', "spatial-tensor-batch", 'Dtype', "double", 'Shape', "H x W x C x N");
            spec.Outputs = struct('Name', "Y", 'Domain', "spatial-tensor-batch", 'Dtype', "double", ...
                'Shape', sprintf('floor((H+2*%d-%d)/%d)+1 x floor((W+2*%d-%d)/%d)+1 x C x N', ...
                    obj.Padding(1), obj.PoolSize(1), obj.Stride(1), obj.Padding(2), obj.PoolSize(2), obj.Stride(2)));
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "avgpool");
            spec.ComposesFrom = {};
        end
    end
end
