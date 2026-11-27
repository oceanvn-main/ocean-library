classdef MaxPool2D < OceanDL.Pooling.IPoolingLayer
    % MAXPOOL2D  2-D max pooling layer (LeCun et al., 1998; Zeiler & Fergus, 2013).
    %
    % THEORY:
    %   FORMULA (per output position (i,j), channel c, batch sample n):
    %     y_{i,j,c,n} = max_{(p,q) in W_{i,j}} x_{p,q,c,n}
    %   where W_{i,j} is the PoolSize(1) x PoolSize(2) window of the (padded)
    %   input starting at row (i-1)*Stride(1)+1, col (j-1)*Stride(2)+1.
    %   Slides a window over each (H x W) spatial plane independently per
    %   channel and batch sample -- no mixing across channels (matches
    %   PyTorch nn.MaxPool2d / TensorFlow MaxPooling2D / MATLAB
    %   maxPooling2dLayer).
    %
    % PADDING:
    %   Padding pads with -Inf (not zero) so a padded cell can NEVER win the
    %   max over a real input cell -- this makes padding transparently a
    %   "no-op that only affects output size", never a hidden bias toward 0,
    %   which zero-padding would silently introduce for all-negative inputs.
    %
    % SWITCHES (buffer-on-self, PyTorch return_indices=True equivalent):
    %   forward() caches, for every output position, the LINEAR INDEX (into
    %   the unpadded H x W spatial plane, sub2ind-style) of the input pixel
    %   that won the max. This is exactly what MaxUnpool2D needs to scatter
    %   values back to their original spatial location (Badrinarayanan,
    %   Kendall, & Cipolla, 2017, "SegNet"). Retrieve it with getSwitches()
    %   after calling forward().
    %
    % BACKWARD:
    %   Gradient routes ONLY to the argmax location of each window (all other
    %   positions get zero gradient). When Stride < PoolSize, windows overlap
    %   and a single input pixel can be the argmax for more than one output
    %   position -- their gradients are SUMMED (accumarray), not overwritten.
    %
    % USAGE:
    %   pool = OceanDL.Pooling.MaxPool2D([2 2]);            % stride = pool size (default)
    %   pool = OceanDL.Pooling.MaxPool2D([3 3], 'Stride', [2 2], 'Padding', [1 1]);
    %   Y    = pool.forward(X);      % X: (H x W x C x N) -> Y: (Ho x Wo x C x N)
    %   dX   = pool.backward(dY);    % dY: (Ho x Wo x C x N) -> dX: (H x W x C x N)
    %   sw   = pool.getSwitches();   % (Ho x Wo x C x N) linear indices into (H x W)
    %
    % See also: OceanDL.Pooling.AvgPool2D
    %           OceanDL.Pooling.MaxUnpool2D
    %
    % Implements the 3-pillar contract (.agents/pipeline-architecture-spec.md
    % §1.1) via getMetadata() and getPipelineSpec(), in addition to forward()/backward().

    properties (Access = private)
        Switches_  = []   % (Ho x Wo x C x N) linear index into (H x W), set by forward()
        InputSize_ = []   % [H W C N] of the last forward() call, set by forward()
    end

    methods
        function obj = MaxPool2D(poolSize, opts)
            % MAXPOOL2D  Construct a max pooling layer.
            %   pool = MaxPool2D(poolSize)
            %   pool = MaxPool2D(poolSize, 'Stride', s, 'Padding', p)
            %   poolSize/Stride/Padding — scalar (square) or [rows cols].
            %   Stride defaults to poolSize (PyTorch/TensorFlow convention).
            arguments
                poolSize (1,:) double {mustBePositive, mustBeInteger}
                opts.Stride  (1,:) double {mustBePositive, mustBeInteger} = poolSize
                opts.Padding (1,:) double {mustBeNonnegative, mustBeInteger} = [0 0]
            end
            obj.PoolSize = obj.normalizePair(poolSize, "PoolSize");
            obj.Stride   = obj.normalizePair(opts.Stride, "Stride");
            obj.Padding  = obj.normalizePair(opts.Padding, "Padding");
            if any(obj.Padding >= (obj.PoolSize + 1) / 2)
                error('OceanDL:Pooling:MaxPool2D:PaddingTooLarge', ...
                    'Padding [%d %d] must be < half of PoolSize [%d %d] (else a window could be entirely padding).', ...
                    obj.Padding(1), obj.Padding(2), obj.PoolSize(1), obj.PoolSize(2));
            end
        end

        function Y = forward(obj, X)
            % FORWARD  Max-pool X: (H x W x C x N) -> Y: (Ho x Wo x C x N).
            arguments
                obj
                X (:,:,:,:) double
            end
            [H, W, C, N] = size(X);
            obj.InputSize_ = [H, W, C, N];
            [Ho, Wo] = obj.outputSize(H, W);

            ph = obj.Padding(1); pw = obj.Padding(2);
            Xp = -inf(H + 2*ph, W + 2*pw, C, N);
            Xp(ph+1:ph+H, pw+1:pw+W, :, :) = X;

            Y = zeros(Ho, Wo, C, N);
            switches = zeros(Ho, Wo, C, N);
            poolArea = obj.PoolSize(1) * obj.PoolSize(2);

            for i = 1:Ho
                for j = 1:Wo
                    [rows, cols] = obj.windowIndices(i, j);
                    window = Xp(rows, cols, :, :);                       % (kH x kW x C x N)
                    windowFlat = reshape(window, poolArea, C, N);         % (kH*kW x C x N)
                    [maxVal, localIdx] = max(windowFlat, [], 1);          % (1 x C x N)
                    Y(i, j, :, :) = maxVal;

                    [dr, dc] = ind2sub(obj.PoolSize, squeeze(localIdx));  % (C x N) each
                    unpadRow = rows(1) - 1 + dr - ph;                     % (C x N), unpadded row in [1,H]
                    unpadCol = cols(1) - 1 + dc - pw;                     % (C x N), unpadded col in [1,W]
                    switches(i, j, :, :) = reshape(sub2ind([H, W], unpadRow, unpadCol), 1, 1, C, N);
                end
            end
            obj.Switches_ = switches;
        end

        function dX = backward(obj, dY)
            % BACKWARD  Route dY: (Ho x Wo x C x N) to dX: (H x W x C x N)
            %   using the switches cached by the most recent forward() call.
            arguments
                obj
                dY (:,:,:,:) double
            end
            if isempty(obj.Switches_)
                error('OceanDL:Pooling:MaxPool2D:NoForwardCache', ...
                    'backward() called before forward() -- no cached switches.');
            end
            H = obj.InputSize_(1); W = obj.InputSize_(2);
            C = obj.InputSize_(3); N = obj.InputSize_(4);

            dX = zeros(H, W, C, N);
            for n = 1:N
                for c = 1:C
                    idx  = obj.Switches_(:, :, c, n);
                    grad = dY(:, :, c, n);
                    dXslice = accumarray(idx(:), grad(:), [H*W, 1]);
                    dX(:, :, c, n) = reshape(dXslice, H, W);
                end
            end
        end

        function sw = getSwitches(obj)
            % GETSWITCHES  Return the (Ho x Wo x C x N) linear-index switches
            %   from the most recent forward() call (for pairing with
            %   MaxUnpool2D). Errors if forward() has not been called yet.
            if isempty(obj.Switches_)
                error('OceanDL:Pooling:MaxPool2D:NoForwardCache', ...
                    'getSwitches() called before forward() -- nothing cached yet.');
            end
            sw = obj.Switches_;
        end

        function sz = getInputSize(obj)
            % GETINPUTSIZE  Return [H W C N] of the most recent forward() call.
            if isempty(obj.InputSize_)
                error('OceanDL:Pooling:MaxPool2D:NoForwardCache', ...
                    'getInputSize() called before forward() -- nothing cached yet.');
            end
            sz = obj.InputSize_;
        end

        function s = name(~)
            s = 'maxpool2d';
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = "y_{i,j,c,n} = \max_{(p,q)\,\in\,\mathcal{W}_{i,j}} x_{p,q,c,n}";
            meta.Symbols = containers.Map();
            meta.Symbols('y_{i,j,c,n}') = "pooled output at spatial position (i,j), channel c, batch sample n";
            meta.Symbols('\mathcal{W}_{i,j}') = sprintf('%dx%d sliding window at stride [%d %d], padding [%d %d]', ...
                obj.PoolSize(1), obj.PoolSize(2), obj.Stride(1), obj.Stride(2), obj.Padding(1), obj.Padding(2));
            meta.Symbols('x_{p,q,c,n}') = "input pixel at spatial position (p,q), channel c, batch sample n";
            meta.Assumptions = {'padding is -Inf (never wins the max), not zero', ...
                                 'pooling is per-channel, per-sample -- no cross-channel mixing', ...
                                 'gradient of a tie among equal max values routes to the FIRST index (MATLAB max() convention)'};
            meta.IsSymbolic = true;
            meta.Reference = "LeCun et al. (1998); Zeiler & Fergus (2013), Visualizing and Understanding CNNs";
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.Inputs = struct('Name', "X", 'Domain', "spatial-tensor-batch", 'Dtype', "double", 'Shape', "H x W x C x N");
            spec.Outputs = struct('Name', "Y", 'Domain', "spatial-tensor-batch", 'Dtype', "double", ...
                'Shape', sprintf('floor((H+2*%d-%d)/%d)+1 x floor((W+2*%d-%d)/%d)+1 x C x N', ...
                    obj.Padding(1), obj.PoolSize(1), obj.Stride(1), obj.Padding(2), obj.PoolSize(2), obj.Stride(2)));
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "maxpool");
            spec.ComposesFrom = {};
        end
    end
end
