classdef Conv2D < OceanDL.Conv.IConvLayer
    % CONV2D  2-D convolution layer (LeCun et al., 1998), im2col-based.
    %
    % THEORY:
    %   FORMULA (cross-correlation, per output position (i,j), out-channel
    %   o, batch sample n -- see Assumptions below for why this is
    %   cross-correlation and not textbook convolution):
    %     y_{i,j,o,n} = b_o + sum_{p,q,c} W_{p,q,c,o} * x_{s_i+p-1, s_j+q-1, c, n}
    %   where (s_i, s_j) is the top-left corner of the KH x KW window at
    %   output position (i,j), i.e. the same sliding-window geometry as
    %   MaxPool2D (Stride, Padding, outputSize formula are identical).
    %
    % IMPLEMENTATION (im2col, vectorized over the KH*KW kernel-offset loop,
    %   NOT over Ho*Wo like the Pooling classes -- kernel size is small and
    %   fixed while Ho*Wo grows with image size, so this loop nesting scales
    %   far better for realistic image inputs):
    %     Col    = im2col(Xp)                     (KH*KW*Cin) x (Ho*Wo*N)
    %     WMat   = reshape(Weight, KH*KW*Cin, Cout)
    %     Y      = reshape(WMat' * Col, Cout,Ho,Wo,N) permuted to (Ho,Wo,Cout,N), + bias
    %   Both Col and WMat use MATLAB's natural column-major flattening of a
    %   (KH,KW,Cin,...) array, so the row (kh,kw,c) index order is IDENTICAL
    %   in both -- this is the one invariant the forward/backward math
    %   depends on; see buildPatches()/scatterPatches().
    %
    % GRADIENT NORMALIZATION (deliberately different from BackpropEngine):
    %   dWeight/dBias returned by backward() are the RAW SUM over the batch
    %   (matching PyTorch/TF Conv2d backward), NOT pre-divided by N like
    %   OceanDL.Simulation.BackpropEngine's dense dW/db. A
    %   primitive op should not silently bake in a per-sample-mean
    %   normalization -- that decision belongs to the loss function, once,
    %   at the top of the graph. Dividing here AND in the loss would
    %   silently double-normalize.
    %
    % PADDING: zero-padding (unlike MaxPool2D's -Inf padding -- zero is the
    %   mathematically correct additive identity for a weighted sum, and is
    %   the universal convolution convention).
    %
    % BACKWARD (col2im): gradient wrt the input is scattered back through
    %   the same window geometry and SUMMED over overlapping windows
    %   (Stride < KernelSize), exactly like MaxPool2D's accumarray -- see
    %   scatterPatches().
    %
    % USAGE:
    %   conv = OceanDL.Conv.Conv2D(3, 8);                        % 3x3 kernel, 8 out-channels, Cin inferred at first forward()
    %   conv = OceanDL.Conv.Conv2D([3 3], 8, 'InChannels', 3, 'Stride', [2 2], 'Padding', [1 1]);
    %   Y    = conv.forward(X);      % X: (H x W x Cin x N) -> Y: (Ho x Wo x Cout x N)
    %   dX   = conv.backward(dY);    % dY: (Ho x Wo x Cout x N) -> dX: (H x W x Cin x N)
    %   [dW, db] = conv.getWeightGrad();
    %
    % See also: OceanDL.Conv.IConvLayer
    %           OceanDL.Pooling.MaxPool2D
    %           OceanDL.Autograd.Tape

    properties (SetAccess = private)
        Weight    % (KH x KW x Cin x Cout)
        Bias      % (Cout x 1), only meaningful when HasBias
        HasBias (1,1) logical = true
    end

    properties (Access = private)
        X_          = []   % cached input, (H x W x Cin x N), for backward()
        Col_        = []   % cached im2col matrix, (KH*KW*Cin x Ho*Wo*N)
        InputSize_  = []   % [H W Cin N] of the most recent forward()
        OutputSize_ = []   % [Ho Wo] of the most recent forward()
        dWeight_    = []
        dBias_      = []
    end

    methods
        function obj = Conv2D(kernelSize, outChannels, opts)
            % CONV2D  Construct a 2-D convolution layer.
            %   conv = Conv2D(kernelSize, outChannels)
            %   conv = Conv2D(kernelSize, outChannels, 'InChannels', c, ...
            %                 'Stride', s, 'Padding', p, 'HasBias', tf, 'Weight', W, 'Bias', b)
            %
            %   InChannels — if omitted, inferred (and Weight/Bias
            %   initialized He-normal) from the first forward() call's
            %   input. Supplying 'Weight' (and optionally 'Bias') loads
            %   pretrained parameters directly and fixes InChannels.
            arguments
                kernelSize (1,:) double {mustBePositive, mustBeInteger}
                outChannels (1,1) double {mustBePositive, mustBeInteger}
                opts.InChannels (1,1) double {mustBeNonnegative, mustBeInteger} = 0
                opts.Stride  (1,:) double {mustBePositive, mustBeInteger} = [1 1]
                opts.Padding (1,:) double {mustBeNonnegative, mustBeInteger} = [0 0]
                opts.HasBias (1,1) logical = true
                opts.Weight = []
                opts.Bias = []
                opts.Seed = []
            end
            obj.KernelSize = obj.normalizePair(kernelSize, "KernelSize");
            obj.Stride     = obj.normalizePair(opts.Stride, "Stride");
            obj.Padding    = obj.normalizePair(opts.Padding, "Padding");
            obj.OutChannels = outChannels;
            obj.HasBias    = opts.HasBias;

            if ~isempty(opts.Weight)
                W = opts.Weight;
                if ndims(W) > 4 || size(W,1) ~= obj.KernelSize(1) || size(W,2) ~= obj.KernelSize(2) || size(W,4) ~= outChannels
                    error('OceanDL:Conv:Conv2D:BadWeightShape', ...
                        'Weight must be (KH x KW x Cin x Cout) = (%d x %d x Cin x %d).', ...
                        obj.KernelSize(1), obj.KernelSize(2), outChannels);
                end
                obj.InChannels = size(W, 3);
                obj.Weight = W;
                if opts.HasBias
                    if isempty(opts.Bias)
                        obj.Bias = zeros(outChannels, 1);
                    else
                        obj.Bias = reshape(opts.Bias, outChannels, 1);
                    end
                end
            elseif opts.InChannels > 0
                obj.InChannels = opts.InChannels;
                obj.initParams(opts.Seed);
            else
                obj.InChannels = 0; % deferred until first forward()
            end
        end

        function Y = forward(obj, X)
            arguments
                obj
                X (:,:,:,:) double
            end
            [H, W, Cin, N] = size(X);
            if obj.InChannels == 0
                obj.InChannels = Cin;
                obj.initParams([]);
            elseif Cin ~= obj.InChannels
                error('OceanDL:Conv:Conv2D:ChannelMismatch', ...
                    'Input has %d channels but this layer was built/loaded for %d.', Cin, obj.InChannels);
            end

            KH = obj.KernelSize(1); KW = obj.KernelSize(2);
            [Ho, Wo] = obj.outputSize(H, W);
            ph = obj.Padding(1); pw = obj.Padding(2);

            Xp = zeros(H + 2*ph, W + 2*pw, Cin, N);
            Xp(ph+1:ph+H, pw+1:pw+W, :, :) = X;

            Col = obj.buildPatches(Xp, KH, KW, Ho, Wo);   % (KH*KW*Cin) x (Ho*Wo*N)
            WMat = reshape(obj.Weight, KH*KW*Cin, obj.OutChannels);

            Ymat = WMat' * Col;                            % (Cout) x (Ho*Wo*N)
            Y = permute(reshape(Ymat, obj.OutChannels, Ho, Wo, N), [2 3 1 4]);
            if obj.HasBias
                Y = Y + reshape(obj.Bias, 1, 1, obj.OutChannels, 1);
            end

            obj.X_ = X;
            obj.Col_ = Col;
            obj.InputSize_ = [H, W, Cin, N];
            obj.OutputSize_ = [Ho, Wo];
        end

        function dX = backward(obj, dY)
            arguments
                obj
                dY (:,:,:,:) double
            end
            if isempty(obj.Col_)
                error('OceanDL:Conv:Conv2D:NoForwardCache', ...
                    'backward() called before forward() -- no cached im2col buffer.');
            end
            KH = obj.KernelSize(1); KW = obj.KernelSize(2);
            H = obj.InputSize_(1); W = obj.InputSize_(2);
            Cin = obj.InputSize_(3); N = obj.InputSize_(4);
            Ho = obj.OutputSize_(1); Wo = obj.OutputSize_(2);
            ph = obj.Padding(1); pw = obj.Padding(2);

            dYmat = reshape(permute(dY, [3 1 2 4]), obj.OutChannels, Ho*Wo*N);   % (Cout) x (Ho*Wo*N)

            % ---- Parameter gradients (raw sum over batch -- see class doc) ----
            WMat = reshape(obj.Weight, KH*KW*Cin, obj.OutChannels);
            dWeightMat = obj.Col_ * dYmat';                 % (KH*KW*Cin) x (Cout)
            obj.dWeight_ = reshape(dWeightMat, KH, KW, Cin, obj.OutChannels);
            if obj.HasBias
                obj.dBias_ = sum(dYmat, 2);                 % (Cout x 1)
            else
                obj.dBias_ = [];
            end

            % ---- Input gradient (col2im scatter-add) ----
            dCol = WMat * dYmat;                            % (KH*KW*Cin) x (Ho*Wo*N)
            dXp = zeros(H + 2*ph, W + 2*pw, Cin, N);
            dXp = obj.scatterPatches(dXp, dCol, KH, KW, Ho, Wo);
            dX = dXp(ph+1:ph+H, pw+1:pw+W, :, :);
        end

        function [dW, db] = getWeightGrad(obj)
            % GETWEIGHTGRAD  Return [dWeight, dBias] cached by the most
            %   recent backward() call. dBias is [] when HasBias is false.
            if isempty(obj.dWeight_)
                error('OceanDL:Conv:Conv2D:NoBackwardCache', ...
                    'getWeightGrad() called before backward() -- nothing cached yet.');
            end
            dW = obj.dWeight_;
            db = obj.dBias_;
        end

        function applyUpdate(obj, dW, db, lr)
            % APPLYUPDATE  In-place SGD step: Weight -= lr*dW, Bias -= lr*db.
            arguments
                obj
                dW (:,:,:,:) double
                db = []
                lr (1,1) double = 1
            end
            obj.Weight = obj.Weight - lr * dW;
            if obj.HasBias && ~isempty(db)
                obj.Bias = obj.Bias - lr * reshape(db, obj.OutChannels, 1);
            end
        end

        function sz = getInputSize(obj)
            if isempty(obj.InputSize_)
                error('OceanDL:Conv:Conv2D:NoForwardCache', ...
                    'getInputSize() called before forward() -- nothing cached yet.');
            end
            sz = obj.InputSize_;
        end

        function s = name(~)
            s = 'conv2d';
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = "y_{i,j,o,n} = b_o + \sum_{p,q,c} W_{p,q,c,o}\, x_{s_i+p-1,\, s_j+q-1,\, c,\, n}";
            meta.Symbols = containers.Map();
            meta.Symbols('y_{i,j,o,n}') = "output at spatial position (i,j), out-channel o, batch sample n";
            meta.Symbols('W_{p,q,c,o}') = sprintf('%dx%d kernel, in-channel c, out-channel o', obj.KernelSize(1), obj.KernelSize(2));
            meta.Symbols('b_o') = "bias for out-channel o (0 when HasBias=false)";
            meta.Symbols('x_{...}') = sprintf('input pixel, stride [%d %d], padding [%d %d]', ...
                obj.Stride(1), obj.Stride(2), obj.Padding(1), obj.Padding(2));
            meta.Assumptions = {'this is CROSS-CORRELATION (no kernel flip), matching PyTorch nn.Conv2d / TF Conv2D -- NOT textbook mathematical convolution', ...
                                 'zero-padding (additive identity), unlike MaxPool2D''s -Inf padding', ...
                                 'backward() returns dWeight/dBias as a RAW SUM over the batch, not pre-divided by N'};
            meta.IsSymbolic = true;
            meta.Reference = "LeCun et al. (1998), Gradient-Based Learning Applied to Document Recognition";
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.Inputs = struct('Name', "X", 'Domain', "spatial-tensor-batch", 'Dtype', "double", 'Shape', "H x W x Cin x N");
            spec.Outputs = struct('Name', "Y", 'Domain', "spatial-tensor-batch", 'Dtype', "double", ...
                'Shape', sprintf('floor((H+2*%d-%d)/%d)+1 x floor((W+2*%d-%d)/%d)+1 x %d x N', ...
                    obj.Padding(1), obj.KernelSize(1), obj.Stride(1), obj.Padding(2), obj.KernelSize(2), obj.Stride(2), obj.OutChannels));
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "conv2d");
            spec.ComposesFrom = {};
        end
    end

    methods (Access = private)
        function initParams(obj, seedVal)
            % He-normal init (ReLU-paired convention, matching NNGraph's
            % documented dense-layer initialization policy).
            if ~isempty(seedVal)
                rng(seedVal);
            end
            KH = obj.KernelSize(1); KW = obj.KernelSize(2);
            fanIn = KH * KW * obj.InChannels;
            obj.Weight = randn(KH, KW, obj.InChannels, obj.OutChannels) * sqrt(2 / fanIn);
            if obj.HasBias
                obj.Bias = zeros(obj.OutChannels, 1);
            end
        end

        function Col = buildPatches(obj, Xp, KH, KW, Ho, Wo)
            % BUILDPATCHES  im2col via a loop over the (small) KH*KW kernel
            %   offsets -- see class doc for why this loop nesting (not
            %   Ho*Wo) is used, and why the resulting row order matches
            %   reshape(Weight, KH*KW*Cin, Cout) exactly: both flatten a
            %   (KH,KW,Cin,...) array in MATLAB's natural column-major order.
            [~, ~, Cin, N] = size(Xp);
            strideH = obj.Stride(1); strideW = obj.Stride(2);
            Patches = zeros(KH, KW, Cin, Ho, Wo, N);
            for kh = 1:KH
                rowIdx = kh : strideH : kh + strideH*(Ho-1);
                for kw = 1:KW
                    colIdx = kw : strideW : kw + strideW*(Wo-1);
                    patch = Xp(rowIdx, colIdx, :, :);                  % (Ho, Wo, Cin, N)
                    patch = permute(patch, [3 1 2 4]);                 % (Cin, Ho, Wo, N)
                    Patches(kh, kw, :, :, :, :) = reshape(patch, 1, 1, Cin, Ho, Wo, N);
                end
            end
            Col = reshape(Patches, KH*KW*Cin, Ho*Wo*N);
        end

        function dXp = scatterPatches(obj, dXp, dCol, KH, KW, Ho, Wo)
            % SCATTERPATCHES  col2im: inverse of buildPatches(), SUMMING
            %   overlapping windows (Stride < KernelSize) -- mirrors
            %   MaxPool2D backward's accumarray-based accumulation.
            Cin = size(dXp, 3); N = size(dXp, 4);
            strideH = obj.Stride(1); strideW = obj.Stride(2);
            dPatches = reshape(dCol, KH, KW, Cin, Ho, Wo, N);
            for kh = 1:KH
                rowIdx = kh : strideH : kh + strideH*(Ho-1);
                for kw = 1:KW
                    colIdx = kw : strideW : kw + strideW*(Wo-1);
                    patchGrad = reshape(dPatches(kh, kw, :, :, :, :), Cin, Ho, Wo, N);
                    patchGrad = permute(patchGrad, [2 3 1 4]);         % (Ho, Wo, Cin, N)
                    dXp(rowIdx, colIdx, :, :) = dXp(rowIdx, colIdx, :, :) + patchGrad;
                end
            end
        end
    end
end
