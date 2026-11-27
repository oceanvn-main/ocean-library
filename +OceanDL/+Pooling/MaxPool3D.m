classdef MaxPool3D < OceanDL.Pooling.IPoolingLayer3D
    % MAXPOOL3D  3-D max pooling layer (volumetric/video/medical-imaging CNNs).
    %
    % THEORY:
    %   FORMULA (per output position (k,i,j), channel c, batch sample n):
    %     y_{k,i,j,c,n} = max_{(d,p,q) in W_{k,i,j}} x_{d,p,q,c,n}
    %   where W_{k,i,j} is the PoolSize(1) x PoolSize(2) x PoolSize(3) window
    %   of the (padded) input. Matches PyTorch nn.MaxPool3d / MATLAB
    %   maxPooling3dLayer. See MaxPool2D's header comment for the full
    %   rationale behind every design choice replicated here (padding=-Inf,
    %   switches buffer, overlapping-window gradient accumulation, etc.).
    %
    % USAGE:
    %   pool = OceanDL.Pooling.MaxPool3D([2 2 2]);
    %   Y    = pool.forward(X);      % X: (D x H x W x C x N) -> Y: (Do x Ho x Wo x C x N)
    %   dX   = pool.backward(dY);
    %   sw   = pool.getSwitches();   % (Do x Ho x Wo x C x N) linear index into (D x H x W)
    %
    % See also: OceanDL.Pooling.AvgPool3D
    %           OceanDL.Pooling.MaxUnpool3D
    %           OceanDL.Pooling.MaxPool2D

    properties (Access = private)
        Switches_  = []   % (Do x Ho x Wo x C x N) linear index into (D x H x W), set by forward()
        InputSize_ = []   % [D H W C N] of the last forward() call, set by forward()
    end

    methods
        function obj = MaxPool3D(poolSize, opts)
            arguments
                poolSize (1,:) double {mustBePositive, mustBeInteger}
                opts.Stride  (1,:) double {mustBePositive, mustBeInteger} = poolSize
                opts.Padding (1,:) double {mustBeNonnegative, mustBeInteger} = [0 0 0]
            end
            obj.PoolSize = obj.normalizeTriple(poolSize, "PoolSize");
            obj.Stride   = obj.normalizeTriple(opts.Stride, "Stride");
            obj.Padding  = obj.normalizeTriple(opts.Padding, "Padding");
            if any(obj.Padding >= (obj.PoolSize + 1) / 2)
                error('OceanDL:Pooling:MaxPool3D:PaddingTooLarge', ...
                    'Padding [%d %d %d] must be < half of PoolSize [%d %d %d] (else a window could be entirely padding).', ...
                    obj.Padding(1), obj.Padding(2), obj.Padding(3), ...
                    obj.PoolSize(1), obj.PoolSize(2), obj.PoolSize(3));
            end
        end

        function Y = forward(obj, X)
            arguments
                obj
                X (:,:,:,:,:) double
            end
            [D, H, W, C, N] = size(X);
            obj.InputSize_ = [D, H, W, C, N];
            [Do, Ho, Wo] = obj.outputSize(D, H, W);

            pd = obj.Padding(1); ph = obj.Padding(2); pw = obj.Padding(3);
            Xp = -inf(D + 2*pd, H + 2*ph, W + 2*pw, C, N);
            Xp(pd+1:pd+D, ph+1:ph+H, pw+1:pw+W, :, :) = X;

            Y = zeros(Do, Ho, Wo, C, N);
            switches = zeros(Do, Ho, Wo, C, N);
            poolVol = obj.PoolSize(1) * obj.PoolSize(2) * obj.PoolSize(3);

            for k = 1:Do
                for i = 1:Ho
                    for j = 1:Wo
                        [ds, hs, ws] = obj.windowIndices(k, i, j);
                        window = Xp(ds, hs, ws, :, :);                      % (kD x kH x kW x C x N)
                        windowFlat = reshape(window, poolVol, C, N);         % (kD*kH*kW x C x N)
                        [maxVal, localIdx] = max(windowFlat, [], 1);         % (1 x C x N)
                        Y(k, i, j, :, :) = maxVal;

                        [dd, dh, dw] = ind2sub(obj.PoolSize, squeeze(localIdx));  % (C x N) each
                        unpadD = ds(1) - 1 + dd - pd;
                        unpadH = hs(1) - 1 + dh - ph;
                        unpadW = ws(1) - 1 + dw - pw;
                        switches(k, i, j, :, :) = reshape(sub2ind([D, H, W], unpadD, unpadH, unpadW), 1, 1, 1, C, N);
                    end
                end
            end
            obj.Switches_ = switches;
        end

        function dX = backward(obj, dY)
            arguments
                obj
                dY (:,:,:,:,:) double
            end
            if isempty(obj.Switches_)
                error('OceanDL:Pooling:MaxPool3D:NoForwardCache', ...
                    'backward() called before forward() -- no cached switches.');
            end
            D = obj.InputSize_(1); H = obj.InputSize_(2); W = obj.InputSize_(3);
            C = obj.InputSize_(4); N = obj.InputSize_(5);

            dX = zeros(D, H, W, C, N);
            for n = 1:N
                for c = 1:C
                    idx  = obj.Switches_(:, :, :, c, n);
                    grad = dY(:, :, :, c, n);
                    dXslice = accumarray(idx(:), grad(:), [D*H*W, 1]);
                    dX(:, :, :, c, n) = reshape(dXslice, D, H, W);
                end
            end
        end

        function sw = getSwitches(obj)
            if isempty(obj.Switches_)
                error('OceanDL:Pooling:MaxPool3D:NoForwardCache', ...
                    'getSwitches() called before forward() -- nothing cached yet.');
            end
            sw = obj.Switches_;
        end

        function sz = getInputSize(obj)
            if isempty(obj.InputSize_)
                error('OceanDL:Pooling:MaxPool3D:NoForwardCache', ...
                    'getInputSize() called before forward() -- nothing cached yet.');
            end
            sz = obj.InputSize_;
        end

        function s = name(~)
            s = 'maxpool3d';
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = "y_{k,i,j,c,n} = \max_{(d,p,q)\,\in\,\mathcal{W}_{k,i,j}} x_{d,p,q,c,n}";
            meta.Symbols = containers.Map();
            meta.Symbols('y_{k,i,j,c,n}') = "pooled output at spatial position (k,i,j), channel c, batch sample n";
            meta.Symbols('\mathcal{W}_{k,i,j}') = sprintf('%dx%dx%d sliding window at stride [%d %d %d], padding [%d %d %d]', ...
                obj.PoolSize(1), obj.PoolSize(2), obj.PoolSize(3), ...
                obj.Stride(1), obj.Stride(2), obj.Stride(3), ...
                obj.Padding(1), obj.Padding(2), obj.Padding(3));
            meta.Symbols('x_{d,p,q,c,n}') = "input voxel at spatial position (d,p,q), channel c, batch sample n";
            meta.Assumptions = {'padding is -Inf (never wins the max), not zero', ...
                                 'pooling is per-channel, per-sample -- no cross-channel mixing', ...
                                 'gradient of a tie among equal max values routes to the FIRST index (MATLAB max() convention)'};
            meta.IsSymbolic = true;
            meta.Reference = "LeCun et al. (1998); extended to 3-D for volumetric/video CNNs (e.g. C3D, Tran et al., 2015)";
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.Inputs = struct('Name', "X", 'Domain', "volumetric-tensor-batch", 'Dtype', "double", 'Shape', "D x H x W x C x N");
            spec.Outputs = struct('Name', "Y", 'Domain', "volumetric-tensor-batch", 'Dtype', "double", ...
                'Shape', sprintf('floor((D+2*%d-%d)/%d)+1 x floor((H+2*%d-%d)/%d)+1 x floor((W+2*%d-%d)/%d)+1 x C x N', ...
                    obj.Padding(1), obj.PoolSize(1), obj.Stride(1), ...
                    obj.Padding(2), obj.PoolSize(2), obj.Stride(2), ...
                    obj.Padding(3), obj.PoolSize(3), obj.Stride(3)));
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "maxpool3d");
            spec.ComposesFrom = {};
        end
    end
end
