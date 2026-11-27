classdef AvgPool3D < OceanDL.Pooling.IPoolingLayer3D
    % AVGPOOL3D  3-D average pooling layer (volumetric/video/medical-imaging CNNs).
    %
    % THEORY:
    %   FORMULA (per output position (k,i,j), channel c, batch sample n):
    %     y_{k,i,j,c,n} = (1/|W_{k,i,j}|) * sum_{(d,p,q) in W_{k,i,j}} x_{d,p,q,c,n}
    %   Zero-padded; CountIncludePad behaves exactly as in AvgPool2D (see
    %   its header comment for the full rationale).
    %
    % USAGE:
    %   pool = OceanDL.Pooling.AvgPool3D([2 2 2]);
    %   Y    = pool.forward(X);      % X: (D x H x W x C x N) -> Y: (Do x Ho x Wo x C x N)
    %   dX   = pool.backward(dY);
    %
    % See also: OceanDL.Pooling.MaxPool3D
    %           OceanDL.Pooling.MaxUnpool3D
    %           OceanDL.Pooling.AvgPool2D

    properties (SetAccess = private)
        CountIncludePad (1,1) logical = true
    end

    properties (Access = private)
        Divisors_  = []   % (Do x Ho x Wo) window divisor, set by forward() (shared across C,N)
        InputSize_ = []   % [D H W C N] of the last forward() call, set by forward()
    end

    methods
        function obj = AvgPool3D(poolSize, opts)
            arguments
                poolSize (1,:) double {mustBePositive, mustBeInteger}
                opts.Stride  (1,:) double {mustBePositive, mustBeInteger} = poolSize
                opts.Padding (1,:) double {mustBeNonnegative, mustBeInteger} = [0 0 0]
                opts.CountIncludePad (1,1) logical = true
            end
            obj.PoolSize = obj.normalizeTriple(poolSize, "PoolSize");
            obj.Stride   = obj.normalizeTriple(opts.Stride, "Stride");
            obj.Padding  = obj.normalizeTriple(opts.Padding, "Padding");
            obj.CountIncludePad = opts.CountIncludePad;
            if any(obj.Padding >= (obj.PoolSize + 1) / 2)
                error('OceanDL:Pooling:AvgPool3D:PaddingTooLarge', ...
                    'Padding [%d %d %d] must be < half of PoolSize [%d %d %d] (else a window could be entirely padding, and CountIncludePad=false would divide by zero).', ...
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
            Xp = zeros(D + 2*pd, H + 2*ph, W + 2*pw, C, N);
            Xp(pd+1:pd+D, ph+1:ph+H, pw+1:pw+W, :, :) = X;
            poolVol = obj.PoolSize(1) * obj.PoolSize(2) * obj.PoolSize(3);

            Y = zeros(Do, Ho, Wo, C, N);
            divisors = zeros(Do, Ho, Wo);
            for k = 1:Do
                for i = 1:Ho
                    for j = 1:Wo
                        [ds, hs, ws] = obj.windowIndices(k, i, j);
                        window = Xp(ds, hs, ws, :, :);
                        Y(k, i, j, :, :) = sum(window, [1 2 3]);

                        if obj.CountIncludePad
                            divisors(k, i, j) = poolVol;
                        else
                            validD = ds > pd & ds <= pd + D;
                            validH = hs > ph & hs <= ph + H;
                            validW = ws > pw & ws <= pw + W;
                            divisors(k, i, j) = nnz(validD) * nnz(validH) * nnz(validW);
                        end
                        Y(k, i, j, :, :) = Y(k, i, j, :, :) / divisors(k, i, j);
                    end
                end
            end
            obj.Divisors_ = divisors;
        end

        function dX = backward(obj, dY)
            arguments
                obj
                dY (:,:,:,:,:) double
            end
            if isempty(obj.Divisors_)
                error('OceanDL:Pooling:AvgPool3D:NoForwardCache', ...
                    'backward() called before forward() -- no cached divisors.');
            end
            D = obj.InputSize_(1); H = obj.InputSize_(2); W = obj.InputSize_(3);
            C = obj.InputSize_(4); N = obj.InputSize_(5);
            pd = obj.Padding(1); ph = obj.Padding(2); pw = obj.Padding(3);
            [Do, Ho, Wo] = size(obj.Divisors_);

            dXp = zeros(D + 2*pd, H + 2*ph, W + 2*pw, C, N);
            for k = 1:Do
                for i = 1:Ho
                    for j = 1:Wo
                        [ds, hs, ws] = obj.windowIndices(k, i, j);
                        contrib = dY(k, i, j, :, :) / obj.Divisors_(k, i, j);   % (1x1x1xCxN), broadcasts over the window
                        dXp(ds, hs, ws, :, :) = dXp(ds, hs, ws, :, :) + contrib;
                    end
                end
            end
            dX = dXp(pd+1:pd+D, ph+1:ph+H, pw+1:pw+W, :, :);
        end

        function s = name(~)
            s = 'avgpool3d';
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = "y_{k,i,j,c,n} = \frac{1}{|\mathcal{W}_{k,i,j}|} \sum_{(d,p,q)\,\in\,\mathcal{W}_{k,i,j}} x_{d,p,q,c,n}";
            meta.Symbols = containers.Map();
            meta.Symbols('y_{k,i,j,c,n}') = "pooled output at spatial position (k,i,j), channel c, batch sample n";
            meta.Symbols('\mathcal{W}_{k,i,j}') = sprintf('%dx%dx%d sliding window at stride [%d %d %d], padding [%d %d %d]', ...
                obj.PoolSize(1), obj.PoolSize(2), obj.PoolSize(3), ...
                obj.Stride(1), obj.Stride(2), obj.Stride(3), ...
                obj.Padding(1), obj.Padding(2), obj.Padding(3));
            if obj.CountIncludePad
                meta.Symbols('|\mathcal{W}_{k,i,j}|') = "PoolSize(1)*PoolSize(2)*PoolSize(3), always (zero-padding counted in the divisor)";
            else
                meta.Symbols('|\mathcal{W}_{k,i,j}|') = "number of REAL (non-padding) cells in the window";
            end
            meta.Assumptions = {'padding is zero-valued', ...
                                 'pooling is per-channel, per-sample -- no cross-channel mixing'};
            meta.IsSymbolic = true;
            meta.Reference = "LeCun, Bottou, Bengio, & Haffner (1998); extended to 3-D for volumetric/video CNNs";
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
            spec.DiagramSymbol = struct('Type', "default", 'Label', "avgpool3d");
            spec.ComposesFrom = {};
        end
    end
end
