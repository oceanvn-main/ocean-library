classdef MaxUnpool3D < handle
    % MAXUNPOOL3D  3-D max unpooling -- the SegNet-style inverse of MaxPool3D.
    %   Same design as MaxUnpool2D (see its header comment for the full
    %   theory/rationale), one more axis.
    %
    % USAGE:
    %   pool   = OceanDL.Pooling.MaxPool3D([2 2 2]);
    %   Y      = pool.forward(X);                 % X: (D x H x W x C x N)
    %   sw     = pool.getSwitches();
    %   unpool = OceanDL.Pooling.MaxUnpool3D();
    %   Xhat   = unpool.forward(Y, sw, pool.getInputSize());   % (D x H x W x C x N), sparse
    %   dY     = unpool.backward(dXhat);
    %
    % See also: OceanDL.Pooling.MaxPool3D
    %           OceanDL.Pooling.MaxUnpool2D

    properties (Access = private)
        Switches_   = []   % (Do x Ho x Wo x C x N) linear index into (D x H x W), cached by the last forward()
        OutputSize_ = []   % [D H W C N] target size passed to the last forward()
        PoolShape_  = []   % [Do Ho Wo] shape of the pooled input to the last forward()
    end

    methods
        function Y = forward(obj, X, switches, outputSize)
            arguments
                obj
                X (:,:,:,:,:) double
                switches (:,:,:,:,:) double {mustBeInteger, mustBePositive}
                outputSize (1,:) double {mustBeInteger, mustBePositive}
            end
            [Do, Ho, Wo, C, N] = size(X);
            [Ds, Hs, Ws, Cs, Ns] = size(switches);
            if ~isequal([Ds, Hs, Ws, Cs, Ns], [Do, Ho, Wo, C, N])
                error('OceanDL:Pooling:MaxUnpool3D:SwitchesShapeMismatch', ...
                    'switches must be the same size as X (%s), got (%s).', ...
                    mat2str([Do Ho Wo C N]), mat2str([Ds Hs Ws Cs Ns]));
            end
            if numel(outputSize) < 3
                error('OceanDL:Pooling:MaxUnpool3D:BadOutputSize', ...
                    'outputSize must have at least [D H W], got %d element(s).', numel(outputSize));
            end
            D = outputSize(1); H = outputSize(2); W = outputSize(3);

            Y = zeros(D, H, W, C, N);
            for n = 1:N
                for c = 1:C
                    idx  = switches(:, :, :, c, n);
                    vals = X(:, :, :, c, n);
                    Yslice = accumarray(idx(:), vals(:), [D*H*W, 1]);
                    Y(:, :, :, c, n) = reshape(Yslice, D, H, W);
                end
            end

            obj.Switches_   = switches;
            obj.OutputSize_ = [D, H, W, C, N];
            obj.PoolShape_  = [Do, Ho, Wo];
        end

        function dX = backward(obj, dY)
            arguments
                obj
                dY (:,:,:,:,:) double
            end
            if isempty(obj.Switches_)
                error('OceanDL:Pooling:MaxUnpool3D:NoForwardCache', ...
                    'backward() called before forward() -- no cached switches.');
            end
            [Hd1, Hd2, Hd3, Cd, Nd] = size(dY);
            if ~isequal([Hd1, Hd2, Hd3, Cd, Nd], obj.OutputSize_)
                error('OceanDL:Pooling:MaxUnpool3D:GradShapeMismatch', ...
                    'dY must be the same size as the last forward() output (%s), got (%s).', ...
                    mat2str(obj.OutputSize_), mat2str([Hd1 Hd2 Hd3 Cd Nd]));
            end
            Do = obj.PoolShape_(1); Ho = obj.PoolShape_(2); Wo = obj.PoolShape_(3);
            C = obj.OutputSize_(4); N = obj.OutputSize_(5);

            dX = zeros(Do, Ho, Wo, C, N);
            for n = 1:N
                for c = 1:C
                    idx = obj.Switches_(:, :, :, c, n);
                    dYslice = dY(:, :, :, c, n);
                    dX(:, :, :, c, n) = reshape(dYslice(idx(:)), Do, Ho, Wo);
                end
            end
        end

        function s = name(~)
            s = 'maxunpool3d';
        end

        function meta = getMetadata(~)
            meta = struct();
            meta.SymbolicLatex = "y_{\mathrm{switches}(k,i,j,c,n)} = x_{k,i,j,c,n} \quad\text{(else } y = 0\text{)}";
            meta.Symbols = containers.Map();
            meta.Symbols('x_{k,i,j,c,n}') = "pooled-tensor value at spatial position (k,i,j), channel c, batch sample n";
            meta.Symbols('\mathrm{switches}(k,i,j,c,n)') = "linear (D x H x W) index of the input voxel that won that pooling window, from a paired MaxPool3D.getSwitches()";
            meta.Symbols('y') = "the reconstructed (D x H x W x C x N) sparse tensor, zero everywhere except at the switches";
            meta.Assumptions = {'switches must come from a MaxPool3D call over the SAME spatial size this unpools back to', ...
                                 'colliding switches (possible when the paired pooling used overlapping windows) are summed'};
            meta.IsSymbolic = true;
            meta.Reference = "Badrinarayanan, Kendall, & Cipolla (2017), SegNet: A Deep Convolutional Encoder-Decoder Architecture";
        end

        function spec = getPipelineSpec(~)
            spec = struct();
            spec.Inputs = [ ...
                struct('Name', "X", 'Domain', "volumetric-tensor-batch", 'Dtype', "double", 'Shape', "Do x Ho x Wo x C x N"), ...
                struct('Name', "switches", 'Domain', "volumetric-index-map", 'Dtype', "double", 'Shape', "Do x Ho x Wo x C x N"), ...
                struct('Name', "outputSize", 'Domain', "shape-vector", 'Dtype', "double", 'Shape', "1 x 3 (or 1 x 5)")];
            spec.Outputs = struct('Name', "Y", 'Domain', "volumetric-tensor-batch", 'Dtype', "double", 'Shape', "D x H x W x C x N");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "maxunpool3d");
            spec.ComposesFrom = {};
        end
    end
end
