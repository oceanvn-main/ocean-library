classdef MaxUnpool2D < handle
    % MAXUNPOOL2D  2-D max UNpooling -- the SegNet-style inverse of MaxPool2D
    %   (Badrinarayanan, Kendall, & Cipolla, 2017).
    %
    % THEORY:
    %   MaxPool2D discards, for every pooling window, everything except the
    %   winning value AND (via getSwitches()) the winner's original spatial
    %   location. MaxUnpool2D re-inserts a smaller tensor back into a larger
    %   spatial grid at EXACTLY those remembered locations, leaving every
    %   other position zero -- it is a scatter, not a learned upsampling
    %   (contrast with a transposed/"deconvolution" layer, which has learned
    %   weights; unpooling has none):
    %
    %     y_{switches(i,j,c,n)} = x_{i,j,c,n}     (all other positions of y = 0)
    %
    %   This is NOT a general-purpose module on its own -- it must be paired
    %   with the SAME switches a MaxPool2D produced (typically the mirrored
    %   encoder layer in an encoder-decoder / autoencoder architecture), which
    %   is why forward() takes `switches` and `outputSize` as EXPLICIT
    %   arguments rather than reading another object's hidden state (PyTorch-
    %   style functional core: the dependency on a specific MaxPool2D's
    %   switches is visible at the call site, never implicit).
    %
    %   COLLISION HANDLING: if a Stride < PoolSize MaxPool2D produced switches
    %   where more than one output position maps to the SAME input pixel
    %   (possible with overlapping windows), forward() here SUMS the
    %   colliding values via accumarray (the mathematically consistent choice
    %   -- it is also exactly what backward() of MaxPool2D itself does when
    %   accumulating gradients through overlapping windows).
    %
    % USAGE:
    %   pool   = OceanDL.Pooling.MaxPool2D([2 2]);
    %   Y      = pool.forward(X);                 % X: (H x W x C x N)
    %   sw     = pool.getSwitches();
    %   unpool = OceanDL.Pooling.MaxUnpool2D();
    %   Xhat   = unpool.forward(Y, sw, pool.getInputSize());   % (H x W x C x N), sparse
    %   dY     = unpool.backward(dXhat);          % gather, inverse of forward's scatter
    %
    % See also: OceanDL.Pooling.MaxPool2D
    %           OceanDL.Pooling.AvgPool2D
    %
    % Implements the 3-pillar contract (.agents/pipeline-architecture-spec.md
    % §1.1) via getMetadata() and getPipelineSpec(), in addition to forward()/backward().

    properties (Access = private)
        Switches_   = []   % (Ho x Wo x C x N) linear index into (H x W), cached by the last forward()
        OutputSize_ = []   % [H W C N] target size passed to the last forward()
        PoolShape_  = []   % [Ho Wo] shape of the pooled input to the last forward()
    end

    methods
        function Y = forward(obj, X, switches, outputSize)
            % FORWARD  Scatter X: (Ho x Wo x C x N) into Y: (H x W x C x N)
            %   at the spatial locations recorded in `switches` (as produced
            %   by MaxPool2D.getSwitches()). Positions not hit by any switch
            %   are zero.
            %
            %   X          — (Ho x Wo x C x N)
            %   switches   — (Ho x Wo x C x N) linear index into (H x W),
            %                e.g. pool.getSwitches()
            %   outputSize — [H W] or [H W C N] (C, N default to X's own),
            %                e.g. pool.getInputSize()
            arguments
                obj
                X (:,:,:,:) double
                switches (:,:,:,:) double {mustBeInteger, mustBePositive}
                outputSize (1,:) double {mustBeInteger, mustBePositive}
            end
            [Ho, Wo, C, N] = size(X);
            % Compare dimension-by-dimension (not whole size() vectors) --
            % MATLAB silently drops TRAILING singleton dims, so a (2x2x1x1)
            % and a (2x2) array have DIFFERENT-LENGTH size() vectors despite
            % being the same logical shape; isequal(size(a),size(b)) would
            % wrongly reject that as a mismatch.
            [Hs, Ws, Cs, Ns] = size(switches);
            if ~isequal([Hs, Ws, Cs, Ns], [Ho, Wo, C, N])
                error('OceanDL:Pooling:MaxUnpool2D:SwitchesShapeMismatch', ...
                    'switches must be the same size as X (%s), got (%s).', ...
                    mat2str([Ho Wo C N]), mat2str([Hs Ws Cs Ns]));
            end
            if numel(outputSize) < 2
                error('OceanDL:Pooling:MaxUnpool2D:BadOutputSize', ...
                    'outputSize must have at least [H W], got %d element(s).', numel(outputSize));
            end
            H = outputSize(1); W = outputSize(2);

            Y = zeros(H, W, C, N);
            for n = 1:N
                for c = 1:C
                    idx  = switches(:, :, c, n);
                    vals = X(:, :, c, n);
                    Yslice = accumarray(idx(:), vals(:), [H*W, 1]);
                    Y(:, :, c, n) = reshape(Yslice, H, W);
                end
            end

            obj.Switches_   = switches;
            obj.OutputSize_ = [H, W, C, N];
            obj.PoolShape_  = [Ho, Wo];
        end

        function dX = backward(obj, dY)
            % BACKWARD  Gather dY: (H x W x C x N) back down to dX: (Ho x Wo
            %   x C x N) at the switches cached by the most recent forward()
            %   call -- the exact inverse operation of forward()'s scatter
            %   (a pure lookup, needs no accumulation).
            arguments
                obj
                dY (:,:,:,:) double
            end
            if isempty(obj.Switches_)
                error('OceanDL:Pooling:MaxUnpool2D:NoForwardCache', ...
                    'backward() called before forward() -- no cached switches.');
            end
            [Hd, Wd, Cd, Nd] = size(dY);
            if ~isequal([Hd, Wd, Cd, Nd], obj.OutputSize_)
                error('OceanDL:Pooling:MaxUnpool2D:GradShapeMismatch', ...
                    'dY must be the same size as the last forward() output (%s), got (%s).', ...
                    mat2str(obj.OutputSize_), mat2str([Hd Wd Cd Nd]));
            end
            Ho = obj.PoolShape_(1); Wo = obj.PoolShape_(2);
            C = obj.OutputSize_(3); N = obj.OutputSize_(4);
            H = obj.OutputSize_(1); W = obj.OutputSize_(2);

            dX = zeros(Ho, Wo, C, N);
            for n = 1:N
                for c = 1:C
                    idx = obj.Switches_(:, :, c, n);
                    dYslice = dY(:, :, c, n);
                    dX(:, :, c, n) = reshape(dYslice(idx(:)), Ho, Wo);
                end
            end
        end

        function s = name(~)
            s = 'maxunpool2d';
        end

        function meta = getMetadata(~)
            meta = struct();
            meta.SymbolicLatex = "y_{\mathrm{switches}(i,j,c,n)} = x_{i,j,c,n} \quad\text{(else } y = 0\text{)}";
            meta.Symbols = containers.Map();
            meta.Symbols('x_{i,j,c,n}') = "pooled-tensor value at spatial position (i,j), channel c, batch sample n";
            meta.Symbols('\mathrm{switches}(i,j,c,n)') = "linear (H x W) index of the input pixel that won that pooling window, from a paired MaxPool2D.getSwitches()";
            meta.Symbols('y') = "the reconstructed (H x W x C x N) sparse tensor, zero everywhere except at the switches";
            meta.Assumptions = {'switches must come from a MaxPool2D call over the SAME spatial size this unpools back to', ...
                                 'colliding switches (possible when the paired pooling used overlapping windows) are summed'};
            meta.IsSymbolic = true;
            meta.Reference = "Badrinarayanan, Kendall, & Cipolla (2017), SegNet: A Deep Convolutional Encoder-Decoder Architecture";
        end

        function spec = getPipelineSpec(~)
            spec = struct();
            spec.Inputs = [ ...
                struct('Name', "X", 'Domain', "spatial-tensor-batch", 'Dtype', "double", 'Shape', "Ho x Wo x C x N"), ...
                struct('Name', "switches", 'Domain', "spatial-index-map", 'Dtype', "double", 'Shape', "Ho x Wo x C x N"), ...
                struct('Name', "outputSize", 'Domain', "shape-vector", 'Dtype', "double", 'Shape', "1 x 2 (or 1 x 4)")];
            spec.Outputs = struct('Name', "Y", 'Domain', "spatial-tensor-batch", 'Dtype', "double", 'Shape', "H x W x C x N");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "maxunpool");
            spec.ComposesFrom = {};
        end
    end
end
