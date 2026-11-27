classdef MaxUnpool1D < handle
    % MAXUNPOOL1D  1-D max unpooling -- the SegNet-style inverse of MaxPool1D.
    %   Same design as MaxUnpool2D (see its header comment for the full
    %   theory/rationale), one axis instead of two.
    %
    % USAGE:
    %   pool   = OceanDL.Pooling.MaxPool1D(2);
    %   Y      = pool.forward(X);                 % X: (L x C x N)
    %   sw     = pool.getSwitches();
    %   unpool = OceanDL.Pooling.MaxUnpool1D();
    %   Xhat   = unpool.forward(Y, sw, pool.getInputSize());   % (L x C x N), sparse
    %   dY     = unpool.backward(dXhat);
    %
    % See also: OceanDL.Pooling.MaxPool1D
    %           OceanDL.Pooling.MaxUnpool2D

    properties (Access = private)
        Switches_   = []   % (Lo x C x N) linear index into (L), cached by the last forward()
        OutputSize_ = []   % [L C N] target size passed to the last forward()
        PoolShape_  = []   % Lo, length of the pooled input to the last forward()
    end

    methods
        function Y = forward(obj, X, switches, outputSize)
            arguments
                obj
                X (:,:,:) double
                switches (:,:,:) double {mustBeInteger, mustBePositive}
                outputSize (1,:) double {mustBeInteger, mustBePositive}
            end
            [Lo, C, N] = size(X);
            [Ls, Cs, Ns] = size(switches);
            if ~isequal([Ls, Cs, Ns], [Lo, C, N])
                error('OceanDL:Pooling:MaxUnpool1D:SwitchesShapeMismatch', ...
                    'switches must be the same size as X (%s), got (%s).', ...
                    mat2str([Lo C N]), mat2str([Ls Cs Ns]));
            end
            if isempty(outputSize)
                error('OceanDL:Pooling:MaxUnpool1D:BadOutputSize', ...
                    'outputSize must have at least [L], got 0 elements.');
            end
            L = outputSize(1);

            Y = zeros(L, C, N);
            for n = 1:N
                for c = 1:C
                    idx  = switches(:, c, n);
                    vals = X(:, c, n);
                    Y(:, c, n) = accumarray(idx(:), vals(:), [L, 1]);
                end
            end

            obj.Switches_   = switches;
            obj.OutputSize_ = [L, C, N];
            obj.PoolShape_  = Lo;
        end

        function dX = backward(obj, dY)
            arguments
                obj
                dY (:,:,:) double
            end
            if isempty(obj.Switches_)
                error('OceanDL:Pooling:MaxUnpool1D:NoForwardCache', ...
                    'backward() called before forward() -- no cached switches.');
            end
            [Ld, Cd, Nd] = size(dY);
            if ~isequal([Ld, Cd, Nd], obj.OutputSize_)
                error('OceanDL:Pooling:MaxUnpool1D:GradShapeMismatch', ...
                    'dY must be the same size as the last forward() output (%s), got (%s).', ...
                    mat2str(obj.OutputSize_), mat2str([Ld Cd Nd]));
            end
            Lo = obj.PoolShape_;
            C = obj.OutputSize_(2); N = obj.OutputSize_(3);

            dX = zeros(Lo, C, N);
            for n = 1:N
                for c = 1:C
                    idx = obj.Switches_(:, c, n);
                    dYslice = dY(:, c, n);
                    dX(:, c, n) = dYslice(idx(:));
                end
            end
        end

        function s = name(~)
            s = 'maxunpool1d';
        end

        function meta = getMetadata(~)
            meta = struct();
            meta.SymbolicLatex = "y_{\mathrm{switches}(i,c,n)} = x_{i,c,n} \quad\text{(else } y = 0\text{)}";
            meta.Symbols = containers.Map();
            meta.Symbols('x_{i,c,n}') = "pooled-tensor value at sequence position i, channel c, batch sample n";
            meta.Symbols('\mathrm{switches}(i,c,n)') = "linear (L) index of the input element that won that pooling window, from a paired MaxPool1D.getSwitches()";
            meta.Symbols('y') = "the reconstructed (L x C x N) sparse tensor, zero everywhere except at the switches";
            meta.Assumptions = {'switches must come from a MaxPool1D call over the SAME length this unpools back to', ...
                                 'colliding switches (possible when the paired pooling used overlapping windows) are summed'};
            meta.IsSymbolic = true;
            meta.Reference = "Badrinarayanan, Kendall, & Cipolla (2017), SegNet: A Deep Convolutional Encoder-Decoder Architecture";
        end

        function spec = getPipelineSpec(~)
            spec = struct();
            spec.Inputs = [ ...
                struct('Name', "X", 'Domain', "sequence-tensor-batch", 'Dtype', "double", 'Shape', "Lo x C x N"), ...
                struct('Name', "switches", 'Domain', "sequence-index-map", 'Dtype', "double", 'Shape', "Lo x C x N"), ...
                struct('Name', "outputSize", 'Domain', "shape-vector", 'Dtype', "double", 'Shape', "1 x 1 (or 1 x 3)")];
            spec.Outputs = struct('Name', "Y", 'Domain', "sequence-tensor-batch", 'Dtype', "double", 'Shape', "L x C x N");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "maxunpool1d");
            spec.ComposesFrom = {};
        end
    end
end
