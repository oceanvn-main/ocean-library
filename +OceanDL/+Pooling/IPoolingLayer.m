classdef (Abstract) IPoolingLayer < handle
    % IPOOLINGLAYER  Abstract interface for spatial pooling layers.
    %
    %   Implements the mandatory 3-Pillar Component Contract O = (phi, f, D)
    %   from .agents/pipeline-architecture-spec.md:
    %     Pillar 1 (phi) : Closed-form LaTeX formula, Symbol dict (getMetadata)
    %     Pillar 2 (f)   : Numeric forward/backward pass (forward, backward)
    %     Pillar 3 (D)   : PortSpec signature, DiagramSymbol (getPipelineSpec)
    %
    %   TENSOR CONVENTION (matches MATLAB Deep Learning Toolbox dlarray SSCB,
    %   NOT the (d x N) column-major convention used by +OceanDL's
    %   dense layers -- pooling is spatial, dense layers are not):
    %     X — (H x W x C x N) double.  H,W = spatial height/width, C = channels,
    %         N = batch size. A single sample is simply N = 1.
    %     Y — (Ho x Wo x C x N), Ho/Wo computed from PoolSize/Stride/Padding.
    %
    %   STATEFUL FORWARD/BACKWARD PAIRING (buffer-on-self, PyTorch-style):
    %     forward(X) caches whatever backward(dY) needs (e.g. MaxPool2D caches
    %     the argmax "switches") directly on the object as a private buffer --
    %     never returned as a separate state struct. backward() must be called
    %     with the SAME object, after the matching forward() call, exactly once
    %     per forward (mirrors a single autograd step, not a general tape).
    %
    %   USAGE:
    %     pool = OceanDL.Pooling.MaxPool2D([2 2]);
    %     Y    = pool.forward(X);     % X: (H x W x C x N)
    %     dX   = pool.backward(dY);   % dY: (Ho x Wo x C x N), same shape as Y
    %
    %   See also: OceanDL.Pooling.MaxPool2D
    %             OceanDL.Pooling.AvgPool2D
    %             OceanDL.Pooling.MaxUnpool2D

    properties (SetAccess = protected)
        PoolSize (1,2) double {mustBePositive, mustBeInteger} = [2 2]
        Stride   (1,2) double {mustBePositive, mustBeInteger} = [2 2]
        Padding  (1,2) double {mustBeNonnegative, mustBeInteger} = [0 0]
    end

    methods (Abstract)
        Y = forward(obj, X)
        % FORWARD  Pool X: (H x W x C x N) -> Y: (Ho x Wo x C x N).
        %   Caches whatever backward() needs as a private buffer on obj.

        dX = backward(obj, dY)
        % BACKWARD  Route upstream gradient dY: (Ho x Wo x C x N) back to
        %   dX: (H x W x C x N), using the buffer cached by the most recent
        %   forward() call.

        meta = getMetadata(obj)
        % GETMETADATA  Pillar 1: SymbolicLatex, Symbols, Assumptions, Reference.

        spec = getPipelineSpec(obj)
        % GETPIPELINESPEC  Pillar 3: PortSpec Inputs/Outputs, DiagramSymbol.

        s = name(obj)
        % NAME  Lowercase string identifier, e.g. 'maxpool2d'.
    end

    methods (Access = protected)
        function [Ho, Wo] = outputSize(obj, H, W)
            % OUTPUTSIZE  Standard convolution/pooling output-size formula.
            %   Ho = floor((H + 2*padH - poolH) / strideH) + 1, likewise Wo.
            Ho = floor((H + 2*obj.Padding(1) - obj.PoolSize(1)) / obj.Stride(1)) + 1;
            Wo = floor((W + 2*obj.Padding(2) - obj.PoolSize(2)) / obj.Stride(2)) + 1;
            if Ho < 1 || Wo < 1
                error('OceanDL:Pooling:IPoolingLayer:InputTooSmall', ...
                    'Input (%d x %d) with Padding=[%d %d] is too small for PoolSize=[%d %d].', ...
                    H, W, obj.Padding(1), obj.Padding(2), obj.PoolSize(1), obj.PoolSize(2));
            end
        end

        function [rows, cols] = windowIndices(obj, i, j)
            % WINDOWINDICES  Padded-input (row, col) index ranges covered by
            %   output position (i, j), for i = 1..Ho, j = 1..Wo.
            rowStart = (i - 1) * obj.Stride(1) + 1;
            colStart = (j - 1) * obj.Stride(2) + 1;
            rows = rowStart : rowStart + obj.PoolSize(1) - 1;
            cols = colStart : colStart + obj.PoolSize(2) - 1;
        end

    end

    methods (Static, Access = protected)
        function pv = normalizePair(v, fieldName)
            % NORMALIZEPAIR  Accept a scalar or a 2-vector and return [a b].
            arguments
                v (1,:) double
                fieldName (1,1) string = "value"
            end
            if isscalar(v)
                pv = [v v];
            elseif numel(v) == 2
                pv = v;
            else
                error('OceanDL:Pooling:IPoolingLayer:BadPairSize', ...
                    '%s must be a scalar or a 1x2 vector, got %d elements.', fieldName, numel(v));
            end
        end
    end
end
