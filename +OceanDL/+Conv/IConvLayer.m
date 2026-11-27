classdef (Abstract) IConvLayer < handle
    % ICONVLAYER  Abstract interface for spatial convolution layers.
    %
    %   Implements the mandatory 3-Pillar Component Contract O = (phi, f, D)
    %   from .agents/pipeline-architecture-spec.md (see IPoolingLayer.m for
    %   the identical contract applied to pooling).
    %
    %   TENSOR CONVENTION (matches OceanDL.Pooling exactly):
    %     X — (H x W x Cin x N) double.
    %     Y — (Ho x Wo x Cout x N), Ho/Wo from KernelSize/Stride/Padding.
    %     Weight — (KH x KW x Cin x Cout).
    %     Bias   — (Cout x 1), only when HasBias.
    %
    %   STATEFUL FORWARD/BACKWARD PAIRING (buffer-on-self, PyTorch-style):
    %     forward(X) caches whatever backward(dY) needs directly on the
    %     object -- never returned as a separate state struct. backward()
    %     must be called with the SAME object, after the matching forward()
    %     call, exactly once per forward.
    %
    %   USAGE:
    %     conv = OceanDL.Conv.Conv2D(3, 8, 'Padding', [1 1]);
    %     Y  = conv.forward(X);      % X: (H x W x Cin x N)
    %     dX = conv.backward(dY);    % dY: (Ho x Wo x Cout x N), same shape as Y
    %
    %   See also: OceanDL.Conv.Conv2D
    %             OceanDL.Pooling.IPoolingLayer

    properties (SetAccess = protected)
        KernelSize (1,2) double {mustBePositive, mustBeInteger} = [3 3]
        Stride     (1,2) double {mustBePositive, mustBeInteger} = [1 1]
        Padding    (1,2) double {mustBeNonnegative, mustBeInteger} = [0 0]
        InChannels  (1,1) double {mustBeNonnegative, mustBeInteger} = 1
        % NOTE: 0 is a valid sentinel meaning "not yet known -- inferred
        % from the first forward() call" (see Conv2D's deferred-InChannels
        % constructor path).
        OutChannels (1,1) double {mustBePositive, mustBeInteger} = 1
    end

    methods (Abstract)
        Y = forward(obj, X)
        % FORWARD  Convolve X: (H x W x Cin x N) -> Y: (Ho x Wo x Cout x N).
        %   Caches whatever backward() needs as a private buffer on obj.

        dX = backward(obj, dY)
        % BACKWARD  Route upstream gradient dY: (Ho x Wo x Cout x N) back to
        %   dX: (H x W x Cin x N), and cache dWeight/dBias (see getGrad*),
        %   using the buffer cached by the most recent forward() call.

        meta = getMetadata(obj)
        % GETMETADATA  Pillar 1: SymbolicLatex, Symbols, Assumptions, Reference.

        spec = getPipelineSpec(obj)
        % GETPIPELINESPEC  Pillar 3: PortSpec Inputs/Outputs, DiagramSymbol.

        s = name(obj)
        % NAME  Lowercase string identifier, e.g. 'conv2d'.
    end

    methods (Access = protected)
        function [Ho, Wo] = outputSize(obj, H, W)
            % OUTPUTSIZE  Standard convolution output-size formula.
            Ho = floor((H + 2*obj.Padding(1) - obj.KernelSize(1)) / obj.Stride(1)) + 1;
            Wo = floor((W + 2*obj.Padding(2) - obj.KernelSize(2)) / obj.Stride(2)) + 1;
            if Ho < 1 || Wo < 1
                error('OceanDL:Conv:IConvLayer:InputTooSmall', ...
                    'Input (%d x %d) with Padding=[%d %d] is too small for KernelSize=[%d %d].', ...
                    H, W, obj.Padding(1), obj.Padding(2), obj.KernelSize(1), obj.KernelSize(2));
            end
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
                error('OceanDL:Conv:IConvLayer:BadPairSize', ...
                    '%s must be a scalar or a 1x2 vector, got %d elements.', fieldName, numel(v));
            end
        end
    end
end
