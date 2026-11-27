classdef (Abstract) IPoolingLayer1D < handle
    % IPOOLINGLAYER1D  Abstract interface for 1-D (sequence) pooling layers.
    %
    %   Sibling of IPoolingLayer (which is 2-D) -- kept as a SEPARATE class
    %   rather than generalized into one N-D interface because the window
    %   loop nesting genuinely differs per rank (single loop here vs. nested
    %   loops in 2-D/3-D), matching how PyTorch itself keeps nn.MaxPool1d/
    %   2d/3d as independent classes rather than one parameterized-by-rank
    %   implementation.
    %
    %   TENSOR CONVENTION:
    %     X — (L x C x N) double.  L = sequence length, C = channels,
    %         N = batch size. A single sample is simply N = 1.
    %     Y — (Lo x C x N), Lo computed from PoolSize/Stride/Padding.
    %
    %   See also: OceanDL.Pooling.IPoolingLayer (2-D)
    %             OceanDL.Pooling.IPoolingLayer3D
    %             OceanDL.Pooling.MaxPool1D
    %             OceanDL.Pooling.AvgPool1D
    %             OceanDL.Pooling.MaxUnpool1D

    properties (SetAccess = protected)
        PoolSize (1,1) double {mustBePositive, mustBeInteger} = 2
        Stride   (1,1) double {mustBePositive, mustBeInteger} = 2
        Padding  (1,1) double {mustBeNonnegative, mustBeInteger} = 0
    end

    methods (Abstract)
        Y = forward(obj, X)
        dX = backward(obj, dY)
        meta = getMetadata(obj)
        spec = getPipelineSpec(obj)
        s = name(obj)
    end

    methods (Access = protected)
        function Lo = outputSize(obj, L)
            % OUTPUTSIZE  Lo = floor((L + 2*padding - poolSize) / stride) + 1.
            Lo = floor((L + 2*obj.Padding - obj.PoolSize) / obj.Stride) + 1;
            if Lo < 1
                error('OceanDL:Pooling:IPoolingLayer1D:InputTooSmall', ...
                    'Input length %d with Padding=%d is too small for PoolSize=%d.', ...
                    L, obj.Padding, obj.PoolSize);
            end
        end

        function idxs = windowIndices(obj, i)
            % WINDOWINDICES  Padded-input index range covered by output
            %   position i, for i = 1..Lo.
            start = (i - 1) * obj.Stride + 1;
            idxs = start : start + obj.PoolSize - 1;
        end
    end
end
