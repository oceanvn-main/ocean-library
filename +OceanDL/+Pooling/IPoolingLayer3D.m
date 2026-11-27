classdef (Abstract) IPoolingLayer3D < handle
    % IPOOLINGLAYER3D  Abstract interface for 3-D (volumetric/video) pooling layers.
    %
    %   Sibling of IPoolingLayer (2-D) and IPoolingLayer1D -- kept separate
    %   for the same reason as IPoolingLayer1D (see its header comment):
    %   the window loop nesting genuinely differs per rank.
    %
    %   TENSOR CONVENTION (matches MATLAB Deep Learning Toolbox's 3-D
    %   dlarray convention, "SSSCB" = Spatial,Spatial,Spatial,Channel,Batch):
    %     X — (D x H x W x C x N) double.  D,H,W = spatial depth/height/width,
    %         C = channels, N = batch size. A single sample is simply N = 1.
    %     Y — (Do x Ho x Wo x C x N), each output dim from PoolSize/Stride/Padding.
    %
    %   See also: OceanDL.Pooling.IPoolingLayer (2-D)
    %             OceanDL.Pooling.IPoolingLayer1D
    %             OceanDL.Pooling.MaxPool3D
    %             OceanDL.Pooling.AvgPool3D
    %             OceanDL.Pooling.MaxUnpool3D

    properties (SetAccess = protected)
        PoolSize (1,3) double {mustBePositive, mustBeInteger} = [2 2 2]
        Stride   (1,3) double {mustBePositive, mustBeInteger} = [2 2 2]
        Padding  (1,3) double {mustBeNonnegative, mustBeInteger} = [0 0 0]
    end

    methods (Abstract)
        Y = forward(obj, X)
        dX = backward(obj, dY)
        meta = getMetadata(obj)
        spec = getPipelineSpec(obj)
        s = name(obj)
    end

    methods (Access = protected)
        function [Do, Ho, Wo] = outputSize(obj, D, H, W)
            % OUTPUTSIZE  Standard convolution/pooling output-size formula, per axis.
            Do = floor((D + 2*obj.Padding(1) - obj.PoolSize(1)) / obj.Stride(1)) + 1;
            Ho = floor((H + 2*obj.Padding(2) - obj.PoolSize(2)) / obj.Stride(2)) + 1;
            Wo = floor((W + 2*obj.Padding(3) - obj.PoolSize(3)) / obj.Stride(3)) + 1;
            if Do < 1 || Ho < 1 || Wo < 1
                error('OceanDL:Pooling:IPoolingLayer3D:InputTooSmall', ...
                    'Input (%d x %d x %d) with Padding=[%d %d %d] is too small for PoolSize=[%d %d %d].', ...
                    D, H, W, obj.Padding(1), obj.Padding(2), obj.Padding(3), ...
                    obj.PoolSize(1), obj.PoolSize(2), obj.PoolSize(3));
            end
        end

        function [ds, hs, ws] = windowIndices(obj, k, i, j)
            % WINDOWINDICES  Padded-input (depth,row,col) index ranges
            %   covered by output position (k, i, j), for k=1..Do, i=1..Ho, j=1..Wo.
            dStart = (k - 1) * obj.Stride(1) + 1;
            hStart = (i - 1) * obj.Stride(2) + 1;
            wStart = (j - 1) * obj.Stride(3) + 1;
            ds = dStart : dStart + obj.PoolSize(1) - 1;
            hs = hStart : hStart + obj.PoolSize(2) - 1;
            ws = wStart : wStart + obj.PoolSize(3) - 1;
        end
    end

    methods (Static, Access = protected)
        function pv = normalizeTriple(v, fieldName)
            % NORMALIZETRIPLE  Accept a scalar or a 3-vector and return [a b c].
            arguments
                v (1,:) double
                fieldName (1,1) string = "value"
            end
            if isscalar(v)
                pv = [v v v];
            elseif numel(v) == 3
                pv = v;
            else
                error('OceanDL:Pooling:IPoolingLayer3D:BadTripleSize', ...
                    '%s must be a scalar or a 1x3 vector, got %d elements.', fieldName, numel(v));
            end
        end
    end
end
