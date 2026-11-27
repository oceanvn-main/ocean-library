classdef ResidualProductQuantizer < handle
    %RESIDUALPRODUCTQUANTIZER  Product quantization after a global residual.
    properties (SetAccess=private), Center double=zeros(1,0), Quantizer, IsFitted logical=false, end
    methods
        function obj=ResidualProductQuantizer(options)
            arguments
                options.NumSubspaces (1,1) double {mustBeInteger,mustBePositive}=8
                options.NumCentroids (1,1) double {mustBeInteger,mustBePositive}=256
                options.NumIterations (1,1) double {mustBeInteger,mustBePositive}=20
                options.Seed=[]
            end
            obj.Quantizer=OceanAlgo.Compression.ProductQuantizer('NumSubspaces',options.NumSubspaces,'NumCentroids',options.NumCentroids,'NumIterations',options.NumIterations,'Seed',options.Seed);
        end
        function fit(obj,X)
            arguments
                obj
                X (:,:) double
            end
            obj.Center=mean(X,1); obj.Quantizer.fit(X-obj.Center); obj.IsFitted=true;
        end
        function codes=encode(obj,X)
            arguments
                obj
                X (:,:) double
            end
            obj.check_(); codes=obj.Quantizer.encode(X-obj.Center);
        end
        function X=decode(obj,codes)
            arguments
                obj
                codes (:,:) uint32
            end
            obj.check_(); X=obj.Quantizer.decode(codes)+obj.Center;
        end
        function meta=getMetadata(obj), meta=struct('Name','ResidualProductQuantizer','Base','ProductQuantizer','IsFitted',obj.IsFitted); end
    end
    methods (Access=private)
        function check_(obj)
            if ~obj.IsFitted, error('OceanAlgo:ResidualPQ:NotFitted','Call fit first.'); end
        end
    end
end
