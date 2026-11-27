classdef IVFPQIndex < OceanAlgo.Search.IVectorIndex
    %Inverted File Product Quantization Index IVFPQINDEX  Generic inverted-file index with product-quantized residuals.
    properties (SetAccess=private)
        CoarseCentroids double=zeros(0,0)
        ListIds uint32=zeros(0,1,'uint32')
        Codes uint32=zeros(0,0,'uint32')
        ProductQuantizer
        CoarsePartitioner (1,1) OceanAlgo.Search.ICoarsePartitioner=OceanAlgo.Search.KMeansCoarsePartitioner()
        NumLists (1,1) double
        NumProbe (1,1) double
        IsFitted (1,1) logical=false
    end
    methods
        function obj=IVFPQIndex(options)
            arguments
                options.NumLists (1,1) double {mustBeInteger,mustBePositive}=64
                options.NumProbe (1,1) double {mustBeInteger,mustBePositive}=4
                options.ProductQuantizer (1,1) OceanAlgo.Compression.ProductQuantizer=OceanAlgo.Compression.ProductQuantizer()
                options.CoarsePartitioner (1,1) OceanAlgo.Search.ICoarsePartitioner=OceanAlgo.Search.KMeansCoarsePartitioner()
            end
            obj.NumLists=options.NumLists; obj.NumProbe=options.NumProbe; obj.ProductQuantizer=options.ProductQuantizer; obj.CoarsePartitioner=options.CoarsePartitioner;
        end
        function fit(obj,X)
            arguments
                obj
                X (:,:) double
            end

            %Full-Space
            [n,d]=size(X); if n<obj.NumLists, error('OceanAlgo:IVFPQIndex:FewSamples','Need at least NumLists vectors.'); end
            numLists=min(obj.NumLists,n); 
            obj.CoarsePartitioner.fit(X,numLists); %Run (kmeans)
            obj.CoarseCentroids=obj.CoarsePartitioner.getCentroids(); % K x D centroids
            list=obj.CoarsePartitioner.assign(X); obj.ListIds=uint32(list);

            %Compression
            residuals=X-obj.CoarseCentroids(list,:); 
            obj.ProductQuantizer.fit(residuals);
            obj.Codes=obj.ProductQuantizer.encode(residuals); 
            obj.IsFitted=true;
        end
        function [ids,scores]=search(obj,Q,k,options)
            arguments
                obj
                Q (:,:) double
                k (1,1) double {mustBeInteger,mustBePositive}
                options.NumProbe (1,1) double {mustBeInteger,mustBePositive}=obj.NumProbe
            end
            if ~obj.IsFitted, error('OceanAlgo:IVFPQIndex:NotFitted','Call fit first.'); end
            k=min(k,numel(obj.ListIds)); ids=zeros(size(Q,1),k); scores=-inf(size(Q,1),k);

            for r=1:size(Q,1)
                [~,co]=sort(obj.dist_(Q(r,:),obj.CoarseCentroids),'ascend'); 
                probes=co(1:min(options.NumProbe,numel(co))); %Search only coast lists
                mask=ismember(double(obj.ListIds),probes);
                cand=find(mask); 
                %centroid that was subtracted from each candidate vector during compression
                base=obj.CoarseCentroids(double(obj.ListIds(cand)),:); 
                rq=Q(r,:)-base; %Query to residual
                d=obj.ProductQuantizer.approximateDistances(rq,obj.Codes(cand,:)); 
                d=d(1,:); [sd,o]=sort(d,'ascend'); 
                take=min(k,numel(o)); 
                ids(r,1:take)=cand(o(1:take)); 
                scores(r,1:take)=-sd(1:take); %negative because we want to maximize the score which is negative distance
            end
        end
        function meta=getMetadata(obj)
            meta=struct('Name','IVFPQIndex','IsApproximate',true,'NumLists',obj.NumLists,'NumProbe',obj.NumProbe,'Components',{{class(obj.CoarsePartitioner),'ProductQuantizer'}},'CoarsePartitioner',obj.CoarsePartitioner.getMetadata());
        end
    end
    methods (Access=private)
        function D=dist_(~,A,B), D=sum(A.^2,2)+sum(B.^2,2).'-2*A*B.'; end
    end
end
