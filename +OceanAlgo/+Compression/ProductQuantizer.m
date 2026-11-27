classdef ProductQuantizer < handle
    %PRODUCTQUANTIZER  Generic product quantization for row-vector data.
    %   Splits D-dimensional vectors into M independent subspaces and learns
    %   K centroids per subspace with Lloyd k-means.  The implementation is
    %   domain-neutral: NLP, vision, audio, and scientific vectors use it.
    properties (SetAccess=private)
        NumSubspaces (1,1) double
        NumCentroids (1,1) double
        NumIterations (1,1) double
        Dimension (1,1) double=0
        SubspaceDimension (1,1) double=0
        Codebooks double=zeros(0,0,0) % subspace dimension x K x M
        IsFitted (1,1) logical=false
        Seed
        ReconstructionError (1,1) double=NaN
    end
    methods
        function obj=ProductQuantizer(options)
            arguments
                options.NumSubspaces (1,1) double {mustBeInteger,mustBePositive}=8
                options.NumCentroids (1,1) double {mustBeInteger,mustBePositive}=256
                options.NumIterations (1,1) double {mustBeInteger,mustBePositive}=20
                options.Seed=[]
            end
            obj.NumSubspaces=options.NumSubspaces; obj.NumCentroids=options.NumCentroids;
            obj.NumIterations=options.NumIterations; obj.Seed=options.Seed;
        end
        function fit(obj,X)
            %FIT learns the PQ codebooks from training vector X
            arguments
                obj
                X (:,:) double
            end
            [n,d]=size(X); if n==0, error('OceanAlgo:ProductQuantizer:EmptyData','X must not be empty.'); end
            if mod(d,obj.NumSubspaces)~=0, error('OceanAlgo:ProductQuantizer:NonUniformSubspaces','Dimension must be divisible by NumSubspaces.'); end
            if n<obj.NumCentroids, error('OceanAlgo:ProductQuantizer:FewSamples','Need at least NumCentroids training vectors.'); end
            obj.Dimension=d; obj.SubspaceDimension=d/obj.NumSubspaces;
            obj.Codebooks=zeros(obj.SubspaceDimension,obj.NumCentroids,obj.NumSubspaces);
            stream=obj.makeStream_();
            %Train one-independent quantizer per subspace
            for m=1:obj.NumSubspaces
                cols=(m-1)*obj.SubspaceDimension+(1:obj.SubspaceDimension); A=X(:,cols);
                %Random start
                C=A(randperm(stream,n,obj.NumCentroids),:); % data-derived initialization

                for it=1:obj.NumIterations
                    dist=obj.squaredDistances_(A,C); [~,labels]=min(dist,[],2);
                    for k=1:obj.NumCentroids
                        %Extract
                        members=A(labels==k,:); 
                        if ~isempty(members), 
                        %Update centroids
                        C(k,:)=mean(members,1); end
                    end
                end
                obj.Codebooks(:,:,m)=C.';
            end
            obj.IsFitted=true; codes=obj.encode(X); Xhat=obj.decode(codes);
            obj.ReconstructionError=mean(sum((X-Xhat).^2,2));
        end
        function codes=encode(obj,X)
            arguments
                obj
                X (:,:) double
            end
            obj.checkFitted_(); if size(X,2)~=obj.Dimension, error('OceanAlgo:ProductQuantizer:DimMismatch','Dimension mismatch.'); end
            
            codes=zeros(size(X,1),obj.NumSubspaces,'uint32');
            for m=1:obj.NumSubspaces
                cols=(m-1)*obj.SubspaceDimension+(1:obj.SubspaceDimension);
                %Retrieve subspace m's centroi matrix
                C=obj.Codebooks(:,:,m).';
                [~,codes(:,m)]=min(obj.squaredDistances_(X(:,cols),C),[],2); codes(:,m)=uint32(codes(:,m));
            end
        end
        function X=decode(obj,codes)
            %DECODE recover centroids approximation, NOT origin X
            arguments
                obj
                codes (:,:) uint32
            end
            obj.checkFitted_(); if size(codes,2)~=obj.NumSubspaces, error('OceanAlgo:ProductQuantizer:CodeShape','Code matrix must have one column per subspace.'); end
            X=zeros(size(codes,1),obj.Dimension);
            for m=1:obj.NumSubspaces, cols=(m-1)*obj.SubspaceDimension+(1:obj.SubspaceDimension); X(:,cols)=obj.Codebooks(:,double(codes(:,m)),m).'; end
        end
        function distances=approximateDistances(obj,Q,codes)
            arguments
                obj
                Q (:,:) double
                codes (:,:) uint32
            end
            obj.checkFitted_(); if size(Q,2)~=obj.Dimension, error('OceanAlgo:ProductQuantizer:DimMismatch','Dimension mismatch.'); end
            if size(codes,2)~=obj.NumSubspaces, error('OceanAlgo:ProductQuantizer:CodeShape','Code matrix must have one column per subspace.'); end
            distances=zeros(size(Q,1),size(codes,1));
            for m=1:obj.NumSubspaces
                cols=(m-1)*obj.SubspaceDimension+(1:obj.SubspaceDimension); C=obj.Codebooks(:,:,m).';
                table=obj.squaredDistances_(Q(:,cols),C); distances=distances+table(:,double(codes(:,m)));
            end
        end
        function meta=getMetadata(obj)
            meta=struct('Name','ProductQuantizer','Domain','numeric vectors','NumSubspaces',obj.NumSubspaces,'NumCentroids',obj.NumCentroids,'Dimension',obj.Dimension,'ReconstructionError',obj.ReconstructionError,'Formula','dhat(q,x)=sum_m ||q_m-mu_(m,c_m)||^2');
        end
    end
    methods (Access=private)
        function D=squaredDistances_(~,A,C), D=sum(A.^2,2)+sum(C.^2,2).'-2*(A*C.'); end
        function checkFitted_(obj), if ~obj.IsFitted, error('OceanAlgo:ProductQuantizer:NotFitted','Call fit first.'); end, end
        function s=makeStream_(obj), if isempty(obj.Seed), s=RandStream.getGlobalStream(); else, s=RandStream('mt19937ar','Seed',obj.Seed); end, end
    end
end
