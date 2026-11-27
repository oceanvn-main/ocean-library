classdef KMeansCoarsePartitioner < OceanAlgo.Search.ICoarsePartitioner
    %KMEANSCOARSEPARTITIONER Adapter around OceanStats.Unsupervised.KMeans.
    properties (SetAccess=private)
        Model
        MaxIterations (1,1) double {mustBeInteger,mustBePositive}=50
        NumRestarts (1,1) double {mustBeInteger,mustBePositive}=5
        IsFitted (1,1) logical=false
    end
    methods
        function obj=KMeansCoarsePartitioner(options)
            arguments
                options.MaxIterations (1,1) double {mustBeInteger,mustBePositive}=50
                options.NumRestarts (1,1) double {mustBeInteger,mustBePositive}=5
            end
            obj.MaxIterations=options.MaxIterations;
            obj.NumRestarts=options.NumRestarts;
            obj.Model=OceanStats.Unsupervised.KMeans();
        end
        function fit(obj,X,numPartitions)
            obj.Model.fit(X,numPartitions,MaxIter=obj.MaxIterations,NumRestarts=obj.NumRestarts);
            obj.IsFitted=true;
        end
        function labels=assign(obj,X)
            if ~obj.IsFitted, error('OceanAlgo:KMeansCoarsePartitioner:NotFitted','Call fit first.'); end
            C=obj.Model.Centroids;
            D=sum(X.^2,2)+sum(C.^2,2).'-2*(X*C.'); % N x K 
            [~,labels]=min(max(D,0),[],2); % [], 2 reduce each rows across columns to -> 1 cell value, N x 1
        end
        function centroids=getCentroids(obj)
            if ~obj.IsFitted, centroids=zeros(0,0); else, centroids=obj.Model.Centroids; end
        end
        function meta=getMetadata(obj)
            meta=struct('Name','KMeansCoarsePartitioner','Backend','OceanStats.Unsupervised.KMeans','MaxIterations',obj.MaxIterations,'NumRestarts',obj.NumRestarts,'IsFitted',obj.IsFitted);
        end
    end
end
