classdef (Abstract) ICoarsePartitioner < handle
    %ICOARSEPARTITIONER Pluggable coarse partitioner for inverted indexes.
    methods (Abstract)
        fit(obj, X, numPartitions)
        labels = assign(obj, X)
        centroids = getCentroids(obj)
        meta = getMetadata(obj)
    end
end
