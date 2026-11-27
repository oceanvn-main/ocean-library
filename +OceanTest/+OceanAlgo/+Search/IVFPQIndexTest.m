classdef IVFPQIndexTest < OceanTest.BaseTestCase
    methods (Test)
        function testSearchReturnsCorpusIds(testCase)
            X=[randn(20,4); randn(20,4)+5];
            pq=OceanAlgo.Compression.ProductQuantizer('NumSubspaces',2,'NumCentroids',4,'NumIterations',5,'Seed',3);
            idx=OceanAlgo.Search.IVFPQIndex('NumLists',4,'NumProbe',2,'ProductQuantizer',pq); idx.fit(X);
            [ids,scores]=idx.search(X(1,:),5);
            testCase.verifyTrue(all(ids>0)); testCase.verifyEqual(numel(unique(ids)),5); testCase.verifyTrue(all(isfinite(scores)));
        end
        function testCoarsePartitionerIsPluggable(testCase)
            X=randn(30,4);
            pq=OceanAlgo.Compression.ProductQuantizer('NumSubspaces',2,'NumCentroids',2,'NumIterations',3,'Seed',4);
            partitioner=OceanAlgo.Search.KMeansCoarsePartitioner('MaxIterations',4,'NumRestarts',1);
            idx=OceanAlgo.Search.IVFPQIndex('NumLists',3,'NumProbe',2,'ProductQuantizer',pq,'CoarsePartitioner',partitioner);
            idx.fit(X);
            meta=idx.getMetadata();
            testCase.verifyEqual(meta.CoarsePartitioner.Name,'KMeansCoarsePartitioner');
            testCase.verifyEqual(meta.CoarsePartitioner.Backend,'OceanStats.Unsupervised.KMeans');
        end
    end
end
