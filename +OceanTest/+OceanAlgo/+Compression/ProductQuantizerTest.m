classdef ProductQuantizerTest < OceanTest.BaseTestCase
    methods (Test)
        function testEncodeDecodeAndDistance(testCase)
            X=[-2 -2 0 0; -2 -1 0 1; 2 2 5 5; 1 2 5 4; 0 0 2 2; 0 1 2 3; 3 3 7 7; 2 3 7 6];
            pq=OceanAlgo.Compression.ProductQuantizer('NumSubspaces',2,'NumCentroids',2,'NumIterations',8,'Seed',7);
            pq.fit(X); codes=pq.encode(X); Xhat=pq.decode(codes);
            testCase.verifySize(codes,[8 2]); testCase.verifySize(Xhat,size(X));
            D=pq.approximateDistances(X(1:3,:),codes); testCase.verifyEqual(size(D),[3 8]);
            testCase.verifyEqual(D(1,1),0,'AbsTol',1e-12);
            testCase.verifyGreaterThanOrEqual(pq.ReconstructionError,0);
        end
    end
end
