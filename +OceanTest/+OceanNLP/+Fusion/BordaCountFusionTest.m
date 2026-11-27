classdef BordaCountFusionTest < OceanTest.BaseTestCase
    methods (Test)
        function testBordaVoting(testCase)
            borda = OceanNLP.Fusion.BordaCountFusion();
            [ids, scores] = borda.fuse({[1, 2, 3], [2, 1, 4]});
            
            % List 1 (N=3): 1 gets 2 pts, 2 gets 1 pt, 3 gets 0 pts
            % List 2 (N=3): 2 gets 2 pts, 1 gets 1 pt, 4 gets 0 pts
            % Total: doc 1 (3 pts), doc 2 (3 pts), doc 3 (0 pts), doc 4 (0 pts)
            testCase.verifyEqual(scores(1), 3);
            testCase.verifyEqual(scores(2), 3);
        end

        function testFusionFactory(testCase)
            fRrf = OceanNLP.Fusion.FusionFactory.create('rrf', 'K', 60);
            testCase.verifyClass(fRrf, 'OceanNLP.Fusion.ReciprocalRankFusion');

            fBorda = OceanNLP.Fusion.FusionFactory.create('borda');
            testCase.verifyClass(fBorda, 'OceanNLP.Fusion.BordaCountFusion');
        end
    end
end
