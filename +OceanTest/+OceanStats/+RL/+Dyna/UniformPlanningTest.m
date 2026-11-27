classdef UniformPlanningTest < OceanTest.BaseTestCase
    methods (Test)
        function testControlledDrawsAndLinearPairIDs(testCase)
            mask=logical([1 1;1 0]);
            model=OceanStats.RL.Dyna.Models.DeterministicTabularModel(2,2,mask);
            model.update(struct('StateIndex',1,'ActionIndex',2,'Reward',0,'NextStateIndex',1,'Terminated',false));
            model.update(struct('StateIndex',2,'ActionIndex',1,'Reward',0,'NextStateIndex',1,'Terminated',false));
            planner=OceanStats.RL.Dyna.Planning.UniformPlanning(); stream=RandStream('mt19937ar','Seed',19);
            draws=zeros(1,8);
            for k=1:8, c=planner.next(model,[],[],stream); draws(k)=c.PairID; end
            reference=RandStream('mt19937ar','Seed',19); ids=model.observedPairs(); expected=ids(randi(reference,2,1,8));
            testCase.verifyEqual(draws,expected(:)');
            testCase.verifyEqual(sort(unique(draws(:))),sort(ids(:)));
        end
        function testNoCandidateError(testCase)
            model=OceanStats.RL.Dyna.Models.DeterministicTabularModel(1,1,true);
            planner=OceanStats.RL.Dyna.Planning.UniformPlanning(); stream=RandStream('mt19937ar','Seed',1);
            testCase.verifyError(@()planner.next(model,[],[],stream),'OceanStats:RL:Dyna:Planning:NoCandidate');
        end
    end
end
