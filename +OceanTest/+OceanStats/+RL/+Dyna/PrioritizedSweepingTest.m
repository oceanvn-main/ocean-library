classdef PrioritizedSweepingTest < OceanTest.BaseTestCase
    methods (Test)
        function testThresholdAndPredecessorPropagation(testCase)
            mask=true(3,1); model=OceanStats.RL.Dyna.Models.DeterministicTabularModel(3,1,mask);
            model.update(struct('StateIndex',1,'ActionIndex',1,'Reward',1,'NextStateIndex',2,'Terminated',false));
            model.update(struct('StateIndex',2,'ActionIndex',1,'Reward',2,'NextStateIndex',3,'Terminated',true));
            values=OceanStats.RL.TD.Values.TabularQ(mask,0);
            planner=OceanStats.RL.Dyna.Planning.PrioritizedSweeping(0.5);
            planner.observeReal(struct('StateIndex',2,'ActionIndex',1),model,values,1);
            c=planner.next(model,values,1,RandStream('mt19937ar','Seed',1));
            values.update(2,1,1,2);
            planner.afterUpdate(struct('StateIndex',2),model,values,1);
            pred=planner.next(model,values,1,RandStream('mt19937ar','Seed',2));
            testCase.verifyEqual(c.PairID,2);
            testCase.verifyEqual(c.Priority,2,'AbsTol',1e-14);
            testCase.verifyEqual(pred.PairID,1);
            testCase.verifyEqual(pred.Priority,3,'AbsTol',1e-14);
        end
        function testBelowThresholdCannotPlan(testCase)
            model=OceanStats.RL.Dyna.Models.DeterministicTabularModel(1,1,true);
            model.update(struct('StateIndex',1,'ActionIndex',1,'Reward',0.5,'NextStateIndex',0,'Terminated',true));
            values=OceanStats.RL.TD.Values.TabularQ(true,0);
            planner=OceanStats.RL.Dyna.Planning.PrioritizedSweeping(0.5);
            planner.observeReal(struct('StateIndex',1,'ActionIndex',1),model,values,1);
            testCase.verifyFalse(planner.canPlan(model));
        end
    end
end
