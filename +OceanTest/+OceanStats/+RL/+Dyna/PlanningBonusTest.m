classdef PlanningBonusTest < OceanTest.BaseTestCase
    methods (Test)
        function testNoBonus(testCase)
            bonus=OceanStats.RL.Dyna.Bonuses.NoBonus();
            testCase.verifyEqual(bonus.evaluate(1,1,10,0),0,'AbsTol',0);
        end
        function testRecencyUsesOnlyRealStepClock(testCase)
            bonus=OceanStats.RL.Dyna.Bonuses.RecencyBonus(0.2,true(1,2));
            bonus.onRealVisit(1,1,4);
            testCase.verifyEqual(bonus.evaluate(1,1,13,[]),0.6,'AbsTol',1e-14);
            testCase.verifyEqual(bonus.evaluate(1,2,9,[]),0.6,'AbsTol',1e-14);
        end
        function testFutureVisitRejected(testCase)
            bonus=OceanStats.RL.Dyna.Bonuses.RecencyBonus(0.1,true);
            testCase.verifyError(@()bonus.evaluate(1,1,3,4),'OceanStats:RL:Dyna:Bonuses:FutureVisit');
        end
        function testInvalidActionRejected(testCase)
            bonus=OceanStats.RL.Dyna.Bonuses.RecencyBonus(0.1,logical([1 0]));
            testCase.verifyError(@()bonus.onRealVisit(1,2,1),'OceanStats:RL:Dyna:Bonuses:InvalidAction');
        end
    end
end
