classdef WorldModelTest < OceanTest.BaseTestCase
    % WORLDMODELTEST Focused tests for Dyna tabular world models.

    methods (Test)
        function deterministicLatestOutcomeAndPairs(testCase)
            model = OceanStats.RL.Dyna.Models.DeterministicTabularModel(3, 2, logical([1 1; 1 0; 0 0]));
            model.update(testCase.transition_(1, 2, 4, 2, false));
            testCase.verifyTrue(model.contains(1, 2));
            testCase.verifyEqual(model.observedPairs(), 4); % sub2ind([3 2],1,2)
            testCase.verifyEqual(model.distribution(1, 2), ...
                struct('NextStateIndex',2,'Reward',4,'Terminated',false,'Probability',1));
            model.update(testCase.transition_(1, 2, 7, 3, true));
            snap = model.inspect(1, 2);
            testCase.verifyEqual(snap.ObservationCount, 2);
            testCase.verifyEqual(snap.Outcomes.Reward, 7);
            testCase.verifyEqual(model.predecessors(3).PairID, 4);
            testCase.verifyEmpty(model.predecessors(2)); % stale edge removed
        end

        function invalidAndUnobservedPairsFail(testCase)
            model = OceanStats.RL.Dyna.Models.DeterministicTabularModel(2, 2, logical([1 0; 0 0]));
            testCase.verifyError(@() model.update(testCase.transition_(1,2,0,2,false)), ...
                'OceanStats:RL:Dyna:Models:InvalidAction');
            testCase.verifyError(@() model.distribution(1,1), ...
                'OceanStats:RL:Dyna:Models:UnobservedPair');
        end

        function empiricalProbabilitiesRewardsAndPredecessors(testCase)
            model = OceanStats.RL.Dyna.Models.EmpiricalTabularModel(3, 1, logical([1;1;0]));
            model.update(testCase.transition_(1,1,2,2,false));
            model.update(testCase.transition_(1,1,4,2,false));
            model.update(testCase.transition_(1,1,9,3,true));
            d = model.distribution(1,1);
            testCase.verifyEqual([d.Probability], [2/3 1/3], 'AbsTol', 1e-14);
            testCase.verifyEqual([d.Reward], [3 9], 'AbsTol', 1e-14);
            p2 = model.predecessors(2); p3 = model.predecessors(3);
            testCase.verifyEqual([p2.Probability p2.MeanReward], [2/3 3], 'AbsTol', 1e-14);
            testCase.verifyEqual([p3.Probability p3.MeanReward], [1/3 9], 'AbsTol', 1e-14);
            testCase.verifyEqual(model.inspect(1,1).ObservationCount, 3);
        end

        function empiricalSamplingUsesOwnedStreamOnly(testCase)
            model = OceanStats.RL.Dyna.Models.EmpiricalTabularModel(3,1,logical([1;0;0]));
            model.update(testCase.transition_(1,1,0,2,false));
            model.update(testCase.transition_(1,1,1,3,true));
            rng(41); expected = rand(); rng(41);
            s1 = RandStream('mt19937ar','Seed',8); s2 = RandStream('mt19937ar','Seed',8);
            a = zeros(1,40); b = zeros(1,40);
            for k=1:40
                draw=model.sample(1,1,s1); a(k)=draw.NextStateIndex;
                draw=model.sample(1,1,s2); b(k)=draw.NextStateIndex;
            end
            testCase.verifyEqual(a,b);
            testCase.verifyEqual(rand(), expected); % no global RNG consumption
        end

        function resetClearsStatisticsButKeepsMask(testCase)
            mask = logical([1 0; 1 1]);
            model = OceanStats.RL.Dyna.Models.EmpiricalTabularModel(2,2,mask);
            model.update(testCase.transition_(2,2,1,1,false)); oldVersion=model.Version;
            model.reset();
            testCase.verifyEmpty(model.observedPairs());
            testCase.verifyEqual(model.ValidActionMask,mask);
            testCase.verifyGreaterThan(oldVersion,uint64(0));
            testCase.verifyEqual(model.Version,uint64(0));
        end

        function exactAdapterMatchesRealModelEnvironment(testCase)
            env = OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner',3);
            model = OceanStats.RL.Dyna.Models.ExactModelAdapter(env);
            testCase.verifyEqual(model.NumStates,5);
            testCase.verifyEqual(model.observedPairs(),find(model.ValidActionMask));
            d=model.distribution(4,2);
            [next,p,r]=env.get_transitions(4,2);
            testCase.verifyEqual([d.NextStateIndex]',next);
            testCase.verifyEqual([d.Probability]',p);
            testCase.verifyEqual([d.Reward]',r);
            testCase.verifyTrue(d.Terminated);
            testCase.verifyError(@() model.update(testCase.transition_(2,1,0,1,true)), ...
                'OceanStats:RL:Dyna:Models:ExactModelReadOnly');
        end
    end

    methods (Static, Access=private)
        function t=transition_(s,a,r,n,terminated)
            t=struct('StateIndex',s,'ActionIndex',a,'Reward',r,'NextStateIndex',n, ...
                'Terminated',logical(terminated),'Truncated',false,'Origin',"real", ...
                'RealStep',0,'PlanningStep',0);
        end
    end
end
