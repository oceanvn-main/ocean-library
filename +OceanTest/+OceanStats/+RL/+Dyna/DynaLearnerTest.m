classdef DynaLearnerTest < OceanTest.BaseTestCase
    methods (Test)
        function realAndPlanningUpdatesShareTrace(testCase)
            env=OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner',3);
            mask=OceanStats.RL.Dyna.Models.ExactModelAdapter.deriveMask(env);
            model=OceanStats.RL.Dyna.Models.DeterministicTabularModel(5,2,mask);
            recorder=OceanStats.RL.Dyna.Observers.Recorder();
            learner=OceanStats.RL.Dyna.DynaLearner(env,'Model',model,'PlanningSteps',2, ...
                'StepSize',OceanStats.RL.TD.StepSize.ConstantStepSize(0.5), ...
                'Exploration',OceanStats.RL.TD.Exploration.EpsilonGreedyExploration(0), ...
                'Observer',recorder,'Gamma',1,'Seed',7);
            result=learner.run('NumEpisodes',2,'MaxStepsPerEpisode',20);
            testCase.verifyEqual(result.RealSteps,numel(recorder.Transitions));
            testCase.verifyGreaterThan(result.PlanningSteps,0);
            origins=cellfun(@(x)x.Origin,recorder.UpdateEvents);
            testCase.verifyTrue(any(origins=="real")); testCase.verifyTrue(any(origins=="planning"));
            testCase.verifyEqual(sum(result.Counts(:)),result.RealSteps+result.PlanningSteps);
        end
        function repeatedRunRestartsDeterministically(testCase)
            env=OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner',3);
            mask=OceanStats.RL.Dyna.Models.ExactModelAdapter.deriveMask(env);
            learner=OceanStats.RL.Dyna.DynaLearner(env, ...
                'Model',OceanStats.RL.Dyna.Models.DeterministicTabularModel(5,2,mask), ...
                'PlanningSteps',3,'Seed',11);
            a=learner.run('NumEpisodes',3,'MaxStepsPerEpisode',15);
            b=learner.run('NumEpisodes',3,'MaxStepsPerEpisode',15);
            testCase.verifyEqual(a.Q,b.Q); testCase.verifyEqual(a.Returns,b.Returns);
            testCase.verifyEqual(a.RealSteps,b.RealSteps); testCase.verifyEqual(a.PlanningSteps,b.PlanningSteps);
        end
        function planningZeroCountsOnlyRealUpdates(testCase)
            env=OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner',3);
            mask=OceanStats.RL.Dyna.Models.ExactModelAdapter.deriveMask(env);
            learner=OceanStats.RL.Dyna.DynaLearner(env, ...
                'Model',OceanStats.RL.Dyna.Models.DeterministicTabularModel(5,2,mask), ...
                'PlanningSteps',0,'Seed',3);
            r=learner.run('NumEpisodes',2,'MaxStepsPerEpisode',10);
            testCase.verifyEqual(r.PlanningSteps,0); testCase.verifyEqual(sum(r.Counts(:)),r.RealSteps);
        end
        function prioritizedRealTransitionIsQueuedBeforeDirectUpdate(testCase)
            env=OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner',3,'LeftReward',2,'RightReward',2);
            mask=OceanStats.RL.Dyna.Models.ExactModelAdapter.deriveMask(env);
            learner=OceanStats.RL.Dyna.DynaLearner(env, ...
                'Model',OceanStats.RL.Dyna.Models.DeterministicTabularModel(5,2,mask), ...
                'Planning',OceanStats.RL.Dyna.Planning.PrioritizedSweeping(0), ...
                'StepSize',OceanStats.RL.TD.StepSize.ConstantStepSize(1), ...
                'PlanningSteps',1,'Seed',5);
            r=learner.run('NumEpisodes',1,'MaxStepsPerEpisode',20);
            testCase.verifyGreaterThan(r.PlanningSteps,0);
        end
        function dynaQPlusMakesUntriedActionsEligible(testCase)
            env=OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner',3);
            learner=OceanStats.RL.Dyna.Factory.dynaQPlus(env,'PlanningSteps',0,'Seed',2);
            r=learner.run('NumEpisodes',1,'MaxStepsPerEpisode',1);
            testCase.verifyEqual(numel(r.Model.observedPairs()),2);
        end
        function exactModelAdapterRunsWithoutMutation(testCase)
            env=OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner',3);
            model=OceanStats.RL.Dyna.Models.ExactModelAdapter(env);
            learner=OceanStats.RL.Dyna.DynaLearner(env,'Model',model,'PlanningSteps',1,'Seed',4);
            r=learner.run('NumEpisodes',1,'MaxStepsPerEpisode',10);
            testCase.verifyGreaterThan(r.RealSteps,0);
        end
        function empiricalPrioritizedPlanningUsesExpectedTarget(testCase)
            env=OceanTest.OceanStats.RL.Dyna.Fixtures.AlternatingTerminalEnv();
            model=OceanStats.RL.Dyna.Models.EmpiricalTabularModel(1,1,true);
            learner=OceanStats.RL.Dyna.DynaLearner(env,'Model',model, ...
                'Planning',OceanStats.RL.Dyna.Planning.PrioritizedSweeping(0), ...
                'StepSize',OceanStats.RL.TD.StepSize.ConstantStepSize(1), ...
                'Exploration',OceanStats.RL.TD.Exploration.EpsilonGreedyExploration(0), ...
                'Gamma',1,'PlanningSteps',1,'Seed',1);
            r=learner.run('NumEpisodes',2,'MaxStepsPerEpisode',1);
            testCase.verifyEqual(r.Q,4,'AbsTol',1e-14);
            testCase.verifyEqual(r.Model.inspect(1,1).ObservationCount,2);
        end
        function truncationBootstrapsAndUsesExternalActionLabel(testCase)
            env=OceanTest.OceanStats.RL.Dyna.Fixtures.ContinuingSelfLoopEnv();
            values=OceanStats.RL.TD.Values.TabularQ(true,5);
            model=OceanStats.RL.Dyna.Models.DeterministicTabularModel(1,1,true);
            learner=OceanStats.RL.Dyna.DynaLearner(env,'Model',model,'Values',values, ...
                'StepSize',OceanStats.RL.TD.StepSize.ConstantStepSize(1), ...
                'Exploration',OceanStats.RL.TD.Exploration.EpsilonGreedyExploration(0), ...
                'Gamma',0.9,'PlanningSteps',0,'Seed',3);
            r=learner.run('NumEpisodes',1,'MaxStepsPerEpisode',1);
            testCase.verifyEqual(r.Q,6.5,'AbsTol',1e-14);
            testCase.verifyTrue(r.Truncated);
            testCase.verifyFalse(r.Terminated);
        end
        function repeatedRunPreservesModelVersionTrace(testCase)
            env=OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner',3);
            mask=OceanStats.RL.Dyna.Models.ExactModelAdapter.deriveMask(env);
            recorder=OceanStats.RL.Dyna.Observers.Recorder();
            learner=OceanStats.RL.Dyna.DynaLearner(env, ...
                'Model',OceanStats.RL.Dyna.Models.DeterministicTabularModel(5,2,mask), ...
                'Observer',recorder,'PlanningSteps',2,'Seed',17);
            learner.run('NumEpisodes',2,'MaxStepsPerEpisode',8);
            first=OceanTest.OceanStats.RL.Dyna.DynaLearnerTest.updateVersions_(recorder);
            firstModel=OceanTest.OceanStats.RL.Dyna.DynaLearnerTest.modelVersions_(recorder);
            learner.run('NumEpisodes',2,'MaxStepsPerEpisode',8);
            testCase.verifyEqual(OceanTest.OceanStats.RL.Dyna.DynaLearnerTest.updateVersions_(recorder),first);
            testCase.verifyEqual(OceanTest.OceanStats.RL.Dyna.DynaLearnerTest.modelVersions_(recorder),firstModel);
            testCase.verifyEqual(firstModel,(1:numel(firstModel))');
        end
        function planningSelectionStreamIsIndependent(testCase)
            model=OceanStats.RL.Dyna.Models.DeterministicTabularModel(2,2,true(2));
            OceanTest.OceanStats.RL.Dyna.DynaLearnerTest.populateAllPairs_(model);
            values=OceanStats.RL.TD.Values.TabularQ(true(2),0);
            plannerA=OceanStats.RL.Dyna.Planning.UniformPlanning();
            plannerB=OceanStats.RL.Dyna.Planning.UniformPlanning();
            streamA=RandStream('mrg32k3a','Seed',23);
            streamB=RandStream('mrg32k3a','Seed',23);
            unrelated=RandStream('mrg32k3a','Seed',24);
            idsA=OceanTest.OceanStats.RL.Dyna.DynaLearnerTest.drawPairIDs_(plannerA,model,values,streamA,[],30);
            idsB=OceanTest.OceanStats.RL.Dyna.DynaLearnerTest.drawPairIDs_(plannerB,model,values,streamB,unrelated,30);
            testCase.verifyEqual(idsA,idsB);
        end
        function compositeObserverStopsAtFirstFailure(testCase)
            failing=OceanTest.OceanStats.RL.Dyna.Fixtures.ThrowingObserver();
            downstream=OceanStats.RL.Dyna.Observers.Recorder();
            observer=OceanStats.RL.Dyna.Observers.CompositeObserver({failing,downstream});
            event=OceanStats.RL.Dyna.Core.UpdateEvent('Origin',"real",'StateIndex',1, ...
                'ActionIndex',1,'OldQ',0,'Target',1,'TDError',1,'Alpha',1,'NewQ',1);
            testCase.verifyError(@()observer.valueUpdated(event),'OceanTest:RL:Dyna:ObserverFailure');
            testCase.verifyEmpty(downstream.UpdateEvents);
        end
    end
    methods (Static, Access=private)
        function versions=updateVersions_(recorder)
            versions=cellfun(@(e)e.ModelVersion,recorder.UpdateEvents);
        end
        function versions=modelVersions_(recorder)
            versions=cellfun(@(e)e.ModelVersion,recorder.ModelEvents);
        end
        function populateAllPairs_(model)
            for s=1:2
                for a=1:2
                    t=struct('StateIndex',s,'ActionIndex',a,'Reward',0,'NextStateIndex',s,'Terminated',false);
                    model.update(t);
                end
            end
        end
        function ids=drawPairIDs_(planner,model,values,stream,unrelated,n)
            ids=zeros(n,1);
            for k=1:n
                if ~isempty(unrelated), rand(unrelated,1,7); end
                ids(k)=planner.next(model,values,1,stream).PairID;
            end
        end
    end
end
