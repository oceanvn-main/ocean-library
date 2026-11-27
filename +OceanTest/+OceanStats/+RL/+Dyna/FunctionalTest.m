classdef FunctionalTest < OceanTest.BaseTestCase
    %FUNCTIONALTEST Public-contract tests for Dyna records and pure formulas.

    methods (Test)
        function testQLearningTargetUsesOnlyLegalActions(testCase)
            actual = OceanStats.RL.Dyna.Functional.qLearningTarget(2, [100 4 6], false, 0.5, [false true true]);
            testCase.verifyEqual(actual, 5, AbsTol=1e-14);
        end

        function testQLearningTargetDropsTerminalBootstrap(testCase)
            actual = OceanStats.RL.Dyna.Functional.qLearningTarget(-3, [], true, 0.9, []);
            testCase.verifyEqual(actual, -3, AbsTol=1e-14);
        end

        function testPriorityMagnitudeIsAbsoluteTDError(testCase)
            actual = OceanStats.RL.Dyna.Functional.priorityMagnitude(1, [2 8], 4, false, 0.5, [true false]);
            testCase.verifyEqual(actual, 2, AbsTol=1e-14);
        end

        function testRecencyBonusVectorizes(testCase)
            actual = OceanStats.RL.Dyna.Functional.recencyBonus(0.2, [0 1 9]);
            testCase.verifyEqual(actual, [0 0.2 0.6], AbsTol=1e-14);
        end

        function testCategoricalSampleUsesExplicitHalfOpenIntervals(testCase)
            sample = @OceanStats.RL.Dyna.Functional.categoricalSample;
            testCase.verifyEqual(sample([0.2 0.3 0.5], 0), 1);
            testCase.verifyEqual(sample([0.2 0.3 0.5], 0.2), 2);
            testCase.verifyEqual(sample([0.2 0.3 0.5], 0.999), 3);
        end

        function testPairIDRoundTripMatchesColumnMajorLayout(testCase)
            id = OceanStats.RL.Dyna.Functional.pairID(2, 3, 4, 5);
            [state, action] = OceanStats.RL.Dyna.Functional.pairFromID(id, 4, 5);
            testCase.verifyEqual(id, 10);
            testCase.verifyEqual([state action], [2 3]);
        end

        function testFunctionalValidationUsesStableErrorIDs(testCase)
            testCase.verifyError(@() OceanStats.RL.Dyna.Functional.qLearningTarget(0, [1 2], false, 1, [false false]), ...
                'OceanStats:RL:Dyna:Functional:InvalidInput');
            testCase.verifyError(@() OceanStats.RL.Dyna.Functional.recencyBonus(-1, 0), ...
                'OceanStats:RL:Dyna:Functional:InvalidInput');
            testCase.verifyError(@() OceanStats.RL.Dyna.Functional.categoricalSample([0.2 0.2], 0.1), ...
                'OceanStats:RL:Dyna:Functional:InvalidInput');
            testCase.verifyError(@() OceanStats.RL.Dyna.Functional.pairID(5, 1, 4, 2), ...
                'OceanStats:RL:Dyna:Functional:InvalidPair');
        end

        function testTransitionPreservesTerminationAndClocks(testCase)
            value = OceanStats.RL.Dyna.Core.Transition( ...
                StateIndex=2, ActionIndex=3, Reward=-1, NextStateIndex=0, ...
                Terminated=true, Origin="planning", RealStep=7, PlanningStep=11);
            testCase.verifyEqual(value.StateIndex, 2);
            testCase.verifyEqual(value.NextStateIndex, 0);
            testCase.verifyTrue(value.Terminated);
            testCase.verifyEqual([value.RealStep value.PlanningStep], [7 11]);
        end

        function testTransitionRejectsAmbiguousEndAndBadTruncation(testCase)
            testCase.verifyError(@() OceanStats.RL.Dyna.Core.Transition( ...
                StateIndex=1, ActionIndex=1, Reward=0, NextStateIndex=1, Terminated=true, Truncated=true), ...
                'OceanStats:RL:Dyna:Core:InvalidRecord');
            testCase.verifyError(@() OceanStats.RL.Dyna.Core.Transition( ...
                StateIndex=1, ActionIndex=1, Reward=0, NextStateIndex=0, Truncated=true), ...
                'OceanStats:RL:Dyna:Core:InvalidRecord');
        end

        function testModelOutcomeValidatesProbability(testCase)
            outcome = OceanStats.RL.Dyna.Core.ModelOutcome(3, 1.5, false, 0.25);
            testCase.verifyEqual(outcome.Probability, 0.25, AbsTol=1e-14);
            testCase.verifyError(@() OceanStats.RL.Dyna.Core.ModelOutcome(3, 0, false, 0), ...
                'OceanStats:RL:Dyna:Core:InvalidRecord');
        end

        function testExplainabilityRecordsRetainOperands(testCase)
            event = OceanStats.RL.Dyna.Core.UpdateEvent(Origin="real", StateIndex=1, ActionIndex=2, ...
                OldQ=3, Target=7, TDError=4, Alpha=0.25, NewQ=4, PlanningPriority=4, ...
                RealStep=5, PlanningStep=6, ModelVersion=2);
            modelEvent = OceanStats.RL.Dyna.Core.ModelEvent(StateIndex=1, ActionIndex=2, ObservationCount=3, ...
                PreviousOutcomes=struct([]), NewOutcomes=struct('NextStateIndex', 4), ...
                PredecessorChanges=struct('Added', true), ModelVersion=2);
            testCase.verifyEqual(event.NewQ, 4, AbsTol=1e-14);
            testCase.verifyEqual(event.TDError, event.Target - event.OldQ, AbsTol=1e-14);
            testCase.verifyEqual(modelEvent.NewOutcomes.NextStateIndex, 4);
        end

        function testPlanningCandidateUsesLinearPairIdentity(testCase)
            candidate = OceanStats.RL.Dyna.Core.PlanningCandidate(2, 3, 10, 4.5, 7);
            testCase.verifyEqual([candidate.StateIndex candidate.ActionIndex candidate.PairID], [2 3 10]);
            testCase.verifyEqual(candidate.Priority, 4.5, AbsTol=1e-14);
        end

        function testDynaComponentProvidesThreePillars(testCase)
            component = OceanTest.OceanStats.RL.Dyna.Fixtures.DummyComponent();
            metadata = component.getMetadata();
            pipeline = component.getPipelineSpec();
            testCase.verifyTrue(metadata.IsSymbolic);
            testCase.verifyEqual(metadata.SymbolicLatex, "y=x");
            testCase.verifyEqual(pipeline.Kind, "NONVIRTUAL");
            testCase.verifyEqual(pipeline.DiagramSymbol.Label, "dummy");
        end
    end
end
