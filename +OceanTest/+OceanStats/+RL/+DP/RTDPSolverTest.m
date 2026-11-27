classdef RTDPSolverTest < matlab.unittest.TestCase
    %RTDPSOLVERTEST Independent exact arithmetic and trial behavior checks.
    methods (Test)
        function expectedBackupUsesBothBranchesAndTerminalZero(testCase)
            model = OceanTest.OceanStats.RL.Dyna.Fixtures.TinyRTDPModel();
            solver = OceanStats.RL.DP.RTDPSolver(model, ...
                InitialValue=[0;8;99],DiscountFactor=0.5,Seed=7,MaxDepth=1);
            trace = solver.runTrial(1);
            % Q(1,1)=.25*(2+.5*8)+.75*4=4.5; Q(1,2)=3.
            testCase.verifyEqual(solver.getValue(1),4.5,AbsTol=1e-12);
            testCase.verifyEqual(trace.Steps(1).ActionValues,[4.5 3],AbsTol=1e-12);
            testCase.verifyEqual(solver.getPolicy(1),1);
            testCase.verifyEqual(solver.getValue(3),99,AbsTol=1e-12);
            testCase.verifyEqual(solver.getPolicy(2),0);
        end

        function nonterminalSuccessorGetsItsOwnBackup(testCase)
            model = OceanTest.OceanStats.RL.Dyna.Fixtures.TinyRTDPModel();
            solver = OceanStats.RL.DP.RTDPSolver(model, ...
                InitialValue=[0;8;99],DiscountFactor=0.5,Seed=1,MaxDepth=2);
            solver.runTrial(2);
            testCase.verifyEqual(solver.getValue(2),1,AbsTol=1e-12);
            testCase.verifyEqual(solver.getPolicy(2),1);
            testCase.verifyEqual(solver.getValue(3),99,AbsTol=1e-12);
        end

        function terminalStartDoesNoWork(testCase)
            model = OceanTest.OceanStats.RL.Dyna.Fixtures.TinyRTDPModel();
            solver = OceanStats.RL.DP.RTDPSolver(model,InitialValue=[0;0;7]);
            trace = solver.runTrial(3);
            testCase.verifyEqual(trace.Backups,0);
            testCase.verifyEqual(trace.TerminationReason,"terminal_state");
            testCase.verifyEqual(solver.getValue(3),7,AbsTol=1e-12);
        end

        function solveRespectsBackupBudgetAndDoesNotClaimConvergence(testCase)
            model = OceanTest.OceanStats.RL.Dyna.Fixtures.TinyRTDPModel();
            solver = OceanStats.RL.DP.RTDPSolver(model,MaxDepth=10,Seed=2);
            [values,policy,result] = solver.solve(1,MaxTrials=10,MaxBackups=3);
            testCase.verifyEqual(result.Backups,3);
            testCase.verifyEqual(result.TerminationReason,"backup_budget");
            testCase.verifyFalse(result.Converged);
            testCase.verifyEqual(values,solver.ValueVector,AbsTol=1e-12);
            testCase.verifyEqual(policy,solver.PolicyVector);
            testCase.verifyEqual(solver.getPolicy(3),0);
        end

        function seededTrialsMatchAndDoNotAdvanceGlobalRNG(testCase)
            model = OceanTest.OceanStats.RL.Dyna.Fixtures.TinyRTDPModel();
            first = OceanStats.RL.DP.RTDPSolver(model,Seed=21,MaxDepth=1);
            second = OceanStats.RL.DP.RTDPSolver(model,Seed=21,MaxDepth=1);
            original = rng;
            testCase.addTeardown(@() rng(original));
            rng(37); expected = rand(); rng(37);
            [~,~,a] = first.solve(1,MaxTrials=20,MaxBackups=20);
            [~,~,b] = second.solve(1,MaxTrials=20,MaxBackups=20);
            testCase.verifyEqual(a.LastTrajectory,b.LastTrajectory);
            testCase.verifyEqual(first.ValueVector,second.ValueVector,AbsTol=1e-12);
            testCase.verifyEqual(rand(),expected,AbsTol=1e-12);
        end

        function depthLimitBoundsCyclicTrial(testCase)
            model = OceanStats.RL.Dyna.Models.DeterministicTabularModel(1,1,true);
            transition = struct('StateIndex',1,'ActionIndex',1,'Reward',0, ...
                'NextStateIndex',1,'Terminated',false,'Truncated',false, ...
                'Origin',"real",'RealStep',0,'PlanningStep',0);
            model.update(transition);
            solver = OceanStats.RL.DP.RTDPSolver(model,MaxDepth=4,Seed=3);
            trace = solver.runTrial(1);
            testCase.verifyEqual(trace.Backups,4);
            testCase.verifyEqual(trace.TerminationReason,"depth_limit");
        end

        function invalidProbabilityIsRejected(testCase)
            model = OceanTest.OceanStats.RL.Dyna.Fixtures.TinyRTDPModel("bad_probability");
            solver = OceanStats.RL.DP.RTDPSolver(model,MaxDepth=1);
            testCase.verifyError(@() solver.runTrial(1), ...
                'OceanStats:RL:DP:BadDistribution');
        end

        function unindexedNonterminalSuccessorIsRejected(testCase)
            model = OceanTest.OceanStats.RL.Dyna.Fixtures.TinyRTDPModel("unindexed_successor");
            solver = OceanStats.RL.DP.RTDPSolver(model,MaxDepth=1);
            testCase.verifyError(@() solver.runTrial(1), ...
                'OceanStats:RL:DP:BadDistribution');
        end

        function agreesWithValueIterationOnRandomWalk(testCase)
            env = OceanStats.RL.Env.Classic.RandomWalkEnv(NumInner=3);
            vi = OceanStats.RL.DP.ValueIterationSolver(env, ...
                DiscountFactor=0.9,Theta=1e-12);
            [viValues,viPolicy] = vi.run(MaxIter=100);
            model = OceanStats.RL.Dyna.Models.ExactModelAdapter(env);
            rtdp = OceanStats.RL.DP.RTDPSolver(model,DiscountFactor=0.9,Seed=5);
            rtdp.runTrial(4);
            rtdp.runTrial(3);
            rtdp.runTrial(2);
            testCase.verifyEqual(rtdp.ValueVector,viValues,AbsTol=1e-11);
            testCase.verifyEqual(rtdp.PolicyVector(2:4),viPolicy(2:4));
        end

        function trialBudgetStopsWithoutConvergence(testCase)
            model = OceanTest.OceanStats.RL.Dyna.Fixtures.TinyRTDPModel();
            solver = OceanStats.RL.DP.RTDPSolver(model,MaxDepth=1,Seed=9);
            [~,~,result] = solver.solve(1,MaxTrials=2,MaxBackups=20);
            testCase.verifyEqual(result.Trials,2);
            testCase.verifyEqual(result.Backups,2);
            testCase.verifyEqual(result.TerminationReason,"trial_budget");
            testCase.verifyFalse(result.Converged);
        end

        function tieToleranceSelectsFirstNearOptimalAction(testCase)
            model = OceanTest.OceanStats.RL.Dyna.Fixtures.TinyRTDPModel();
            solver = OceanStats.RL.DP.RTDPSolver(model, ...
                InitialValue=[0;-4.4;0],DiscountFactor=0.5, ...
                TieTolerance=0.1,MaxDepth=1);
            trace = solver.runTrial(1);
            % Q(1,1)=2.95, Q(1,2)=3; first action is within tolerance.
            testCase.verifyEqual(trace.Steps(1).ActionValues,[2.95 3],AbsTol=1e-12);
            testCase.verifyEqual(trace.Steps(1).ActionIndex,1);
            testCase.verifyEqual(solver.getValue(1),3,AbsTol=1e-12);
        end
    end
end
