classdef TDNewComponentsTest < OceanTest.BaseTestCase
    % TDNEWCOMPONENTSTEST  Unit tests for new Tabular and Deep TD components.
    
    methods (Test)
        function testAfterstateTarget(testCase)
            target = OceanStats.RL.TD.Targets.AfterstateTarget();
            testCase.verifyEqual(target.name(), "afterstate");
            testCase.verifyEqual(target.valueKind(), "state");
            testCase.verifyFalse(target.IsOffPolicy);
            
            % Compute non-terminal: r=1, qNext=[5], gamma=0.9 -> y = 1 + 0.9*5 = 5.5
            y = target.compute(1, 5, 1, false, 0.9, [], true);
            testCase.verifyEqual(y, 5.5, 'AbsTol', 1e-10);
            
            % Compute terminal: r=10 -> y = 10
            yTerm = target.compute(10, 5, 1, true, 0.9, [], true);
            testCase.verifyEqual(yTerm, 10, 'AbsTol', 1e-10);
        end
        
        function testDoubleSarsaAndExpectedSarsaTargets(testCase)
            sarsaTarget = OceanStats.RL.TD.Targets.DoubleSarsaTarget();
            testCase.verifyTrue(sarsaTarget.supportsDouble());
            testCase.verifyEqual(sarsaTarget.name(), "doublesarsa");
            
            ySarsa = sarsaTarget.compute(2, [10 20 30], 2, false, 0.9, [], true);
            testCase.verifyEqual(ySarsa, 2 + 0.9*20, 'AbsTol', 1e-10);
            
            expTarget = OceanStats.RL.TD.Targets.DoubleExpectedSarsaTarget();
            testCase.verifyTrue(expTarget.supportsDouble());
            testCase.verifyEqual(expTarget.name(), "doubleexpectedsarsa");
            
            probs = [0.2 0.5 0.3];
            valid = [true true true];
            yExp = expTarget.compute(1, [10 20 30], 1, false, 0.9, probs, valid);
            expVal = 0.2*10 + 0.5*20 + 0.3*30; % 2 + 10 + 9 = 21
            testCase.verifyEqual(yExp, 1 + 0.9*expVal, 'AbsTol', 1e-10);
        end
        
        function testTreeBackupTarget(testCase)
            treeTarget = OceanStats.RL.TD.Targets.TreeBackupTarget();
            testCase.verifyEqual(treeTarget.name(), "treebackup");
            testCase.verifyTrue(treeTarget.IsOffPolicy);
            
            probs = [0.1 0.6 0.3];
            valid = [true true true];
            qNext = [10 20 30];
            y = treeTarget.compute(1, qNext, 2, false, 0.9, probs, valid);
            expectedVal = 0.1*10 + 0.6*20 + 0.3*30; % 21
            testCase.verifyEqual(y, 1 + 0.9*expectedVal, 'AbsTol', 1e-10);
        end
        
        function testQSigmaTarget(testCase)
            qsigma = OceanStats.RL.TD.Targets.QSigmaTarget(0.5);
            testCase.verifyEqual(qsigma.name(), "qsigma");
            
            probs = [0.5 0.5];
            valid = [true true];
            qNext = [10 20];
            % sampVal (a=2) = 20, expVal = 0.5*10 + 0.5*20 = 15
            % blended b = 0.5*20 + 0.5*15 = 17.5
            % y = 1 + 0.9*17.5 = 16.75
            y = qsigma.compute(1, qNext, 2, false, 0.9, probs, valid);
            testCase.verifyEqual(y, 16.75, 'AbsTol', 1e-10);
        end
        
        function testReplayBuffer(testCase)
            buf = OceanStats.RL.TD.Replay.ReplayBuffer(10, 2);
            testCase.verifyEqual(buf.Size, 0);
            
            buf.store([1 2], 1, 5, [3 4], false);
            testCase.verifyEqual(buf.Size, 1);
            
            batch = buf.sampleBatch(1);
            testCase.verifyEqual(batch.Rewards(1), 5);
            testCase.verifyEqual(batch.States(1, :), [1 2]);
        end
        
        function testNeuralQAndNeuralV(testCase)
            % Linear layer network backbone mock
            net = OceanDL.nn.Linear(2, 4);
            nq = OceanStats.RL.TD.Values.NeuralQ(net, 4, 2);
            testCase.verifyEqual(nq.name(), "neuralq");
            testCase.verifyEqual(nq.kind(), "action");
            
            qVals = nq.estimates([1 0]);
            testCase.verifyEqual(numel(qVals), 4);
            
            netV = OceanDL.nn.Linear(2, 1);
            nv = OceanStats.RL.TD.Values.NeuralV(netV, 2);
            testCase.verifyEqual(nv.name(), "neuralv");
            testCase.verifyEqual(nv.kind(), "state");
            
            vVal = nv.estimates([1 0]);
            testCase.verifyEqual(numel(vVal), 1);
        end
    end
end
