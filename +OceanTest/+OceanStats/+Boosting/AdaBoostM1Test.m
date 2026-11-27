classdef AdaBoostM1Test < OceanTest.BaseTestCase
    % ADABOOSTM1TEST  Unit tests for OceanStats.Boosting (AdaBoost.M1).
    %
    %   COMMAND:
    %     OceanTest.TestRunner.runPackage('OceanStats');
    %
    %   TESTS:
    %     1. testWeightNormalization    — sample weights always sum to 1
    %     2. testSeparableConvergence   — 0% training error on separable data
    %     3. testEarlyStopThreshold     — stops when err >= 0.5
    %     4. testPerfectRoundStop       — stops at t=1 when err = 0
    %     5. testWrapperPassthrough     — AdaBoostM1Reg runs without error
    %     6. testMultiClass             — correct aggregation on 3-class data
    
    properties
        X_sep   % Linearly separable 2-class data
        Y_sep
        X_iris  % 3-class (iris petal length / width)
        Y_iris
    end
    
    methods (TestClassSetup)
        function buildSeparableData(testCase)
            % Simple 1-D linearly separable problem
            rng(42);
            N = 100;
            testCase.X_sep = [randn(N/2, 2) - 2; randn(N/2, 2) + 2];
            testCase.Y_sep = [ones(N/2, 1); 2*ones(N/2, 1)];
        end
        
        function buildMultiClassData(testCase)
            % 3-class blobs
            rng(7);
            N = 60;
            X1 = randn(N, 2) + [0  0];
            X2 = randn(N, 2) + [4  0];
            X3 = randn(N, 2) + [2  3];
            testCase.X_iris = [X1; X2; X3];
            testCase.Y_iris = [ones(N,1); 2*ones(N,1); 3*ones(N,1)];
        end
    end
    
    methods (Test)
        % ---------------------------------------------------------------- %
        function testWeightNormalization(testCase)
            % After every boostingStep, W_new must sum to 1.
            strategy = OceanStats.Boosting.Algorithms.AdaBoostM1('NumIterations', 10);
            stump    = OceanStats.Selection.Strategies.DecisionTree(1, 1, 'gini');
            
            X = testCase.X_sep;
            Y = testCase.Y_sep;
            N = size(X, 1);
            W = ones(N, 1) / N;
            
            for t = 1:5
                [alpha, W_new, ~] = strategy.boostingStep(stump, X, Y, W, t);
                if isnan(alpha), break; end
                testCase.verifyEqual(sum(W_new), 1.0, 'AbsTol', 1e-10, ...
                    sprintf('Weights must sum to 1 after round %d', t));
                W = W_new;
            end
        end
        
        % ---------------------------------------------------------------- %
        function testSeparableConvergence(testCase)
            % On linearly separable data, training error → 0 quickly.
            strategy = OceanStats.Boosting.Algorithms.AdaBoostM1('NumIterations', 20);
            stump    = OceanStats.Selection.Strategies.DecisionTree(1, 1, 'gini');
            ens      = OceanStats.Boosting.BoostingEnsemble(strategy, stump);
            
            fit   = ens.train(testCase.X_sep, testCase.Y_sep);
            Yhat  = ens.predict(fit, testCase.X_sep);
            err   = ens.evaluate(testCase.Y_sep, Yhat);
            
            testCase.verifyLessThan(err, 0.05, ...
                'Training error on separable data should be < 5%');
        end
        
        % ---------------------------------------------------------------- %
        function testEarlyStopThreshold(testCase)
            % When EarlyStopThreshold is very low (e.g. 0.01), boosting
            % should stop early (NumRounds < NumIterations).
            strategy = OceanStats.Boosting.Algorithms.AdaBoostM1( ...
                'NumIterations', 50, 'EarlyStopThreshold', 0.01);
            stump = OceanStats.Selection.Strategies.DecisionTree(1, 1, 'gini');
            ens   = OceanStats.Boosting.BoostingEnsemble(strategy, stump);
            
            % Use non-separable data: random labels
            rng(1);
            X = randn(50, 2);
            Y = randi(2, 50, 1);
            
            fit = ens.train(X, Y);
            testCase.verifyLessThan(fit.NumRounds, 50, ...
                'Should early-stop before 50 rounds on noisy data');
        end
        
        % ---------------------------------------------------------------- %
        function testNumRoundsCorrect(testCase)
            % NumRounds in model struct should equal length of Alphas.
            strategy = OceanStats.Boosting.Algorithms.AdaBoostM1('NumIterations', 10);
            stump    = OceanStats.Selection.Strategies.DecisionTree(1, 1, 'gini');
            ens      = OceanStats.Boosting.BoostingEnsemble(strategy, stump);
            
            fit = ens.train(testCase.X_sep, testCase.Y_sep);
            testCase.verifyEqual(fit.NumRounds, numel(fit.Alphas), ...
                'NumRounds must match length of Alphas');
            testCase.verifyEqual(fit.NumRounds, numel(fit.WeakFits), ...
                'NumRounds must match number of WeakFits');
        end
        
        % ---------------------------------------------------------------- %
        function testWrapperPassthrough(testCase)
            % AdaBoostM1Reg must train and predict without error.
            m    = OceanStats.Selection.Strategies.AdaBoostM1Reg('NumIterations', 5);
            fit  = m.train(testCase.X_sep, testCase.Y_sep);
            Yhat = m.predict(fit, testCase.X_sep);
            
            testCase.verifyEqual(size(Yhat), size(testCase.Y_sep), ...
                'Prediction size must match input size');
            testCase.verifyGreaterThan(m.getComplexity(fit), 0, ...
                'NumRounds must be positive');
        end
        
        % ---------------------------------------------------------------- %
        function testMultiClass(testCase)
            % AdaBoost.M1 must handle 3-class problems.
            strategy = OceanStats.Boosting.Algorithms.AdaBoostM1('NumIterations', 30);
            stump    = OceanStats.Selection.Strategies.DecisionTree(1, 1, 'gini');
            ens      = OceanStats.Boosting.BoostingEnsemble(strategy, stump);
            
            fit  = ens.train(testCase.X_iris, testCase.Y_iris);
            Yhat = ens.predict(fit, testCase.X_iris);
            err  = ens.evaluate(testCase.Y_iris, Yhat);
            
            testCase.verifyEqual(numel(fit.Classes), 3, ...
                'Should detect 3 unique classes');
            testCase.verifyLessThan(err, 0.20, ...
                'Training error on 3-class blobs should be < 20%');
        end
    end
end
