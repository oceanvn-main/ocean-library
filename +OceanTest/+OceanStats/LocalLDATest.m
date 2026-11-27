classdef LocalLDATest < OceanTest.BaseTestCase
    % LOCALLDATEST Unit test for OceanStats.Supervised.Discriminant.LocalLDA
    %
    % MIGRATION NOTE:
    %   Replaces the old script: +OceanStats/+tests/test_local_lda.m
    %
    % COMMAND:
    %   OceanTest.TestRunner.runPackage('OceanStats');
    
    properties
        X_train
        y_train
        X_test
        y_test
    end
    
    methods (TestClassSetup)
        function loadDataset(testCase)
            % Load zip.train once for all tests
            data_train = testCase.loadData('zip.train');
            
            % Setup Training Data
            testCase.y_train = data_train(:, 1);
            testCase.X_train = data_train(:, 2:end);
            
            % Load zip.test (subset for speed)
            data_test = testCase.loadData('zip.test');
            n_test = 200; % Same as original script
            testCase.y_test = data_test(1:n_test, 1);
            testCase.X_test = data_test(1:n_test, 2:end);
        end
    end
    
    methods (Test)
        function testInitialization(testCase)
            import OceanStats.Supervised.Discriminant.LocalLDA
            
            model = LocalLDA('Bandwidth', 5.0);
            testCase.verifyEqual(model.Bandwidth, 5.0);
        end
        
        function testFitAndPredict(testCase)
            import OceanStats.Supervised.Discriminant.LocalLDA
            
            lam = 5.0;
            model = LocalLDA('Bandwidth', lam);
            
            % Fit
            model.fit(testCase.X_train, testCase.y_train);
            
            % Predict
            y_pred = model.predict(testCase.X_test);
            
            % Verify Output Dimensions
            testCase.verifyEqual(size(y_pred), size(testCase.y_test));
            
            % Verify Accuracy (Regression Test against known good value)
            % In the original script, lambda=5.0 gave decent accuracy.
            % We just want to ensure it runs without error and gives *reasonable* results.
            acc = mean(y_pred == testCase.y_test);
            testCase.verifyGreaterThan(acc, 0.8, 'Accuracy should be > 80% for lambda=5');
        end
        
        function testParameterSweep(testCase)
            % Replicates the loop from the original script but as a single test
            import OceanStats.Supervised.Discriminant.LocalLDA
            
            lambdas = [1.0, 15.0]; % Test extremes
            
            for k = 1:length(lambdas)
                lam = lambdas(k);
                model = LocalLDA('Bandwidth', lam);
                model.fit(testCase.X_train, testCase.y_train);
                y_pred = model.predict(testCase.X_test);
                
                % Just ensure it doesn't crash
                testCase.verifyNotEmpty(y_pred);
            end
        end
    end
end
