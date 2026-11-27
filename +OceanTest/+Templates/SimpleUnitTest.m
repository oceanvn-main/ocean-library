classdef SimpleUnitTest < OceanTest.BaseTestCase
    % SIMPLEUNITTEST Example of a basic Unit Test.
    %
    % HOW TO USE:
    %   1. Copy this file to your test folder (e.g., +OceanTest/+MyPackage).
    %   2. Rename class to MatchFileTest (e.g., MyFunctionTest).
    %   3. Write your methods.
    %
    % COMMAND:
    %   OceanTest.TestRunner.runTags('Unit');
    
    properties
        TestData
    end
    
    methods (TestClassSetup)
        function loadResources(testCase)
            % Runs once before any tests in this file
            % testCase.TestData = testCase.loadData('zip.train');
        end
    end
    
    methods (Test)
        function testBasicMath(testCase)
            % Verifies 1+1=2
            actual = 1 + 1;
            expected = 2;
            testCase.verifyEqual(actual, expected, 'Math failed');
        end
        
        function testMatrixHelper(testCase)
            % Verifies the custom matrix assertion
            A = eye(3);
            B = eye(3);
            B(1,1) = 1.000000001; % Tiny difference
            
            % Should pass with default tolerance (1e-6)
            testCase.assertMatrixEquals(A, B);
        end
        
        function testWithTags(testCase)
            % Tags allow selective running: TestRunner.runTags('Unit')
            testCase.verifyTrue(true);
        end
    end
end
