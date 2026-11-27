classdef TestRunner
    % OCEANTEST.TESTRUNNER Facade for executing the test suite.
    %
    % EXAMPLES:
    %   OceanTest.TestRunner.runAll();
    %   OceanTest.TestRunner.runPackage('OceanStats');
    %   OceanTest.TestRunner.runTags('Unit');
    %
    % AUTHOR: Ocean Library System
    % DATE: 2026-02-15
    
    methods (Static)
        function results = runAll()
            % RUNALL Executes every test in the +OceanTest package.
            import matlab.unittest.TestSuite
            import matlab.unittest.TestRunner
            import matlab.unittest.plugins.TestReportPlugin
            
            fprintf('\n=== Running ALL OceanTest Suites ===\n');
            
            % Locate all tests that inherit from BaseTestCase
            suite = TestSuite.fromPackage('OceanTest', 'IncludingSubpackages', true);
            
            results = OceanTest.TestRunner.execute(suite);
        end
        
        function results = runPackage(packageName)
            % RUNPACKAGE Executes tests for a specific sub-package.
            % Usage: runPackage('OceanStats') -> runs +OceanTest/+OceanStats tests
            import matlab.unittest.TestSuite
            
            fullPackage = ['OceanTest.' packageName];
            fprintf('\n=== Running Tests for Package: %s ===\n', fullPackage);
            
            try
                suite = TestSuite.fromPackage(fullPackage, 'IncludingSubpackages', true);
                if isempty(suite)
                    warning('No tests found in %s', fullPackage);
                    results = [];
                    return;
                end
                results = OceanTest.TestRunner.execute(suite);
            catch ME
                fprintf('Error locating package: %s\n', ME.message);
                results = [];
            end
        end
        
        function results = runTags(tag)
            % RUNTAGS Executes tests marked with a specific tag.
            % Usage: runTags('Integration')
            import matlab.unittest.TestSuite
            import matlab.unittest.selectors.HasTag
            
            fprintf('\n=== Running Tests with Tag: %s ===\n', tag);
            
            suite = TestSuite.fromPackage('OceanTest', 'IncludingSubpackages', true);
            suite = suite.selectIf(HasTag(tag));
            
            if isempty(suite)
                warning('No tests found with tag: %s', tag);
                results = [];
                return;
            end
            
            results = OceanTest.TestRunner.execute(suite);
        end
    end
    
    methods (Static, Access = private)
        function results = execute(suite)
            import matlab.unittest.TestRunner
            import matlab.unittest.plugins.DiagnosticsRecordingPlugin
            import matlab.unittest.plugins.CodeCoveragePlugin
            
            % Create runner
            runner = TestRunner.withTextOutput();
            
            % Add plugins (Optional: could add coverage reports here)
            % runner.addPlugin(DiagnosticsRecordingPlugin);
            
            % Run
            results = runner.run(suite);
            
            % Summary
            OceanTest.TestRunner.printSummary(results);
        end
        
        function printSummary(results)
            failed = [results.Failed];
            passed = [results.Passed];
            incomplete = [results.Incomplete];
            
            n_fail = sum(failed);
            n_pass = sum(passed);
            n_inc  = sum(incomplete);
            
            fprintf('\n--------------------------------------\n');
            fprintf('Summary: %d Passed, %d Failed, %d Incomplete\n', n_pass, n_fail, n_inc);
            
            if n_fail > 0
                fprintf('\nFAILED TESTS:\n');
                for i = 1:length(results)
                    if results(i).Failed
                        fprintf('  [X] %s\n', results(i).Name);
                    end
                end
                fprintf('\nCheck usage of assertMatrixEquals or logic.\n');
            else
                fprintf('\nSUCCESS: All tests passed.\n');
            end
            fprintf('--------------------------------------\n');
        end
    end
end
