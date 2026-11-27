function results = run_shannon_suite()
    % RUN_SHANNON_SUITE Execute all Shannon Text Approximations test suites.
    %
    % EXAMPLES:
    %   results = OceanTest.run_shannon_suite();
    %
    % STRICTLY NO LaTeX formatting is used in console logging.

    import matlab.unittest.TestSuite
    import matlab.unittest.TestRunner

    fprintf('\n==================================================\n');
    fprintf('   RUNNING SHANNON TEXT APPROXIMATIONS TEST SUITE   \n');
    fprintf('==================================================\n\n');

    % Load specific test suites using metaclass ? operator
    try
        suite1 = TestSuite.fromClass(?OceanTest.OceanMath.test_shannon_processes);
        suite2 = TestSuite.fromClass(?OceanTest.OceanViz.test_shannon_renderers);
        suite3 = TestSuite.fromClass(?OceanTest.OceanViz.test_shannon_sandbox);
        
        fullSuite = [suite1, suite2, suite3];
    catch ME
        fprintf('Error loading test suites: %s\n', ME.message);
        rethrow(ME);
    end

    % Create runner with text output
    runner = TestRunner.withTextOutput();

    % Run
    results = runner.run(fullSuite);

    % Print custom summary (NO LaTeX)
    failed = [results.Failed];
    passed = [results.Passed];
    incomplete = [results.Incomplete];

    n_fail = sum(failed);
    n_pass = sum(passed);
    n_inc  = sum(incomplete);

    fprintf('\n==================================================\n');
    fprintf('   SHANNON TEST SUITE SUMMARY\n');
    fprintf('   Passed:     %d\n', n_pass);
    fprintf('   Failed:     %d\n', n_fail);
    fprintf('   Incomplete: %d\n', n_inc);
    fprintf('==================================================\n\n');

    if n_fail > 0
        fprintf('*** WARNING: %d tests failed! ***\n\n', n_fail);
    else
        fprintf('All Shannon tests passed successfully.\n\n');
    end
end
