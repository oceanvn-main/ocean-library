function plan = buildfile
% BUILDFILE  Build tasks for Ocean Library.
%
%   buildtool test        Run the core test suites (also run in CI)
%   buildtool testAll     Run every suite under +OceanTest
%   buildtool check       Static analysis of all MATLAB source

plan = buildplan(localfunctions);
plan.DefaultTasks = "test";
end

function testTask(~)
% Run the core test suites: packages with no dataset, database or GPU needs.
runSuites(corePackages());
end

function testAllTask(~)
% Run every test suite under +OceanTest.
runSuites("");
end

function checkTask(~)
% Report Code Analyzer errors across the repository.
issues = codeIssues(pwd);
errs = issues.Issues(issues.Issues.Severity == "error", :);
fprintf("%d Code Analyzer issues, %d errors\n", height(issues.Issues), height(errs));
if ~isempty(errs)
    disp(errs(:, ["Location", "Description"]));
end
assert(isempty(errs), "OceanLibrary:build:codeErrors", ...
    "Code Analyzer reported %d errors.", height(errs));
end

function names = corePackages()
names = ["OceanMath", "OceanAlgo"];
end

function runSuites(packages)
import matlab.unittest.TestSuite
import matlab.unittest.TestRunner

addpath(pwd);
if all(packages == "")
    suite = TestSuite.fromPackage("OceanTest", "IncludingSubpackages", true);
else
    parts = arrayfun(@(name) TestSuite.fromPackage("OceanTest." + name, ...
        "IncludingSubpackages", true), packages, "UniformOutput", false);
    suite = [parts{:}];
end
runner = TestRunner.withTextOutput("OutputDetail", "Concise");
results = runner.run(suite);
fprintf("%d passed, %d failed, %d incomplete\n", ...
    nnz([results.Passed]), nnz([results.Failed]), nnz([results.Incomplete]));
assertSuccess(results);
end
