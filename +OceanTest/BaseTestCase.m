classdef BaseTestCase < matlab.unittest.TestCase
    % OCEANTEST.BASETESTCASE
    % The foundational class for all unit tests in the Ocean Library.
    %
    % FEATURES:
    %   - Automatic Path Management: Ensures the library root is on the path.
    %   - Data Loading: Helpers to find and load standard datasets (zip.train, etc).
    %   - Custom Assertions: Specialized checks for numerical stability.
    %
    % AUTHOR: Ocean Library System
    % DATE: 2026-02-15
    
    properties (Constant)
        % Relative path from +OceanTest to the Library Root
        % Level 1: +OceanTest -> Root
        LibraryRootRelative = '..';
    end
    
    properties
        LibraryRoot
    end
    
    methods (TestClassSetup)
        function setLibraryPath(testCase)
            % 1. Determine the absolute path to the library root
            import matlab.unittest.fixtures.PathFixture
            
            [currentFolder, ~, ~] = fileparts(mfilename('fullpath'));
            testCase.LibraryRoot = fullfile(currentFolder, testCase.LibraryRootRelative);
            
            % 2. Add to path using a Fixture (Auto-teardown)
            % This ensures the path is clean after tests run.
            testCase.applyFixture(PathFixture(testCase.LibraryRoot));
        end
    end
    
    methods
        function data = loadData(testCase, filename)
            % LOADDATA Helper to load standard datasets from +OceanStats/data
            % Usage: data = testCase.loadData('zip.train');
            
            targetPath = fullfile(testCase.LibraryRoot, '+OceanStats', 'data', filename);
            
            if ~isfile(targetPath)
                error('OceanTest:DataNotFound', 'Could not find data file: %s', targetPath);
            end
            
            try
                % Attempt to load using standard loader
                [~, ~, ext] = fileparts(filename);
                if strcmp(ext, '.mat')
                    data = load(targetPath);
                else
                    data = load(targetPath, '-ascii'); % Assume text/ascii for .train/.test
                end
            catch ME
                error('OceanTest:LoadFailed', 'Failed to load %s: %s', filename, ME.message);
            end
        end
        
        function assertMatrixEquals(testCase, actual, expected, tol, msg)
            % ASSERTMATRIXEQUALS Custom assertion for matrix equality with tolerance.
            if nargin < 4, tol = 1e-6; end
            if nargin < 5, msg = "Matrices are not equal within tolerance."; end
            
            testCase.verifyEqual(size(actual), size(expected), ...
                msg + " (Dimension Mismatch)");
            
            diff = abs(actual - expected);
            max_diff = max(diff(:));
            
            testCase.verifyTrue(max_diff < tol, ...
                sprintf("%s (Max Diff: %.2e > Tol: %.2e)", msg, max_diff, tol));
        end
    end
end
